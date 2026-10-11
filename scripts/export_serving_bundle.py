r"""Package one checkpoint for serving, and prove the package answers as the checkpoint did.

A bundle (``vulnloc.serving.bundle``) is everything the service needs and nothing it has to
go and find: the weights, the encoder's architecture, the tokenizer, the vocabularies the
structural branch was trained on, the checkpoint's own calibrator and threshold, the
explainer's settings, a model card, and a manifest carrying the provenance — run id, git
sha, config hash, split fingerprint — and a SHA-256 for every file.

**The export is not finished until the bundle has been checked against the checkpoint.**
The service featurizes a function with L1 code of its own, because it may not import the
training layer; nothing but a measurement says the two agree. So the freshly written
bundle is loaded through the serving path and asked about a sample of ``val`` functions,
drawn by content hash, and two things are compared:

* **its input, exactly.** The batch it builds for each function must equal, tensor for
  tensor, the batch ``collate_fused`` builds for the same function from the run's own
  evaluation dataset. There is no tolerance in this one;
* **its output, within a bound.** The line logits must sit within ``export.tolerance`` of
  the ones ``scripts/score_checkpoint.py`` stored — which were themselves checked against
  the run's logged metrics — and agree on which lines the model could score. The stored
  scores were made on a GPU under mixed precision and the service runs in full precision
  on a CPU, so they are not bitwise equal.

A bundle that fails either is removed. The largest difference found, its median and its
99th percentile are written into the manifest.

``val`` only. A bundle is never checked against a held-out partition: that would be a
second scoring of it.

Usage::

    python scripts/export_serving_bundle.py +explainer=template \
        +export.run_id=ba7dcc5ff8354cc4b6d7e7b76f876fd9 \
        +export.scores_dir=./artifacts/phase6/scores_retrained \
        +export.calibration_report=./artifacts/phase6/retrained/calibration_report.json
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
import numpy as np
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoTokenizer

from vulnloc.data.cache import iter_shard
from vulnloc.representation import CodeUnit
from vulnloc.serving.bundle import MANIFEST_NAME, SCHEMA_VERSION, file_digest
from vulnloc.serving.inference import Analyzer, featurize
from vulnloc.training.fused_datasets import collate_fused
from vulnloc.training.graph_datasets import load_node_vocab, load_subtoken_vocab
from vulnloc.training.line_scores import load_scores
from vulnloc.training.semantic_cache import corpus_fingerprints
from vulnloc.training.tracking import start_run
from vulnloc.utils.config import run_id_text, strip_comments_enabled
from vulnloc.utils.exceptions import ConfigError, ModelError
from vulnloc.utils.hashing import stable_sample
from vulnloc.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_LOGGER = get_logger(__name__)

REPO = Path(__file__).resolve().parent.parent
CHECK_SPLIT = "val"
"""The only partition a bundle is checked against."""

DEFAULT_TOLERANCE = 0.2
"""Largest line-logit difference from the stored scores a bundle may show.

Set between two measurements made on 2026-10-10 with ``ba7dcc5f…``. The same checkpoint,
full precision on a CPU against its stored mixed-precision GPU scores: at most 0.090 over
3,272 lines of 48 functions (median 0.0004, 99th percentile 0.005). A *different*
checkpoint — another seed of the same arm — differs by more than 0.33 on some line of
every one of the 488 ``val`` functions. ``scripts/explain_checkpoint.py``'s 0.05 is for a
re-run on the same device and precision, and is too tight here.
"""


def _load_script(name: str) -> Any:
    """Import a sibling script by path; ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        f"vulnloc_scripts_{name}", REPO / "scripts" / f"{name}.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_manifest(run_cfg: DictConfig, *, node_size: int, subtoken_size: int) -> dict[str, Any]:
    """The part of the manifest that says what the model is.

    Args:
        run_cfg: The run's archived, resolved config.
        node_size: Node-type embedding rows the run was built with.
        subtoken_size: Identifier subtoken rows, ``0`` when the run used none.

    Returns:
        The ``text``, ``llm``, ``graph``, ``gnn``, ``fusion`` and ``head`` blocks.
    """
    llm, gnn, fusion, head = run_cfg.llm, run_cfg.gnn, run_cfg.fusion, run_cfg.head
    return {
        "text": {
            "normalize_whitespace": bool(run_cfg.representation.normalize_whitespace),
            "strip_comments": strip_comments_enabled(run_cfg),
        },
        "llm": {
            "model_name": str(llm.model_name),
            "max_length": int(llm.max_length),
            "stride": int(llm.stride),
            "windowing": bool(llm.windowing),
            "n_lines": int(llm.n_lines),
            "dropout": float(llm.dropout),
        },
        "graph": {"builder": str(gnn.builder), "language": str(gnn.language)},
        "gnn": {
            "kind": str(gnn.kind),
            "hidden_dim": int(gnn.hidden_dim),
            "depth": int(gnn.depth),
            "heads": int(gnn.heads),
            "dropout": float(gnn.dropout),
            "residual": bool(gnn.residual),
            "layer_norm": bool(gnn.layer_norm),
            "edge_handling": str(gnn.edge_handling),
            "node_vocab_size": int(node_size),
            "subtoken_vocab_size": int(subtoken_size),
        },
        "fusion": {
            "kind": str(fusion.kind),
            "args": OmegaConf.to_container(fusion.args, resolve=True),
            "trim_line_axis": bool(fusion.trim_line_axis),
        },
        "head": {"hidden_dim": int(head.hidden_dim), "dropout": float(head.dropout)},
    }


def write_files(
    bundle_dir: Path,
    run_cfg: DictConfig,
    *,
    checkpoint: Path,
    node_vocab: Path,
    subtoken_vocab: Path | None,
    model_card: Path,
) -> dict[str, str]:
    """Write every file of the bundle except its manifest.

    Args:
        bundle_dir: The (empty) bundle directory.
        run_cfg: The run's archived config.
        checkpoint: The run's ``best.pt``.
        node_vocab: The node-type vocabulary artefact the run was built on.
        subtoken_vocab: The subtoken vocabulary artefact, or ``None``.
        model_card: The model card to ship with the weights.

    Returns:
        ``relative path -> SHA-256`` for every file written.
    """
    # Our own artefact, written by this repository's training loop (Rules.md §2). What is
    # shipped is the state dict alone, so the service can load it with `weights_only`.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    torch.save(payload["model_state"], bundle_dir / "model.pt")
    del payload
    AutoConfig.from_pretrained(str(run_cfg.llm.model_name)).save_pretrained(bundle_dir / "encoder")
    AutoTokenizer.from_pretrained(str(run_cfg.llm.tokenizer_name)).save_pretrained(
        bundle_dir / "tokenizer"
    )
    shutil.copyfile(node_vocab, bundle_dir / "node_vocab.json")
    if subtoken_vocab is not None:
        shutil.copyfile(subtoken_vocab, bundle_dir / "subtoken_vocab.json")
    shutil.copyfile(model_card, bundle_dir / "MODEL_CARD.md")
    return {
        path.relative_to(bundle_dir).as_posix(): file_digest(path)
        for path in sorted(bundle_dir.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    }


def check_against_scores(
    analyzer: Analyzer,
    sources: Mapping[str, tuple[str, str]],
    stored: Mapping[str, Any],
    unit_ids: Sequence[str],
) -> dict[str, Any]:
    """Ask the bundle about scored functions and compare with what was stored.

    Args:
        analyzer: The bundle, loaded through the serving path.
        sources: ``unit_id -> (raw source, language)``.
        stored: ``unit_id -> stored scores``.
        unit_ids: The units to check.

    Returns:
        ``{"units", "lines", "max_logit_deviation", "median_logit_deviation",
        "p99_logit_deviation", "logit_spread", "scoreable_mismatches", "same_top_line",
        "seconds_median", "seconds_max"}``. ``logit_spread`` is the standard deviation of
        the stored logits over the same lines — the scale a deviation is read against.
        ``same_top_line`` counts the units whose top-ranked line is the same under both;
        informative only: the stored logits are half-precision, so exact ties move
        (``Memory.md`` §7 #63).
    """
    mismatches, same_top = 0, 0
    deviations: list[np.ndarray] = []
    kept_logits: list[np.ndarray] = []
    seconds: list[float] = []
    for unit_id in unit_ids:
        source, language = sources[unit_id]
        analysis = analyzer.analyze(source, language=language, unit_id=unit_id)
        saved = stored[unit_id]
        usable = saved.scoreable.astype(bool)
        if analysis.logits.shape != saved.logits.shape or not np.array_equal(
            analysis.scoreable, usable
        ):
            mismatches += 1
            continue
        if usable.any():
            deviations.append(np.abs(analysis.logits[usable] - saved.logits[usable]))
            kept_logits.append(saved.logits[usable])
            fresh = np.where(usable, analysis.logits, -np.inf)
            kept = np.where(usable, saved.logits, -np.inf)
            same_top += int(np.argmax(fresh) == np.argmax(kept))
        seconds.append(float(analysis.diagnostics["seconds"]))
    every = np.concatenate(deviations) if deviations else np.zeros(0)
    return {
        "units": len(unit_ids),
        "lines": int(every.size),
        "max_logit_deviation": float(every.max()) if every.size else 0.0,
        "median_logit_deviation": float(np.median(every)) if every.size else 0.0,
        "p99_logit_deviation": float(np.percentile(every, 99)) if every.size else 0.0,
        "logit_spread": float(np.concatenate(kept_logits).std()) if kept_logits else 0.0,
        "scoreable_mismatches": mismatches,
        "same_top_line": same_top,
        "seconds_median": float(np.median(seconds)) if seconds else None,
        "seconds_max": float(np.max(seconds)) if seconds else None,
    }


def check_batches(
    analyzer: Analyzer,
    training_units: Mapping[str, Any],
    sources: Mapping[str, tuple[str, str]],
    unit_ids: Sequence[str],
) -> list[str]:
    """Compare the batch the service builds with the batch training collated, exactly.

    This is the check with no tolerance in it. The logits can differ by precision; the
    *input* cannot: for the same function the serving path and ``collate_fused`` must
    produce the same tensors, or the service is scoring something the model was not
    trained on.

    Args:
        analyzer: The bundle, loaded through the serving path.
        training_units: ``unit_id -> FusedUnit`` from the run's own evaluation dataset.
        sources: ``unit_id -> (raw source, language)``.
        unit_ids: The units to check.

    Returns:
        ``"<unit_id>: <key>"`` for every tensor that differs; empty when none does.
    """
    differing: list[str] = []
    for unit_id in unit_ids:
        source, language = sources[unit_id]
        unit = CodeUnit.from_source(
            unit_id,
            source,
            language=language,
            normalize_whitespace=analyzer.normalize_whitespace,
            strip_comments=analyzer.strip_comments,
        )
        served, _, _ = featurize(
            unit,
            analyzer.tokenizer,
            analyzer.parser,
            analyzer.graph_builder,
            analyzer.node_vocab,
            analyzer.subtoken_vocab,
            language_policy=analyzer.language_policy,
            n_lines=analyzer.n_lines,
        )
        trained = collate_fused([training_units[unit_id]])
        for key, value in served.items():
            same = (
                torch.equal(value, trained[key])
                if torch.is_tensor(value)
                else value == trained[key]
            )
            if not same:
                differing.append(f"{unit_id}: {key}")
        missing = {k for k in trained if k.startswith(("sem_", "str_"))} - set(served)
        differing += [f"{unit_id}: {key} (absent from the served batch)" for key in sorted(missing)]
    return differing


def build_bundle(cfg: DictConfig) -> tuple[dict[str, Any], Path]:
    """Build a bundle for one run and check it against the run's stored scores.

    Args:
        cfg: Composed config. Requires ``+explainer=<name>``, ``export.run_id``,
            ``export.scores_dir`` and ``export.calibration_report``; reads
            ``export.output_dir`` (default ``./artifacts/serving/<run id, 8 characters>``),
            ``export.model_card`` (default ``./docs/MODEL_CARD.md``), ``export.check_units``
            (default 16), ``export.check_seed``, ``export.tolerance`` and
            ``export.checkpoint_dir``.

    Returns:
        The manifest, as written to ``bundle.json``, and the bundle directory.

    Raises:
        ConfigError: For a missing input, a run that is not a result, or an output
            directory that already exists.
        ModelError: If the bundle does not reproduce the checkpoint's stored scores. The
            bundle directory is removed.
    """
    transformers.logging.set_verbosity_error()  # type: ignore[no-untyped-call]

    def option(name: str, default: Any = None) -> Any:
        value = OmegaConf.select(cfg, f"export.{name}")
        return default if value is None else value

    explainer_cfg = OmegaConf.select(cfg, "explainer")
    if explainer_cfg is None:
        raise ConfigError("no explainer selected", hint="Pass '+explainer=template'.")
    for required in ("run_id", "scores_dir", "calibration_report"):
        if option(required) is None:
            raise ConfigError("a required option is missing", option=f"export.{required}")

    scoring, explaining = _load_script("score_checkpoint"), _load_script("explain_checkpoint")
    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    checkpoint_root = Path(str(option("checkpoint_dir", "./artifacts/checkpoints")))
    run_id = scoring.resolve_run_id(run_id_text(option("run_id")), checkpoint_root)
    run = mlflow.get_run(run_id)
    scoring.refuse_unusable_run(run)
    run_cfg = scoring.load_run_config(run_id)

    gnn = run_cfg.gnn
    _, node_meta = load_node_vocab(
        str(gnn.vocab_path), expect_digest=str(gnn.expect_vocab_digest) or None
    )
    uses_subtokens = bool(OmegaConf.select(run_cfg, "gnn.use_subtokens"))
    subtoken_size = 0
    if uses_subtokens:
        _, subtoken_meta = load_subtoken_vocab(
            str(gnn.subtoken_vocab_path), expect_digest=str(gnn.expect_subtoken_digest) or None
        )
        subtoken_size = int(subtoken_meta.get("size", 0))
    calibrator, threshold = explaining.load_calibration(
        Path(str(option("calibration_report"))), run_id
    )
    model_card = Path(str(option("model_card", "./docs/MODEL_CARD.md")))
    if not model_card.is_file():
        raise ConfigError(
            "a checkpoint is not released without its model card (Rules.md §7.7)",
            path=str(model_card),
        )
    scores_path = Path(str(option("scores_dir"))) / f"{run_id}_{CHECK_SPLIT}.npz"
    if not scores_path.is_file():
        raise ConfigError(
            "the run has no stored validation scores to check a bundle against",
            path=str(scores_path),
            hint="Run scripts/score_checkpoint.py for this run first.",
        )

    bundle_dir = Path(str(option("output_dir", f"./artifacts/serving/{run_id[:8]}")))
    if bundle_dir.exists():
        raise ConfigError(
            "the output directory already exists; a bundle is never edited in place",
            path=str(bundle_dir),
        )
    bundle_dir.mkdir(parents=True)
    files = write_files(
        bundle_dir,
        run_cfg,
        checkpoint=checkpoint_root / run_id / "best.pt",
        node_vocab=Path(str(gnn.vocab_path)),
        subtoken_vocab=Path(str(gnn.subtoken_vocab_path)) if uses_subtokens else None,
        model_card=model_card,
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "run_id": run_id,
        "git_sha": str(run.data.tags.get("git_sha", "")),
        "config_hash": str(run.data.tags.get("config_hash", "")),
        "seed": int(run_cfg.seed),
        # Training runs do not tag it; it is read from the split artefact the run's own
        # config points at, which is the one its datasets were built from.
        "split_fingerprint": corpus_fingerprints(run_cfg, CHECK_SPLIT)[0],
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **model_manifest(run_cfg, node_size=int(node_meta["size"]), subtoken_size=subtoken_size),
        "calibration": {
            **calibrator.to_dict(),
            "threshold": threshold,
            "fitted_on": CHECK_SPLIT,
            "report": Path(str(option("calibration_report"))).as_posix(),
        },
        "explainer": {
            "name": str(explainer_cfg.name),
            "args": OmegaConf.to_container(explainer_cfg.args, resolve=True),
            "evidence": OmegaConf.to_container(explainer_cfg.evidence, resolve=True),
        },
        "vocabularies": {
            "node_digest": str(node_meta["digest"]),
            "subtoken_digest": str(subtoken_meta["digest"]) if uses_subtokens else "",
        },
        "files": files,
    }
    manifest_path = bundle_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    stored = {unit.unit_id: unit for unit in load_scores(scores_path)}
    chosen = sorted(
        stable_sample(
            sorted(stored),
            min(int(option("check_units", 16)), len(stored)),
            seed=int(option("check_seed", 1337)),
        )
    )
    wanted = set(chosen)
    sources = {
        unit.unit_id: (unit.source, unit.language)
        for unit in iter_shard(Path(cfg.data.cache.output_dir), CHECK_SPLIT)
        if unit.unit_id in wanted
    }
    tolerance = float(option("tolerance", DEFAULT_TOLERANCE))
    analyzer = Analyzer.load(bundle_dir)
    training_set = scoring.build_dataset(run_cfg, CHECK_SPLIT, scoring._load_train_fused())
    training_units = {
        training_set[index].unit_id: training_set[index] for index in range(len(training_set))
    }
    differing = check_batches(analyzer, training_units, sources, chosen)
    check = check_against_scores(analyzer, sources, stored, chosen)
    check.update(
        {
            "split": CHECK_SPLIT,
            "tolerance": tolerance,
            "scores": scores_path.as_posix(),
            "batch_tensors_differing": len(differing),
        }
    )
    if differing or check["scoreable_mismatches"] or check["max_logit_deviation"] > tolerance:
        shutil.rmtree(bundle_dir)
        raise ModelError(
            "the bundle does not reproduce the checkpoint; it was removed",
            run_id=run_id,
            differing=differing[:8],
            **check,
        )
    manifest["verification"] = check
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest, bundle_dir


def export(cfg: DictConfig) -> dict[str, Any]:
    """Build a bundle, check it, and log the run that carries it.

    Args:
        cfg: Composed config — see :func:`build_bundle`.

    Returns:
        The manifest.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )
    manifest, bundle_dir = build_bundle(cfg)
    run_id, check = str(manifest["run_id"]), manifest["verification"]
    manifest_path = bundle_dir / MANIFEST_NAME
    with start_run(cfg, run_name=f"serving-bundle-{run_id[:8]}") as export_run:
        mlflow.set_tags({"artefact": "serving-bundle", "source_run": run_id, "split": CHECK_SPLIT})
        mlflow.log_params(
            {
                "export.source_run": run_id,
                "export.bundle_dir": bundle_dir.as_posix(),
                "export.check_units": check["units"],
                "export.tolerance": check["tolerance"],
                "export.threshold": manifest["calibration"]["threshold"],
            }
        )
        mlflow.log_metrics(
            {
                "max_logit_deviation": check["max_logit_deviation"],
                "same_top_line": float(check["same_top_line"]),
                "checked_lines": float(check["lines"]),
                "seconds_median": float(check["seconds_median"] or 0.0),
            }
        )
        mlflow.log_artifact(str(manifest_path), artifact_path="serving")
    _LOGGER.info(
        "Serving bundle written and checked",
        extra={
            "bundle": str(bundle_dir),
            "source_run": run_id,
            "export_run": export_run,
            "max_logit_deviation": round(check["max_logit_deviation"], 5),
            "same_top_line": f"{check['same_top_line']} of {check['units']}",
        },
    )
    return manifest


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    export(cfg)


if __name__ == "__main__":
    main()
