r"""Score a partition with a trained fused checkpoint and keep every line's score.

Phase 6 needs calibrated probabilities and a threshold; Phase 7 needs lines to explain;
Phase 8 needs ranking metrics the training loop never logged. All of them are functions of
the same table — *(unit, line) -> logit, label, scoreable* — so this script produces that
table once per checkpoint and partition (``vulnloc.training.line_scores``) and nothing
downstream loads the model again.

Three things it refuses to do, each because the alternative is a number nobody can trust:

**It does not take the model's configuration from the command line.** The run's own
resolved config, archived by ``tracking.start_run`` when the run began, is read back from
the tracking store and the model and dataset are rebuilt from that. Re-typing nine
overrides is how a checkpoint gets scored under a config it was not trained with.

**It does not keep scores that fail to reproduce the run.** On the run's own evaluation
split the re-scored metrics are compared with the best epoch the run logged, and a
disagreement beyond float noise is an error, raised before anything is written. A dump
that is not the scores the run reported is worse than no dump.

**It does not touch `test` or `clean_eval` by default.** ``Rules.md`` §5.3: the held-out
partitions are evaluated once, at the end, on the frozen configuration. They need
``+score.final=true``, and the run this script logs is then tagged ``held_out``.

**One thing it does on request, and says so in everything it writes: score a
checkpoint on text it was not trained on.** ``+representation.strip_comments=true``
strips every comment from the partition before scoring (``Memory.md`` §1.25: with
comments in place the corpus text carries a formatting leak the checkpoints read). For
a run trained with comments this is *evaluation-time* stripping: the metrics are not
expected to reproduce the run's own, so that check is skipped, the manifest records
``strip_comments`` and ``strip_comments_at_training`` separately, and the files go to
``scores_stripped/`` so they can never be mistaken for the scores the run reported.

Usage::

    python scripts/score_checkpoint.py +score.run_id=80fc5238fd6f4e3e852db9e6c604d930
    python scripts/score_checkpoint.py +score.run_id=80fc5238 +score.split=val
    python scripts/score_checkpoint.py +score.run_id=80fc5238 +score.split=test +score.final=true
    python scripts/score_checkpoint.py +score.run_id=80fc5238 +representation.strip_comments=true
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import hydra
import mlflow
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from vulnloc.representation.tokenization import LineTokenizer
from vulnloc.training.device import resolve_device
from vulnloc.training.fused_datasets import FusedLineDataset
from vulnloc.training.graph_datasets import load_node_vocab, load_subtoken_vocab
from vulnloc.training.line_scores import UnitScores, compare_metrics, save_scores
from vulnloc.training.metrics import MetricAccumulator
from vulnloc.training.semantic_cache import (
    SemanticFeatureCache,
    cache_inputs,
    corpus_fingerprints,
    encoder_revision,
)
from vulnloc.training.tracking import start_run
from vulnloc.utils.config import strip_comments_enabled
from vulnloc.utils.exceptions import ConfigError, ModelError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

REPO = Path(__file__).resolve().parent.parent

HELD_OUT_SPLITS = frozenset({"test", "clean_eval"})
"""Partitions that are evaluated once, at the end (``Rules.md`` §5.3)."""

REPRODUCED_KEYS = (
    "line/top_1_accuracy",
    "line/top_5_accuracy",
    "line/ifa_mean",
    "line/precision",
    "line/recall",
    "line/f1",
    "line/scored_units",
    "coverage/flagged_line_coverage_pct",
)
"""What a re-scoring of the run's own evaluation split must agree with the run about."""

DEFAULT_TOLERANCE = 1e-4
"""Smaller than one function's worth of Top-5 on `val` (1 / 482 = 0.0021) by a factor of 20."""


def _load_train_fused() -> Any:
    """Import ``scripts/train_fused.py`` by path; ``scripts/`` is not a package.

    The model is composed by the function that composed it for training, so the two cannot
    drift apart.
    """
    spec = importlib.util.spec_from_file_location(
        "vulnloc_scripts_train_fused", REPO / "scripts" / "train_fused.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_scoreable_split(split: str, *, final: bool) -> None:
    """Refuse a held-out partition unless the caller has said this is the final evaluation.

    Args:
        split: The partition to score.
        final: ``score.final``.

    Raises:
        ConfigError: For ``test`` or ``clean_eval`` without ``final``.
    """
    if split in HELD_OUT_SPLITS and not final:
        raise ConfigError(
            "this partition is evaluated once, at the end, on the frozen configuration",
            split=split,
            hint="Rules.md §5.3. Add '+score.final=true' only when that moment has come.",
        )


def resolve_run_id(prefix: str, checkpoint_root: Path) -> str:
    """Expand a run-id prefix against the checkpoint directories on disk.

    Args:
        prefix: A full run id or an unambiguous leading part of one.
        checkpoint_root: The directory holding one folder per run.

    Returns:
        The full 32-character run id.

    Raises:
        ConfigError: If nothing matches or more than one run does.
    """
    matches = (
        sorted(p.name for p in checkpoint_root.glob(f"{prefix}*") if p.is_dir())
        if checkpoint_root.is_dir()
        else []
    )
    if len(matches) != 1:
        raise ConfigError(
            "run id does not name exactly one checkpoint directory",
            run_id=prefix,
            matches=matches[:5],
            checkpoint_root=str(checkpoint_root),
        )
    return matches[0]


def refuse_unusable_run(run: Any) -> None:
    """Refuse to score a run that is not a result.

    Args:
        run: The MLflow run.

    Raises:
        ConfigError: If the run did not finish, is tagged void, was trained on a
            subsample, or is an overfit check.
    """
    params, tags = run.data.params, run.data.tags
    reasons = []
    if run.info.status != "FINISHED":
        reasons.append(f"status is {run.info.status}, not FINISHED")
    if tags.get("void"):
        reasons.append(f"tagged void: {tags['void']}")
    if params.get("train.limit", "0") not in {"0", "None", ""}:
        reasons.append(f"train.limit={params['train.limit']} — a subsample")
    if params.get("train.overfit", "False") == "True":
        reasons.append("train.overfit=true — a debugging instrument")
    if reasons:
        raise ConfigError("this run is not a result to score", run_id=run.info.run_id, why=reasons)


def load_run_config(run_id: str) -> DictConfig:
    """Read back the resolved config the run archived when it started.

    Args:
        run_id: The run.

    Returns:
        The run's full resolved config.

    Raises:
        ConfigError: If the run carries no archived config.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="config/resolved.yaml", dst_path=tmp
            )
        except mlflow.exceptions.MlflowException as error:
            raise ConfigError(
                "the run has no archived config to rebuild its model from", run_id=run_id
            ) from error
        loaded = OmegaConf.load(local)
    if not isinstance(loaded, DictConfig):
        raise ConfigError("the run's archived config is not a mapping", run_id=run_id)
    return loaded


def evaluation_config(run_cfg: DictConfig, cfg: DictConfig) -> tuple[DictConfig, bool]:
    """The config a partition is built from, and whether its text differs from training.

    Args:
        run_cfg: The run's archived config.
        cfg: This invocation's config.

    Returns:
        ``(config, text_changed)``. The run's own config, unchanged, unless this
        invocation asks for comments to be stripped and the run was trained with them
        in place — then a copy with ``representation.strip_comments`` set, and ``True``.
    """
    if not strip_comments_enabled(cfg) or strip_comments_enabled(run_cfg):
        return run_cfg, False
    stripped = OmegaConf.merge(run_cfg, {"representation": {"strip_comments": True}})
    assert isinstance(stripped, DictConfig)
    return stripped, True


def build_dataset(run_cfg: DictConfig, split: str, train_fused: Any) -> FusedLineDataset:
    """Build ``split`` exactly as the run's own evaluation set was built.

    Mirrors ``scripts/train_fused.py::train`` line for line, from the run's config: same
    tokenizer, vocabularies, graph builder, line cap and population filter, and the same
    semantic cache if the run read one. ``representation.strip_comments`` is read from
    the same config — pass :func:`evaluation_config`'s result to strip at evaluation —
    and it is part of the semantic cache's key, so features cached from text with
    comments are never joined to graphs built without them.

    Args:
        run_cfg: The run's resolved config.
        split: The partition.
        train_fused: The imported training script.

    Returns:
        The dataset, every unit of it.
    """
    fusion_cfg = train_fused.require_group(run_cfg, "fusion", hint="+fusion=cross_attention")
    gnn_cfg = train_fused.require_group(run_cfg, "gnn", hint="+gnn=gcn")
    vocab, _ = load_node_vocab(
        str(gnn_cfg.vocab_path), expect_digest=str(gnn_cfg.expect_vocab_digest) or None
    )
    subtoken_vocab: dict[str, int] | None = None
    if bool(OmegaConf.select(run_cfg, "gnn.use_subtokens")):
        subtoken_vocab, _ = load_subtoken_vocab(
            str(gnn_cfg.subtoken_vocab_path),
            expect_digest=str(gnn_cfg.expect_subtoken_digest) or None,
        )
    tokenizer = LineTokenizer(
        AutoTokenizer.from_pretrained(str(run_cfg.llm.tokenizer_name)),
        max_length=int(run_cfg.llm.max_length),
        stride=int(run_cfg.llm.stride),
        windowing=bool(run_cfg.llm.windowing),
    )
    semantic_cache: SemanticFeatureCache | None = None
    if train_fused.semantic_cache_enabled(run_cfg) and str(fusion_cfg.kind) != "gnn_only":
        fingerprint, shard_digest = corpus_fingerprints(run_cfg, split)
        semantic_cache = SemanticFeatureCache.load(
            str(run_cfg.train.semantic_cache.dir),
            split,
            cache_inputs(
                run_cfg,
                split=split,
                split_fingerprint=fingerprint,
                shard_digest=shard_digest,
                encoder_revision=encoder_revision(str(run_cfg.llm.model_name)),
            ),
        )
    return FusedLineDataset(
        Path(run_cfg.data.cache.output_dir),
        split,
        tokenizer,
        vocab,
        subtoken_vocab=subtoken_vocab,
        builder=str(gnn_cfg.builder),
        language=str(gnn_cfg.language),
        n_lines=int(run_cfg.llm.n_lines),
        limit=0,
        line_labelled_only=bool(run_cfg.train.line_labelled_only),
        normalize_whitespace=bool(run_cfg.representation.normalize_whitespace),
        strip_comments=strip_comments_enabled(run_cfg),
        semantic_cache=semantic_cache,
    )


@torch.no_grad()
def score_units(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    threshold: float,
    use_amp: bool,
) -> tuple[list[UnitScores], dict[str, float]]:
    """Run the model over ``loader`` and keep each unit's lines.

    The forward pass and the metric accumulation are the training loop's ``evaluate``,
    statement for statement; the only addition is that the logits are kept.

    Args:
        model: The model, weights loaded.
        loader: Batches in a fixed order.
        device: Where to run.
        threshold: Decision threshold for the threshold-dependent metrics.
        use_amp: Run under autocast, as the run's own evaluation did.

    Returns:
        ``(units, metrics)`` — per-unit line scores, and the metrics over all of them.
    """
    model.eval()
    accumulator = MetricAccumulator(threshold=threshold)
    units: list[UnitScores] = []
    for raw in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw.items()
        }
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(batch)
        line_logits = output["line_logits"].float()  # [B, L]
        accumulator.update(
            line_logits,
            batch["line_labels"],
            output["line_mask"],
            batch["line_valid"],
            output["function_logits"].float(),
            batch["targets"],
        )
        valid = batch["line_valid"].cpu()  # [B, L]
        logits = line_logits.cpu()
        labels = (batch["line_labels"] > 0).cpu()
        scoreable = (output["line_mask"] & batch["line_valid"]).cpu()
        for index, unit_id in enumerate(batch["unit_ids"]):
            keep = valid[index]
            units.append(
                UnitScores(
                    unit_id=str(unit_id),
                    logits=logits[index][keep].numpy(),
                    labels=labels[index][keep].numpy().astype("int8"),
                    scoreable=scoreable[index][keep].numpy(),
                )
            )
    return units, accumulator.result()


def score(cfg: DictConfig) -> dict[str, Any]:
    """Score one partition with one run's best checkpoint.

    Args:
        cfg: Composed config. Reads ``score.run_id``, ``score.split``, ``score.final``,
            ``score.output_dir``, ``score.checkpoint_dir`` and ``score.tolerance``; the
            tracking store and device come from the base config, everything about the
            model from the run.

    Returns:
        The manifest written beside the score file.

    Raises:
        ConfigError: If the run or partition may not be scored.
        ModelError: If the re-scoring does not reproduce the run's logged metrics.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )
    transformers.logging.set_verbosity_error()
    requested = OmegaConf.select(cfg, "score.run_id")
    if not requested:
        raise ConfigError("no run to score", hint="Pass '+score.run_id=<run id>'.")
    final = bool(OmegaConf.select(cfg, "score.final"))
    tolerance = float(OmegaConf.select(cfg, "score.tolerance") or DEFAULT_TOLERANCE)
    requested_output = OmegaConf.select(cfg, "score.output_dir")

    checkpoint_root = Path(
        str(OmegaConf.select(cfg, "score.checkpoint_dir") or "./artifacts/checkpoints")
    )

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    run_id = resolve_run_id(str(requested), checkpoint_root)
    run = mlflow.get_run(run_id)
    refuse_unusable_run(run)
    trained_cfg = load_run_config(run_id)
    run_cfg, text_changed = evaluation_config(trained_cfg, cfg)
    output_dir = Path(
        str(requested_output or f"./artifacts/phase6/scores{'_stripped' if text_changed else ''}")
    )
    split = str(OmegaConf.select(cfg, "score.split") or run_cfg.train.eval_split)
    require_scoreable_split(split, final=final)

    train_fused = _load_train_fused()
    device = resolve_device(str(cfg.device.kind), index=int(cfg.device.index))
    dataset = build_dataset(run_cfg, split, train_fused)
    gnn_cfg = train_fused.require_group(run_cfg, "gnn", hint="+gnn=gcn")
    _, vocab_meta = load_node_vocab(
        str(gnn_cfg.vocab_path), expect_digest=str(gnn_cfg.expect_vocab_digest) or None
    )
    subtoken_size = 0
    if bool(OmegaConf.select(run_cfg, "gnn.use_subtokens")):
        _, subtoken_meta = load_subtoken_vocab(
            str(gnn_cfg.subtoken_vocab_path),
            expect_digest=str(gnn_cfg.expect_subtoken_digest) or None,
        )
        subtoken_size = int(subtoken_meta.get("size", 0))
    model = train_fused.build_model(
        run_cfg, vocab_size=vocab_meta["size"], device=device, subtoken_vocab_size=subtoken_size
    )

    checkpoint_dir = checkpoint_root / run_id
    checkpoint = checkpoint_dir / "best.pt"
    if not checkpoint.is_file():
        raise ConfigError("the run has no best.pt", run_id=run_id, checkpoint=str(checkpoint))
    # Our own artefact, written by this repository's training loop (Rules.md §2).
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    del payload

    loader = train_fused.build_loader(dataset, run_cfg, shuffle=False, seed=int(run_cfg.seed))
    units, metrics = score_units(
        model,
        loader,
        device,
        threshold=float(run_cfg.train.threshold),
        use_amp=bool(run_cfg.train.amp) and device.type == "cuda",
    )

    reproduced: bool | None = None
    # On text the run was not trained on there is nothing logged to reproduce.
    if split == str(run_cfg.train.eval_split) and not text_changed:
        summary = json.loads((checkpoint_dir / "summary.json").read_text(encoding="utf-8"))
        problems = compare_metrics(
            metrics, summary.get("best") or {}, REPRODUCED_KEYS, tolerance=tolerance
        )
        if problems:
            raise ModelError(
                "the re-scoring does not reproduce the metrics this run logged; "
                "nothing was written",
                run_id=run_id,
                split=split,
                disagreements=problems,
            )
        reproduced = True

    score_path = output_dir / f"{run_id}_{split}.npz"
    save_scores(score_path, units)
    manifest: dict[str, Any] = {
        "generated_by": "scripts/score_checkpoint.py",
        "source_run": run_id,
        "seed": int(run_cfg.seed),
        "fusion_kind": str(run_cfg.fusion.kind),
        # Ablation A6's arms share a fusion kind and differ only here.
        "fusion_direction": str(OmegaConf.select(run_cfg, "fusion.args.direction") or ""),
        "frozen": bool(run_cfg.llm.frozen),
        "split": split,
        "held_out": split in HELD_OUT_SPLITS,
        "strip_comments": strip_comments_enabled(run_cfg),
        "strip_comments_at_training": strip_comments_enabled(trained_cfg),
        "checkpoint": str(checkpoint),
        "units": len(units),
        "lines": int(sum(unit.logits.shape[0] for unit in units)),
        "scoreable_lines": int(sum(int(unit.scoreable.sum()) for unit in units)),
        "flagged_lines": int(sum(int(unit.labels.sum()) for unit in units)),
        "threshold": float(run_cfg.train.threshold),
        "reproduces_logged_metrics": reproduced,
        "tolerance": tolerance,
        "metrics": metrics,
        "scores": str(score_path),
    }
    manifest_path = score_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with start_run(cfg, run_name=f"score-{split}"):
        mlflow.set_tags(
            {"artefact": "scores", "source_run": run_id, "held_out": str(split in HELD_OUT_SPLITS)}
        )
        mlflow.log_params(
            {
                "score.source_run": run_id,
                "score.split": split,
                "score.fusion_kind": manifest["fusion_kind"],
                "score.seed": manifest["seed"],
                "score.units": manifest["units"],
                "score.reproduces_logged_metrics": str(reproduced),
                "score.strip_comments": str(manifest["strip_comments"]),
                "score.strip_comments_at_training": str(manifest["strip_comments_at_training"]),
            }
        )
        mlflow.log_metrics({f"score/{key}": float(value) for key, value in metrics.items()})
        mlflow.log_artifact(str(manifest_path), artifact_path="scores")
    _LOGGER.info(
        "Scores written",
        extra={
            "source_run": run_id,
            "split": split,
            "units": manifest["units"],
            "lines": manifest["lines"],
            "top_5": round(metrics["line/top_5_accuracy"], 4),
            "top_1": round(metrics["line/top_1_accuracy"], 4),
            "reproduces_logged_metrics": reproduced,
            "scores": str(score_path),
        },
    )
    return manifest


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    score(cfg)


if __name__ == "__main__":
    main()
