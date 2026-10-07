r"""Do a checkpoint's line scores ride on a formatting tell? — the model-side leak audit.

``audit_label_leakage.py`` asks whether a line's label can be read off its formatting. This
script asks the question that follows: **did the trained models read it?** It needs no
model — only the per-line scores ``score_checkpoint.py`` stored and the partition's text —
and reports, per checkpoint and per arm:

* ``all`` — Top-1 / Top-5 over every unit, recomputed from the stored scores. This must
  equal what the run logged; it is how the reader knows the rest is computed on the same
  numbers.
* ``tell_units`` — units in which the tell fires on a flagged line, i.e. where the answer
  is written in the formatting. ``top1_is_tell_line`` counts how often the checkpoint's
  top-ranked line there *is* such a line.
* ``other_units`` — every other unit. **A subgroup, not a counterfactual**: these units
  may be easier or harder for reasons that have nothing to do with the tell.
* ``tell_lines_withheld`` — every unit again, with the lines either comment tell fires on
  removed from the ranking and from the labels. A unit left with no flagged line drops
  out. This is what the checkpoint ranks when it cannot be handed those lines.
* three rule-only baselines that never see a model: source order, the tell first, and the
  tell first with intact comment ends last.

None of this is the leak-free number. That needs models trained and evaluated on repaired
text. What this gives is the size of the problem, from files already on disk.

``test`` and ``clean_eval`` score files are refused without ``+audit.final=true``
(``Rules.md`` §5.3).

Usage::

    python scripts/audit_score_leakage.py
    python scripts/audit_score_leakage.py '+audit.runs=[80fc5238,c193e3bc,829c3555]'
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf

from vulnloc.data.cache import iter_shard
from vulnloc.representation.line_map import split_lines
from vulnloc.training.line_scores import UnitScores, load_scores
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import get_logger, setup_logging
from vulnloc.utils.text import normalize_source_whitespace

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_LOGGER = get_logger(__name__)

REPO = Path(__file__).resolve().parent.parent
HELD_OUT_SPLITS = frozenset({"test", "clean_eval"})
TOP_KS = (1, 5)
GROUPS = ("all", "tell_units", "other_units", "tell_lines_withheld")
BASELINES = ("source_order", "tell_first", "tell_first_intact_last")


def _load_label_audit() -> Any:
    """Import ``scripts/audit_label_leakage.py`` by path, for its tells."""
    spec = importlib.util.spec_from_file_location(
        "vulnloc_scripts_audit_label_leakage", REPO / "scripts" / "audit_label_leakage.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def topk_hit(
    scores: np.ndarray, labels: np.ndarray, usable: np.ndarray, k: int
) -> tuple[bool, bool]:
    """Whether a unit counts towards Top-k, and whether it is a hit.

    Args:
        scores: ``[n_lines]`` — higher is more suspicious.
        labels: ``[n_lines]`` — non-zero where the line is flagged.
        usable: ``[n_lines]`` — lines that may be ranked.
        k: How many top lines to inspect.

    Returns:
        ``(counts, hit)``. A unit with no usable flagged line does not count. Ties keep
        source order, as ``training/metrics.py`` does.
    """
    index = np.flatnonzero(usable)
    if index.size == 0 or not labels[index].any():
        return False, False
    order = index[np.argsort(-scores[index], kind="stable")]
    return True, bool(labels[order[:k]].any())


class _Tally:
    """Top-k counts for one group of units."""

    def __init__(self) -> None:
        self.units = 0
        self.hits = dict.fromkeys(TOP_KS, 0)

    def add(self, scores: np.ndarray, labels: np.ndarray, usable: np.ndarray) -> None:
        counted = False
        for k in TOP_KS:
            counts, hit = topk_hit(scores, labels, usable, k)
            counted = counts
            self.hits[k] += hit
        self.units += counted

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"units": self.units}
        for k in TOP_KS:
            result[f"top_{k}"] = self.hits[k] / self.units if self.units else 0.0
        return result


def audit_run(
    scored: Sequence[UnitScores],
    texts: Mapping[str, Sequence[str]],
    *,
    tell: Callable[[str], bool],
    mirror: Callable[[str], bool],
) -> dict[str, Any]:
    """Stratify one checkpoint's stored scores by a formatting tell.

    Args:
        scored: The checkpoint's units.
        texts: ``unit_id -> lines``, as the model saw them.
        tell: The positive oracle — fires on lines readable as flagged.
        mirror: The negative oracle — fires on lines readable as not flagged.

    Returns:
        One section per group in :data:`GROUPS` and per baseline in :data:`BASELINES`,
        plus ``top1_is_tell_line`` over the ``tell_units``.

    Raises:
        ConfigError: If a scored unit has no text or more scored lines than text.
    """
    tallies = {name: _Tally() for name in (*GROUPS, *BASELINES)}
    top1_is_tell = 0
    for unit in scored:
        lines = texts.get(unit.unit_id)
        n = int(unit.logits.shape[0])
        if lines is None or len(lines) < n:
            raise ConfigError(
                "a scored unit does not match the partition's text",
                unit_id=unit.unit_id,
                scored_lines=n,
                text_lines=None if lines is None else len(lines),
            )
        usable = unit.scoreable.astype(bool)
        labels = unit.labels.astype(bool)
        fires = np.array([tell(text) for text in lines[:n]], dtype=bool)
        intact = np.array([mirror(text) for text in lines[:n]], dtype=bool)
        logits = unit.logits.astype(np.float64)

        tallies["all"].add(logits, labels, usable)
        answered = bool((fires & labels & usable).any())
        tallies["tell_units" if answered else "other_units"].add(logits, labels, usable)
        tallies["tell_lines_withheld"].add(logits, labels, usable & ~fires & ~intact)

        position = -np.arange(n, dtype=np.float64)
        tallies["source_order"].add(position, labels, usable)
        tallies["tell_first"].add(position + n * fires, labels, usable)
        tallies["tell_first_intact_last"].add(position + n * fires - n * intact, labels, usable)
        if answered:
            index = np.flatnonzero(usable)
            best = index[np.argmax(logits[index])]
            top1_is_tell += bool(fires[best])
    report: dict[str, Any] = {name: tally.to_dict() for name, tally in tallies.items()}
    report["top1_is_tell_line"] = top1_is_tell
    return report


def aggregate(per_run: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean and sample standard deviation over seeds, per group and metric.

    Args:
        per_run: :func:`audit_run` reports, one per seed of one arm.

    Returns:
        ``group -> {metric: {"mean", "std"}, "units": [...]}``. ``std`` is ``None`` for a
        single seed.
    """
    summary: dict[str, Any] = {"n_seeds": len(per_run)}
    for group in (*GROUPS, *BASELINES):
        section: dict[str, Any] = {"units": [run[group]["units"] for run in per_run]}
        for k in TOP_KS:
            values = [run[group][f"top_{k}"] for run in per_run]
            section[f"top_{k}"] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
            }
        summary[group] = section
    summary["top1_is_tell_line"] = [run["top1_is_tell_line"] for run in per_run]
    return summary


def load_texts(cache_dir: Path, split: str, *, normalize: bool) -> dict[str, list[str]]:
    """The partition's line-labelled units as lists of lines.

    Args:
        cache_dir: Directory holding the preprocessed shards.
        split: The partition.
        normalize: Apply ADR-006 normalization, as every reported run did.

    Returns:
        ``unit_id -> lines``.
    """
    texts: dict[str, list[str]] = {}
    for unit in iter_shard(cache_dir, split):
        if unit.vulnerable_lines:
            source = normalize_source_whitespace(unit.source) if normalize else unit.source
            texts[unit.unit_id] = split_lines(source)
    return texts


def audit(cfg: DictConfig) -> dict[str, Any]:
    """Audit every scored checkpoint of one partition.

    Args:
        cfg: Composed config. Reads ``audit.split`` (default ``val``), ``audit.runs``
            (default: every manifest in the scores directory for that split),
            ``audit.scores_dir``, ``audit.output_dir`` and ``audit.final``.

    Returns:
        The report, as written to disk.

    Raises:
        ConfigError: For a held-out partition without ``audit.final``, or no score files.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )

    def option(name: str, default: Any) -> Any:
        value = OmegaConf.select(cfg, f"audit.{name}")
        return default if value is None else value

    split = str(option("split", "val"))
    if split in HELD_OUT_SPLITS and not bool(option("final", False)):
        raise ConfigError(
            "this partition is evaluated once, at the end, on the frozen configuration",
            split=split,
            hint="Rules.md §5.3. Add '+audit.final=true' only when that moment has come.",
        )
    scores_dir = Path(str(option("scores_dir", "./artifacts/phase6/scores")))
    output_dir = Path(str(option("output_dir", "./artifacts/phase6")))
    wanted = [str(run) for run in option("runs", [])]

    manifests = []
    for path in sorted(scores_dir.glob(f"*_{split}.manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not wanted or any(manifest["source_run"].startswith(prefix) for prefix in wanted):
            manifests.append(manifest)
    if not manifests:
        raise ConfigError(
            "no score files to audit", scores_dir=str(scores_dir), split=split, runs=wanted
        )

    label_audit = _load_label_audit()
    tell = label_audit.TELLS["comment_end_lost_slash"]
    mirror = label_audit.TELLS["comment_end_intact"]
    texts = load_texts(
        Path(cfg.data.cache.output_dir),
        split,
        normalize=bool(cfg.representation.normalize_whitespace),
    )

    runs: dict[str, Any] = {}
    arms: dict[str, list[dict[str, Any]]] = {}
    for manifest in manifests:
        run_id = str(manifest["source_run"])
        scored = load_scores(scores_dir / f"{run_id}_{split}.npz")
        report = audit_run(scored, texts, tell=tell, mirror=mirror)
        report.update(
            {
                "seed": int(manifest["seed"]),
                "fusion_kind": str(manifest["fusion_kind"]),
                "frozen": bool(manifest["frozen"]),
            }
        )
        runs[run_id] = report
        arm = f"{manifest['fusion_kind']}{'_frozen' if manifest['frozen'] else ''}"
        arms.setdefault(arm, []).append(report)
        _LOGGER.info(
            "Checkpoint audited",
            extra={
                "source_run": run_id,
                "arm": arm,
                "seed": report["seed"],
                "top_5_all": round(report["all"]["top_5"], 4),
                "top_5_tell_units": round(report["tell_units"]["top_5"], 4),
                "top_5_other_units": round(report["other_units"]["top_5"], 4),
                "tell_units": report["tell_units"]["units"],
                "top1_is_tell_line": report["top1_is_tell_line"],
            },
        )

    payload: dict[str, Any] = {
        "generated_by": "scripts/audit_score_leakage.py",
        "split": split,
        "tell": "comment_end_lost_slash",
        "mirror": "comment_end_intact",
        "arms": {arm: aggregate(reports) for arm, reports in sorted(arms.items())},
        "runs": runs,
        "note": (
            "other_units is a subgroup, not a counterfactual; the leak-free number needs "
            "models trained and evaluated on repaired text."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"score_leakage_{split}.json"
    with start_run(cfg, run_name="score-leakage-audit") as run_id:
        payload["run_id"] = run_id
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        mlflow.set_tags({"artefact": "score_leakage_audit", "split": split})
        mlflow.log_params(
            {
                "score_leakage.split": split,
                "score_leakage.tell": payload["tell"],
                "score_leakage.checkpoints": len(runs),
            }
        )
        for arm, summary in payload["arms"].items():
            for group in (*GROUPS, *BASELINES):
                for k in TOP_KS:
                    mlflow.log_metric(
                        f"{arm}/{group}/top_{k}_mean", summary[group][f"top_{k}"]["mean"]
                    )
        mlflow.log_artifact(str(output_path), artifact_path="score_leakage")
    _LOGGER.info("Report written", extra={"path": str(output_path), "run_id": payload["run_id"]})
    return payload


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    audit(cfg)


if __name__ == "__main__":
    main()
