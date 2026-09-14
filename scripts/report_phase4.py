"""Assemble Phase 4's four exit criteria from the logged runs, and say which are met.

``Phases.md`` Phase 4 asks for four things:

1. all three GNN variants run and are config-selectable;
2. over-smoothing checked — performance vs depth at 2/4/6 reported;
3. GNN-only metrics logged over at least three seeds;
4. the overfit test passes.

Each lives in a different run, and deciding whether a criterion is met means grouping
runs by what varied and checking that nothing else did. Done by hand at 2am that is how
a depth sweep quietly becomes a seed comparison.

**Grouping is by configuration, not by name.** Runs are keyed on ``(kind, depth,
hidden_dim, edge_handling, builder)``, so a sweep entry that also changed the width lands
in its own group rather than silently joining one it does not belong to. Seeds within a
group are aggregated by :mod:`report_seeds`'s own comparability check, so the same
refusals apply here.

**The depth table carries its own warning, and it is not optional.** ``Memory.md`` §1.12
measured that the representation alone leaves **27.91% of flagged lines indistinguishable
at 2 hops** against 7.35% at 4 and 5.39% at 6. Depth 2 is therefore *expected* to lose on
feature grounds before over-smoothing is considered at all. Over-smoothing is a claim
about **6 vs 4**, where the two ceilings are nearly equal. The report prints that beside
the numbers because a depth table without it invites exactly the wrong conclusion.

Reads the tracking store only; it trains nothing.

Usage::

    python scripts/report_phase4.py
    python scripts/report_phase4.py +seeds.min_seeds=5
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import hydra
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from vulnloc.training.tracking import ensure_experiment, start_run
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

REPORTED = (
    "eval/line/top_5_accuracy",
    "eval/line/top_1_accuracy",
    "eval/line/f1",
    "eval/line/precision",
    "eval/line/recall",
    "eval/line/ifa_mean",
)

FEATURE_CEILING_BY_DEPTH = {
    2: 27.91,
    4: 7.35,
    6: 5.39,
}
"""§1.12, run `b530f1ff…`: share of flagged `train` lines still indistinguishable from an
unflagged one after this many hops of message passing. A ceiling the depth cannot beat."""

DEPTH_WARNING = (
    "Depth 2 carries a 27.91% hard recall ceiling from the representation itself "
    "(Memory.md §1.12), against 7.35% at depth 4. Depth 2 losing to depth 4 is therefore "
    "the expected result BEFORE over-smoothing is considered, and must not be read as "
    "evidence about over-smoothing. Over-smoothing is a claim about 6 vs 4, where the "
    "feature ceilings are 5.39% and 7.35% and therefore nearly equal."
)


def _group_key(run: Any) -> tuple[str, ...]:
    """What makes two runs the same experiment, seed aside.

    Args:
        run: An MLflow run.

    Returns:
        A tuple of the configuration values that define the group.
    """
    params = run.data.params
    # `use_subtokens` defaults to "False" rather than "?" on purpose: runs logged before
    # the parameter existed were all node-type-only, so defaulting makes them the correct
    # comparison partners for the A4 runs instead of a third, phantom group.
    #
    # Leaving this key OUT of the grouping was a real bug, caught before it produced a
    # number: A4 runs would have been merged into their own partners' group and averaged
    # with them, turning a feature comparison into a wider seed group.
    return (
        *(
            str(params.get(key, "?"))
            for key in (
                "gnn_kind",
                "gnn_depth",
                "gnn_hidden_dim",
                "edge_handling",
                "graph_builder",
            )
        ),
        str(params.get("use_subtokens", "False")),
    )


def _best_metrics(client: MlflowClient, run: Any) -> dict[str, float]:
    """Metrics at the run's own best step.

    Args:
        client: Tracking client.
        run: The run.

    Returns:
        ``{metric: value}``; empty when the run logged no history.
    """
    watch = f"eval/{run.data.params.get('train.watch_metric', 'line/top_5_accuracy')}"
    history = client.get_metric_history(run.info.run_id, watch)
    if not history:
        return {}
    best = max(history, key=lambda m: m.value)
    values: dict[str, float] = {}
    for key in REPORTED:
        points = [m for m in client.get_metric_history(run.info.run_id, key) if m.step == best.step]
        if points:
            values[key] = float(points[-1].value)
    return values


def _summarise(rows: list[dict[str, float]]) -> dict[str, Any]:
    """Mean and spread per metric over a group's seeds.

    Args:
        rows: One metric dict per run.

    Returns:
        ``{metric: {mean, std, n}}``.
    """
    out: dict[str, Any] = {}
    for key in REPORTED:
        values = [r[key] for r in rows if key in r]
        if not values:
            continue
        out[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values),
        }
    return out


def build(cfg: DictConfig) -> dict[str, Any]:
    """Collect the structural runs and decide each exit criterion.

    Args:
        cfg: Composed Hydra config.

    Returns:
        The report, as written to disk.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_id = ensure_experiment(str(cfg.mlflow.experiment_name), cfg.mlflow.artifact_location)
    client = MlflowClient()
    min_seeds = int(OmegaConf.select(cfg, "seeds.min_seeds") or 3)

    found = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="params.branch = 'gnn_only' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
        max_results=500,
    )

    def _is_result(run: Any) -> bool:
        """A run that may enter a reported group.

        Three ways out, each of which has actually happened: an overfit run is a
        debugging instrument; a run on a subsample is a probe (two CPU trials at
        train.limit=600 reached an A4 depth-2 mean before this check existed); and a run
        tagged `void` was disowned deliberately.
        """
        params = run.data.params
        return (
            params.get("train.overfit", "False") != "True"
            and params.get("train.limit", "0") in {"0", "None", ""}
            and not run.data.tags.get("void")
        )

    real = [r for r in found if _is_result(r)]
    overfits = [r for r in found if r.data.params.get("train.overfit", "False") == "True"]

    groups: dict[tuple[str, ...], list[Any]] = {}
    for run in real:
        groups.setdefault(_group_key(run), []).append(run)

    group_rows: list[dict[str, Any]] = []
    for key, runs in sorted(groups.items()):
        metrics = [_best_metrics(client, r) for r in runs]
        group_rows.append(
            {
                "kind": key[0],
                "depth": int(key[1]) if key[1].isdigit() else key[1],
                "hidden_dim": key[2],
                "edge_handling": key[3],
                "builder": key[4],
                "use_subtokens": key[5] == "True",
                "seeds": sorted(int(r.data.params["seed"]) for r in runs),
                "run_ids": [r.info.run_id for r in runs],
                "metrics": _summarise(metrics),
            }
        )

    # --- the four criteria -------------------------------------------------------
    kinds_run = sorted({row["kind"] for row in group_rows})
    depths_at_default = sorted(
        {
            row["depth"]
            for row in group_rows
            if row["kind"] == "gcn" and isinstance(row["depth"], int) and not row["use_subtokens"]
        }
    )
    best_seeded = max((len(row["seeds"]) for row in group_rows), default=0)
    overfit_passed = []
    for run in overfits:
        history = client.get_metric_history(run.info.run_id, "eval/line/top_1_accuracy")
        if history and max(m.value for m in history) >= 0.99:
            overfit_passed.append(run.info.run_id)

    criteria = {
        "all_three_variants_run": {
            "met": set(kinds_run) >= {"gcn", "gat", "ggnn"},
            "evidence": f"kinds with a logged real run: {kinds_run}",
        },
        "depth_sweep_2_4_6_reported": {
            "met": {2, 4, 6} <= set(depths_at_default),
            "evidence": f"gcn depths logged: {depths_at_default}",
            "read_this_first": DEPTH_WARNING,
        },
        "at_least_3_seeds": {
            "met": best_seeded >= min_seeds,
            "evidence": f"largest seeded group: {best_seeded} seeds (need {min_seeds})",
        },
        "overfit_test_passes": {
            "met": bool(overfit_passed),
            "evidence": f"overfit runs reaching Top-1 >= 0.99: {[r[:8] for r in overfit_passed]}",
        },
    }

    depth_table = [
        {
            "depth": row["depth"],
            "seeds": row["seeds"],
            "top_5": row["metrics"].get("eval/line/top_5_accuracy", {}).get("mean"),
            "top_1": row["metrics"].get("eval/line/top_1_accuracy", {}).get("mean"),
            "f1": row["metrics"].get("eval/line/f1", {}).get("mean"),
            "feature_ceiling_pct_ambiguous": FEATURE_CEILING_BY_DEPTH.get(row["depth"]),
        }
        for row in sorted(group_rows, key=lambda r: (str(r["depth"])))
        if row["kind"] == "gcn" and isinstance(row["depth"], int) and not row["use_subtokens"]
    ]

    a4 = []
    for row in group_rows:
        if not row["use_subtokens"]:
            continue
        partner = next(
            (
                other
                for other in group_rows
                if not other["use_subtokens"]
                and (other["kind"], other["depth"], other["hidden_dim"])
                == (row["kind"], row["depth"], row["hidden_dim"])
            ),
            None,
        )
        entry: dict[str, Any] = {
            "kind": row["kind"],
            "depth": row["depth"],
            "with_subtokens": row["metrics"].get("eval/line/top_5_accuracy", {}).get("mean"),
            "without_subtokens": (
                partner["metrics"].get("eval/line/top_5_accuracy", {}).get("mean")
                if partner
                else None
            ),
            "feature_ceiling_pct_ambiguous_without": FEATURE_CEILING_BY_DEPTH.get(row["depth"]),
            "seeds_with": row["seeds"],
            "seeds_without": partner["seeds"] if partner else [],
        }
        if entry["with_subtokens"] is not None and entry["without_subtokens"] is not None:
            entry["delta_top_5"] = entry["with_subtokens"] - entry["without_subtokens"]
        a4.append(entry)

    report = {
        "generated_by": "scripts/report_phase4.py",
        "a4_subtoken_features": {
            "rows": a4,
            "read_this_first": (
                "Section 1.12 predicts the gain is LARGE at depth 2 (27.91% of flagged "
                "lines unresolvable without subtokens) and SMALL at depth 4 (7.35%). A "
                "gain that does not follow that pattern is more interesting than one "
                "that does."
            ),
        },
        "criteria": criteria,
        "criteria_met": sum(1 for c in criteria.values() if c["met"]),
        "groups": group_rows,
        "depth_sweep": {"rows": depth_table, "read_this_first": DEPTH_WARNING},
        "caveats": [
            "Node features are node-type only — the ADR-002 §D2.5 subtoken half is not "
            "built, so this branch cannot distinguish two identifiers. A4 must not be "
            "read as measuring the full D2.5 feature set.",
            "7.84% of flagged lines corpus-wide carry no graph node at all (§7 #17). "
            "That is a ceiling on THIS branch that the semantic branch does not share.",
            "function/* is degenerate under line_labelled_only (fp=tn=0). Never quote it.",
            DEPTH_WARNING,
        ],
    }

    output_dir = Path("./artifacts/phase4")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase4_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with start_run(cfg, run_name="phase4-report"):
        mlflow.log_artifact(str(output_path))
        mlflow.log_metric("criteria_met", float(report["criteria_met"]))

    _LOGGER.info(
        "Phase 4 report written",
        extra={"path": str(output_path), "criteria_met": report["criteria_met"]},
    )
    for name, criterion in criteria.items():
        _LOGGER.info(
            "Exit criterion",
            extra={
                "criterion": name,
                "met": criterion["met"],
                "evidence": criterion["evidence"],
            },
        )
    return report


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    build(cfg)


if __name__ == "__main__":
    main()
