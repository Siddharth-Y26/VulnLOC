r"""The ablation tables, from the seed reports that already exist — by script, never by hand.

Phase 8 asks that *ablations confirm or refute each novelty claim explicitly* and that
*every table cell is traceable to a run ID*. ``docs/ablations.yaml`` is the catalogue: for
each of ``ResearchPlan.md`` §5's ten ablations, which seed reports hold its runs, under
which conditions they were made, and which difference would count as the claim holding.
This script renders that catalogue. It trains nothing, loads no model and reads no
partition; its inputs are ``report_seeds.py``'s and ``report_phase4.py``'s artefacts and,
read-only, the MLflow stores those runs were logged to.

A table is one comparison group under one set of conditions — encoder regime, text
variant, machine. Arms measured under different conditions never share one
(``Memory.md`` §7 #49, #61).

**It refuses to write** when

* a run did not log the parameters its arm and its table declare. A parameter a run never
  logged counts as not enabled (``Memory.md`` §7 #39), so "comments stripped" must have
  been logged to be believed;
* a seed report was selected on a partition other than ``val``, or its headline mean is
  not the mean of the runs it lists;
* a run is void, an overfit check or a probe, or appears in two arms of one table;
* a table marked ``run`` is missing a report, or one marked ``pending`` has one.

**What "resolved" means here** is what it means everywhere in this repository: a
difference of at least 2 pooled standard deviations. With three seeds an arm, two arms
with no difference reach that about 7 times in 100, and this document makes several dozen
comparisons — so it counts them and prints the count beside every crossing
(``Memory.md`` §7 #64). An outcome is one of *supported*, *contradicted*, *differs* or *not
resolved*; fewer than three seeds on either side is *not compared* (§7 #37).

Usage::

    python scripts/make_ablation_tables.py
    python scripts/make_ablation_tables.py +ablations.output_dir=./artifacts/phase8
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from vulnloc.evaluation.statistics import (
    RESOLVED_AT,
    compare_arms,
    pooled_sd,
    spread,
    threshold_p_value,
)
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import ConfigError, DataError
from vulnloc.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_LOGGER = get_logger(__name__)

METRICS: dict[str, str] = {
    "top_5": "eval/line/top_5_accuracy",
    "top_1": "eval/line/top_1_accuracy",
    "ifa": "eval/line/ifa_mean",
}
"""The threshold-free metrics every training run logs. Line F1 is left out on purpose: the
runs log it at 0.5 on uncalibrated scores, and a gap in it is the threshold moving
(``Memory.md`` §1.24)."""

METRIC_LABELS = {"top_5": "Top-5", "top_1": "Top-1", "ifa": "IFA"}
CLAIM_METRIC = "top_5"
"""The one metric a claim is judged on. The other two are shown beside it."""

SELECTION_SPLIT = "val"
MIN_SEEDS = 3
RECONCILE_TOLERANCE = 1e-9

FALSY = frozenset({"false", "0", "", "none"})
"""Expected values an absent parameter satisfies: a knob that did not exist when a run was
made was not enabled in it (``Memory.md`` §7 #39)."""

REFUSED_PARAMS = {"train.overfit": "True"}
TEXT_VARIANTS = {"as_shipped": "False", "stripped": "True"}
ENCODER_REGIMES = {"frozen": "True", "fine_tuned": "False", "none": None}
STATUSES = frozenset({"run", "pending"})
EXPECTATIONS = frozenset({"first_higher", "none"})


def read_run_facts(database: Path, run_ids: Sequence[str]) -> dict[str, dict[str, dict[str, str]]]:
    """Logged parameters and tags of the given runs, read without taking a write lock.

    The store is opened read-only: a queue may be training against it, and a report must
    never be the reason a run's own logging waits.

    Args:
        database: An MLflow SQLite store.
        run_ids: Full run ids.

    Returns:
        ``run_id -> {"params": {...}, "tags": {...}}``.

    Raises:
        ConfigError: If the store does not exist.
        DataError: If a run is not in it.
    """
    if not database.exists():
        raise ConfigError("an MLflow store named in the catalogue is missing", path=str(database))
    facts: dict[str, dict[str, dict[str, str]]] = {}
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=30.0)
    try:
        for run_id in run_ids:
            if (
                connection.execute("SELECT 1 FROM runs WHERE run_uuid = ?", (run_id,)).fetchone()
                is None
            ):
                raise DataError(
                    "a run is not in the store its table names", run_id=run_id, store=str(database)
                )
            facts[run_id] = {
                table: {
                    str(key): str(value)
                    for key, value in connection.execute(
                        f"SELECT key, value FROM {table} WHERE run_uuid = ?",  # noqa: S608 - a fixed table name
                        (run_id,),
                    )
                }
                for table in ("params", "tags")
            }
    finally:
        connection.close()
    return facts


def param_mismatches(
    params: Mapping[str, str], expected: Mapping[str, str]
) -> dict[str, dict[str, str]]:
    """Declared parameters a run did not log with the declared value.

    Args:
        params: The run's logged parameters.
        expected: ``key -> value`` the catalogue declares.

    Returns:
        ``key -> {"expected", "logged"}`` for every disagreement; empty when the run is
        what the catalogue says it is.
    """
    problems: dict[str, dict[str, str]] = {}
    for key, value in expected.items():
        logged = params.get(key)
        if logged is None and value.lower() in FALSY:
            continue
        if logged != value:
            problems[key] = {"expected": value, "logged": "absent" if logged is None else logged}
    return problems


def table_expectations(table: Mapping[str, Any]) -> dict[str, str]:
    """The parameters a table's conditions imply for every one of its runs.

    Args:
        table: A catalogue table.

    Returns:
        ``strip_comments`` from the text variant and, for fused runs, ``frozen`` from the
        encoder regime.

    Raises:
        ConfigError: For a text variant or an encoder regime the catalogue does not define.
    """
    text, encoder = str(table.get("text")), str(table.get("encoder"))
    if text not in TEXT_VARIANTS or encoder not in ENCODER_REGIMES:
        raise ConfigError(
            "a table declares conditions this script does not know",
            table=table.get("key"),
            text=text,
            encoder=encoder,
        )
    expected = {"strip_comments": TEXT_VARIANTS[text]}
    frozen = ENCODER_REGIMES[encoder]
    if frozen is not None:
        expected["frozen"] = frozen
    return expected


def _summary(mean: float, std: float | None, n: int) -> dict[str, Any]:
    return {"mean": float(mean), "std": None if std is None else float(std), "n": int(n)}


def load_seed_report(path: Path) -> dict[str, Any]:
    """One arm from a ``report_seeds.py`` artefact, reconciled with the runs it lists.

    Args:
        path: The seed report.

    Returns:
        ``{"source", "runs": [{"run_id", "seed"}], "metrics": {name: summary},
        "by_seed": {name: {seed: value}}, "drift_accepted": [...]}``.

    Raises:
        DataError: If the report was selected on another partition, lists a seed twice, or
            its headline mean is not the mean of its runs.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("eval_split") != SELECTION_SPLIT:
        raise DataError(
            "ablations are validation-only", report=str(path), eval_split=report.get("eval_split")
        )
    runs = [{"run_id": str(run["run_id"]), "seed": int(run["seed"])} for run in report["runs"]]
    seeds = [run["seed"] for run in runs]
    if len(set(seeds)) != len(seeds) or len(runs) != int(report["n_seeds"]):
        raise DataError("a seed report's runs are not one per seed", report=str(path), seeds=seeds)
    metrics: dict[str, dict[str, Any]] = {}
    by_seed: dict[str, dict[int, float]] = {}
    for name, key in METRICS.items():
        values = {int(run["seed"]): float(run["metrics"][key]) for run in report["runs"]}
        summary = spread(list(values.values()))
        headline = report["headline"][key]
        if abs(summary["mean"] - float(headline["mean"])) > RECONCILE_TOLERANCE:
            raise DataError(
                "a seed report's headline is not the mean of its runs",
                report=str(path),
                metric=key,
                headline=headline["mean"],
                recomputed=summary["mean"],
            )
        metrics[name] = _summary(summary["mean"], summary["std"], len(values))
        by_seed[name] = values
    drift = report.get("provenance", {}).get("src_vulnloc_drift_accepted") or {}
    return {
        "source": path.as_posix(),
        "runs": runs,
        "metrics": metrics,
        "by_seed": by_seed,
        "drift_accepted": sorted({name for files in drift.values() for name in files}),
    }


def load_phase4_group(path: Path, selector: Mapping[str, Any]) -> dict[str, Any]:
    """One arm from a group of ``report_phase4.py``'s artefact — summary statistics only.

    Args:
        path: ``phase4_report.json``.
        selector: ``{"kind", "depth", "use_subtokens"}``.

    Returns:
        The same shape as :func:`load_seed_report`, with ``by_seed`` empty — that report
        keeps a mean and a standard deviation per group, not each seed's value — and with
        every run's ``seed`` left ``None``: its ``seeds`` list is sorted and its
        ``run_ids`` list is not, so the two cannot be zipped. :func:`check_runs` fills the
        seed in from what each run logged.

    Raises:
        DataError: Unless exactly one group matches.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        group
        for group in report["groups"]
        if all(group.get(key) == value for key, value in selector.items())
    ]
    if len(matches) != 1:
        raise DataError(
            "a Phase 4 group selector must match exactly one group",
            report=str(path),
            selector=dict(selector),
            matched=len(matches),
        )
    group = matches[0]
    return {
        "source": f"{path.as_posix()} {json.dumps(dict(selector), sort_keys=True)}",
        "runs": [{"run_id": str(run_id), "seed": None} for run_id in group["run_ids"]],
        "metrics": {
            name: _summary(
                group["metrics"][key]["mean"],
                group["metrics"][key]["std"],
                group["metrics"][key]["n"],
            )
            for name, key in METRICS.items()
        },
        "by_seed": {},
        "drift_accepted": [],
    }


def phase4_expectations(selector: Mapping[str, Any]) -> dict[str, str]:
    """The logged parameters a Phase 4 group selector implies.

    Args:
        selector: ``{"kind", "depth", "use_subtokens"}``.

    Returns:
        ``gnn_kind``, ``gnn_depth`` and ``use_subtokens`` as the runs log them.
    """
    return {
        "gnn_kind": str(selector["kind"]),
        "gnn_depth": str(selector["depth"]),
        "use_subtokens": str(bool(selector["use_subtokens"])),
    }


def check_runs(
    arm: Mapping[str, Any],
    facts: Mapping[str, Mapping[str, Mapping[str, str]]],
    expected: Mapping[str, str],
    *,
    where: str,
) -> list[dict[str, Any]]:
    """Check an arm's runs against what the catalogue declares, and annotate them.

    Args:
        arm: From :func:`load_seed_report` or :func:`load_phase4_group`.
        facts: From :func:`read_run_facts`.
        expected: Declared parameters, table-level and arm-level together.
        where: ``<table>/<arm>``, for the error.

    Returns:
        The arm's runs in seed order, each with ``seed`` (as the run logged it),
        ``git_sha``, ``dirty``, ``resumed_at`` and ``gpu``.

    Raises:
        DataError: If a run is void, an overfit check or a probe, did not log a declared
            parameter with the declared value, or logged a seed other than the one its
            report lists it under — or if two runs of the arm share a seed.
    """
    annotated = []
    for run in arm["runs"]:
        params, tags = facts[run["run_id"]]["params"], facts[run["run_id"]]["tags"]
        refused = {key: value for key, value in REFUSED_PARAMS.items() if params.get(key) == value}
        if "void" in tags or refused or params.get("train.limit", "0") != "0":
            raise DataError(
                "a void run, an overfit check or a probe is not a result",
                arm=where,
                run_id=run["run_id"],
            )
        problems = param_mismatches(params, expected)
        seed = int(params["seed"])
        if run["seed"] is not None and int(run["seed"]) != seed:
            problems["seed"] = {"expected": str(run["seed"]), "logged": str(seed)}
        if problems:
            raise DataError(
                "a run is not what the catalogue says its arm is",
                arm=where,
                run_id=run["run_id"],
                problems=problems,
            )
        sha = tags.get("git_sha", "")
        annotated.append(
            {
                "run_id": run["run_id"],
                "seed": seed,
                "git_sha": sha,
                "dirty": sha.endswith("-dirty"),
                "resumed_at": tags.get("resumed_at"),
                "gpu": params.get("resolved_gpu"),
            }
        )
    if len({run["seed"] for run in annotated}) != len(annotated):
        raise DataError("two runs of one arm logged the same seed", arm=where)
    return sorted(annotated, key=lambda run: run["seed"])


def compare_summaries(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """``first - second`` from means and standard deviations alone.

    Used where a report keeps no per-seed values. The scale is the same pooled standard
    deviation; what is missing is the pairing by seed.

    Args:
        first: ``{"mean", "std", "n"}`` of the first arm.
        second: The same for the second.

    Returns:
        The shape of :func:`vulnloc.evaluation.statistics.compare_arms`, with ``by_seed``
        and ``same_sign_on_every_seed`` set to ``None``.
    """
    difference = float(first["mean"]) - float(second["mean"])
    pooled = (
        None
        if first["std"] is None or second["std"] is None
        else pooled_sd(float(first["std"]), float(second["std"]))
    )
    scaled = difference / pooled if pooled else None
    return {
        "difference": difference,
        "pooled_sd": pooled,
        "difference_in_pooled_sd": scaled,
        "resolved": scaled is not None and abs(scaled) >= RESOLVED_AT,
        "by_seed": None,
        "same_sign_on_every_seed": None,
        "n_seeds": min(int(first["n"]), int(second["n"])),
    }


def outcome(scaled: float | None, n_seeds: int, expects: str) -> str:
    """What one comparison says about the claim it was declared to test.

    Args:
        scaled: ``first - second`` on the claim metric, in pooled standard deviations.
        n_seeds: The smaller number of seeds of the two arms.
        expects: ``first_higher`` or ``none``.

    Returns:
        ``not compared`` under three seeds; otherwise ``not resolved`` below the
        threshold, and above it ``supported`` / ``contradicted`` when a direction was
        predicted or ``differs`` when none was.
    """
    if n_seeds < MIN_SEEDS or scaled is None:
        return "not compared"
    if abs(scaled) < RESOLVED_AT:
        return "not resolved"
    if expects == "none":
        return "differs"
    return "supported" if scaled > 0 else "contradicted"


def build_comparison(
    declared: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]], *, table: str
) -> dict[str, Any]:
    """One declared comparison, on every metric.

    Args:
        declared: ``{"first", "second", "expects"}`` from the catalogue.
        arms: The table's loaded arms.
        table: The table key, for the error.

    Returns:
        ``{"first", "second", "expects", "metrics": {name: comparison}, "outcome"}``.

    Raises:
        ConfigError: For an unknown arm or expectation.
    """
    first, second, expects = (
        str(declared["first"]),
        str(declared["second"]),
        str(declared["expects"]),
    )
    if first not in arms or second not in arms or expects not in EXPECTATIONS:
        raise ConfigError(
            "a comparison names an arm or an expectation its table does not have",
            table=table,
            comparison=dict(declared),
        )
    metrics: dict[str, Any] = {}
    for name in METRICS:
        a, b = arms[first]["by_seed"].get(name), arms[second]["by_seed"].get(name)
        if a and b and set(a) == set(b):
            metrics[name] = compare_arms(a, b)
        else:
            metrics[name] = compare_summaries(
                arms[first]["metrics"][name], arms[second]["metrics"][name]
            )
    claim = metrics[CLAIM_METRIC]
    return {
        "first": first,
        "second": second,
        "expects": expects,
        "metrics": metrics,
        "outcome": outcome(claim["difference_in_pooled_sd"], claim["n_seeds"], expects),
    }


def _load_arm(spec: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Load one declared arm and the parameters its source implies."""
    if "seed_report" in spec:
        return load_seed_report(root / str(spec["seed_report"])), {}
    if "phase4_group" in spec:
        selector = dict(spec["phase4_group"])
        return load_phase4_group(root / str(spec["report"]), selector), phase4_expectations(
            selector
        )
    raise ConfigError("an arm names no source", arm=dict(spec))


def _arm_paths(table: Mapping[str, Any], root: Path) -> list[Path]:
    return [
        root / str(spec.get("seed_report") or spec.get("report")) for spec in table["arms"].values()
    ]


def build_table(
    table: Mapping[str, Any], stores: Mapping[str, Mapping[str, Any]], root: Path
) -> dict[str, Any]:
    """Load, check and compare one catalogue table.

    Args:
        table: The catalogue entry.
        stores: The catalogue's ``stores`` block.
        root: Directory the catalogue's paths are relative to.

    Returns:
        The table as rendered: conditions, arms with their annotated runs, comparisons.
        A ``pending`` table carries no arms.

    Raises:
        ConfigError: If the entry is malformed, or its status disagrees with what is on disk.
        DataError: From the loaders and from :func:`check_runs`.
    """
    key, status = str(table["key"]), str(table.get("status"))
    if status not in STATUSES or table.get("store") not in stores:
        raise ConfigError("a table has no valid status or store", table=key)
    head = {
        name: table.get(name)
        for name in (
            "key",
            "title",
            "store",
            "encoder",
            "text",
            "status",
            "memory",
            "note",
            "reference",
        )
    }
    head["store_label"] = stores[str(table["store"])]["label"]
    paths = _arm_paths(table, root)
    present = [path for path in paths if path.exists()]
    if status == "pending":
        if present:
            raise ConfigError(
                "a table marked pending has reports on disk — mark it run",
                table=key,
                present=[path.as_posix() for path in present],
            )
        return {**head, "arms": {}, "comparisons": []}
    if len(present) != len(paths):
        raise ConfigError(
            "a table marked run is missing a report",
            table=key,
            missing=[path.as_posix() for path in paths if not path.exists()],
        )

    shared = table_expectations(table)
    loaded: dict[str, dict[str, Any]] = {}
    expected: dict[str, dict[str, str]] = {}
    for name, spec in table["arms"].items():
        arm, implied = _load_arm(spec, root)
        loaded[str(name)] = arm
        declared = {str(k): str(v) for k, v in (spec.get("expect") or {}).items()}
        expected[str(name)] = {**shared, **implied, **declared}
    run_ids = [run["run_id"] for arm in loaded.values() for run in arm["runs"]]
    if len(set(run_ids)) != len(run_ids):
        raise DataError("a run appears in two arms of one table", table=key)
    facts = read_run_facts(root / str(stores[str(table["store"])]["mlflow_db"]), run_ids)
    for name, arm in loaded.items():
        arm["runs"] = check_runs(arm, facts, expected[name], where=f"{key}/{name}")
        arm["declared"] = expected[name]
    if str(table["reference"]) not in loaded:
        raise ConfigError("a table's reference is not one of its arms", table=key)
    comparisons = [
        build_comparison(entry, loaded, table=key) for entry in table.get("comparisons") or []
    ]
    return {**head, "arms": loaded, "comparisons": comparisons}


def summarise(ablations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count the comparisons the document makes, and name the ones past the threshold.

    Args:
        ablations: The built ablations.

    Returns:
        ``{"comparisons", "claim_comparisons", "crossings", "threshold", "chance"}`` —
        ``chance`` is how often two arms with no difference reach the threshold at three
        seeds, in any one comparison. A comparison of the same two sets of runs that two
        ablations both show (the full model against ``llm_only`` is A1's and A2's) is
        counted once, under the first.
    """
    total, claims, crossings = 0, 0, []
    counted: set[tuple[str, str, str]] = set()
    for ablation in ablations:
        for table in ablation["tables"]:
            for comparison in table["comparisons"]:
                sources = tuple(
                    table["arms"][comparison[side]]["source"] for side in ("first", "second")
                )
                for name, entry in comparison["metrics"].items():
                    if entry["difference_in_pooled_sd"] is None or entry["n_seeds"] < MIN_SEEDS:
                        continue
                    if (sources[0], sources[1], name) in counted:
                        continue
                    counted.add((sources[0], sources[1], name))
                    total += 1
                    claims += name == CLAIM_METRIC
                    if entry["resolved"]:
                        crossings.append(
                            {
                                "ablation": ablation["id"],
                                "table": table["key"],
                                "comparison": f"{comparison['first']} - {comparison['second']}",
                                "metric": name,
                                "difference_in_pooled_sd": entry["difference_in_pooled_sd"],
                            }
                        )
    return {
        "comparisons": total,
        "claim_comparisons": claims,
        "crossings": crossings,
        "threshold": RESOLVED_AT,
        "chance": threshold_p_value(MIN_SEEDS),
    }


def build_tables(catalogue: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Render the whole catalogue into a payload. Pure: reads artefacts, writes nothing.

    Args:
        catalogue: ``docs/ablations.yaml``, parsed.
        root: Directory its paths are relative to.

    Returns:
        ``{"ablations": [...], "summary": {...}}``.

    Raises:
        ConfigError: For a malformed catalogue.
        DataError: When an artefact does not reconcile with what the catalogue declares.
    """
    if int(catalogue.get("schema", 0)) != 1:
        raise ConfigError("unknown ablation catalogue schema", schema=catalogue.get("schema"))
    stores = catalogue["stores"]
    ablations = []
    seen: set[str] = set()
    for entry in catalogue["ablations"]:
        tables = [build_table(table, stores, root) for table in entry.get("tables") or []]
        keys = [str(table["key"]) for table in tables]
        if seen.intersection(keys) or len(set(keys)) != len(keys):
            raise ConfigError("table keys must be unique", ablation=entry["id"], keys=keys)
        seen.update(keys)
        ablations.append(
            {
                "id": str(entry["id"]),
                "name": str(entry["name"]),
                "claim": str(entry["claim"]),
                "note": entry.get("note"),
                "tables": tables,
                "not_run": [dict(item) for item in entry.get("not_run") or []],
            }
        )
    return {"ablations": ablations, "summary": summarise(ablations)}


def _pm(summary: Mapping[str, Any], digits: int) -> str:
    if summary["std"] is None:
        return f"{summary['mean']:.{digits}f} (one seed)"
    return f"{summary['mean']:.{digits}f} ± {summary['std']:.{digits}f}"


def _delta(entry: Mapping[str, Any], digits: int) -> str:
    scaled = entry["difference_in_pooled_sd"]
    scale = "one seed" if scaled is None else f"{scaled:+.2f} pooled sd"
    return f"{entry['difference']:+.{digits}f} ({scale})"


def _by_seed(entry: Mapping[str, Any]) -> str:
    if entry["by_seed"] is None:
        return "not paired"
    return ", ".join(f"{seed}: {value:+.4f}" for seed, value in entry["by_seed"].items())


_TEXT_BANNERS = {
    "as_shipped": (
        "**Text as shipped — it carries the comment-terminator tell** (`Memory.md` §1.25). "
        "A fine-tuned encoder and every arm with a structural branch can read it; nothing "
        "in this table is a leak-free number."
    ),
    "stripped": "Comment-stripped text (ADR-010): trained and selected without the tell.",
}


def _render_table(table: Mapping[str, Any]) -> list[str]:
    """One table: its conditions, its arms, its comparisons, its footnotes."""
    lines = [f"### {table['title']}", ""]
    lines.append(
        f"`{table['key']}` · encoder: {str(table['encoder']).replace('_', '-')} · "
        f"{table['store_label']} · `Memory.md` {table['memory']}"
    )
    lines += ["", _TEXT_BANNERS[str(table["text"])]]
    if table.get("note"):
        lines += ["", str(table["note"]).strip()]
    if table["status"] == "pending":
        return [*lines, "", "**Not yet run.** No report for this table exists.", ""]
    lines += [
        "",
        "| Arm | Seeds | Top-5 | Top-1 | IFA | Runs (by seed) |",
        "|---|---|---|---|---|---|",
    ]
    notes: list[str] = []
    resumed: dict[str, str] = {}
    dirty: dict[str, str] = {}
    for name, arm in table["arms"].items():
        cells = []
        for run in arm["runs"]:
            short = run["run_id"][:8]
            marks = ("†" if run["resumed_at"] else "") + ("‡" if run["dirty"] else "")
            cells.append(f"{run['seed']}: `{short}`{marks}")
            if run["resumed_at"]:
                resumed[short] = f"`{short}` at `{run['resumed_at']}`"
            if run["dirty"]:
                dirty[short] = f"`{short}` (`{run['git_sha']}`)"
        if arm["drift_accepted"]:
            notes.append(
                f"`{name}`: its seed report accepted source drift between seeds in "
                + ", ".join(f"`{path}`" for path in arm["drift_accepted"])
                + " (`Memory.md` §7 #62)."
            )
        reference = " (reference)" if name == table["reference"] else ""
        metrics = arm["metrics"]
        lines.append(
            f"| {name}{reference} | {metrics['top_5']['n']} | {_pm(metrics['top_5'], 4)} | "
            f"{_pm(metrics['top_1'], 4)} | {_pm(metrics['ifa'], 2)} | {', '.join(cells)} |"
        )
    if table["comparisons"]:
        lines += [
            "",
            "| First − second | Top-5 | Top-1 | IFA | Top-5 by seed | On Top-5 |",
            "|---|---|---|---|---|---|",
        ]
        for comparison in table["comparisons"]:
            metrics = comparison["metrics"]
            verdict = comparison["outcome"]
            shown = f"**{verdict}**" if verdict not in {"not resolved", "not compared"} else verdict
            lines.append(
                f"| {comparison['first']} − {comparison['second']} | "
                f"{_delta(metrics['top_5'], 4)} | {_delta(metrics['top_1'], 4)} | "
                f"{_delta(metrics['ifa'], 2)} | {_by_seed(metrics['top_5'])} | {shown} |"
            )
    if resumed:
        notes.append("† resumed (`Memory.md` §7 #53): " + "; ".join(resumed.values()) + ".")
    if dirty:
        notes.append("‡ made from an uncommitted tree: " + ", ".join(dirty.values()) + ".")
    if notes:
        lines += ["", *[line for note in notes for line in (note, "")][:-1]]
    return [*lines, ""]


def _claim_line(ablation: Mapping[str, Any]) -> str:
    """The status of an ablation for the overview table: one clause per table."""
    if not ablation["tables"]:
        return "not run"
    parts = []
    for table in ablation["tables"]:
        if table["status"] == "pending":
            parts.append(f"`{table['key']}`: not yet run")
            continue
        outcomes = "; ".join(
            f"{comparison['first']} − {comparison['second']} {comparison['outcome']}"
            for comparison in table["comparisons"]
        )
        leaky = " (text with the tell)" if table["text"] == "as_shipped" else ""
        parts.append(f"`{table['key']}`{leaky}: {outcomes}")
    return "<br>".join(parts)


_PREAMBLE = (
    "Generated by `scripts/make_ablation_tables.py` from `{catalogue}`, MLflow run `{run_id}`. "
    "Every number is validation-selected on `val` — the epoch was chosen by early stopping on "
    "the same partition — and none is a held-out number. The values are the ones each training "
    "run logged (equal scores in `argsort` order, `Memory.md` §7 #63), so a Top-1 here can "
    "differ in the fourth decimal from `evaluation_val.md`, which breaks ties in source order. "
    "Mean ± sample sd over seeds; differences are first minus second. IFA is lower-is-better; "
    "a claim is judged on Top-5."
)

_HOW_TO_READ = (
    "**How to read an outcome.** A difference under {threshold:g} pooled sd is *not resolved*; "
    "fewer than three seeds on either side is *not compared*. This document makes {made} scaled "
    "comparisons ({claims} of them on Top-5) and {crossed} reach the threshold. With three seeds "
    "an arm, two arms with no difference reach it with probability about {chance:.2f} in any one "
    "comparison — about {expected:.1f} of {made} if they were independent, which they are not. "
    "A crossing is a candidate, not a finding (`Memory.md` §7 #64)."
)


def render_markdown(payload: Mapping[str, Any]) -> str:
    """The tables, as markdown.

    Args:
        payload: From :func:`build_tables`, with ``run_id`` and ``catalogue`` added.

    Returns:
        The document.
    """
    summary = payload["summary"]
    made, chance = summary["comparisons"], summary["chance"]
    lines = [
        "# Ablations — what each one tested, and what the logged runs say",
        "",
        _PREAMBLE.format(catalogue=payload["catalogue"], run_id=payload["run_id"]),
        "",
        "## Overview",
        "",
        "| | Ablation | Claim tested | What the runs say (on Top-5) |",
        "|---|---|---|---|",
    ]
    for ablation in payload["ablations"]:
        cells = (ablation["id"], ablation["name"], ablation["claim"], _claim_line(ablation))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        _HOW_TO_READ.format(
            threshold=summary["threshold"],
            made=made,
            claims=summary["claim_comparisons"],
            crossed=len(summary["crossings"]),
            chance=chance,
            expected=made * chance,
        ),
    ]
    if summary["crossings"]:
        lines += ["", "Past the threshold:", ""]
        lines += [
            f"- {item['ablation']} `{item['table']}` — {item['comparison']} on "
            f"{METRIC_LABELS[item['metric']]}: {item['difference_in_pooled_sd']:+.2f} pooled sd"
            for item in summary["crossings"]
        ]
    for ablation in payload["ablations"]:
        lines += [
            "",
            f"## {ablation['id']} — {ablation['name']}",
            "",
            f"**Claim tested.** {ablation['claim']}",
            "",
        ]
        if ablation.get("note"):
            lines += [str(ablation["note"]).strip(), ""]
        for table in ablation["tables"]:
            lines += _render_table(table)
        for item in ablation["not_run"]:
            lines += [f"**Not run — {str(item['what']).strip()}** {str(item['why']).strip()}", ""]
    return "\n".join(lines).rstrip() + "\n"


def make(cfg: DictConfig) -> dict[str, Any]:
    """Build the tables, write them and log the run that carries them.

    Args:
        cfg: Composed config. Reads ``ablations.catalogue`` (default
            ``./docs/ablations.yaml``) and ``ablations.output_dir`` (default
            ``./artifacts/phase8``).

    Returns:
        The payload, as written to ``<output_dir>/ablations.json``.

    Raises:
        ConfigError: If the catalogue is missing or malformed.
        DataError: When an artefact does not reconcile with the catalogue.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )
    catalogue_path = Path(
        str(OmegaConf.select(cfg, "ablations.catalogue") or "./docs/ablations.yaml")
    )
    output_dir = Path(str(OmegaConf.select(cfg, "ablations.output_dir") or "./artifacts/phase8"))
    if not catalogue_path.exists():
        raise ConfigError("the ablation catalogue is missing", path=str(catalogue_path))
    catalogue: dict[str, Any] = OmegaConf.to_container(OmegaConf.load(catalogue_path), resolve=True)  # type: ignore[assignment]
    payload = build_tables(catalogue, Path())
    payload.update(
        {"generated_by": "scripts/make_ablation_tables.py", "catalogue": catalogue_path.as_posix()}
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = output_dir / "ablations.json", output_dir / "ablations.md"
    with start_run(cfg, run_name="ablation-tables") as run_id:
        payload["run_id"] = run_id
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        markdown_path.write_text(render_markdown(payload), encoding="utf-8")
        mlflow.set_tags({"artefact": "ablation-tables", "split": SELECTION_SPLIT})
        summary = payload["summary"]
        mlflow.log_params(
            {
                "ablations.catalogue": catalogue_path.as_posix(),
                "ablations.tables_run": sum(
                    table["status"] == "run"
                    for entry in payload["ablations"]
                    for table in entry["tables"]
                ),
                "ablations.comparisons": summary["comparisons"],
                "ablations.crossings": len(summary["crossings"]),
            }
        )
        mlflow.log_artifact(str(json_path), artifact_path="ablations")
        mlflow.log_artifact(str(markdown_path), artifact_path="ablations")
        mlflow.log_artifact(str(catalogue_path), artifact_path="ablations")
    _LOGGER.info(
        "Ablation tables written",
        extra={
            "tables": str(markdown_path),
            "run_id": run_id,
            "comparisons": summary["comparisons"],
        },
    )
    return payload


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    make(cfg)


if __name__ == "__main__":
    main()
