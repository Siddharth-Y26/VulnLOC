"""Split-building pipeline: corpus in, versioned partition out (L0).

This is the step that turns the pieces of Phase 1 — the loader, the ADR-003 membership
measured by :mod:`vulnloc.data.audit`, and the strategies in
:mod:`vulnloc.data.splits` — into an artefact everything downstream reads. After this
runs, no later phase re-derives a split: it loads ``assignment.csv`` and trusts it.

Three Phase 1 exit criteria are discharged here rather than in a notebook:

* *statistics reported* — per-split composition, including the two denominators that
  are easy to conflate (functions vs. line-labelled functions) and the two line counts
  (source lines vs. flagged lines).
* *leakage check run* — :func:`~vulnloc.data.splits.check_leakage` on the real corpus,
  with its findings written to the report rather than printed.
* *re-running reproduces byte-identical splits* — :func:`write_split_artifacts`
  compares against whatever is already on disk and refuses to overwrite a differing
  assignment, so a drift in the loader, the corpus, or the strategy surfaces as a
  failure instead of a quietly different experiment.

The membership file is deliberately an *input*, not something recomputed here: the
clean-evaluation and excluded hashes are a reported result (ADR-003 §D3.3, the 19.2%
noise rate) and must come from the audit run that produced them.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vulnloc.data.registry import DATASET_REGISTRY
from vulnloc.data.splits import (
    CLEAN_EVAL,
    EXCLUDED,
    TRAIN,
    SplitResult,
    SplitSpec,
    Unit,
    assign_splits,
)
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger

__all__ = [
    "ASSIGNMENT_FILENAME",
    "REPORT_FILENAME",
    "Adr003Membership",
    "SplitComposition",
    "build_splits",
    "clean_eval_overlap",
    "compose_report",
    "load_adr003_membership",
    "load_units",
    "project_breakdown",
    "split_composition",
    "write_split_artifacts",
]

_LOGGER = get_logger(__name__)

ASSIGNMENT_FILENAME = "assignment.csv"
REPORT_FILENAME = "split_report.json"

_ASSIGNMENT_HEADER = ("unit_id", "split")


@dataclass(frozen=True)
class Adr003Membership:
    """The hold-out and exclusion sets ADR-003 defines, keyed by ``func_hash``.

    Attributes:
        clean_eval: Hashes PrimeVul confirms vulnerable — the clean evaluation set
            (§D3.1). Held out of training entirely.
        excluded: Hashes PrimeVul explicitly contradicts (§D3.2). Dropped.
    """

    clean_eval: frozenset[int] = frozenset()
    excluded: frozenset[int] = frozenset()

    def as_dict(self) -> dict[str, int]:
        """Return sizes only; the hashes themselves live in the audit report."""
        return {"clean_eval_hashes": len(self.clean_eval), "excluded_hashes": len(self.excluded)}


def load_adr003_membership(report_path: Path | str) -> Adr003Membership:
    """Read the clean-evaluation and excluded hash sets from an audit report.

    Args:
        report_path: Path to ``audit_report.json`` as written by
            ``scripts/audit_data.py``.

    Returns:
        The membership sets.

    Raises:
        DataError: If the file is missing, or was produced with the PrimeVul
            cross-reference skipped. A silent empty membership would train on the
            functions ADR-003 excludes and evaluate on nothing, while still producing
            a plausible-looking report.
    """
    path = Path(report_path)
    if not path.exists():
        raise DataError(
            f"Audit report not found: {path}. Run scripts/audit_data.py first — the "
            f"ADR-003 membership is a measured result and is not recomputed here.",
            path=str(path),
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if "clean_eval_hashes" not in report:
        raise DataError(
            f"{path.name} carries no clean_eval_hashes; it was produced with "
            f"data.audit.skip_overlap=true. Re-run the audit with the PrimeVul "
            f"cross-reference enabled, or set data.splits.apply_adr003=false to build "
            f"a split without the ADR-003 hold-out and say so in the report.",
            path=str(path),
        )
    return Adr003Membership(
        clean_eval=frozenset(int(h) for h in report["clean_eval_hashes"]),
        excluded=frozenset(int(h) for h in report.get("excluded_hashes", [])),
    )


def load_units(dataset: str, raw_dir: Path | str, **loader_kwargs: Any) -> list[Unit]:
    """Materialize every unit of a registered dataset.

    Only metadata is loaded, never source text, so the whole 188k-function corpus fits
    comfortably in memory — which is what splitting needs, since project packing and
    leakage detection are both global operations.

    Args:
        dataset: Registered dataset name (``bigvul``).
        raw_dir: Directory holding the raw corpus files.
        **loader_kwargs: Forwarded to the loader constructor.

    Returns:
        Every unit in the corpus.

    Raises:
        RegistryError: If ``dataset`` is not registered.
        DataError: If the corpus yields no units, which means the files are present but
            unreadable — a case that would otherwise produce an empty, "clean" split.
    """
    loader = DATASET_REGISTRY.get(dataset)(raw_dir, **loader_kwargs)
    units = list(loader.iter_units())
    if not units:
        raise DataError(f"Dataset {dataset!r} yielded no units from {raw_dir}.", dataset=dataset)
    _LOGGER.info("Loaded units", extra={"dataset": dataset, "units": len(units)})
    return units


@dataclass
class SplitComposition:
    """Measured composition of one partition.

    Both denominators are carried explicitly. ``functions`` is the function-level
    denominator; ``line_labelled`` is the line-level one, and it is far smaller —
    24.4% of Big-Vul's positives carry no line annotation at all. Reporting a
    line-level metric against the function-level denominator would inflate it, so the
    two never share a field.

    Attributes:
        split: Partition name.
        functions: Units in the partition.
        vulnerable: Units with ``target == 1``.
        line_labelled: Vulnerable units that carry line-level supervision.
        source_lines: Total source lines across the partition.
        flagged_lines: Total lines flagged vulnerable.
        projects: Distinct projects present.
    """

    split: str
    functions: int = 0
    vulnerable: int = 0
    line_labelled: int = 0
    source_lines: int = 0
    flagged_lines: int = 0
    projects: set[str] = field(default_factory=set)

    def record(self, unit: Unit) -> None:
        """Fold one unit into the running counters.

        Args:
            unit: A unit assigned to this partition.
        """
        self.functions += 1
        self.vulnerable += int(unit.target == 1)
        self.line_labelled += int(unit.target == 1 and unit.has_line_labels)
        self.source_lines += unit.n_lines
        self.flagged_lines += unit.n_flagged_lines
        self.projects.add(unit.project)

    def as_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serializable summary with derived rates."""
        return {
            "split": self.split,
            "functions": self.functions,
            "vulnerable": self.vulnerable,
            "vulnerable_pct": _pct(self.vulnerable, self.functions),
            "line_labelled_functions": self.line_labelled,
            "line_labelled_pct_of_vulnerable": _pct(self.line_labelled, self.vulnerable),
            "source_lines": self.source_lines,
            "mean_lines_per_function": (
                round(self.source_lines / self.functions, 2) if self.functions else 0.0
            ),
            "flagged_lines": self.flagged_lines,
            "flagged_lines_pct": _pct(self.flagged_lines, self.source_lines),
            "distinct_projects": len(self.projects),
        }


def _pct(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 4) if denominator else 0.0


def split_composition(
    units: Sequence[Unit], assignment: dict[str, str]
) -> dict[str, SplitComposition]:
    """Measure the composition of every partition.

    Args:
        units: The units that were assigned.
        assignment: Mapping of ``unit_id`` to split name.

    Returns:
        One :class:`SplitComposition` per partition present in the assignment, plus an
        ``all`` entry covering the corpus.
    """
    compositions: dict[str, SplitComposition] = {"all": SplitComposition("all")}
    for unit in units:
        name = assignment.get(unit.unit_id)
        if name is None:
            continue
        compositions.setdefault(name, SplitComposition(name)).record(unit)
        compositions["all"].record(unit)
    return compositions


def build_splits(
    units: Sequence[Unit],
    *,
    strategy: str = "project",
    spec: SplitSpec | None = None,
    membership: Adr003Membership | None = None,
    dedup_across_splits: bool = True,
) -> SplitResult:
    """Assign every unit to a partition.

    Args:
        units: The corpus.
        strategy: Registered strategy name.
        spec: Proportions and seed.
        membership: ADR-003 hold-out and exclusion sets. ``None`` builds a plain
            train/val/test split with no clean evaluation set.
        dedup_across_splits: Withhold exact duplicates spanning two partitions.

    Returns:
        The assignment, its fingerprint and the leakage report.
    """
    membership = membership or Adr003Membership()
    return assign_splits(
        units,
        strategy=strategy,
        spec=spec,
        clean_eval_hashes=membership.clean_eval,
        excluded_hashes=membership.excluded,
        dedup_across_splits=dedup_across_splits,
    )


def compose_report(
    result: SplitResult,
    compositions: dict[str, SplitComposition],
    *,
    dataset: str,
    membership: Adr003Membership,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON split report.

    Args:
        result: The split outcome.
        compositions: Per-partition statistics.
        dataset: Registered dataset name the split was built from.
        membership: The ADR-003 sets applied.
        extra: Additional provenance to record (run id, source paths).

    Returns:
        A JSON-serializable report.
    """
    return {
        "dataset": dataset,
        **result.as_dict(),
        "adr003": membership.as_dict(),
        "composition": {name: comp.as_dict() for name, comp in sorted(compositions.items())},
        **(extra or {}),
    }


def _read_assignment(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if tuple(header or ()) != _ASSIGNMENT_HEADER:
            raise DataError(
                f"{path.name} is not a split assignment: expected header "
                f"{list(_ASSIGNMENT_HEADER)}, got {header}.",
                path=str(path),
            )
        return {row[0]: row[1] for row in reader if row}


def write_split_artifacts(
    output_dir: Path | str,
    result: SplitResult,
    report: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    r"""Write the assignment and report, checking reproducibility against any existing one.

    The assignment is written sorted by ``unit_id`` with ``\n`` line endings, so two
    runs over the same corpus produce byte-identical files — the Phase 1 exit criterion
    is byte-identity, not merely equivalent membership.

    Args:
        output_dir: Directory to write into.
        result: The split outcome.
        report: The report to write alongside it.
        overwrite: Permit replacing an assignment whose fingerprint differs. Intended
            for deliberate changes (a new strategy or seed), never for routine reruns.

    Returns:
        The report, with a ``reproduced`` field recording the comparison: ``True`` if a
        previous assignment was present and matched, ``False`` if it differed and was
        overwritten, ``None`` if there was nothing to compare against.

    Raises:
        DataError: If an existing assignment differs and ``overwrite`` is False.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    assignment_path = directory / ASSIGNMENT_FILENAME

    reproduced: bool | None = None
    if assignment_path.exists():
        previous = _read_assignment(assignment_path)
        reproduced = previous == result.assignment
        if not reproduced and not overwrite:
            differing = sum(
                1
                for uid in set(previous) | set(result.assignment)
                if previous.get(uid) != result.assignment.get(uid)
            )
            raise DataError(
                f"The split changed: {differing} of {len(result.assignment)} units are "
                f"assigned differently than in {assignment_path}. Splitting is "
                f"content-hashed and must be reproducible (Phases.md Phase 1), so this "
                f"means the corpus, the loader or the strategy changed. Investigate, "
                f"then pass data.splits.overwrite=true if the change is intended.",
                path=str(assignment_path),
                differing_units=differing,
                previous_units=len(previous),
                current_units=len(result.assignment),
            )
        _LOGGER.info(
            "Compared against the existing assignment",
            extra={"reproduced": reproduced, "path": str(assignment_path)},
        )

    # newline="" keeps csv from turning "\n" into "\r\n" on Windows; the fingerprint is
    # computed over content, but the exit criterion is about the file on disk.
    with assignment_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_ASSIGNMENT_HEADER)
        for unit_id in sorted(result.assignment):
            writer.writerow((unit_id, result.assignment[unit_id]))

    report = {**report, "reproduced": reproduced}
    (directory / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def project_breakdown(
    units: Iterable[Unit], assignment: dict[str, str], *, top_n: int = 15
) -> dict[str, list[tuple[str, int]]]:
    """Return the largest projects in each partition.

    Big-Vul's two dominant projects necessarily land in training (`Memory.md` §7 #12),
    which makes val and test *smaller-project* sets rather than merely unseen-project
    ones. That is a threat to validity, so it is measured and recorded rather than
    left implicit.

    Args:
        units: The units that were assigned.
        assignment: Mapping of ``unit_id`` to split name.
        top_n: Projects to list per partition.

    Returns:
        Partition name to ``(project, functions)`` pairs, largest first.
    """
    per_split: dict[str, Counter[str]] = {}
    for unit in units:
        name = assignment.get(unit.unit_id)
        if name in (None, EXCLUDED):
            continue
        per_split.setdefault(str(name), Counter())[unit.project] += 1
    return {name: counts.most_common(top_n) for name, counts in sorted(per_split.items())}


def clean_eval_overlap(units: Sequence[Unit], assignment: dict[str, str]) -> dict[str, Any]:
    """Measure how far the clean evaluation set overlaps training *projects*.

    ADR-003 selects the clean evaluation set by label confidence, not by project, so it
    is **not** project-disjoint from training (`Memory.md` §7 #13). Reporting it as a
    generalization result would be wrong. The overlap is quantified here so the
    DATA_CARD can state the size of the caveat rather than gesture at it.

    Args:
        units: The units that were assigned.
        assignment: Mapping of ``unit_id`` to split name.

    Returns:
        Counts of clean-evaluation units whose project also appears in training.
    """
    train_projects = {u.project for u in units if assignment.get(u.unit_id) == TRAIN}
    clean = [u for u in units if assignment.get(u.unit_id) == CLEAN_EVAL]
    shared = [u for u in clean if u.project in train_projects]
    return {
        "clean_eval_functions": len(clean),
        "clean_eval_flagged_lines": sum(u.n_flagged_lines for u in clean),
        "in_a_training_project": len(shared),
        "in_a_training_project_pct": _pct(len(shared), len(clean)),
        "distinct_projects": len({u.project for u in clean}),
    }
