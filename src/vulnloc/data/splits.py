"""Deterministic, leakage-checked dataset splitting (L0).

Three strategies, selected by config (`Rules.md` §5.5 — ablations are config-only):

* ``random`` — the field's default. Leaks: Big-Vul contains near-duplicate functions
  from the same project, and a random split scatters them across train and test.
  Retained only to reproduce published numbers, which are labelled ``(reported)``.
* ``project`` — no project appears in two splits. The honest generalization number,
  and what `ResearchPlan.md` §3.4 says to lead with.
* ``temporal`` — train on the past, evaluate on the future. Cuts on date boundaries so
  no single date straddles two splits.

Determinism (a Phase 1 exit criterion: *re-running reproduces byte-identical splits*) is
achieved by **content hashing, not RNG shuffling**. Each unit's bucket is a function of
``sha256(seed, key)``, so the assignment does not depend on row order, dict iteration
order, pandas version, or platform. A shuffle-based split satisfies "reproducible with
the same seed" but silently changes when the input is reordered; this does not.

ADR-003 adds two axes on top of the train/val/test split:

* ``excluded`` — PrimeVul explicitly contradicts the label. Dropped entirely.
* ``clean_eval`` — PrimeVul confirms the label. Held out of training and used for the
  headline line-level results.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from vulnloc.utils.exceptions import ConfigError, DataError
from vulnloc.utils.hashing import unit_interval
from vulnloc.utils.logging import get_logger
from vulnloc.utils.registry import Registry

__all__ = [
    "SPLIT_REGISTRY",
    "LeakageReport",
    "SplitResult",
    "SplitSpec",
    "SplitStrategy",
    "Unit",
    "assign_splits",
    "check_leakage",
    "register_split",
    "withhold_cross_split_duplicates",
]

_LOGGER = get_logger(__name__)

TRAIN, VAL, TEST = "train", "val", "test"
CLEAN_EVAL, EXCLUDED = "clean_eval", "excluded"
DUPLICATE = "duplicate"
_SPLIT_NAMES = (TRAIN, VAL, TEST)

# Partitions that hold data but are never trained on or evaluated against, and so
# cannot leak. Leakage checks skip them.
_WITHHELD = (EXCLUDED, DUPLICATE)

# Ratio arithmetic on floats is inexact; anything inside this of 1.0 is accepted.
_RATIO_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Unit:
    """The split-relevant metadata of one code unit.

    Deliberately not the full :class:`~vulnloc.representation.CodeUnit`: splitting needs
    identity, provenance and time, never source text. Keeping them apart means splits
    can be computed and tested without loading 10 GB of code.

    The two size counters are the exception, and they are here for a measurement reason
    rather than a splitting one: `Phases.md` Phase 1 requires per-split class balance
    (*mean lines/function*, *%vulnerable lines*) to be reported, and the units pass is
    the only cheap pass over the corpus. Both are derived from columns the loader
    already reads, so carrying them costs nothing; recovering them later would cost a
    second traversal of 10 GB.

    Args:
        unit_id: Stable identifier, unique within the corpus.
        project: Owning project, used by the ``project`` strategy.
        func_hash: Normalized-content hash, used for duplicate detection and for the
            PrimeVul join (ADR-003).
        date: Sortable date string (e.g. ``"2019-03-14"``), used by ``temporal``.
        target: 1 if the unit is vulnerable.
        has_line_labels: Whether line-level supervision exists for this unit.
        n_lines: Source lines in the function body.
        n_flagged_lines: Lines flagged vulnerable, before any range validation. This
            counts what the corpus *claims*; :mod:`vulnloc.data.labeling` counts what
            survives verification, and the two differing is itself a finding.
    """

    unit_id: str
    project: str
    func_hash: int
    date: str | None = None
    target: int = 0
    has_line_labels: bool = False
    n_lines: int = 0
    n_flagged_lines: int = 0


@dataclass(frozen=True)
class SplitSpec:
    """Split proportions and the seed that makes them reproducible.

    Args:
        train: Share of units for training.
        val: Share for validation.
        test: Share for the held-out test split.
        seed: Salt for the content hash. Changing it changes the assignment.

    Raises:
        ConfigError: If the ratios are negative or do not sum to 1.
    """

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        ratios = {TRAIN: self.train, VAL: self.val, TEST: self.test}
        negative = sorted(name for name, value in ratios.items() if value < 0)
        if negative:
            raise ConfigError(f"Split ratios must be non-negative; {negative} are not.")
        total = sum(ratios.values())
        if abs(total - 1.0) > _RATIO_TOLERANCE:
            raise ConfigError(f"Split ratios must sum to 1.0, got {total}.", ratios=ratios)

    @property
    def cumulative(self) -> list[tuple[str, float]]:
        """Return ``(split, upper_bound)`` pairs for bucketing a value in ``[0, 1)``."""
        bounds: list[tuple[str, float]] = []
        running = 0.0
        for name, ratio in ((TRAIN, self.train), (VAL, self.val), (TEST, self.test)):
            running += ratio
            bounds.append((name, running))
        return bounds


class SplitStrategy(Protocol):
    """Assigns units to ``train``/``val``/``test``."""

    def __call__(self, units: Sequence[Unit], spec: SplitSpec) -> dict[str, str]:
        """Return a mapping of ``unit_id`` to split name."""
        ...


SPLIT_REGISTRY: Registry[Any] = Registry("split_strategy")
register_split = SPLIT_REGISTRY.register


def _bucket(value: float, spec: SplitSpec) -> str:
    for name, upper in spec.cumulative:
        if value < upper:
            return name
    return TEST  # only reachable through floating-point drift at exactly 1.0


@register_split("random")
def random_split(units: Sequence[Unit], spec: SplitSpec) -> dict[str, str]:
    """Assign each unit independently by content hash.

    Warning:
        Leaks. Near-duplicate functions from the same project land on both sides, which
        is why published numbers on this split are inflated. Use for reproduction only.

    Args:
        units: Units to assign.
        spec: Proportions and seed.

    Returns:
        Mapping of ``unit_id`` to split name.
    """
    return {unit.unit_id: _bucket(unit_interval(spec.seed, unit.unit_id), spec) for unit in units}


@register_split("project")
def project_split(units: Sequence[Unit], spec: SplitSpec) -> dict[str, str]:
    """Assign whole projects, packing greedily to hit the target proportions.

    Hashing the project name would keep projects intact but not control their sizes, and
    Big-Vul cannot tolerate that: two projects hold ~66% of the corpus, so a hash-based
    assignment routinely lands both in the same split and starves the others.

    Instead, projects are sorted largest-first and each is placed in whichever split is
    furthest below its quota. That keeps projects intact *and* the proportions close,
    which is the only way ``ResearchPlan.md`` §3.4's project-wise split is usable here.

    Ties are broken by project name so the result never depends on input order.

    Args:
        units: Units to assign.
        spec: Proportions and seed.

    Returns:
        Mapping of ``unit_id`` to split name.
    """
    sizes = Counter(unit.project for unit in units)
    total = sum(sizes.values())
    quotas = {name: ratio * total for name, ratio in zip(_SPLIT_NAMES, _ratios(spec), strict=True)}
    filled = dict.fromkeys(_SPLIT_NAMES, 0.0)

    # Largest first; the hash breaks ties among equal-sized projects deterministically
    # without reintroducing an order dependence.
    ordered = sorted(
        sizes.items(), key=lambda item: (-item[1], unit_interval(spec.seed, item[0]), item[0])
    )

    project_to_split: dict[str, str] = {}
    for project, size in ordered:
        # "Most room left" rather than "least full": with an active quota of zero a split
        # must never be chosen, or an empty val/test would silently receive projects.
        candidate = max(
            (name for name in _SPLIT_NAMES if quotas[name] > 0),
            key=lambda name: (quotas[name] - filled[name], name),
            default=TRAIN,
        )
        project_to_split[project] = candidate
        filled[candidate] += size

    return {unit.unit_id: project_to_split[unit.project] for unit in units}


@register_split("temporal")
def temporal_split(units: Sequence[Unit], spec: SplitSpec) -> dict[str, str]:
    """Train on the past, evaluate on the future, cutting on date boundaries.

    Cut points are advanced to the next distinct date so no single date straddles two
    splits — otherwise commits landing on the boundary day leak across it.

    Args:
        units: Units to assign. Every unit must carry a ``date``.
        spec: Proportions and seed (the seed only breaks ties).

    Returns:
        Mapping of ``unit_id`` to split name.

    Raises:
        DataError: If any unit lacks a date, since a silent fallback would produce a
            split that looks temporal but is not.
    """
    undated = [unit.unit_id for unit in units if not unit.date]
    if undated:
        raise DataError(
            f"temporal split requires a date on every unit; {len(undated)} lack one. "
            f"Drop them explicitly or choose another strategy.",
            missing=len(undated),
            examples=undated[:5],
        )

    by_date: Counter[str] = Counter(str(unit.date) for unit in units)
    dates = sorted(by_date)
    total = sum(by_date.values())

    boundaries: dict[str, str] = {}
    running = 0
    ratios = _ratios(spec)
    train_cut = ratios[0] * total
    val_cut = (ratios[0] + ratios[1]) * total
    for date in dates:
        running += by_date[date]
        # A date is assigned by where the *end* of its block falls, keeping it whole.
        if running <= train_cut:
            boundaries[date] = TRAIN
        elif running <= val_cut:
            boundaries[date] = VAL
        else:
            boundaries[date] = TEST

    return {unit.unit_id: boundaries[str(unit.date)] for unit in units}


def _ratios(spec: SplitSpec) -> tuple[float, float, float]:
    return (spec.train, spec.val, spec.test)


def withhold_cross_split_duplicates(
    units: Sequence[Unit], assignment: dict[str, str]
) -> tuple[dict[str, str], int]:
    """Move exactly-duplicated functions out of train/val/test entirely.

    A project-wise split guarantees no *project* spans two partitions. It does not
    guarantee no *function* does, and on Big-Vul it demonstrably does not: forks and
    vendored copies put byte-identical functions in different projects (Chrome/Android,
    php/php-src, ImageMagick/ImageMagick6). Measured on the real corpus: 207 hashes,
    426 functions, with zero shared projects.

    Withholding rather than reassigning is the only option that preserves both
    invariants. Reassigning a duplicate to its twin's partition would drag a unit of a
    test-only project into training and break project disjointness; withholding costs
    0.23% of the corpus and keeps both guarantees exact.

    Args:
        units: The units that were assigned.
        assignment: Mapping of ``unit_id`` to split name. Not mutated.

    Returns:
        The updated assignment and the number of units withheld.
    """
    hash_splits: dict[int, set[str]] = {}
    for unit in units:
        split = assignment.get(unit.unit_id)
        if split in _SPLIT_NAMES:
            hash_splits.setdefault(unit.func_hash, set()).add(str(split))

    spanning = {digest for digest, splits in hash_splits.items() if len(splits) > 1}
    if not spanning:
        return dict(assignment), 0

    updated = dict(assignment)
    withheld = 0
    for unit in units:
        if unit.func_hash in spanning and updated.get(unit.unit_id) in _SPLIT_NAMES:
            updated[unit.unit_id] = DUPLICATE
            withheld += 1
    return updated, withheld


@dataclass
class LeakageReport:
    """Cross-split contamination findings (`Rules.md` §5.4).

    Attributes:
        duplicate_hashes: Content hashes appearing in more than one partition.
        duplicated_units: How many units those hashes account for.
        shared_projects: Projects appearing in more than one of train/val/test. The
            clean-evaluation partition is deliberately excluded: ADR-003 selects it by
            label confidence, so it shares projects with training *by construction*
            (`Memory.md` §7 #13). Counting that here would report a designed property
            as a defect and drown the real signal — the overlap is quantified
            separately by :func:`vulnloc.data.pipeline.clean_eval_overlap`.
        clean_eval_in_train: Clean-evaluation hashes that also appear in training —
            must always be empty, or the headline numbers are invalid.
        intra_split_duplicates: Hashes duplicated *within* a single partition.
    """

    duplicate_hashes: set[int] = field(default_factory=set)
    duplicated_units: int = 0
    shared_projects: set[str] = field(default_factory=set)
    clean_eval_in_train: set[int] = field(default_factory=set)
    intra_split_duplicates: int = 0

    @property
    def is_clean(self) -> bool:
        """True when nothing leaks across splits."""
        return not (self.duplicate_hashes or self.shared_projects or self.clean_eval_in_train)

    def as_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serializable summary."""
        return {
            "duplicate_hashes": len(self.duplicate_hashes),
            "duplicated_units": self.duplicated_units,
            "shared_projects": sorted(self.shared_projects),
            "clean_eval_in_train": len(self.clean_eval_in_train),
            "intra_split_duplicates": self.intra_split_duplicates,
            "is_clean": self.is_clean,
        }


def check_leakage(
    units: Sequence[Unit],
    assignment: Mapping[str, str],
    *,
    clean_eval_hashes: Iterable[int] = (),
) -> LeakageReport:
    """Detect cross-split contamination.

    Exact duplicates only. Near-duplicate detection needs tokenization and belongs to
    Phase 2; this establishes the floor, and the floor is already non-zero for Big-Vul.

    Args:
        units: The units that were assigned.
        assignment: Mapping of ``unit_id`` to split name.
        clean_eval_hashes: Hashes reserved for the clean evaluation set (ADR-003).

    Returns:
        The leakage findings.
    """
    hash_splits: dict[int, set[str]] = {}
    hash_counts: Counter[int] = Counter()
    project_splits: dict[str, set[str]] = {}

    for unit in units:
        split = assignment.get(unit.unit_id)
        if split is None or split in _WITHHELD:
            continue
        hash_splits.setdefault(unit.func_hash, set()).add(split)
        hash_counts[unit.func_hash] += 1
        if split in _SPLIT_NAMES:
            project_splits.setdefault(unit.project, set()).add(split)

    report = LeakageReport()
    for digest, splits in hash_splits.items():
        if len(splits) > 1:
            report.duplicate_hashes.add(digest)
            report.duplicated_units += hash_counts[digest]
    report.intra_split_duplicates = sum(
        count - 1 for digest, count in hash_counts.items() if count > 1
    )
    report.shared_projects = {
        project for project, splits in project_splits.items() if len(splits) > 1
    }

    reserved = set(clean_eval_hashes)
    if reserved:
        train_hashes = {unit.func_hash for unit in units if assignment.get(unit.unit_id) == TRAIN}
        report.clean_eval_in_train = reserved & train_hashes

    return report


@dataclass
class SplitResult:
    """The outcome of splitting a corpus.

    Attributes:
        assignment: ``unit_id`` -> one of train/val/test/clean_eval/excluded/duplicate.
        strategy: Name of the strategy used.
        spec: The proportions and seed used.
        leakage: Contamination findings.
        withheld_duplicates: Units moved out of train/val/test because an exact copy
            of them appeared in another partition.
    """

    assignment: dict[str, str]
    strategy: str
    spec: SplitSpec
    leakage: LeakageReport
    withheld_duplicates: int = 0

    def counts(self) -> dict[str, int]:
        """Return the number of units in each split, including empty ones."""
        counts = dict.fromkeys((*_SPLIT_NAMES, CLEAN_EVAL, EXCLUDED, DUPLICATE), 0)
        for split in self.assignment.values():
            counts[split] = counts.get(split, 0) + 1
        return counts

    def fingerprint(self) -> str:
        """Return a hash of the assignment.

        The Phase 1 exit criterion is that re-running reproduces *byte-identical*
        splits; comparing fingerprints is how that is checked in CI rather than by
        eyeballing counts.

        Returns:
            SHA-256 hex digest over the sorted ``unit_id -> split`` pairs.
        """
        digest = hashlib.sha256()
        for unit_id in sorted(self.assignment):
            digest.update(f"{unit_id}={self.assignment[unit_id]}\n".encode())
        return digest.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary (the assignment itself is excluded)."""
        return {
            "strategy": self.strategy,
            "seed": self.spec.seed,
            "ratios": {TRAIN: self.spec.train, VAL: self.spec.val, TEST: self.spec.test},
            "counts": self.counts(),
            "withheld_duplicates": self.withheld_duplicates,
            "fingerprint": self.fingerprint(),
            "leakage": self.leakage.as_dict(),
        }


def assign_splits(
    units: Sequence[Unit],
    *,
    strategy: str = "project",
    spec: SplitSpec | None = None,
    clean_eval_hashes: Iterable[int] = (),
    excluded_hashes: Iterable[int] = (),
    dedup_across_splits: bool = True,
) -> SplitResult:
    """Split a corpus, applying the ADR-003 exclusions and hold-out first.

    Order matters and is not interchangeable:

    1. Drop ``excluded`` units — a higher-accuracy source contradicts their label.
    2. Reserve ``clean_eval`` units — held out of training entirely.
    3. Split what remains by ``strategy``.
    4. Withhold any function that still ended up in two partitions.

    Doing this in the other order would let the clean evaluation set influence the
    train/val/test proportions and, worse, allow a duplicate of a clean-eval function
    into training.

    Args:
        units: The corpus to split.
        strategy: Registered strategy name (``random``, ``project``, ``temporal``).
        spec: Proportions and seed; defaults to 80/10/10 at seed 42.
        clean_eval_hashes: Hashes reserved for the clean evaluation set (ADR-003 §D3.1).
        excluded_hashes: Hashes to drop entirely (ADR-003 §D3.2).
        dedup_across_splits: Withhold exact duplicates that span two partitions. Turning
            this off reproduces the field's usual setup, in which train/test share
            byte-identical functions; any number produced that way is inflated and must
            be labelled as such.

    Returns:
        The assignment, its fingerprint, and a leakage report.

    Raises:
        RegistryError: If ``strategy`` is unknown.
        DataError: If ``unit_id`` values are not unique, since the assignment is keyed
            by them and collisions would silently drop units.
    """
    spec = spec or SplitSpec()
    duplicated_ids = [uid for uid, count in Counter(u.unit_id for u in units).items() if count > 1]
    if duplicated_ids:
        raise DataError(
            f"unit_id must be unique; {len(duplicated_ids)} are repeated.",
            examples=sorted(duplicated_ids)[:5],
        )

    strategy_fn = SPLIT_REGISTRY.get(strategy)
    reserved = set(clean_eval_hashes)
    dropped = set(excluded_hashes)

    assignment: dict[str, str] = {}
    remaining: list[Unit] = []
    for unit in units:
        if unit.func_hash in dropped:
            assignment[unit.unit_id] = EXCLUDED
        elif unit.func_hash in reserved:
            assignment[unit.unit_id] = CLEAN_EVAL
        else:
            remaining.append(unit)

    assignment.update(strategy_fn(remaining, spec))

    withheld = 0
    if dedup_across_splits:
        assignment, withheld = withhold_cross_split_duplicates(units, assignment)

    result = SplitResult(
        assignment=assignment,
        strategy=strategy,
        spec=spec,
        leakage=check_leakage(units, assignment, clean_eval_hashes=reserved),
        withheld_duplicates=withheld,
    )
    _LOGGER.info("Split assigned", extra=result.as_dict())
    if not result.leakage.is_clean:
        _LOGGER.warning(
            "Split leaks across partitions; investigate before reporting any number",
            extra=result.leakage.as_dict(),
        )
    return result
