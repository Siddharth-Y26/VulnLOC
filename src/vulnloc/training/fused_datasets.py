"""Shards → batches carrying both branches for the same units (L3, Phase 5).

Experiment A1 compares fusion strategies, so the one thing this module must guarantee is
that the semantic branch and the structural branch of a batch describe **the same
functions, in the same order, on the same line axis**. Nothing downstream checks it and
nothing downstream could: swap two units between the branches and the model trains
happily on one function's tokens fused with another's control flow, converging to
something plausible and meaningless.

So the agreement is enforced twice, at both places it could break.

**At construction**, by intersecting on ``unit_id`` rather than by trusting two shard
walks to produce the same list. They need not: ``LineLabelledDataset`` keeps every
line-labelled unit, while ``GraphLineDataset`` drops any whose graph will not build —
measured at **0 failures on `train` and `val`** (``Memory.md`` §1.15's runs log
``graph_build_failures=0`` at 5,457 / 488 units), which is
exactly why an index-based join would have passed every test and then broken silently on
the first partition where it did not hold. Every dropped unit is counted and logged,
because a fused arm training on a smaller population than the baselines it is compared
against is a confound, not a detail.

**At collation**, by rebuilding both sub-batches through the two existing collate
functions and asserting they agree on ``unit_ids``, ``line_labels``, ``line_valid`` and
``targets``. Reusing those functions rather than reimplementing them is deliberate: the
fused arm must batch its windows and its nodes exactly as the two baselines batched
theirs, or the comparison picks up the difference between two collation routines.

**The batch is flat and prefixed** — ``sem_*``, ``str_*``, shared keys unprefixed. See
:mod:`vulnloc.models.fused` for why nesting it would half-move it to the GPU in silence.

**Batching is bounded by both budgets at once.** A unit costs windows on one side and
nodes on the other, and the two are uncorrelated — a 60-line function with a deep AST is
cheap for the encoder and expensive for the GNN. :class:`FusedBudgetBatchSampler` closes a
batch when *either* budget would be exceeded, which is the only way peak memory stays a
property of the sampler on a 4 GiB card (``Memory.md`` §1.9).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset, Sampler

from vulnloc.training.datasets import EncodedUnit, LineLabelledDataset, collate_units
from vulnloc.training.graph_datasets import GraphLineDataset, GraphUnit, collate_graphs
from vulnloc.training.semantic_cache import CachedLines, SemanticFeatureCache
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from vulnloc.representation.tokenization import LineTokenizer

__all__ = [
    "FusedBudgetBatchSampler",
    "FusedLineDataset",
    "FusedUnit",
    "collate_fused",
    "fused_line_coverage",
    "join_branches",
]

_LOGGER = get_logger(__name__)

_SHARED_KEYS = ("line_labels", "line_valid", "targets", "unit_ids", "n_units")
"""Keys both branches produce, which must be identical and appear once in a fused batch."""


@dataclass(frozen=True)
class FusedUnit:
    """One function, encoded for both branches.

    Attributes:
        unit_id: Stable corpus identifier. Equal on both halves by construction.
        semantic: The tokenized half.
        structural: The graph half.
    """

    unit_id: str
    semantic: EncodedUnit
    structural: GraphUnit
    sem_cached: CachedLines | None = None
    """The unit's frozen-encoder line features, when the dataset was given a cache (ADR-009)."""

    @property
    def n_windows(self) -> int:
        """Windows this unit costs the semantic branch."""
        return self.semantic.n_windows

    @property
    def n_nodes(self) -> int:
        """Nodes this unit costs the structural branch."""
        return self.structural.n_nodes

    @property
    def flagged_beyond_cap(self) -> int:
        """Flagged lines past the line cap. Identical on both halves."""
        return self.semantic.flagged_beyond_cap


def join_branches(semantic: LineLabelledDataset, structural: GraphLineDataset) -> list[FusedUnit]:
    """Intersect the two branches on ``unit_id``, in the semantic branch's order.

    Module-level and public because it is the single place the A1 comparison can be
    silently invalidated, and a test of it should not have to reach into a class to get
    at it.

    Args:
        semantic: The tokenized dataset.
        structural: The graph dataset.

    Returns:
        The units present in both, each carrying the same ``unit_id`` on both halves.

    Raises:
        DataError: If a ``unit_id`` repeats within a branch, the join is empty, or the two
            branches disagree about a unit's labels.
    """
    by_id: dict[str, GraphUnit] = {}
    for graph_unit in structural.units:
        if graph_unit.unit_id in by_id:
            raise DataError(
                "duplicate unit_id in the structural branch; the join would be ambiguous",
                unit_id=graph_unit.unit_id,
                split=structural.split,
            )
        by_id[graph_unit.unit_id] = graph_unit

    seen: set[str] = set()
    joined: list[FusedUnit] = []
    for encoded in semantic.units:
        if encoded.unit_id in seen:
            raise DataError(
                "duplicate unit_id in the semantic branch; the join would be ambiguous",
                unit_id=encoded.unit_id,
                split=semantic.split,
            )
        seen.add(encoded.unit_id)
        graph = by_id.get(encoded.unit_id)
        if graph is None:
            continue
        # The two branches derived these from the same `CodeUnit` by the same rules. If
        # they ever differ, one of the two paths changed and the line axis no longer means
        # the same thing on both sides — which is the failure this module exists to make
        # impossible.
        if (
            encoded.line_labels != graph.line_labels
            or encoded.n_lines != graph.n_lines
            or encoded.target != graph.target
        ):
            raise DataError(
                "the two branches disagree about a unit's line supervision",
                unit_id=encoded.unit_id,
                semantic_n_lines=encoded.n_lines,
                structural_n_lines=graph.n_lines,
                semantic_target=encoded.target,
                structural_target=graph.target,
                labels_match=encoded.line_labels == graph.line_labels,
            )
        joined.append(FusedUnit(unit_id=encoded.unit_id, semantic=encoded, structural=graph))

    if not joined:
        raise DataError(
            "no unit is present in both branches",
            split=semantic.split,
            semantic_units=len(semantic),
            structural_units=len(structural),
        )
    return joined


class FusedLineDataset(Dataset[FusedUnit]):
    """Units present in **both** branches, joined on ``unit_id``.

    Args:
        cache_dir: Directory holding the preprocessed shards.
        split: Shard name — ``train``, ``val``, ``test``, ``clean_eval``.
        tokenizer: The line-preserving tokenizer.
        node_vocab: The corpus-wide ``symbol -> index`` table. Required, for the reason
            :class:`~vulnloc.training.graph_datasets.GraphLineDataset` documents.
        subtoken_vocab: ADR-002 §D2.5's identifier table, or ``None``.
        builder: Graph type — a ``GRAPH_REGISTRY`` key.
        language: Grammar-selection policy (ADR-005).
        n_lines: Line-axis cap ``L``. One value for both branches: it is a property of the
            shared line axis, and two branches capping differently would put a silent
            difference inside the A1 comparison.
        limit: Stop after roughly this many units per branch. ``0`` means the whole shard.
            Applied to each branch independently before the join, so the fused population
            can come out slightly smaller than ``limit``; the log line says by how much.
        line_labelled_only: Keep only functions carrying line labels.
        normalize_whitespace: Strip edge whitespace from every source line first.
            **On by default: leaving it off leaks the label** (ADR-006).
        semantic_cache: A verified frozen-encoder feature cache (ADR-009). Every joined
            unit then carries its cached line features, and batches carry
            ``sem_line_states`` so the model skips the encoder. Tokenization still runs:
            the window budget and the active line axis are computed exactly as on the live
            path, so batch composition is identical whether the cache is used or not.

    Raises:
        DataError: If the join is empty, a ``unit_id`` is duplicated within a branch, the
            two branches disagree about a unit's labels, or a joined unit is absent from
            the cache.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        split: str,
        tokenizer: LineTokenizer,
        node_vocab: Mapping[str, int],
        *,
        subtoken_vocab: Mapping[str, int] | None = None,
        builder: str = "ast",
        language: str = "best_of_two",
        n_lines: int = 512,
        limit: int = 0,
        line_labelled_only: bool = False,
        normalize_whitespace: bool = True,
        semantic_cache: SemanticFeatureCache | None = None,
    ) -> None:
        self.split = split
        self.n_lines = n_lines
        self.semantic_cache_digest = semantic_cache.digest if semantic_cache else ""

        semantic = LineLabelledDataset(
            cache_dir,
            split,
            tokenizer,
            n_lines=n_lines,
            limit=limit,
            line_labelled_only=line_labelled_only,
            normalize_whitespace=normalize_whitespace,
        )
        structural = GraphLineDataset(
            cache_dir,
            split,
            node_vocab,
            subtoken_vocab=subtoken_vocab,
            builder=builder,
            language=language,
            n_lines=n_lines,
            limit=limit,
            line_labelled_only=line_labelled_only,
            normalize_whitespace=normalize_whitespace,
        )
        self.units = join_branches(semantic, structural)
        if semantic_cache is not None:
            # `get` raises on a unit the cache never saw: a population mismatch must not be
            # papered over with zero features.
            self.units = [
                FusedUnit(
                    unit_id=unit.unit_id,
                    semantic=unit.semantic,
                    structural=unit.structural,
                    sem_cached=semantic_cache.get(unit.unit_id),
                )
                for unit in self.units
            ]
        self.flagged_beyond_cap = sum(unit.flagged_beyond_cap for unit in self.units)
        self.flagged_without_node = sum(unit.structural.flagged_without_node for unit in self.units)
        self.build_failures = structural.build_failures
        self.semantic_only = len(semantic) - len(self.units)
        self.structural_only = len(structural) - len(self.units)

        _LOGGER.info(
            "Fused dataset joined",
            extra={
                "split": split,
                "units": len(self.units),
                "semantic_units": len(semantic),
                "structural_units": len(structural),
                # If either of these is non-zero the fused arm is training on a different
                # population from the baseline it will be compared against. It is logged
                # rather than asserted because the honest response is to report it, not to
                # crash — but it must never be discovered at write-up time.
                "dropped_semantic_only": self.semantic_only,
                "dropped_structural_only": self.structural_only,
                "windows": sum(unit.n_windows for unit in self.units),
                "nodes": sum(unit.n_nodes for unit in self.units),
                "positives": sum(unit.semantic.target for unit in self.units),
                "flagged_beyond_line_cap": self.flagged_beyond_cap,
                "flagged_without_a_graph_node": self.flagged_without_node,
                "graph_build_failures": structural.build_failures,
                "n_lines_cap": n_lines,
                "normalize_whitespace": normalize_whitespace,
                "semantic_cache_digest": self.semantic_cache_digest[:12] or None,
            },
        )

    def __len__(self) -> int:
        """Number of units present in both branches."""
        return len(self.units)

    def __getitem__(self, index: int) -> FusedUnit:
        """Return one fused unit."""
        return self.units[index]

    def positive_line_weight(self) -> float:
        """Return ``negatives / positives`` over scoreable lines, for a weighted loss.

        Computed exactly as both baselines compute it, over the joined population — so the
        weight describes the data this arm actually trains on rather than either branch's
        own set.

        Returns:
            The ratio, or ``1.0`` when there are no positive lines.
        """
        positives = 0
        scoreable = 0
        for unit in self.units:
            usable = min(unit.semantic.n_lines, self.n_lines)
            scoreable += usable
            positives += sum(unit.semantic.line_labels[:usable])
        if positives == 0:
            return 1.0
        return (scoreable - positives) / positives


def collate_fused(units: Sequence[FusedUnit]) -> dict[str, Any]:
    """Flatten fused units into one batch carrying both branches.

    Both halves go through the baselines' own collate functions, then the shared keys are
    checked for agreement and emitted once. The check is cheap and decisive: it is the
    last place a mismatch between the branches could still be caught before it becomes a
    trained model.

    Args:
        units: The units in this batch.

    Returns:
        A batch carrying ``sem_input_ids`` ``[W, T]``, ``sem_attention_mask`` ``[W, T]``,
        ``sem_token_to_line`` ``[W, T]``, ``sem_window_to_unit`` ``[W]``,
        ``str_node_type`` ``[N]``, ``str_line`` ``[N]``, ``str_edge_index`` ``[2, E]``,
        ``str_edge_type`` ``[E]``, ``str_batch`` ``[N]``, optionally ``str_subtokens``
        ``[N, K]``, plus the shared ``line_labels`` ``[B, L]``, ``line_valid`` ``[B, L]``,
        ``targets`` ``[B]``, ``unit_ids`` and ``n_units``. When the units carry cached
        features (ADR-009), also ``sem_line_states`` ``[B, R, D]`` (stored dtype) and
        ``sem_line_mask`` ``[B, R]``, where ``R`` is the most rows any unit stored.

    Raises:
        DataError: If ``units`` is empty, the two halves disagree on a shared key, or some
            units carry cached features and others do not.
    """
    if not units:
        raise DataError("cannot collate an empty batch")

    semantic = collate_units([unit.semantic for unit in units])
    structural = collate_graphs([unit.structural for unit in units])

    if semantic["unit_ids"] != structural["unit_ids"]:
        raise DataError(
            "the two branches collated different units",
            semantic=semantic["unit_ids"],
            structural=structural["unit_ids"],
        )
    for key in ("line_labels", "line_valid", "targets"):
        if not torch.equal(semantic[key], structural[key]):
            raise DataError(
                "the two branches collated different line supervision",
                key=key,
                unit_ids=semantic["unit_ids"],
            )

    batch: dict[str, Any] = {key: semantic[key] for key in _SHARED_KEYS}
    batch |= {f"sem_{key}": value for key, value in semantic.items() if key not in _SHARED_KEYS}
    batch |= {f"str_{key}": value for key, value in structural.items() if key not in _SHARED_KEYS}

    cached = [unit.sem_cached for unit in units]
    if any(entry is not None for entry in cached):
        if any(entry is None for entry in cached):
            # Half a batch on cached features and half on the live encoder would put two
            # different semantic paths inside one gradient step.
            raise DataError(
                "some units in the batch carry cached semantic features and some do not",
                unit_ids=semantic["unit_ids"],
            )
        present = [entry for entry in cached if entry is not None]
        rows = max(1, max(entry.n_rows for entry in present))
        width = int(present[0].states.shape[1])
        # The stored dtype, not a hardcoded one — see `encode_units` on why it is fp32.
        states = torch.zeros((len(units), rows, width), dtype=present[0].states.dtype)
        mask = torch.zeros((len(units), rows), dtype=torch.bool)
        for index, entry in enumerate(present):
            states[index, : entry.n_rows] = entry.states
            mask[index, : entry.n_rows] = entry.mask
        batch["sem_line_states"] = states
        batch["sem_line_mask"] = mask
    return batch


class FusedBudgetBatchSampler(Sampler[list[int]]):
    """Groups units by a **window** budget and a **node** budget at once.

    The two costs are uncorrelated — a long flat function is expensive for the encoder and
    cheap for the GNN, a short deeply-nested one the reverse — so respecting only one of
    them leaves peak memory to whichever the batch happened not to be sized by. A batch
    closes when either budget would be exceeded.

    This duplicates the shape of :class:`~vulnloc.training.datasets.WindowBatchSampler` and
    :class:`~vulnloc.training.graph_datasets.NodeBudgetBatchSampler` rather than
    generalising them. That is on purpose: those two produced every logged number in
    Phases 3 and 4, and refactoring them to serve a third caller would put "was it the
    refactor?" between those runs and the fused ones they are about to be compared with.

    A unit over budget on its own still gets its own batch, for the reason both of those
    record: dropping it would discard the largest functions, which carry the most flagged
    lines (``Memory.md`` §1.8).

    Args:
        dataset: The dataset to sample from.
        max_windows: Window budget per batch.
        max_nodes: Node budget per batch.
        shuffle: Shuffle unit order each epoch.
        seed: Base seed; the epoch number is mixed in so each epoch differs reproducibly.
        max_units: Optional cap on units per batch, regardless of either cost.

    Raises:
        DataError: If either budget is not positive.
    """

    def __init__(
        self,
        dataset: FusedLineDataset,
        *,
        max_windows: int = 8,
        max_nodes: int = 4096,
        shuffle: bool = True,
        seed: int = 42,
        max_units: int = 0,
    ) -> None:
        if max_windows <= 0 or max_nodes <= 0:
            raise DataError(
                "both budgets must be positive", max_windows=max_windows, max_nodes=max_nodes
            )
        self.dataset = dataset
        self.max_windows = max_windows
        self.max_nodes = max_nodes
        self.shuffle = shuffle
        self.seed = seed
        self.max_units = max_units
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the shuffle for the next epoch, reproducibly.

        Args:
            epoch: The epoch about to start.
        """
        self.epoch = epoch

    def _order(self) -> list[int]:
        """Return this epoch's unit order."""
        if not self.shuffle:
            return list(range(len(self.dataset)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        """Yield lists of unit indices, each within both budgets where possible."""
        batch: list[int] = []
        windows = 0
        nodes = 0
        for index in self._order():
            unit = self.dataset[index]
            over_windows = batch and windows + unit.n_windows > self.max_windows
            over_nodes = batch and nodes + unit.n_nodes > self.max_nodes
            over_units = self.max_units and len(batch) >= self.max_units
            if over_windows or over_nodes or over_units:
                yield batch
                batch, windows, nodes = [], 0, 0
            if not batch and (unit.n_windows > self.max_windows or unit.n_nodes > self.max_nodes):
                _LOGGER.warning(
                    "Unit exceeds a budget on its own",
                    extra={
                        "unit_id": unit.unit_id,
                        "windows": unit.n_windows,
                        "max_windows": self.max_windows,
                        "nodes": unit.n_nodes,
                        "max_nodes": self.max_nodes,
                    },
                )
            batch.append(index)
            windows += unit.n_windows
            nodes += unit.n_nodes
        if batch:
            yield batch

    def __len__(self) -> int:
        """Number of batches this epoch. Counted by walking, since it depends on order."""
        return sum(1 for _ in self.__iter__())


def fused_line_coverage(batch: dict[str, Any], output: Mapping[str, Any]) -> dict[str, int]:
    """Count flagged lines each branch could reach, and how many the fusion recovers.

    The per-batch form of the claim fusion is supposed to justify. 7.84% of flagged lines
    carry no graph node (``Memory.md`` §7 #17) and the structural branch cannot score them
    at all; the fused line mask is the union, so it can. ``recovered_by_union`` is the size
    of that effect measured on this batch rather than argued from the corpus.

    Args:
        batch: A collated fused batch.
        output: The model's output, carrying ``line_mask``, ``sem_line_mask`` and
            ``str_line_mask``.

    Returns:
        ``{"flagged", "flagged_with_features", "flagged_semantic_only",
        "flagged_structural_only", "recovered_by_union"}``.
    """
    flagged = batch["line_labels"] > 0
    fused = output["line_mask"].to(flagged.device)
    sem = output["sem_line_mask"].to(flagged.device)
    structural = output["str_line_mask"].to(flagged.device)
    return {
        "flagged": int(flagged.sum()),
        "flagged_with_features": int((flagged & fused).sum()),
        "flagged_semantic_only": int((flagged & sem & ~structural).sum()),
        "flagged_structural_only": int((flagged & structural & ~sem).sum()),
        "recovered_by_union": int((flagged & fused & ~(sem & structural)).sum()),
    }
