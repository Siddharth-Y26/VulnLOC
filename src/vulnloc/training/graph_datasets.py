"""Shards → graph batches for the structural branch (L3, so it may read both L0 and L1).

The mirror of :mod:`vulnloc.training.datasets`, and deliberately shaped like it: same
shard source, same line cap, same coverage accounting, same "a unit is never split"
rule. Phase 5 has to put both branches in one batch on one line axis, so the two paths
agree on ``line_labels``, ``line_valid``, ``targets``, ``unit_ids`` and ``n_units``
meaning exactly the same thing here as there.

Four decisions shape this module, and each of them is a measurement rather than a
preference.

**Graphs are built per epoch, not cached.** Measured over every line-labelled unit in
both partitions: **train 3.23 ms/unit — 17.6 s for all 5,457 — and val 6.50 ms/unit,
3.2 s for 488**, with zero build failures. Against roughly 1,000 s for a semantic epoch
that is under 2%, so a graph cache would buy a second artefact with a second digest and
a second way to go stale, for nothing. Two caveats that both cost time to find: val is
**twice** train's per-unit cost because its functions are larger, which is the same
distribution trap ``Memory.md`` §7 #12 records for truncation — never size this from a
corpus average; and the figures only hold once the lazy imports inside ``to_pyg`` are
warm, because the first call pays ~7 s of ``torch_geometric`` import that is startup
cost, not per-unit cost.

**Batching is in nodes, not units.** An AST reaches 56,266 nodes and averages 252
(``Memory.md`` §7 #18), so a batch of 8 *units* varies in cost by orders of magnitude and
is bounded by its worst member. :class:`NodeBudgetBatchSampler` fills to a node budget,
exactly as :class:`~vulnloc.training.datasets.WindowBatchSampler` fills to a window
budget, for the same reason: on a 4 GiB card memory has to be a property of the sampler
rather than a gamble.

**The node-type vocabulary comes from the corpus artefact, never from the graph.**
``to_pyg`` will happily build a vocabulary from a single graph when none is passed, and
the result trains without complaint — with a different index for the same symbol in every
batch. :func:`load_node_vocab` reads the artefact ``scripts/build_node_vocab.py`` wrote
and verifies its digest, so a changed table is a hard failure rather than a slow
mystery. ``Memory.md`` §7 #21 records why the table must be scanned rather than read off
one grammar's symbol list.

**Flagged lines the graph cannot reach are counted, not dropped.** 7.84% of flagged lines
carry no AST node at all (``Memory.md`` §7 #17), and 84.8% of those sit in units no
grammar parses cleanly. That is a ceiling on the structural branch's recall, not a model
failure, and it belongs in the dataset's own log line where it cannot be discovered late.
Units whose graph will not build at all are counted the same way and reported as a rate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset, Sampler

from vulnloc.data.cache import iter_shard
from vulnloc.representation import GRAPH_REGISTRY, TreeSitterParser
from vulnloc.representation.to_pyg import to_pyg
from vulnloc.representation.units import CodeUnit
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

__all__ = [
    "GraphLineDataset",
    "GraphUnit",
    "NodeBudgetBatchSampler",
    "collate_graphs",
    "graph_line_coverage",
    "load_node_vocab",
    "load_subtoken_vocab",
]

_LOGGER = get_logger(__name__)

DEFAULT_VOCAB_PATH = Path("./artifacts/representation/node_vocab_ast.json")
"""Where ``scripts/build_node_vocab.py`` writes the AST vocabulary of record."""

DEFAULT_SUBTOKEN_VOCAB_PATH = Path("./artifacts/representation/subtoken_vocab_ast.json")
"""Where ``scripts/build_subtoken_vocab.py`` writes the identifier table of record."""


def load_node_vocab(
    path: Path | str = DEFAULT_VOCAB_PATH, *, expect_digest: str | None = None
) -> tuple[dict[str, int], dict[str, Any]]:
    """Load the corpus-wide node-type vocabulary and verify it.

    Args:
        path: The artefact written by ``scripts/build_node_vocab.py``.
        expect_digest: If given, the digest the caller believes it is loading. A
            mismatch raises. Pin this in a training config and two checkpoints stay
            comparable; leave it out and a rebuilt table silently renumbers the
            embedding rows, which is the same class of failure the split fingerprint
            exists to catch one layer down.

    Returns:
        ``(vocab, metadata)`` — the ``symbol -> index`` mapping, and the artefact's own
        provenance fields (run id, digest, size, builder, language policy).

    Raises:
        DataError: If the artefact is missing, malformed, or fails either check.
    """
    artefact = Path(path)
    if not artefact.exists():
        raise DataError(
            "node-type vocabulary not found",
            path=str(artefact),
            hint="Run: python scripts/build_node_vocab.py",
        )
    payload = json.loads(artefact.read_text(encoding="utf-8"))
    vocab = payload.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise DataError("vocabulary artefact carries no 'vocab' mapping", path=str(artefact))

    # Recomputed, not trusted. The digest in the file is only evidence if it is checked
    # against the table beside it.
    recomputed = hashlib.sha256(
        "\n".join(f"{name}:{index}" for name, index in sorted(vocab.items())).encode("utf-8")
    ).hexdigest()
    recorded = str(payload.get("digest", ""))
    if recorded and recomputed != recorded:
        raise DataError(
            "vocabulary digest does not match its own table; the artefact was edited",
            path=str(artefact),
            recorded=recorded[:12],
            recomputed=recomputed[:12],
        )
    if expect_digest and not recomputed.startswith(expect_digest.rstrip("…").rstrip(".")):
        raise DataError(
            "vocabulary digest is not the one this run expects",
            path=str(artefact),
            expected=expect_digest,
            found=recomputed[:12],
            hint="Two checkpoints built on different tables are not comparable.",
        )

    metadata = {
        "run_id": payload.get("run_id"),
        "digest": recomputed,
        "size": int(payload.get("size", len(vocab) + 1)),
        "builder": payload.get("builder"),
        "language_policy": payload.get("language_policy"),
        "units_scanned": payload.get("units_scanned"),
    }
    return {str(k): int(v) for k, v in vocab.items()}, metadata


def load_subtoken_vocab(
    path: Path | str = DEFAULT_SUBTOKEN_VOCAB_PATH, *, expect_digest: str | None = None
) -> tuple[dict[str, int], dict[str, Any]]:
    """Load the identifier subtoken vocabulary and verify it.

    Same checks as :func:`load_node_vocab` — it is the same class of artefact, and the
    same silent failure if a rebuilt table renumbers the embedding rows.

    One difference matters and is asserted here rather than assumed: this table is built
    from the **training split only**. Identifiers from `val`/`test` in it would give the
    model embedding rows for names that appear only in evaluation data. Grammar symbols
    are a property of the parser and are correctly scanned corpus-wide; these are not.

    Args:
        path: The artefact written by ``scripts/build_subtoken_vocab.py``.
        expect_digest: The digest the caller believes it is loading, if it wants the
            table pinned.

    Returns:
        ``(vocab, metadata)``.

    Raises:
        DataError: If the artefact is missing, malformed, fails a check, or was not
            built from `train` alone.
    """
    vocab, metadata = load_node_vocab(path, expect_digest=expect_digest)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not payload.get("scanned_from_train_only"):
        raise DataError(
            "subtoken vocabulary was not built from the training split alone",
            path=str(path),
            split_scanned=payload.get("split_scanned"),
            hint="Identifiers from val/test in the table leak evaluation data into the model.",
        )
    metadata["min_count"] = payload.get("min_count")
    metadata["distinct_subtokens_seen"] = payload.get("distinct_subtokens_seen")
    return vocab, metadata


@dataclass(frozen=True)
class GraphUnit:
    """One function as a graph, with its line supervision.

    Attributes:
        unit_id: Stable corpus identifier.
        node_type: ``[N]`` vocabulary indices.
        subtokens: ``[N, MAX_SUBTOKENS]`` identifier subtoken indices, or ``None`` when
            the ADR-002 §D2.5 second half is off. Padded with 1; 0 is the unknown slot.
        node_line: ``[N]`` **1-indexed** source line per node.
        edge_index: ``[2, E]`` node indices, local to this unit.
        edge_type: ``[E]`` edge-type indices. Carried even where the encoder ignores it,
            because ablation A3 selects on it and a tensor that is not plumbed through is
            a knob that silently does nothing (``Memory.md`` §7 #22).
        line_labels: ``[n_lines]`` of 0/1, 1-indexed line ``i`` at position ``i - 1``.
        n_lines: Source lines in the function, before the line cap.
        target: 1 if the function is vulnerable.
        flagged_beyond_cap: Flagged lines past the line cap, which can never be
            predicted. Reported, not hidden.
        flagged_without_node: Flagged lines inside the cap that carry no graph node, so
            the structural branch cannot score them at all. The per-unit form of
            ``Memory.md`` §7 #17.
    """

    unit_id: str
    node_type: torch.Tensor
    subtokens: torch.Tensor | None
    node_line: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    line_labels: tuple[int, ...]
    n_lines: int
    target: int
    flagged_beyond_cap: int
    flagged_without_node: int

    @property
    def n_nodes(self) -> int:
        """How many nodes this unit contributes to a batch."""
        return int(self.node_type.numel())


class GraphLineDataset(Dataset[GraphUnit]):
    """Units from one shard, parsed and built into graphs once and held in memory.

    Args:
        cache_dir: Directory holding the preprocessed shards.
        split: Shard name — ``train``, ``val``, ``test``, ``clean_eval``.
        node_vocab: The corpus-wide ``symbol -> index`` table. Required: passing ``None``
            through to ``to_pyg`` would build a per-graph vocabulary and give the same
            symbol a different index in every batch.
        subtoken_vocab: The identifier table from ``scripts/build_subtoken_vocab.py``,
            enabling ADR-002 §D2.5's second half. ``None`` leaves node features as
            node-type only, which reproduces every number logged before it existed.
        builder: Graph type — a ``GRAPH_REGISTRY`` key (``ast``/``cfg``/``dfg``/``cpg``).
        language: Grammar-selection policy. ``best_of_two`` is ADR-005's default and the
            only one any current number was measured under.
        n_lines: Line-axis cap, ``L``.
        limit: Stop after this many usable units. ``0`` means the whole shard.
        line_labelled_only: Keep only functions carrying line labels, as the line-level
            task requires (24.4% of Big-Vul's positives have none).
        normalize_whitespace: Strip edge whitespace from every source line before
            parsing. **On by default: leaving it off leaks the label** (ADR-006).
        strip_comments: Remove every comment before anything sees the text, keeping the
            line count (``Memory.md`` §1.25). **Off by default**: every number logged
            before 2026-10-07 was measured with comments in place.

    Raises:
        DataError: If the shard yields no usable units.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        split: str,
        node_vocab: Mapping[str, int],
        *,
        subtoken_vocab: Mapping[str, int] | None = None,
        builder: str = "ast",
        language: str = "best_of_two",
        n_lines: int = 512,
        limit: int = 0,
        line_labelled_only: bool = False,
        normalize_whitespace: bool = True,
        strip_comments: bool = False,
    ) -> None:
        self.split = split
        self.n_lines = n_lines
        self.builder_name = builder
        self.language = language
        self.units: list[GraphUnit] = []

        parser = TreeSitterParser()
        graph_builder = GRAPH_REGISTRY.get(builder)()
        flagged_lost = 0
        flagged_nodeless = 0
        build_failures = 0
        considered = 0

        for labelled in iter_shard(Path(cache_dir), split):
            if limit and len(self.units) >= limit:
                break
            if line_labelled_only and not labelled.vulnerable_lines:
                continue
            considered += 1
            unit = CodeUnit.from_labelled_function(
                labelled,
                normalize_whitespace=normalize_whitespace,
                strip_comments=strip_comments,
            )
            try:
                parsed = parser.parse_with_policy(
                    unit.source,
                    policy=language,
                    declared=unit.language,
                    unit_id=unit.unit_id,
                )
                graph = graph_builder.build(parsed)
                data = to_pyg(graph, node_vocab, subtoken_vocab)
            except Exception:  # noqa: BLE001 - counted and reported below, never silent
                # A unit whose graph will not build contributes nothing the structural
                # branch can read. It is counted rather than ignored: if this rate is
                # large the dataset is quietly smaller than the semantic one, and the two
                # branches would then be evaluated on different populations.
                build_failures += 1
                continue

            labels = [0] * n_lines
            beyond = 0
            for line in unit.vulnerable_lines:
                if 1 <= line <= n_lines:
                    labels[line - 1] = 1
                else:
                    beyond += 1
            flagged_lost += beyond

            lines_with_nodes = {int(line) for line in data.line.tolist()}
            nodeless = sum(
                1
                for line in unit.vulnerable_lines
                if 1 <= line <= n_lines and line not in lines_with_nodes
            )
            flagged_nodeless += nodeless

            self.units.append(
                GraphUnit(
                    unit_id=unit.unit_id,
                    node_type=data.node_type,
                    # Kept only when a vocabulary was supplied. `to_pyg` always emits the
                    # tensor so the layout never depends on the setting, but storing an
                    # all-padding tensor per unit would cost memory for no signal.
                    subtokens=data.subtokens if subtoken_vocab is not None else None,
                    node_line=data.line,
                    edge_index=data.edge_index,
                    edge_type=data.edge_type,
                    line_labels=tuple(labels),
                    n_lines=unit.n_lines,
                    target=unit.target,
                    flagged_beyond_cap=beyond,
                    flagged_without_node=nodeless,
                )
            )

        if not self.units:
            raise DataError(
                "shard produced no graph units",
                split=split,
                cache_dir=str(cache_dir),
                builder=builder,
                considered=considered,
                build_failures=build_failures,
            )

        self.flagged_beyond_cap = flagged_lost
        self.flagged_without_node = flagged_nodeless
        self.build_failures = build_failures
        total_flagged = sum(sum(u.line_labels) for u in self.units) + flagged_lost
        scoreable = total_flagged - flagged_lost
        _LOGGER.info(
            "Graph dataset built",
            extra={
                "split": split,
                "builder": builder,
                "language_policy": language,
                "units": len(self.units),
                "nodes": sum(u.n_nodes for u in self.units),
                "edges": sum(int(u.edge_index.shape[1]) for u in self.units),
                "positives": sum(u.target for u in self.units),
                "flagged_lines": total_flagged,
                "flagged_beyond_line_cap": flagged_lost,
                # The §7 #17 ceiling, measured on this split rather than quoted at it.
                "flagged_without_a_graph_node": flagged_nodeless,
                "flagged_without_a_graph_node_pct": (
                    round(100.0 * flagged_nodeless / scoreable, 2) if scoreable else 0.0
                ),
                "graph_build_failures": build_failures,
                "graph_build_failure_pct": (
                    round(100.0 * build_failures / considered, 2) if considered else 0.0
                ),
                "n_lines_cap": n_lines,
                "normalize_whitespace": normalize_whitespace,
                "strip_comments": strip_comments,
            },
        )

    def __len__(self) -> int:
        """Number of units."""
        return len(self.units)

    def __getitem__(self, index: int) -> GraphUnit:
        """Return one graph unit."""
        return self.units[index]

    def positive_line_weight(self) -> float:
        """Return ``negatives / positives`` over scoreable lines, for a weighted loss.

        Computed exactly as the semantic dataset computes it, so a loss weight does not
        change meaning between the two branches.

        Returns:
            The ratio, or ``1.0`` when there are no positive lines.
        """
        positives = 0
        scoreable = 0
        for unit in self.units:
            usable = min(unit.n_lines, self.n_lines)
            scoreable += usable
            positives += sum(unit.line_labels[:usable])
        if positives == 0:
            return 1.0
        return (scoreable - positives) / positives


def collate_graphs(units: Sequence[GraphUnit]) -> dict[str, Any]:
    """Concatenate graphs into one batch, PyG style.

    Nodes are concatenated and every unit's ``edge_index`` is shifted by the running node
    offset. Getting that offset wrong is the characteristic bug of hand-rolled graph
    batching and it does not raise — it wires one function's statements to another's, and
    the model trains on the result — so the offsets are asserted against the node count.

    Args:
        units: The units in this batch.

    Returns:
        A batch carrying ``node_type`` ``[N]``, ``line`` ``[N]``, ``edge_index``
        ``[2, E]``, ``edge_type`` ``[E]``, ``batch`` ``[N]``, ``n_units``, plus the
        line-axis keys shared with :func:`~vulnloc.training.datasets.collate_units`:
        ``line_labels`` ``[B, L]``, ``line_valid`` ``[B, L]``, ``targets`` ``[B]`` and
        ``unit_ids``.

    Raises:
        DataError: If ``units`` is empty, or an edge index escapes its own graph.
    """
    if not units:
        raise DataError("cannot collate an empty batch")

    carries_subtokens = [u.subtokens is not None for u in units]
    if any(carries_subtokens) and not all(carries_subtokens):
        raise DataError(
            "some units carry subtokens and some do not; the batch has no single feature set",
            with_subtokens=sum(carries_subtokens),
            without=len(units) - sum(carries_subtokens),
        )

    node_types: list[torch.Tensor] = []
    node_lines: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    batch_index: list[torch.Tensor] = []

    offset = 0
    for position, unit in enumerate(units):
        n_nodes = unit.n_nodes
        if unit.edge_index.numel() and int(unit.edge_index.max()) >= n_nodes:
            raise DataError(
                "edge index points outside its own graph",
                unit_id=unit.unit_id,
                max_index=int(unit.edge_index.max()),
                n_nodes=n_nodes,
            )
        node_types.append(unit.node_type)
        node_lines.append(unit.node_line)
        edge_indices.append(unit.edge_index + offset)
        edge_types.append(unit.edge_type)
        batch_index.append(torch.full((n_nodes,), position, dtype=torch.long))
        offset += n_nodes

    n_lines = len(units[0].line_labels)
    # `line_valid` is the *label* mask: lines this unit has. It is not the feature mask
    # the model produces — a line can exist and carry no node, which is precisely the
    # §7 #17 ceiling, and conflating the two would excuse the branch from lines it never
    # saw.
    line_valid = torch.zeros((len(units), n_lines), dtype=torch.bool)
    for index, unit in enumerate(units):
        line_valid[index, : min(unit.n_lines, n_lines)] = True

    batch: dict[str, Any] = {
        "node_type": torch.cat(node_types),
        "line": torch.cat(node_lines),
        "edge_index": (
            torch.cat(edge_indices, dim=1)
            if edge_indices
            else torch.empty((2, 0), dtype=torch.long)
        ),
        "edge_type": torch.cat(edge_types),
        "batch": torch.cat(batch_index),
        "n_units": len(units),
        "line_labels": torch.tensor([u.line_labels for u in units], dtype=torch.float32),
        "line_valid": line_valid,
        "targets": torch.tensor([u.target for u in units], dtype=torch.float32),
        "unit_ids": [u.unit_id for u in units],
    }
    if all(carries_subtokens):
        batch["subtokens"] = torch.cat([u.subtokens for u in units if u.subtokens is not None])
    return batch


class NodeBudgetBatchSampler(Sampler[list[int]]):
    """Groups units into batches by a **node** budget, never splitting a unit.

    The graph counterpart of :class:`~vulnloc.training.datasets.WindowBatchSampler`, and
    for the same reason. An AST averages 252 nodes and reaches 56,266 (``Memory.md``
    §7 #18), so a batch sized in units is bounded by its worst member and peak memory is
    unpredictable. Sizing in nodes makes the budget mean something on a 4 GiB card.

    A unit larger than the budget still gets its own batch rather than being dropped.
    Dropping it would discard the largest functions, which are exactly the ones holding
    the most flagged lines (§1.8) — so the budget is a target, not a guarantee, and an
    over-budget batch is logged rather than hidden.

    Args:
        dataset: The dataset to sample from.
        max_nodes: Node budget per batch.
        shuffle: Shuffle unit order each epoch.
        seed: Base seed; the epoch number is mixed in so each epoch differs reproducibly.
        max_units: Optional cap on units per batch, regardless of node count.

    Raises:
        DataError: If ``max_nodes`` is not positive.
    """

    def __init__(
        self,
        dataset: GraphLineDataset,
        *,
        max_nodes: int = 8192,
        shuffle: bool = True,
        seed: int = 42,
        max_units: int = 0,
    ) -> None:
        if max_nodes <= 0:
            raise DataError("max_nodes must be positive", max_nodes=max_nodes)
        self.dataset = dataset
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
        """Yield lists of unit indices, each within the node budget where possible."""
        batch: list[int] = []
        nodes = 0
        for index in self._order():
            needed = self.dataset[index].n_nodes
            over_budget = batch and nodes + needed > self.max_nodes
            over_units = self.max_units and len(batch) >= self.max_units
            if over_budget or over_units:
                yield batch
                batch, nodes = [], 0
            if not batch and needed > self.max_nodes:
                _LOGGER.warning(
                    "Unit exceeds the node budget on its own",
                    extra={
                        "unit_id": self.dataset[index].unit_id,
                        "nodes": needed,
                        "max_nodes": self.max_nodes,
                    },
                )
            batch.append(index)
            nodes += needed
        if batch:
            yield batch

    def __len__(self) -> int:
        """Number of batches this epoch. Counted by walking, since it depends on order."""
        return sum(1 for _ in self.__iter__())


def graph_line_coverage(batch: dict[str, Any], line_mask: torch.Tensor) -> dict[str, int]:
    """Count flagged lines the structural branch had no node for.

    The per-batch form of ``Memory.md`` §7 #17: a flagged line with no graph node cannot
    be scored, so it caps recall rather than indicating a model failure. Counting it
    during training keeps the ceiling visible instead of letting it surface at evaluation.

    Args:
        batch: A collated graph batch.
        line_mask: ``[B, L]`` mask of lines that received node features.

    Returns:
        ``{"flagged": n, "flagged_with_features": n}``.
    """
    flagged = batch["line_labels"] > 0
    return {
        "flagged": int(flagged.sum()),
        "flagged_with_features": int((flagged & line_mask.to(flagged.device)).sum()),
    }
