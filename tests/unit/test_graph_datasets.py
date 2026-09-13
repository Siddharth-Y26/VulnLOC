"""Phase 4's data path, and the two ways it can be wrong without raising.

The first is **edge offsets**. PyG batches graphs by concatenating nodes, so every unit's
``edge_index`` has to shift by the running node count. Get it wrong and no exception
fires — one function's statements are wired to another's, message passing happily mixes
them, and the model trains on a graph that does not exist. Several tests here exist only
to pin that.

The second is **the node-type vocabulary**. ``to_pyg`` will build one from a single graph
when none is passed, which means the same grammar symbol gets a different index in every
batch and the embedding table learns nothing. It also trains without complaint. So the
vocabulary is loaded from the corpus artefact, its digest is recomputed rather than
trusted, and that is pinned too.

The rest is accounting: flagged lines past the line cap and flagged lines with no graph
node are ceilings on recall (``Memory.md`` §7 #17, §7 #26), and a dataset that loses them
quietly turns a known limit into an unexplained score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from vulnloc.data.cache import write_shards  # noqa: E402
from vulnloc.data.labeling import LabelledFunction  # noqa: E402
from vulnloc.models.alignment import align_nodes_to_lines  # noqa: E402
from vulnloc.models.structural import build_structural_encoder  # noqa: E402
from vulnloc.training.graph_datasets import (  # noqa: E402
    GraphLineDataset,
    GraphUnit,
    NodeBudgetBatchSampler,
    collate_graphs,
    graph_line_coverage,
    load_node_vocab,
)
from vulnloc.utils.exceptions import DataError, ModelError  # noqa: E402

SOURCE = "int f(int n) {\n    int x = n + 1;\n    return x;\n}\n"


def _unit(
    unit_id: str, *, source: str = SOURCE, vulnerable: tuple[int, ...] = (2,)
) -> LabelledFunction:
    """One shard record."""
    return LabelledFunction(
        unit_id=unit_id,
        source=source,
        language="C",
        target=1 if vulnerable else 0,
        vulnerable_lines=frozenset(vulnerable),
        n_lines=len(source.split("\n")),
        metadata={"project": "test"},
    )


def _vocab_file(tmp_path: Path, vocab: dict[str, int] | None = None) -> Path:
    """Write a vocabulary artefact with a correct digest."""
    table = vocab or {name: index for index, name in enumerate(sorted(_ALL_TYPES), start=1)}
    digest = hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in sorted(table.items())).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "node_vocab_ast.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "deadbeef",
                "builder": "ast",
                "language_policy": "best_of_two",
                "units_scanned": 3,
                "size": len(table) + 1,
                "digest": digest,
                "vocab": table,
            }
        ),
        encoding="utf-8",
    )
    return path


# A superset of the C node types the fixture source produces. Real runs load the corpus
# artefact; slot 0 stays reserved for symbols this table has never seen.
_ALL_TYPES = [
    "translation_unit",
    "function_definition",
    "primitive_type",
    "function_declarator",
    "identifier",
    "parameter_list",
    "parameter_declaration",
    "compound_statement",
    "declaration",
    "init_declarator",
    "binary_expression",
    "number_literal",
    "return_statement",
    "{",
    "}",
    "(",
    ")",
    ";",
    "=",
    "+",
]


def _write_shard(tmp_path: Path, units: list[LabelledFunction]) -> Path:
    """Write the units into a `train` shard and return the cache directory."""
    cache = tmp_path / "cache"
    write_shards(units, {u.unit_id: "train" for u in units}, cache)
    return cache


def _dataset(tmp_path: Path, units: list[LabelledFunction], **kwargs: Any) -> GraphLineDataset:
    """Write a shard and build a dataset over it."""
    cache = _write_shard(tmp_path, units)
    vocab, _ = load_node_vocab(_vocab_file(tmp_path))
    return GraphLineDataset(cache, "train", vocab, **kwargs)


# --- the vocabulary ---------------------------------------------------------------


def test_a_missing_vocabulary_raises_rather_than_defaulting(tmp_path: Path) -> None:
    """Falling back to a per-graph table would renumber every symbol per batch."""
    with pytest.raises(DataError, match="not found"):
        load_node_vocab(tmp_path / "absent.json")


def test_a_tampered_vocabulary_is_caught_by_recomputing_its_digest(tmp_path: Path) -> None:
    """The digest in the file is evidence only if it is checked against the table."""
    path = _vocab_file(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["vocab"]["identifier"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataError, match="digest does not match"):
        load_node_vocab(path)


def test_pinning_the_wrong_digest_raises(tmp_path: Path) -> None:
    """Two checkpoints built on different tables are not comparable."""
    with pytest.raises(DataError, match="not the one this run expects"):
        load_node_vocab(_vocab_file(tmp_path), expect_digest="0000000000")


def test_loading_returns_the_artefact_s_provenance(tmp_path: Path) -> None:
    vocab, metadata = load_node_vocab(_vocab_file(tmp_path))
    assert vocab["identifier"] > 0
    assert metadata["builder"] == "ast"
    assert metadata["language_policy"] == "best_of_two"
    assert metadata["size"] == len(vocab) + 1
    # Pinning the returned digest must then succeed.
    load_node_vocab(_vocab_file(tmp_path), expect_digest=metadata["digest"][:12])


# --- edge offsets: the silent one --------------------------------------------------


def _toy(unit_id: str, n_nodes: int, edges: list[tuple[int, int]], lines: list[int]) -> GraphUnit:
    """A hand-built unit, so the expected offsets are arithmetic rather than a guess."""
    return GraphUnit(
        unit_id=unit_id,
        node_type=torch.ones(n_nodes, dtype=torch.long),
        subtokens=None,
        node_line=torch.tensor(lines, dtype=torch.long),
        edge_index=(
            torch.tensor([[s for s, _ in edges], [t for _, t in edges]], dtype=torch.long)
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        ),
        edge_type=torch.zeros(len(edges), dtype=torch.long),
        line_labels=(0, 1, 0),
        n_lines=3,
        target=1,
        flagged_beyond_cap=0,
        flagged_without_node=0,
    )


def test_edges_are_offset_so_no_edge_crosses_between_graphs() -> None:
    """The characteristic silent bug: graph B's edges must not point into graph A."""
    a = _toy("a", 3, [(0, 1), (1, 2)], [1, 2, 3])
    b = _toy("b", 2, [(0, 1)], [1, 2])

    batch = collate_graphs([a, b])

    assert batch["node_type"].numel() == 5
    assert batch["edge_index"].tolist() == [[0, 1, 3], [1, 2, 4]]
    assert batch["batch"].tolist() == [0, 0, 0, 1, 1]
    # No edge may join a node of graph 0 to a node of graph 1.
    source_graph = batch["batch"][batch["edge_index"][0]]
    target_graph = batch["batch"][batch["edge_index"][1]]
    assert torch.equal(source_graph, target_graph)


def test_an_edge_pointing_outside_its_own_graph_raises() -> None:
    """Caught at collation, where the unit id is still available to name."""
    broken = _toy("broken", 2, [(0, 5)], [1, 2])
    with pytest.raises(DataError, match="outside its own graph"):
        collate_graphs([broken])


def test_a_graph_with_no_edges_still_batches() -> None:
    """A single-node function is legal and must not break the shape contract."""
    batch = collate_graphs([_toy("a", 1, [], [1]), _toy("b", 2, [(0, 1)], [1, 2])])
    assert batch["edge_index"].shape == (2, 1)
    assert batch["edge_index"].tolist() == [[1], [2]]


def test_collating_nothing_raises() -> None:
    with pytest.raises(DataError, match="empty batch"):
        collate_graphs([])


def test_the_line_axis_keys_match_the_semantic_path() -> None:
    """Phase 5 puts both branches in one batch; the shared keys must mean one thing."""
    batch = collate_graphs([_toy("a", 3, [(0, 1)], [1, 2, 3]), _toy("b", 2, [(0, 1)], [1, 2])])
    assert batch["line_labels"].shape == (2, 3)
    assert batch["line_valid"].shape == (2, 3)
    assert batch["targets"].tolist() == [1.0, 1.0]
    assert batch["unit_ids"] == ["a", "b"]
    assert batch["n_units"] == 2


# --- node-budget batching ----------------------------------------------------------


class _Sized:
    """A dataset stub whose only property is how many nodes each unit has."""

    def __init__(self, sizes: list[int]) -> None:
        self.sizes = sizes

    def __len__(self) -> int:
        return len(self.sizes)

    def __getitem__(self, index: int):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        return SimpleNamespace(n_nodes=self.sizes[index], unit_id=f"u{index}")


def test_batches_respect_the_node_budget() -> None:
    sampler = NodeBudgetBatchSampler(_Sized([100, 100, 100, 100]), max_nodes=250, shuffle=False)
    assert list(sampler) == [[0, 1], [2, 3]]


def test_a_unit_over_budget_gets_its_own_batch_rather_than_being_dropped() -> None:
    """The largest functions hold the most flagged lines; dropping them skews the split."""
    sampler = NodeBudgetBatchSampler(_Sized([50, 9000, 50]), max_nodes=1000, shuffle=False)
    batches = list(sampler)

    assert batches == [[0], [1], [2]]
    assert sum(len(b) for b in batches) == 3


def test_the_unit_cap_applies_even_under_budget() -> None:
    sampler = NodeBudgetBatchSampler(
        _Sized([1, 1, 1, 1, 1]), max_nodes=10_000, shuffle=False, max_units=2
    )
    assert list(sampler) == [[0, 1], [2, 3], [4]]


def test_shuffling_is_reproducible_and_changes_with_the_epoch() -> None:
    sampler = NodeBudgetBatchSampler(_Sized([1] * 20), max_nodes=3, seed=7)
    first = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_a_non_positive_budget_raises() -> None:
    with pytest.raises(DataError, match="max_nodes must be positive"):
        NodeBudgetBatchSampler(_Sized([1]), max_nodes=0)


# --- the dataset, over a real shard ------------------------------------------------


def test_the_dataset_parses_a_shard_into_graphs(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [_unit("a"), _unit("b")], n_lines=16)

    assert len(dataset) == 2
    assert dataset.build_failures == 0
    unit = dataset[0]
    assert unit.n_nodes > 0
    assert unit.edge_index.shape[0] == 2
    assert unit.edge_type.numel() == unit.edge_index.shape[1]
    # Line 2 is the flagged one; the label vector is 1-indexed at position i-1.
    assert unit.line_labels[1] == 1


def test_flagged_lines_past_the_line_cap_are_counted_not_dropped(tmp_path: Path) -> None:
    """The §7 #26 ceiling. A cap that silently loses labels turns a limit into a score."""
    source = "int f() {\n" + "    int a = 0;\n" * 20 + "    return 0;\n}\n"
    dataset = _dataset(tmp_path, [_unit("a", source=source, vulnerable=(2, 21))], n_lines=4)

    assert dataset.flagged_beyond_cap == 1
    assert dataset[0].flagged_beyond_cap == 1


def test_flagged_lines_with_no_graph_node_are_counted(tmp_path: Path) -> None:
    """The §7 #17 ceiling, measured on this split rather than quoted at it."""
    source = "int f() {\n\n    return 0;\n}\n"
    dataset = _dataset(tmp_path, [_unit("a", source=source, vulnerable=(2,))], n_lines=16)

    # Line 2 is blank, so no node claims it and the branch cannot score it.
    assert dataset.flagged_without_node == 1
    assert dataset[0].flagged_without_node == 1


def test_an_empty_shard_raises_rather_than_yielding_an_empty_dataset(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no graph units"):
        _dataset(tmp_path, [_unit("a", vulnerable=())], line_labelled_only=True)


def test_line_labelled_only_filters_the_way_the_semantic_path_does(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path, [_unit("a"), _unit("b", vulnerable=())], line_labelled_only=True, n_lines=16
    )
    assert [u.unit_id for u in dataset.units] == ["a"]


def test_positive_line_weight_counts_only_scoreable_lines(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [_unit("a")], n_lines=16)
    weight = dataset.positive_line_weight()
    # 5 source lines, 1 flagged, so 4 negatives to 1 positive.
    assert weight == pytest.approx(4.0)


# --- the whole path ----------------------------------------------------------------


def test_shard_to_graph_to_encoder_to_line_axis(tmp_path: Path) -> None:
    """The path Phase 4 needs end to end: it must land on [B, L, D] with a mask."""
    vocab, metadata = load_node_vocab(_vocab_file(tmp_path))
    cache = _write_shard(tmp_path, [_unit("a"), _unit("b")])
    dataset = GraphLineDataset(cache, "train", vocab, n_lines=8)

    batch = collate_graphs([dataset[0], dataset[1]])
    encoder = build_structural_encoder("gcn", vocab_size=metadata["size"], hidden_dim=16, depth=2)
    states = encoder(batch)
    line_states, line_mask = align_nodes_to_lines(
        states, batch["line"], batch["batch"], n_lines=8, n_units=2
    )

    assert states.shape == (batch["node_type"].numel(), 16)
    assert line_states.shape == (2, 8, 16)
    assert line_mask.shape == (2, 8)
    assert torch.isfinite(line_states).all()

    coverage = graph_line_coverage(batch, line_mask)
    assert coverage["flagged"] == 2
    assert coverage["flagged_with_features"] <= coverage["flagged"]


def test_coverage_reports_the_gap_rather_than_hiding_it() -> None:
    """A flagged line the branch had no node for is a recall ceiling, not a miss."""
    batch = collate_graphs([_toy("a", 2, [(0, 1)], [1, 3])])
    # Line 2 is flagged but no node claims it.
    mask = torch.tensor([[True, False, True]])
    assert graph_line_coverage(batch, mask) == {"flagged": 1, "flagged_with_features": 0}


# --- the GNN-only baseline ---------------------------------------------------------


def test_the_structural_model_scores_every_line_of_every_unit() -> None:
    """Shards → graphs → encoder → shared line axis → shared head."""
    from vulnloc.models.heads.line_head import LineHead
    from vulnloc.models.structural import build_structural_encoder
    from vulnloc.models.structural.baseline import StructuralLineModel

    batch = collate_graphs(
        [_toy("a", 3, [(0, 1), (1, 2)], [1, 2, 3]), _toy("b", 2, [(0, 1)], [1, 2])]
    )
    encoder = build_structural_encoder("gcn", vocab_size=8, hidden_dim=16, depth=2)
    model = StructuralLineModel(encoder, LineHead(16, hidden_dim=8), n_lines=3)

    out = model(batch)

    assert out["line_logits"].shape == (2, 3)
    assert out["function_logits"].shape == (2,)
    assert out["line_mask"].shape == (2, 3)
    assert torch.isfinite(out["line_logits"]).all()


@pytest.mark.parametrize("kind", ["gcn", "gat", "ggnn"])
def test_gradient_reaches_the_node_type_embedding(kind: str) -> None:
    """The branch has to actually learn from structure, not just return the right shape.

    A model whose embedding never receives gradient scores lines from the head's bias and
    looks entirely healthy doing it.
    """
    from vulnloc.models.heads.line_head import LineHead
    from vulnloc.models.structural import build_structural_encoder
    from vulnloc.models.structural.baseline import StructuralLineModel
    from vulnloc.training.losses import LineLoss

    batch = collate_graphs([_toy("a", 4, [(0, 1), (1, 2), (2, 3)], [1, 2, 3, 3])])
    encoder = build_structural_encoder(kind, vocab_size=8, hidden_dim=16, depth=2)
    model = StructuralLineModel(encoder, LineHead(16, hidden_dim=8), n_lines=3)

    out = model(batch)
    losses = LineLoss(kind="bce", pos_weight=2.0)(
        out["line_logits"].float(),
        batch["line_labels"],
        out["line_mask"] & batch["line_valid"],
        out["function_logits"].float(),
        batch["targets"],
    )
    losses["loss"].backward()

    embedding_grad = model.encoder.features.embedding.weight.grad
    assert embedding_grad is not None
    assert torch.any(embedding_grad != 0)


def test_a_head_that_does_not_match_the_encoder_raises() -> None:
    """Caught at construction, not as a shape error somewhere inside the first batch."""
    from vulnloc.models.heads.line_head import LineHead
    from vulnloc.models.structural import build_structural_encoder
    from vulnloc.models.structural.baseline import StructuralLineModel

    encoder = build_structural_encoder("gcn", vocab_size=8, hidden_dim=16, depth=2)
    with pytest.raises(ModelError, match="head width"):
        StructuralLineModel(encoder, LineHead(32, hidden_dim=8), n_lines=3)


def test_the_structural_model_requires_the_keys_that_tie_nodes_to_units() -> None:
    """Without `batch`, every graph would be scored as if it were one function."""
    from vulnloc.models.heads.line_head import LineHead
    from vulnloc.models.structural import build_structural_encoder
    from vulnloc.models.structural.baseline import StructuralLineModel

    encoder = build_structural_encoder("gcn", vocab_size=8, hidden_dim=16, depth=2)
    model = StructuralLineModel(encoder, LineHead(16, hidden_dim=8), n_lines=3)
    with pytest.raises(ModelError, match="missing a required key"):
        model({"node_type": torch.tensor([1, 2]), "edge_index": torch.tensor([[0], [1]])})


def test_both_datasets_satisfy_what_the_shared_training_job_needs() -> None:
    """One loop for both branches, so a difference between A1's arms cannot be plumbing."""
    from vulnloc.training.datasets import LineLabelledDataset
    from vulnloc.training.job import LineDataset

    for dataset_type in (GraphLineDataset, LineLabelledDataset):
        assert hasattr(dataset_type, "positive_line_weight")
        assert hasattr(dataset_type, "__len__")
    # `flagged_beyond_cap` is set in __init__ rather than declared, so it is checked on an
    # instance in the dataset tests above; the protocol is what names the contract.
    assert set(LineDataset.__annotations__) == {"flagged_beyond_cap"}


# --- ADR-002 §D2.5, second half: subtoken features ---------------------------------


def _sub(unit_id: str, rows: list[list[int]]) -> GraphUnit:
    """A unit carrying explicit subtoken rows."""
    n = len(rows)
    return GraphUnit(
        unit_id=unit_id,
        node_type=torch.ones(n, dtype=torch.long),
        subtokens=torch.tensor(rows, dtype=torch.long),
        node_line=torch.arange(1, n + 1, dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_type=torch.empty((0,), dtype=torch.long),
        line_labels=(0, 1, 0),
        n_lines=3,
        target=1,
        flagged_beyond_cap=0,
        flagged_without_node=0,
    )


def test_subtokens_concatenate_across_a_batch() -> None:
    batch = collate_graphs([_sub("a", [[2, 3, 1], [4, 1, 1]]), _sub("b", [[5, 1, 1]])])

    assert batch["subtokens"].shape == (3, 3)
    assert batch["subtokens"].tolist() == [[2, 3, 1], [4, 1, 1], [5, 1, 1]]


def test_a_batch_cannot_be_half_subtokened() -> None:
    """Two units with different feature sets are not one batch, and it must not average."""
    with pytest.raises(DataError, match="no single feature set"):
        collate_graphs([_sub("a", [[2, 1, 1]]), _toy("b", 2, [(0, 1)], [1, 2])])


def test_a_batch_without_subtokens_omits_the_key_entirely() -> None:
    """So an encoder built without the feature never silently receives padding as signal."""
    assert "subtokens" not in collate_graphs([_toy("a", 2, [(0, 1)], [1, 2])])


def test_the_subtoken_vocabulary_must_come_from_train_alone(tmp_path: Path) -> None:
    """Identifiers from val/test would hand the model embeddings for evaluation-only names.

    Grammar symbols are a property of the parser and are correctly scanned corpus-wide.
    Identifiers are not, and the artefact has to say which it was.
    """
    from vulnloc.training.graph_datasets import load_subtoken_vocab

    path = _vocab_file(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scanned_from_train_only"] = False
    payload["split_scanned"] = "all"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataError, match="not built from the training split alone"):
        load_subtoken_vocab(path)


def test_subtoken_features_change_the_node_states() -> None:
    """The feature has to do something, or it is a knob that looks like it works (§7 #22)."""
    from vulnloc.models.structural.encoder import NodeFeatures

    torch.manual_seed(0)
    features = NodeFeatures(8, 16, dropout=0.0, subtoken_vocab_size=12)
    features.eval()
    node_type = torch.tensor([3, 3])
    with torch.no_grad():
        # Same node type, different identifiers: the states must differ.
        a = features(node_type, torch.tensor([[2, 1, 1, 1, 1], [2, 1, 1, 1, 1]]))
        b = features(node_type, torch.tensor([[2, 1, 1, 1, 1], [7, 1, 1, 1, 1]]))

    assert torch.allclose(a[0], b[0])
    assert not torch.allclose(a[1], b[1])


def test_padding_does_not_dilute_a_short_identifier() -> None:
    """Mean over real slots only. Dividing by K would scale a one-word name down by K."""
    from vulnloc.models.structural.encoder import NodeFeatures

    torch.manual_seed(0)
    features = NodeFeatures(8, 16, dropout=0.0, subtoken_vocab_size=12)
    features.eval()
    with torch.no_grad():
        short = features(torch.tensor([3]), torch.tensor([[5, 1, 1, 1, 1]]))
        wider = features(torch.tensor([3]), torch.tensor([[5, 1, 1, 1, 1, 1, 1]]))

    assert torch.allclose(short, wider)


def test_a_node_with_no_identifier_contributes_exactly_zero() -> None:
    """Operators and interior nodes carry no name; a learned bias must not stand in."""
    from vulnloc.models.structural.encoder import NodeFeatures

    torch.manual_seed(0)
    features = NodeFeatures(8, 16, dropout=0.0, subtoken_vocab_size=12)
    features.eval()
    with torch.no_grad():
        with_subtokens = features(torch.tensor([3]), torch.tensor([[1, 1, 1, 1, 1]]))
        type_only = features.embedding(torch.tensor([3]))

    assert torch.allclose(with_subtokens, type_only)


def test_enabling_subtokens_without_them_in_the_batch_raises() -> None:
    from vulnloc.models.structural.encoder import NodeFeatures
    from vulnloc.utils.exceptions import ModelError as _ModelError

    features = NodeFeatures(8, 16, subtoken_vocab_size=12)
    with pytest.raises(_ModelError, match="carries no 'subtokens'"):
        features(torch.tensor([3]))


def test_an_out_of_range_subtoken_raises_rather_than_clamping() -> None:
    from vulnloc.models.structural.encoder import NodeFeatures
    from vulnloc.utils.exceptions import ModelError as _ModelError

    features = NodeFeatures(8, 16, subtoken_vocab_size=12)
    with pytest.raises(_ModelError, match="subtoken index outside"):
        features(torch.tensor([3]), torch.tensor([[99, 1, 1, 1, 1]]))
