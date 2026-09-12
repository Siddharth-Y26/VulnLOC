"""Phase 4's structural branch: the three convolutions, and what they must not do.

The failure modes worth pinning here are silent ones. A vocabulary built from the wrong
corpus folds unseen grammar symbols into slot 0 and trains happily; an encoder that drops
the residual collapses every node onto one vector and still returns the right shape. Both
produce a model that looks fine and has learned nothing about structure.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from vulnloc.models.alignment import align_nodes_to_lines  # noqa: E402
from vulnloc.models.registry import STRUCTURAL_REGISTRY  # noqa: E402
from vulnloc.models.structural import (  # noqa: E402
    CONVOLUTIONS,
    NodeFeatures,
    StructuralEncoder,
    build_structural_encoder,
)
from vulnloc.utils.exceptions import ModelError  # noqa: E402

VOCAB = 32


def _batch(n_nodes: int = 12, n_lines: int = 5) -> dict[str, torch.Tensor]:
    """A small graph batch shaped like `to_pyg` output."""
    generator = torch.Generator().manual_seed(0)
    return {
        "node_type": torch.randint(1, VOCAB, (n_nodes,), generator=generator),
        "edge_index": torch.stack(
            [
                torch.arange(n_nodes - 1),
                torch.arange(1, n_nodes),
            ]
        ),
        "line": torch.randint(1, n_lines + 1, (n_nodes,), generator=generator),
        "batch": torch.zeros(n_nodes, dtype=torch.long),
    }


@pytest.mark.parametrize("kind", CONVOLUTIONS)
def test_every_variant_runs_and_is_registered(kind: str) -> None:
    """Phase 4 exit criterion: all three run and are config-selectable."""
    assert kind in STRUCTURAL_REGISTRY
    encoder = build_structural_encoder(kind, vocab_size=VOCAB, hidden_dim=16, depth=2)
    states = encoder(_batch())
    assert states.shape == (12, 16)
    assert torch.isfinite(states).all()


@pytest.mark.parametrize("kind", CONVOLUTIONS)
@pytest.mark.parametrize("depth", [2, 4, 6])
def test_the_over_smoothing_sweep_runs(kind: str, depth: int) -> None:
    """Phase 4 owes a performance-vs-depth report, so 2/4/6 must be a config change."""
    encoder = build_structural_encoder(kind, vocab_size=VOCAB, hidden_dim=16, depth=depth)
    states = encoder(_batch())
    assert states.shape == (12, 16)
    assert torch.isfinite(states).all()


def test_unknown_convolution_raises() -> None:
    with pytest.raises(ModelError):
        StructuralEncoder(VOCAB, kind="graphsage", hidden_dim=16)


def test_depth_must_be_positive() -> None:
    with pytest.raises(ModelError):
        StructuralEncoder(VOCAB, kind="gcn", hidden_dim=16, depth=0)


def test_gat_heads_must_divide_the_width() -> None:
    """Otherwise the width silently changes and the depth sweep stops being comparable."""
    with pytest.raises(ModelError):
        StructuralEncoder(VOCAB, kind="gat", hidden_dim=18, heads=4)


def test_out_of_vocabulary_index_raises_rather_than_clamping() -> None:
    """A vocabulary built from the wrong corpus must fail loudly.

    Clamping would train the unknown slot on whatever fell off the end — the exact
    silent failure ``Memory.md`` §7 records for grammar-symbol-list vocabularies.
    """
    features = NodeFeatures(VOCAB, 8)
    with pytest.raises(ModelError):
        features(torch.tensor([VOCAB + 3]))


def test_unknown_slot_is_padding_and_stays_zero() -> None:
    """Slot 0 means "symbol unseen when the vocabulary was built"; it carries no signal."""
    features = NodeFeatures(VOCAB, 8)
    assert torch.equal(features.embedding.weight[0], torch.zeros(8))


def test_missing_key_raises() -> None:
    encoder = build_structural_encoder("gcn", vocab_size=VOCAB, hidden_dim=16, depth=2)
    with pytest.raises(ModelError):
        encoder({"node_type": torch.tensor([1, 2])})


def test_residual_preserves_node_distinctness_at_depth() -> None:
    """The point of the residual: 6 layers must not collapse every node onto one vector.

    Over-smoothing is the named risk for these graphs — the AST of a corpus function
    averages 252 nodes, so a deep stack's receptive field covers most of a function. The
    test compares against the same encoder without residuals, so it measures the wiring
    rather than an arbitrary threshold.
    """
    batch = _batch(n_nodes=24)

    def spread(residual: bool) -> float:
        torch.manual_seed(0)
        encoder = StructuralEncoder(
            VOCAB, kind="gcn", hidden_dim=32, depth=6, residual=residual, dropout=0.0
        )
        encoder.eval()
        with torch.no_grad():
            states = encoder(batch)
        # Mean pairwise distance from the batch centroid: 0 means every node is identical.
        return float((states - states.mean(dim=0, keepdim=True)).norm(dim=1).mean())

    assert spread(residual=True) > spread(residual=False)


def test_node_states_project_onto_the_line_axis() -> None:
    """The branch's output has to reach the shared line axis, through the shared primitive."""
    batch = _batch(n_nodes=12, n_lines=5)
    encoder = build_structural_encoder("gcn", vocab_size=VOCAB, hidden_dim=16, depth=2)
    states = encoder(batch)
    line_states, line_mask = align_nodes_to_lines(
        states, batch["line"], batch["batch"], n_lines=5, n_units=1
    )
    assert line_states.shape == (1, 5, 16)
    assert line_mask.shape == (1, 5)
    # Every line a node claimed must be marked, and nothing else.
    assert set(torch.nonzero(line_mask[0]).flatten().tolist()) == {
        int(line) - 1 for line in batch["line"]
    }


def test_two_graphs_in_one_batch_stay_separate() -> None:
    """PyG concatenates nodes; `batch` is what keeps one function's lines out of another's."""
    encoder = build_structural_encoder("gcn", vocab_size=VOCAB, hidden_dim=8, depth=2)
    node_type = torch.tensor([1, 2, 3, 4])
    batch = {
        "node_type": node_type,
        "edge_index": torch.tensor([[0, 2], [1, 3]]),
    }
    states = encoder(batch)
    line = torch.tensor([1, 2, 1, 2])
    batch_index = torch.tensor([0, 0, 1, 1])
    line_states, line_mask = align_nodes_to_lines(states, line, batch_index, n_lines=3, n_units=2)
    assert line_states.shape == (2, 3, 8)
    assert line_mask[0].tolist() == [True, True, False]
    assert line_mask[1].tolist() == [True, True, False]
