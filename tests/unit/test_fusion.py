"""Phase 5's fusion layer, and the ways A1 could be confounded without anything raising.

Four of these tests are Phase 5 exit criteria stated as code: gradient reaches both
branches, attention maps come back with the right shape and are retrievable, every arm is
selectable by flag alone, and the arms differ only in the fusion mechanism.

The rest guard failures that produce a trained model rather than an error:

**A unit with no graph nodes NaNs the attention.** ``softmax`` over a fully-masked row is
``0/0``; ``MultiheadAttention`` does not guard it and the NaN reaches every parameter on
the first backward pass. Such units are real — see ``Memory.md`` §7 #17 — so the empty
branch is in almost every fixture here rather than in one test of its own.

**A branch could contribute through its bias.** ``Linear(0)`` is not ``0``, so a line the
semantic branch never saw would otherwise arrive at the head carrying a learnable constant
that correlates with the very coverage gap the label noise sits in.

**The single-branch ablations could inherit the other branch's coverage.** If ``llm_only``
were scored on the union mask it would be allowed to predict lines it has no tokens for,
which makes an ablation of a branch into a measurement of the other one.

**Trimming the line axis could change a number.** It is an optimization — 512 slots for a
31-line mean — and an optimization that changes a result is not one.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from vulnloc.models import FUSION_REGISTRY  # noqa: E402
from vulnloc.models.fused import FusedLineModel  # noqa: E402
from vulnloc.models.fusion import build_fusion  # noqa: E402
from vulnloc.models.heads.line_head import MASKED_LOGIT, LineHead  # noqa: E402
from vulnloc.models.structural import build_structural_encoder  # noqa: E402
from vulnloc.training.losses import LineLoss  # noqa: E402
from vulnloc.utils.exceptions import ModelError  # noqa: E402

ARMS = ["cross_attention", "concat", "sum", "gated", "llm_only", "gnn_only"]
SEM_DIM = 12
STR_DIM = 8
FUSED_DIM = 8
N_LINES = 10
VOCAB = 16


def fusion_kwargs(kind: str) -> dict[str, Any]:
    """Constructor arguments for one arm, mirroring what the config group supplies."""
    kwargs: dict[str, Any] = {
        "sem_dim": SEM_DIM,
        "str_dim": STR_DIM,
        "hidden_dim": FUSED_DIM,
        "dropout": 0.0,
    }
    if kind == "cross_attention":
        kwargs |= {"heads": 2, "n_layers": 1}
    return kwargs


@pytest.fixture
def branches() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two line-aligned branches whose coverage differs, including one empty unit.

    Unit 0 has both branches. Unit 1 has tokens but **no graph nodes at all** — the
    ``Memory.md`` §7 #17 case, and the one that NaNs an unguarded softmax.
    """
    generator = torch.Generator().manual_seed(7)
    sem = torch.randn(2, N_LINES, SEM_DIM, generator=generator)
    structural = torch.randn(2, N_LINES, STR_DIM, generator=generator)
    sem_mask = torch.zeros(2, N_LINES, dtype=torch.bool)
    sem_mask[0, :6] = True
    sem_mask[1, :4] = True
    str_mask = torch.zeros(2, N_LINES, dtype=torch.bool)
    str_mask[0, 2:8] = True  # covers lines 6 and 7 that the semantic branch does not
    return sem, structural, sem_mask, str_mask


# --- the registry and the flag-flip criterion ---------------------------------------


def test_the_registry_holds_exactly_the_arms_phase_5_defines() -> None:
    """`fusion=cross_attention` vs `fusion=concat` by flag alone is an exit criterion."""
    assert FUSION_REGISTRY.keys() == sorted(ARMS)


@pytest.mark.parametrize("kind", ARMS)
def test_every_arm_is_constructible_from_the_same_arguments(kind: str) -> None:
    """An arm needing bespoke construction is an arm that cannot be swept by a flag."""
    fusion = build_fusion(kind, **fusion_kwargs(kind))

    assert fusion.output_dim == FUSED_DIM


def test_a_setting_the_arm_does_not_read_is_a_hard_failure() -> None:
    """`heads` belongs to cross_attention alone; silently ignoring it is §7 #22."""
    with pytest.raises(ModelError, match="does not accept"):
        build_fusion("concat", **fusion_kwargs("cross_attention"))


# --- shapes, masks and the empty branch ---------------------------------------------


@pytest.mark.parametrize("kind", ARMS)
def test_every_arm_returns_the_same_shape_and_stays_finite(kind, branches) -> None:
    """Unit 1 has no structural keys at all — the fully-masked-softmax NaN case."""
    sem, structural, sem_mask, str_mask = branches
    output = build_fusion(kind, **fusion_kwargs(kind))(sem, structural, sem_mask, str_mask)

    assert output.fused.shape == (2, N_LINES, FUSED_DIM)
    assert output.line_mask.shape == (2, N_LINES)
    assert torch.isfinite(output.fused).all()


@pytest.mark.parametrize("kind", ["cross_attention", "concat", "sum", "gated"])
def test_fusion_scores_lines_either_branch_reached(kind, branches) -> None:
    """The union is the concrete thing fusion buys over the structural arm (§7 #17)."""
    sem, structural, sem_mask, str_mask = branches
    output = build_fusion(kind, **fusion_kwargs(kind))(sem, structural, sem_mask, str_mask)

    assert torch.equal(output.line_mask, sem_mask | str_mask)
    assert int((output.line_mask & ~sem_mask).sum()) > 0


@pytest.mark.parametrize(("kind", "index"), [("llm_only", 0), ("gnn_only", 1)])
def test_a_single_branch_arm_is_scored_on_its_own_coverage(kind, index, branches) -> None:
    """Scoring llm_only on the union would let it predict lines it has no tokens for."""
    sem, structural, sem_mask, str_mask = branches
    output = build_fusion(kind, **fusion_kwargs(kind))(sem, structural, sem_mask, str_mask)

    assert torch.equal(output.line_mask, (sem_mask, str_mask)[index])


@pytest.mark.parametrize("kind", ARMS)
def test_a_line_a_branch_never_saw_contributes_nothing(kind, branches) -> None:
    """`Linear(0)` is the bias, and a bias on an unseen line is a learnable feature."""
    sem, structural, sem_mask, str_mask = branches
    fusion = build_fusion(kind, **fusion_kwargs(kind)).eval()
    before = fusion(sem, structural, sem_mask, str_mask).fused

    # Change the structural features only where the structural branch saw nothing.
    noise = torch.randn_like(structural) * 100.0
    perturbed = torch.where(str_mask.unsqueeze(-1), structural, noise)
    after = fusion(sem, perturbed, sem_mask, str_mask).fused

    assert torch.allclose(before, after, atol=1e-6)


# --- attention maps: the Phase 7 export contract -------------------------------------


def test_attention_maps_are_not_produced_unless_asked(branches) -> None:
    """A [B, H, L, L] map per direction is the largest thing in a forward pass."""
    sem, structural, sem_mask, str_mask = branches
    output = build_fusion("cross_attention", **fusion_kwargs("cross_attention"))(
        sem, structural, sem_mask, str_mask
    )

    assert not output.has_attention


def test_attention_maps_have_the_documented_shape_and_normalisation(branches) -> None:
    """Phase 5 exit criterion: correct shape, retrievable, readable as line -> line."""
    sem, structural, sem_mask, str_mask = branches
    fusion = build_fusion("cross_attention", **fusion_kwargs("cross_attention")).eval()

    output = fusion(sem, structural, sem_mask, str_mask, return_attention=True)

    assert output.attn_sem2str is not None
    assert output.attn_str2sem is not None
    assert output.attn_sem2str.shape == (2, 2, N_LINES, N_LINES)
    assert output.attn_str2sem.shape == (2, 2, N_LINES, N_LINES)
    # Unit 0 has structural keys, so every query row is a distribution over them.
    rows = output.attn_sem2str[0].sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)
    # Unit 1 has none, so its map is zeroed rather than NaN.
    assert torch.equal(output.attn_sem2str[1], torch.zeros_like(output.attn_sem2str[1]))


def test_attention_never_reaches_a_line_the_source_branch_did_not_cover(branches) -> None:
    """An attention map is exported as evidence; weight on an absent line is not evidence."""
    sem, structural, sem_mask, str_mask = branches
    fusion = build_fusion("cross_attention", **fusion_kwargs("cross_attention")).eval()

    maps = fusion(sem, structural, sem_mask, str_mask, return_attention=True).attn_sem2str

    assert maps is not None
    leaked = maps[0][:, :, ~str_mask[0]].detach().abs().max()
    assert float(leaked) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("kind", ["concat", "sum", "gated", "llm_only", "gnn_only"])
def test_a_comparator_says_it_has_no_attention_rather_than_returning_zeros(kind, branches):
    """Zeros that look like evidence are worse than an honest absence."""
    sem, structural, sem_mask, str_mask = branches
    fusion = build_fusion(kind, **fusion_kwargs(kind))

    assert not fusion.produces_attention
    assert not fusion(sem, structural, sem_mask, str_mask, return_attention=True).has_attention


# --- validation ---------------------------------------------------------------------


def test_branches_on_different_line_axes_raise(branches) -> None:
    """The whole architecture rests on position `i` meaning the same line on both sides."""
    sem, structural, sem_mask, str_mask = branches
    fusion = build_fusion("concat", **fusion_kwargs("concat"))

    with pytest.raises(ModelError, match="not on the same line axis"):
        fusion(sem, structural[:, :-1], sem_mask, str_mask[:, :-1])


def test_a_branch_of_the_wrong_width_raises(branches) -> None:
    sem, structural, sem_mask, str_mask = branches
    fusion = build_fusion("concat", **fusion_kwargs("concat"))

    with pytest.raises(ModelError, match="width does not match"):
        fusion(sem[..., :-1], structural, sem_mask, str_mask)


def test_a_head_count_that_does_not_divide_the_width_raises() -> None:
    """Otherwise a sweep over head count silently sweeps the width too."""
    with pytest.raises(ModelError, match="divide hidden_dim"):
        build_fusion("cross_attention", sem_dim=SEM_DIM, str_dim=STR_DIM, hidden_dim=8, heads=3)


# --- the composed model -------------------------------------------------------------


class StubSemanticEncoder(torch.nn.Module):
    """A tiny stand-in for ``PretrainedCodeEncoder``.

    The real encoder is 124M parameters behind a network download. What these tests need
    of it is its contract — ``hidden_size`` and ``forward(batch) -> [W, T, D]`` — and a
    parameter whose gradient can be checked.
    """

    def __init__(self, hidden: int = SEM_DIM) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB, hidden)
        self.projection = torch.nn.Linear(hidden, hidden)
        self._hidden = hidden

    @property
    def hidden_size(self) -> int:
        """Width of the token states."""
        return self._hidden

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Return ``[W, T, D]`` token states."""
        return torch.as_tensor(self.projection(self.embedding(batch["input_ids"])))


def make_batch() -> dict[str, Any]:
    """A two-unit fused batch, hand-built so the line arithmetic is inspectable.

    Unit 0: 6 lines, tokens and nodes. Unit 1: 4 lines, tokens but **no nodes** — so the
    batch carries the empty-branch case through the whole model, not only the fusion.
    """
    generator = torch.Generator().manual_seed(11)
    n_units, n_windows, n_tokens = 2, 3, 5
    token_to_line = torch.tensor(
        [
            [-1, 1, 2, 3, -1],
            [-1, 4, 5, 6, -1],
            [-1, 1, 2, 3, -1],
        ]
    )
    node_line = torch.tensor([1, 2, 3, 3, 6])
    line_labels = torch.zeros(n_units, N_LINES)
    line_labels[0, 1] = 1.0  # line 2 of unit 0
    line_labels[1, 2] = 1.0  # line 3 of unit 1
    line_valid = torch.zeros(n_units, N_LINES, dtype=torch.bool)
    line_valid[0, :6] = True
    line_valid[1, :4] = True
    return {
        "sem_input_ids": torch.randint(0, VOCAB, (n_windows, n_tokens), generator=generator),
        "sem_attention_mask": torch.ones(n_windows, n_tokens, dtype=torch.long),
        "sem_token_to_line": token_to_line,
        "sem_window_to_unit": torch.tensor([0, 0, 1]),
        "str_node_type": torch.tensor([1, 2, 3, 4, 5]),
        "str_line": node_line,
        "str_edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        "str_edge_type": torch.zeros(4, dtype=torch.long),
        # Every node belongs to unit 0; unit 1 contributes none.
        "str_batch": torch.zeros(5, dtype=torch.long),
        "line_labels": line_labels,
        "line_valid": line_valid,
        "targets": torch.tensor([1.0, 1.0]),
        "unit_ids": ["a", "b"],
        "n_units": n_units,
    }


def make_model(kind: str, *, trim: bool = True) -> FusedLineModel:
    """Compose a small fused model around one fusion arm."""
    torch.manual_seed(3)
    return FusedLineModel(
        StubSemanticEncoder(),  # type: ignore[arg-type]
        build_structural_encoder(
            "gcn",
            vocab_size=VOCAB,
            hidden_dim=STR_DIM,
            depth=2,
            dropout=0.0,
            edge_handling="ignore",
        ),
        build_fusion(kind, **fusion_kwargs(kind)),
        LineHead(FUSED_DIM, hidden_dim=0, dropout=0.0),
        n_lines=N_LINES,
        trim_line_axis=trim,
    )


@pytest.mark.parametrize("kind", ARMS)
def test_the_model_returns_the_batch_contract_both_baselines_produce(kind) -> None:
    """The loop, the loss and the metrics must not be able to tell the arms apart."""
    output = make_model(kind).eval()(make_batch())

    assert output["line_logits"].shape == (2, N_LINES)
    assert output["function_logits"].shape == (2,)
    assert output["line_mask"].shape == (2, N_LINES)
    assert torch.isfinite(output["line_logits"]).all()


@pytest.mark.parametrize("kind", ARMS)
def test_a_line_with_no_features_cannot_win_a_ranking(kind) -> None:
    """Padding beyond a unit's length must not be scoreable at any threshold."""
    output = make_model(kind).eval()(make_batch())
    unscored = output["line_logits"][~output["line_mask"]]

    assert torch.equal(unscored, torch.full_like(unscored, MASKED_LOGIT))


def test_gradient_reaches_both_branches() -> None:
    """Phase 5 exit criterion. A dead branch trains, converges and looks like a result."""
    model = make_model("cross_attention")
    batch = make_batch()

    output = model(batch)
    loss = LineLoss(kind="bce")(
        output["line_logits"],
        batch["line_labels"],
        output["line_mask"] & batch["line_valid"],
        output["function_logits"],
        batch["targets"],
    )["loss"]
    loss.backward()

    semantic = model.semantic.projection.weight.grad
    structural = model.structural.features.embedding.weight.grad
    assert semantic is not None and float(semantic.abs().sum()) > 0.0
    assert structural is not None and float(structural.abs().sum()) > 0.0


@pytest.mark.parametrize(("kind", "dead"), [("llm_only", "structural"), ("gnn_only", "semantic")])
def test_a_disabled_branch_receives_no_gradient_at_all(kind, dead) -> None:
    """The ablation has to be a fact about the graph, not a multiplication by zero."""
    model = make_model(kind)
    batch = make_batch()

    model(batch)["line_logits"].sum().backward()

    grads = [p.grad for p in getattr(model, dead).parameters()]
    assert all(grad is None or float(grad.abs().sum()) == 0.0 for grad in grads)


@pytest.mark.parametrize("kind", ARMS)
def test_trimming_the_line_axis_changes_nothing(kind) -> None:
    """An optimization that changes a number is not an optimization."""
    trimmed = make_model(kind, trim=True).eval()
    full = make_model(kind, trim=False).eval()
    full.load_state_dict(trimmed.state_dict())
    batch = make_batch()

    a = trimmed(batch)
    b = full(batch)

    assert a["n_lines_active"] < N_LINES
    assert b["n_lines_active"] == N_LINES
    assert torch.allclose(a["line_logits"], b["line_logits"], atol=1e-5)
    assert torch.allclose(a["function_logits"], b["function_logits"], atol=1e-5)
    assert torch.equal(a["line_mask"], b["line_mask"])


def test_the_active_axis_covers_every_feature_not_just_every_label() -> None:
    """Taking the trim from the labels alone would silently drop a node past them."""
    batch = make_batch()
    # A node on line 9, beyond unit 0's six labelled lines.
    batch["str_line"] = torch.tensor([1, 2, 3, 3, 9])

    output = make_model("concat").eval()(batch)

    assert output["n_lines_active"] >= 9
    assert bool(output["line_mask"][0, 8])


def test_explain_returns_maps_and_leaves_the_training_flag_alone() -> None:
    """Phase 7 reads these, and must not pay for them with a mode change.

    Flipping a model out of train mode behind the caller's back would change dropout for
    every batch after it.
    """
    model = make_model("cross_attention").train()

    output = model.explain(make_batch())

    assert model.training
    assert output["attn_sem2str"] is not None
    active = output["n_lines_active"]
    assert output["attn_sem2str"].shape == (2, 2, active, active)


def test_a_missing_branch_key_raises_rather_than_encoding_half_the_batch() -> None:
    batch = make_batch()
    del batch["str_node_type"]

    with pytest.raises(ModelError, match="missing a required key"):
        make_model("concat")(batch)


def test_a_disabled_branch_does_not_require_its_keys() -> None:
    """`gnn_only` must be runnable without paying for the semantic half at all."""
    batch = {k: v for k, v in make_batch().items() if not k.startswith("sem_")}

    output = make_model("gnn_only").eval()(batch)

    assert output["line_logits"].shape == (2, N_LINES)


def test_the_head_width_must_match_the_fusion_output() -> None:
    with pytest.raises(ModelError, match="head width does not match"):
        FusedLineModel(
            StubSemanticEncoder(),  # type: ignore[arg-type]
            build_structural_encoder(
                "gcn", vocab_size=VOCAB, hidden_dim=STR_DIM, depth=1, edge_handling="ignore"
            ),
            build_fusion("concat", **fusion_kwargs("concat")),
            LineHead(FUSED_DIM + 1, hidden_dim=0),
            n_lines=N_LINES,
        )
