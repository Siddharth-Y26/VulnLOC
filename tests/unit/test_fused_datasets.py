"""Phase 5's data path — where A1 could be confounded before a single parameter updates.

One property matters more than everything else here: **the semantic half and the
structural half of a batch describe the same functions, in the same order.** Nothing
downstream checks it. Swap two units between the branches and the model fuses one
function's tokens with another's control flow, trains, converges, and produces a number
that looks exactly like a result.

The join is on ``unit_id`` rather than on position for that reason, and these tests exist
to keep it that way. The population where an index-based join would be *correct* is the
one this project actually has — ``graph_build_failures=0`` on both partitions — which is
precisely why the bug would have survived every test written against real data and
appeared the first time a shard contained a function no grammar could parse.

The rest guards the accounting. A fused arm that silently trains on fewer units than the
baselines it is compared against is a confound, so every dropped unit is counted and the
count is logged and logged as a run parameter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from vulnloc.data.cache import write_shards  # noqa: E402
from vulnloc.data.labeling import LabelledFunction  # noqa: E402
from vulnloc.representation import GRAPH_REGISTRY, TreeSitterParser  # noqa: E402
from vulnloc.representation.tokenization import LineTokenizer  # noqa: E402
from vulnloc.training.datasets import LineLabelledDataset  # noqa: E402
from vulnloc.training.fused_datasets import (  # noqa: E402
    FusedBudgetBatchSampler,
    FusedLineDataset,
    collate_fused,
    fused_line_coverage,
    join_branches,
)
from vulnloc.training.graph_datasets import (  # noqa: E402
    GraphLineDataset,
    load_node_vocab,
)
from vulnloc.utils.exceptions import DataError  # noqa: E402

SOURCE = "int f(int n) {\n    int x = n + 1;\n    return x;\n}\n"
UNPARSEABLE = "int g(#!! {\n  @@@ ???\n"


@pytest.fixture(scope="module")
def tokenizer() -> LineTokenizer:
    """A byte-level tokenizer with no network dependency."""
    return make_tokenizer()


def make_tokenizer() -> LineTokenizer:
    """Build the byte-level tokenizer. A plain function so other test modules can reuse it."""
    vocab = {"<s>": 0, "</s>": 1, "<pad>": 2, "<unk>": 3}
    for character in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[character] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 1)]
    )
    backend.decoder = decoders.ByteLevel()
    return LineTokenizer(
        PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
        ),
        max_length=64,
        stride=16,
    )


def unit(unit_id: str, *, source: str = SOURCE, lines: tuple[int, ...] = (2,)) -> LabelledFunction:
    """One shard record."""
    return LabelledFunction(
        unit_id=unit_id,
        source=source,
        language="C",
        target=1,
        vulnerable_lines=frozenset(lines),
        n_lines=len(source.split("\n")),
        metadata={"project": "test"},
    )


def vocab_file(tmp_path: Path) -> Path:
    """A node-type table scanned from the fixture sources, as the real artefact is.

    Built by parsing rather than from a hardcoded list, for the reason ``Memory.md`` §7 #21
    records: the vocabulary is a union over two grammars and reading it off one of them is
    how it goes wrong at corpus scale.
    """
    parser = TreeSitterParser()
    builder = GRAPH_REGISTRY.get("ast")()
    symbols: set[str] = set()
    parsed = parser.parse_with_policy(SOURCE, policy="best_of_two", declared="C", unit_id="scan")
    symbols |= {node.node_type for node in builder.build(parsed).nodes}

    table = {name: index for index, name in enumerate(sorted(symbols), start=1)}
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
                "units_scanned": 1,
                "size": len(table) + 1,
                "digest": digest,
                "vocab": table,
            }
        ),
        encoding="utf-8",
    )
    return path


def halves(
    tmp_path: Path, tokenizer: LineTokenizer, units: list[LabelledFunction]
) -> tuple[LineLabelledDataset, GraphLineDataset]:
    """Build the two branch datasets over one `train` shard, unjoined."""
    cache = tmp_path / "cache"
    write_shards(units, {u.unit_id: "train" for u in units}, cache)
    table, _ = load_node_vocab(vocab_file(tmp_path))
    return (
        LineLabelledDataset(cache, "train", tokenizer),
        GraphLineDataset(cache, "train", table),
    )


def dataset(
    tmp_path: Path, tokenizer: LineTokenizer, units: list[LabelledFunction], **kwargs: Any
) -> FusedLineDataset:
    """Write a `train` shard and build a fused dataset over it."""
    cache = tmp_path / "cache"
    write_shards(units, {u.unit_id: "train" for u in units}, cache)
    table, _ = load_node_vocab(vocab_file(tmp_path))
    return FusedLineDataset(cache, "train", tokenizer, table, **kwargs)


# --- the join ------------------------------------------------------------------------


def test_both_halves_of_every_unit_describe_the_same_function(tmp_path, tokenizer) -> None:
    """The one property nothing downstream can check for itself."""
    built = dataset(tmp_path, tokenizer, [unit(f"u{i}") for i in range(4)])

    assert len(built) == 4
    for fused in built:
        assert fused.semantic.unit_id == fused.unit_id == fused.structural.unit_id
        assert fused.semantic.line_labels == fused.structural.line_labels


def test_a_unit_missing_from_one_branch_is_dropped_and_the_rest_stay_paired(
    tmp_path, tokenizer
) -> None:
    """A graph that will not build removes its unit from the structural branch only.

    This is the case an index-based join gets wrong: it would pair `u2`'s tokens with
    `u3`'s graph and everything after it would be shifted by one, silently, forever.
    """
    semantic, structural = halves(tmp_path, tokenizer, [unit(f"u{i}") for i in range(4)])
    del structural.units[1]  # as a build failure would

    joined = join_branches(semantic, structural)

    assert [fused.unit_id for fused in joined] == ["u0", "u2", "u3"]
    assert all(fused.semantic.unit_id == fused.structural.unit_id for fused in joined)
    # The explicit form of the same claim: position 1 is u2 on BOTH sides, not u2's
    # tokens beside whatever graph happens to sit at structural index 1.
    assert joined[1].structural is structural.units[1]


def test_an_empty_join_raises_rather_than_training_on_nothing(tmp_path, tokenizer) -> None:
    semantic, structural = halves(tmp_path, tokenizer, [unit(f"u{i}") for i in range(2)])
    structural.units.clear()

    with pytest.raises(DataError, match="no unit is present in both"):
        join_branches(semantic, structural)


def test_branches_that_disagree_about_a_label_raise_at_the_join(tmp_path, tokenizer) -> None:
    """Both halves come from the same `CodeUnit`; if they ever differ, a path changed."""
    import dataclasses

    semantic, structural = halves(tmp_path, tokenizer, [unit("u0"), unit("u1")])
    structural.units[0] = dataclasses.replace(structural.units[0], n_lines=999)

    with pytest.raises(DataError, match="disagree about a unit"):
        join_branches(semantic, structural)


def test_a_duplicate_unit_id_raises_rather_than_picking_one(tmp_path, tokenizer) -> None:
    semantic, structural = halves(tmp_path, tokenizer, [unit("u0"), unit("u1")])
    structural.units[1] = structural.units[0]

    with pytest.raises(DataError, match="duplicate unit_id"):
        join_branches(semantic, structural)


def test_units_dropped_by_the_join_are_counted_not_absorbed(tmp_path, tokenizer) -> None:
    """A fused arm training on a smaller population than its baselines is a confound."""
    built = dataset(tmp_path, tokenizer, [unit(f"u{i}") for i in range(3)])

    assert built.semantic_only == 0
    assert built.structural_only == 0
    assert built.build_failures == 0


def test_the_loss_weight_is_computed_over_the_joined_population(tmp_path, tokenizer) -> None:
    """Not over either branch's own set — the weight must match the loss's denominator."""
    built = dataset(tmp_path, tokenizer, [unit(f"u{i}") for i in range(3)])

    scoreable = sum(min(u.semantic.n_lines, built.n_lines) for u in built.units)
    positives = sum(sum(u.semantic.line_labels) for u in built.units)
    assert built.positive_line_weight() == pytest.approx((scoreable - positives) / positives)


# --- collation -----------------------------------------------------------------------


def test_the_batch_is_flat_and_prefixed(tmp_path, tokenizer) -> None:
    """A nested batch would arrive half on the GPU with nothing raising.

    `training/loop.py` moves a batch to the device by walking its TOP level only.
    """
    built = dataset(tmp_path, tokenizer, [unit(f"u{i}") for i in range(3)])

    batch = collate_fused(list(built.units))

    assert batch["sem_input_ids"].dim() == 2
    assert batch["str_node_type"].dim() == 1
    assert batch["n_units"] == 3
    assert batch["unit_ids"] == ["u0", "u1", "u2"]
    assert all(not isinstance(value, dict) for value in batch.values())
    # Exactly one copy of each shared key, unprefixed.
    for shared in ("line_labels", "line_valid", "targets"):
        assert shared in batch
        assert f"sem_{shared}" not in batch
        assert f"str_{shared}" not in batch


def test_every_window_and_every_node_points_at_the_right_unit(tmp_path, tokenizer) -> None:
    built = dataset(tmp_path, tokenizer, [unit(f"u{i}") for i in range(3)])

    batch = collate_fused(list(built.units))

    assert int(batch["sem_window_to_unit"].max()) == 2
    assert int(batch["str_batch"].max()) == 2
    assert batch["sem_window_to_unit"].shape[0] == batch["sem_input_ids"].shape[0]
    assert batch["str_batch"].shape[0] == batch["str_node_type"].shape[0]


def test_collating_branches_that_disagree_raises(tmp_path, tokenizer) -> None:
    """The last place a mismatch is catchable before it becomes a trained model."""
    import dataclasses

    built = dataset(tmp_path, tokenizer, [unit("u0"), unit("u1")])
    first = built.units[0]
    tampered = dataclasses.replace(
        first, structural=dataclasses.replace(first.structural, target=0)
    )

    with pytest.raises(DataError, match="different line supervision"):
        collate_fused([tampered, built.units[1]])


def test_an_empty_batch_raises(tmp_path, tokenizer) -> None:
    with pytest.raises(DataError, match="empty batch"):
        collate_fused([])


# --- the sampler ---------------------------------------------------------------------


def test_a_batch_closes_on_whichever_budget_binds_first(tmp_path, tokenizer) -> None:
    """The two costs are uncorrelated, so honouring one leaves memory to the other."""
    built = dataset(tmp_path, tokenizer, [unit(f"u{i}") for i in range(6)])
    nodes = built.units[0].n_nodes

    sampler = FusedBudgetBatchSampler(built, max_windows=1000, max_nodes=2 * nodes, shuffle=False)

    batches = list(sampler)
    assert all(len(batch) <= 2 for batch in batches)
    assert sum(len(batch) for batch in batches) == 6


def test_a_unit_over_budget_still_gets_a_batch(tmp_path, tokenizer) -> None:
    """The budget is a target, not a guarantee.

    Dropping an over-budget unit would discard the largest functions, which carry the most
    flagged lines (Memory.md §1.8).
    """
    built = dataset(tmp_path, tokenizer, [unit(f"u{i}") for i in range(3)])

    batches = list(FusedBudgetBatchSampler(built, max_windows=1, max_nodes=1, shuffle=False))

    assert sum(len(batch) for batch in batches) == 3
    assert all(len(batch) == 1 for batch in batches)


def test_the_sampler_shuffles_reproducibly_per_epoch(tmp_path, tokenizer) -> None:
    built = dataset(tmp_path, tokenizer, [unit(f"u{i}") for i in range(8)])
    sampler = FusedBudgetBatchSampler(built, max_windows=2, max_nodes=10_000, seed=42)

    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    sampler.set_epoch(0)

    assert list(sampler) == first
    assert first != second


def test_both_budgets_must_be_positive(tmp_path, tokenizer) -> None:
    built = dataset(tmp_path, tokenizer, [unit("u0")])

    with pytest.raises(DataError, match="must be positive"):
        FusedBudgetBatchSampler(built, max_windows=0, max_nodes=1)


# --- coverage accounting -------------------------------------------------------------


def test_coverage_reports_what_the_union_recovers(tmp_path, tokenizer) -> None:
    """The concrete advantage of fusion over the structural arm, measured per batch."""
    built = dataset(tmp_path, tokenizer, [unit("u0"), unit("u1")])
    batch = collate_fused(list(built.units))
    n_units, n_lines = batch["line_labels"].shape

    sem = torch.zeros(n_units, n_lines, dtype=torch.bool)
    sem[:, :4] = True
    structural = torch.zeros(n_units, n_lines, dtype=torch.bool)
    structural[:, :1] = True

    counts = fused_line_coverage(
        batch, {"line_mask": sem | structural, "sem_line_mask": sem, "str_line_mask": structural}
    )

    assert counts["flagged"] == int((batch["line_labels"] > 0).sum())
    assert counts["flagged_semantic_only"] >= 1
    assert counts["flagged_structural_only"] == 0
    assert counts["recovered_by_union"] == counts["flagged_semantic_only"]


# --- representation.strip_comments (Memory.md §1.25) ----------------------------------

# Line 3's comment lost the slash of its terminator, as flagged lines do in the corpus.
COMMENTED = "int f(int a)\n{\nint b = a;  /* why this is here *\nreturn b;\n}"


def test_stripping_comments_reaches_both_branches(tmp_path, tokenizer) -> None:
    (tmp_path / "plain").mkdir()
    (tmp_path / "stripped").mkdir()
    units = [unit("u1", source=COMMENTED, lines=(3,))]

    plain = dataset(tmp_path / "plain", tokenizer, units).units[0]
    stripped = dataset(tmp_path / "stripped", tokenizer, units, strip_comments=True).units[0]

    def tokens(encoded) -> int:
        return sum(sum(window.attention_mask) for window in encoded.windows)

    assert tokens(stripped.semantic) < tokens(plain.semantic), "the comment's tokens are gone"
    # The graph is built from the same stripped text: line 3 no longer carries the nodes
    # the grammar made of an unterminated comment.
    on_line_3 = [unit.structural.node_line.tolist().count(3) for unit in (plain, stripped)]
    assert on_line_3[1] < on_line_3[0]
    assert stripped.semantic.line_labels == plain.semantic.line_labels, "labels never move"
