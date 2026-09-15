"""Guards batching, collation, early stopping and checkpoints.

The batching tests exist because sizing a batch in units instead of windows is the exact
mistake ``Memory.md`` §7 #28 warns about, and it does not fail — it just makes memory
unpredictable until something OOMs hours in.

The checkpoint tests exist because "checkpoints save/load" is a Phase 3 exit criterion,
and a checkpoint that silently fails to restore the optimizer looks fine until a resumed
run quietly starts from scratch.
"""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from vulnloc.models.heads.line_head import LineHead
from vulnloc.representation.tokenization import LineTokenizer
from vulnloc.training.datasets import (
    EncodedUnit,
    WindowBatchSampler,
    collate_units,
    line_label_coverage,
)
from vulnloc.training.loop import EarlyStopping, Trainer, evaluate
from vulnloc.training.losses import LineLoss
from vulnloc.utils.exceptions import DataError, ModelError


@pytest.fixture(scope="module")
def hf_tokenizer():
    vocab = {"<s>": 0, "</s>": 1, "<pad>": 2, "<unk>": 3}
    for character in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[character] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 1)]
    )
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def make_unit(unit_id: str, n_windows: int, tokenizer: LineTokenizer, n_lines: int = 8):
    """Build an EncodedUnit with a chosen number of windows, without touching a shard."""
    from vulnloc.representation.units import CodeUnit

    source = "\n".join(f"int v{i} = {i};" for i in range(4))
    encoded = tokenizer.encode(CodeUnit.from_source(unit_id, source, language="C"))
    window = encoded.windows[0]
    labels = [0] * n_lines
    labels[0] = 1
    return EncodedUnit(
        unit_id=unit_id,
        windows=tuple(window for _ in range(n_windows)),
        line_labels=tuple(labels),
        n_lines=4,
        target=1,
        flagged_beyond_cap=0,
    )


class FakeDataset:
    """Minimal stand-in with the two attributes the sampler needs."""

    def __init__(self, units):
        self.units = units

    def __len__(self):
        return len(self.units)

    def __getitem__(self, index):
        return self.units[index]


# --- window-budget batching -----------------------------------------------------------


def test_batches_are_filled_to_a_window_budget_not_a_unit_count(hf_tokenizer):
    tokenizer = LineTokenizer(hf_tokenizer, max_length=64, stride=16)
    units = [make_unit(f"u{i}", 4, tokenizer) for i in range(6)]
    sampler = WindowBatchSampler(FakeDataset(units), max_windows=8, shuffle=False)

    batches = list(sampler)

    for batch in batches:
        assert sum(units[i].n_windows for i in batch) <= 8
    assert sum(len(b) for b in batches) == 6


def test_a_unit_larger_than_the_budget_still_gets_scored(hf_tokenizer):
    """A unit bigger than the budget still gets scored.

    Dropping it would silently discard the longest functions — the ones carrying the most
    flagged lines (Memory.md §1.8). The budget is a target, not a guarantee.
    """
    tokenizer = LineTokenizer(hf_tokenizer, max_length=64, stride=16)
    units = [make_unit("big", 20, tokenizer), make_unit("small", 1, tokenizer)]
    sampler = WindowBatchSampler(FakeDataset(units), max_windows=4, shuffle=False)

    batches = list(sampler)

    assert [0] in batches
    assert sum(len(b) for b in batches) == 2


def test_every_unit_appears_exactly_once_per_epoch(hf_tokenizer):
    tokenizer = LineTokenizer(hf_tokenizer, max_length=64, stride=16)
    units = [make_unit(f"u{i}", (i % 3) + 1, tokenizer) for i in range(15)]
    sampler = WindowBatchSampler(FakeDataset(units), max_windows=5, seed=7)

    seen = [index for batch in sampler for index in batch]

    assert sorted(seen) == list(range(15))


def test_shuffling_is_reproducible_and_epoch_dependent(hf_tokenizer):
    tokenizer = LineTokenizer(hf_tokenizer, max_length=64, stride=16)
    units = [make_unit(f"u{i}", 1, tokenizer) for i in range(12)]
    sampler = WindowBatchSampler(FakeDataset(units), max_windows=3, seed=7)

    first = list(sampler)
    again = list(sampler)
    sampler.set_epoch(1)
    later = list(sampler)

    assert first == again, "same epoch must reproduce exactly"
    assert first != later, "a new epoch must reshuffle"


def test_max_units_caps_a_batch_of_tiny_units(hf_tokenizer):
    tokenizer = LineTokenizer(hf_tokenizer, max_length=64, stride=16)
    units = [make_unit(f"u{i}", 1, tokenizer) for i in range(10)]
    sampler = WindowBatchSampler(FakeDataset(units), max_windows=100, shuffle=False, max_units=3)

    assert all(len(batch) <= 3 for batch in sampler)


def test_non_positive_budget_raises(hf_tokenizer):
    with pytest.raises(DataError):
        WindowBatchSampler(FakeDataset([]), max_windows=0)


# --- collation ------------------------------------------------------------------------


def test_collate_flattens_windows_and_records_their_owner(hf_tokenizer):
    tokenizer = LineTokenizer(hf_tokenizer, max_length=64, stride=16)
    units = [make_unit("a", 2, tokenizer), make_unit("b", 3, tokenizer)]

    batch = collate_units(units)

    assert batch["input_ids"].shape == (5, 64)
    assert batch["window_to_unit"].tolist() == [0, 0, 1, 1, 1]
    assert batch["n_units"] == 2
    assert batch["line_labels"].shape == (2, 8)
    assert batch["unit_ids"] == ["a", "b"]


def test_line_valid_marks_only_lines_the_unit_actually_has(hf_tokenizer):
    """`line_valid` is a *label* mask, not the feature mask the model produces.

    Conflating the two would excuse the model from scoring lines it happened to skip.
    """
    tokenizer = LineTokenizer(hf_tokenizer, max_length=64, stride=16)
    batch = collate_units([make_unit("a", 1, tokenizer)])

    assert batch["line_valid"][0, :4].all()
    assert not batch["line_valid"][0, 4:].any()


def test_collating_nothing_raises():
    with pytest.raises(DataError):
        collate_units([])


def test_loader_round_trip(hf_tokenizer):
    tokenizer = LineTokenizer(hf_tokenizer, max_length=64, stride=16)
    units = [make_unit(f"u{i}", 1, tokenizer) for i in range(4)]
    dataset = FakeDataset(units)
    loader = DataLoader(
        dataset,
        batch_sampler=WindowBatchSampler(dataset, max_windows=2, shuffle=False),
        collate_fn=collate_units,
    )

    batches = list(loader)

    assert sum(int(b["n_units"]) for b in batches) == 4


def test_line_label_coverage_counts_unscoreable_flagged_lines(hf_tokenizer):
    tokenizer = LineTokenizer(hf_tokenizer, max_length=64, stride=16)
    batch = collate_units([make_unit("a", 1, tokenizer)])
    no_features = torch.zeros_like(batch["line_valid"])

    counts = line_label_coverage(batch, no_features)

    assert counts["flagged"] == 1
    assert counts["flagged_with_features"] == 0


# --- early stopping -------------------------------------------------------------------


def test_early_stopping_reports_improvement_and_counts_patience():
    stopper = EarlyStopping(patience=2, mode="max")

    assert stopper.update(0.5) is True
    assert stopper.update(0.4) is False
    assert not stopper.should_stop
    assert stopper.update(0.45) is False
    assert stopper.should_stop


def test_early_stopping_resets_patience_on_improvement():
    stopper = EarlyStopping(patience=2, mode="max")
    stopper.update(0.5)
    stopper.update(0.1)
    stopper.update(0.9)

    assert stopper.waited == 0
    assert not stopper.should_stop


def test_min_delta_rejects_noise():
    stopper = EarlyStopping(patience=1, mode="max", min_delta=0.1)
    stopper.update(0.5)

    assert stopper.update(0.55) is False


def test_min_mode_watches_downwards():
    stopper = EarlyStopping(patience=1, mode="min")

    assert stopper.update(1.0) is True
    assert stopper.update(0.5) is True
    assert stopper.update(0.9) is False


# --- checkpoints ----------------------------------------------------------------------


class TinyModel(torch.nn.Module):
    """A head over fixed features — enough to exercise save/load without an encoder."""

    def __init__(self):
        super().__init__()
        self.head = LineHead(4, hidden_dim=0)

    def forward(self, batch):
        logits, function_logits = self.head(batch["features"], batch["line_valid"])
        return {
            "line_logits": logits,
            "function_logits": function_logits,
            "line_mask": batch["line_valid"],
        }


def _trainer(tmp_path, model):
    return Trainer(
        model,
        LineLoss(kind="bce"),
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        torch.device("cpu"),
        checkpoint_dir=tmp_path,
        use_amp=False,
    )


def test_checkpoint_round_trips_weights_and_optimizer(tmp_path):
    torch.manual_seed(0)
    model = TinyModel()
    trainer = _trainer(tmp_path, model)
    trainer.state.global_step = 7
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path, {"line/f1": 0.5}, run_id="abc", git_sha="deadbeef")

    torch.manual_seed(1)
    restored = TinyModel()
    restored_trainer = _trainer(tmp_path, restored)
    payload = restored_trainer.load_checkpoint(path)

    for before, after in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(before, after)
    assert restored_trainer.state.global_step == 7
    assert payload["provenance"]["run_id"] == "abc"
    assert payload["metrics"]["line/f1"] == 0.5


def test_weights_only_load_leaves_the_step_counter_alone(tmp_path):
    model = TinyModel()
    trainer = _trainer(tmp_path, model)
    trainer.state.global_step = 11
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path, {})

    other = _trainer(tmp_path, TinyModel())
    other.state.global_step = 3
    other.load_checkpoint(path, weights_only=True)

    assert other.state.global_step == 3


def test_loading_a_missing_checkpoint_raises(tmp_path):
    trainer = _trainer(tmp_path, TinyModel())

    with pytest.raises(ModelError):
        trainer.load_checkpoint(tmp_path / "absent.pt")


def test_invalid_accumulation_steps_raises(tmp_path):
    model = TinyModel()
    with pytest.raises(ModelError):
        Trainer(
            model,
            LineLoss(),
            torch.optim.AdamW(model.parameters()),
            torch.device("cpu"),
            checkpoint_dir=tmp_path,
            accumulation_steps=0,
        )


def test_training_a_tiny_model_decreases_the_loss(tmp_path):
    """The Phase 3 exit criterion in miniature: it trains end to end and the loss falls."""
    torch.manual_seed(0)
    model = TinyModel()
    trainer = _trainer(tmp_path, model)

    features = torch.randn(4, 6, 4)
    labels = torch.zeros(4, 6)
    labels[:, 0] = 1.0
    valid = torch.ones(4, 6, dtype=torch.bool)
    batch = {
        "features": features,
        "line_valid": valid,
        "line_labels": labels,
        "targets": torch.ones(4),
    }

    first = trainer.train_epoch([batch])["loss"]
    for _ in range(30):
        last = trainer.train_epoch([batch])["loss"]

    assert last < first


# --- per-step checkpointing (Memory.md §7 #31) ----------------------------------------


def _step_trainer(tmp_path, model, every):
    return Trainer(
        model,
        LineLoss(kind="bce"),
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        torch.device("cpu"),
        checkpoint_dir=tmp_path,
        use_amp=False,
        checkpoint_every_n_steps=every,
    )


def _one_batch():
    labels = torch.zeros(2, 5)
    labels[:, 0] = 1.0
    return {
        "features": torch.randn(2, 5, 4),
        "line_valid": torch.ones(2, 5, dtype=torch.bool),
        "line_labels": labels,
        "targets": torch.ones(2),
    }


def test_a_rolling_checkpoint_appears_mid_epoch(tmp_path):
    """The whole point: an interruption must cost minutes, not the epoch."""
    trainer = _step_trainer(tmp_path, TinyModel(), every=2)
    batch = _one_batch()

    trainer.train_epoch([batch, batch, batch, batch])

    assert (tmp_path / "last.pt").exists()


def test_zero_disables_periodic_checkpointing(tmp_path):
    trainer = _step_trainer(tmp_path, TinyModel(), every=0)

    trainer.train_epoch([_one_batch(), _one_batch()])

    assert not (tmp_path / "last.pt").exists()


def test_the_rolling_checkpoint_records_where_it_was(tmp_path):
    trainer = _step_trainer(tmp_path, TinyModel(), every=1)
    trainer.state.epoch = 3
    batch = _one_batch()

    trainer.train_epoch([batch, batch])
    payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)

    assert payload["provenance"]["kind"] == "periodic"
    assert payload["provenance"]["epoch"] == 3
    assert payload["state"]["global_step"] == 2


def test_a_rolling_checkpoint_resumes_weights_and_step(tmp_path):
    torch.manual_seed(0)
    trainer = _step_trainer(tmp_path, TinyModel(), every=1)
    batch = _one_batch()
    trainer.train_epoch([batch, batch, batch])

    torch.manual_seed(99)
    restored = _step_trainer(tmp_path, TinyModel(), every=1)
    restored.load_checkpoint(tmp_path / "last.pt")

    assert restored.state.global_step == 3
    for before, after in zip(trainer.model.parameters(), restored.model.parameters(), strict=True):
        assert torch.equal(before, after)


def test_negative_interval_raises(tmp_path):
    model = TinyModel()
    with pytest.raises(ModelError):
        Trainer(
            model,
            LineLoss(),
            torch.optim.AdamW(model.parameters()),
            torch.device("cpu"),
            checkpoint_dir=tmp_path,
            checkpoint_every_n_steps=-1,
        )


# --- evaluation must not pay for a backward that never happens ----------------------


class _GradRecorder(torch.nn.Module):
    """A minimal model that records whether autograd was recording when it ran."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(1))
        self.grad_was_enabled: list[bool] = []

    def forward(self, batch):  # type: ignore[no-untyped-def]
        self.grad_was_enabled.append(torch.is_grad_enabled())
        n_units, n_lines = batch["line_labels"].shape
        line_logits = self.scale.expand(n_units, n_lines) * 1.0
        return {
            "line_logits": line_logits,
            "function_logits": line_logits.max(dim=-1).values,
            "line_mask": torch.ones(n_units, n_lines, dtype=torch.bool),
        }


def _eval_batch(n_units: int = 2, n_lines: int = 4) -> dict[str, object]:
    labels = torch.zeros(n_units, n_lines)
    labels[0, 1] = 1.0
    return {
        "line_labels": labels,
        "line_valid": torch.ones(n_units, n_lines, dtype=torch.bool),
        "targets": torch.tensor([1.0] + [0.0] * (n_units - 1)),
    }


def test_evaluate_does_not_record_an_autograd_graph():
    """`model.eval()` settles dropout; it does not stop autograd from recording.

    Without `no_grad` every activation needed for a backward that never comes is retained
    until the next batch replaces it. Measured on the fused model, that was the difference
    between the largest `val` unit fitting on the card and being paged to host RAM at
    10-25x the cost (Memory.md §1.17, §7 #29) — and on Windows the paging does not raise.
    """
    model = _GradRecorder()

    evaluate(
        model,
        [_eval_batch(), _eval_batch()],
        LineLoss(kind="bce"),
        torch.device("cpu"),
        use_amp=False,
    )

    assert model.grad_was_enabled == [False, False]


def test_evaluate_leaves_no_gradient_behind():
    """Nothing in the evaluation path reads a gradient, so nothing should produce one."""
    model = _GradRecorder()

    evaluate(model, [_eval_batch()], LineLoss(kind="bce"), torch.device("cpu"), use_amp=False)

    assert model.scale.grad is None


def test_evaluate_still_reports_the_line_metrics():
    """The guard above must not have cost the numbers it was protecting."""
    model = _GradRecorder()

    results = evaluate(
        model, [_eval_batch()], LineLoss(kind="bce"), torch.device("cpu"), use_amp=False
    )

    assert "loss" in results
    assert "line/top_5_accuracy" in results
    assert isinstance(results["loss"], float)
