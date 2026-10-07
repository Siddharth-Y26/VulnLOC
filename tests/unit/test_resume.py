"""Guards the property multi-night training rests on: a resumed run is a continuation.

A fine-tuned run is hours and the machine is not free for hours at a stretch, so the runs
that matter most are exactly the ones that get interrupted. A resume that is *nearly*
right is the dangerous kind: it trains, converges and reports a number, and the number is
from a run that saw part of an epoch twice, forgot its best epoch, or left half its
learning curve under another run id. Each test here is one of those.
"""

from __future__ import annotations

import math

import mlflow
import pytest
import torch
from omegaconf import open_dict

from vulnloc.models.heads.line_head import LineHead
from vulnloc.training import job
from vulnloc.training.loop import (
    PAUSED_EXIT_CODE,
    STOP_FILE_ENV,
    Trainer,
    TrainingPaused,
    TrainState,
)
from vulnloc.training.losses import LineLoss
from vulnloc.utils.exceptions import ConfigError

N_BATCHES = 4


class TinyModel(torch.nn.Module):
    """A head over fixed features: no dropout, so CPU training is exactly repeatable."""

    def __init__(self) -> None:
        super().__init__()
        self.head = LineHead(4, hidden_dim=0, dropout=0.0)

    def forward(self, batch):
        logits, function_logits = self.head(batch["features"], batch["line_valid"])
        return {
            "line_logits": logits,
            "function_logits": function_logits,
            "line_mask": batch["line_valid"],
        }


def _batches(epoch: int) -> list[dict[str, torch.Tensor]]:
    """One epoch's batches — a pure function of the epoch, as the real samplers are."""
    generator = torch.Generator().manual_seed(1000 + epoch)
    batches = []
    for _ in range(N_BATCHES):
        labels = torch.zeros(2, 5)
        labels[:, 0] = 1.0
        batches.append(
            {
                "features": torch.randn(2, 5, 4, generator=generator),
                "line_valid": torch.ones(2, 5, dtype=torch.bool),
                "line_labels": labels,
                "targets": torch.ones(2),
            }
        )
    return batches


def _trainer(tmp_path, model, *, every=1, accumulation=1):
    return Trainer(
        model,
        LineLoss(kind="bce"),
        torch.optim.AdamW(model.parameters(), lr=0.05),
        torch.device("cpu"),
        checkpoint_dir=tmp_path,
        use_amp=False,
        accumulation_steps=accumulation,
        checkpoint_every_n_steps=every,
    )


def _weights(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


class StopAfter:
    """Yield batches, and ask for a pause once ``n`` of them have been handed out."""

    def __init__(self, batches, n, stop_file):
        self.batches = batches
        self.n = n
        self.stop_file = stop_file

    def __iter__(self):
        for index, batch in enumerate(self.batches):
            if index + 1 == self.n:
                self.stop_file.write_text("stop", encoding="utf-8")
            yield batch


# --- the trainer -----------------------------------------------------------------------


def test_a_paused_epoch_resumes_to_the_weights_of_an_uninterrupted_one(tmp_path, monkeypatch):
    """The whole claim, in one assertion: no batch twice, no batch skipped."""
    torch.manual_seed(0)
    reference = TinyModel()
    whole = _trainer(tmp_path / "whole", reference, accumulation=2)
    whole_metrics = whole.train_epoch(_batches(0))

    stop_file = tmp_path / "STOP"
    monkeypatch.setenv(STOP_FILE_ENV, str(stop_file))
    torch.manual_seed(0)
    first = _trainer(tmp_path / "split", TinyModel(), accumulation=2)
    with pytest.raises(TrainingPaused):
        first.train_epoch(StopAfter(_batches(0), 2, stop_file))
    assert first.state.batch_in_epoch == 2

    stop_file.unlink()
    torch.manual_seed(99)
    resumed_model = TinyModel()
    second = _trainer(tmp_path / "split", resumed_model, accumulation=2)
    second.load_checkpoint(tmp_path / "split" / "last.pt")
    resumed_metrics = second.train_epoch(_batches(0), resume=True)

    for expected, actual in zip(_weights(reference), _weights(resumed_model), strict=True):
        assert torch.equal(expected, actual)
    assert second.state.global_step == whole.state.global_step
    # The epoch's train loss covers the batches the first process trained on too.
    assert resumed_metrics["loss"] == pytest.approx(whole_metrics["loss"])


def test_resuming_without_the_flag_would_train_the_first_batches_twice(tmp_path, monkeypatch):
    """The failure the flag exists to prevent, shown to be a real difference."""
    torch.manual_seed(0)
    reference = TinyModel()
    _trainer(tmp_path / "whole", reference).train_epoch(_batches(0))

    stop_file = tmp_path / "STOP"
    monkeypatch.setenv(STOP_FILE_ENV, str(stop_file))
    torch.manual_seed(0)
    first = _trainer(tmp_path / "split", TinyModel())
    with pytest.raises(TrainingPaused):
        first.train_epoch(StopAfter(_batches(0), 2, stop_file))
    stop_file.unlink()

    replayed = TinyModel()
    second = _trainer(tmp_path / "split", replayed)
    second.load_checkpoint(tmp_path / "split" / "last.pt")
    second.train_epoch(_batches(0))

    assert any(
        not torch.equal(expected, actual)
        for expected, actual in zip(_weights(reference), _weights(replayed), strict=True)
    )


def test_a_pause_is_only_honoured_between_optimizer_steps(tmp_path, monkeypatch):
    """With accumulation 2 a request after batch 1 waits for batch 2's step."""
    stop_file = tmp_path / "STOP"
    monkeypatch.setenv(STOP_FILE_ENV, str(stop_file))
    trainer = _trainer(tmp_path, TinyModel(), accumulation=2)

    with pytest.raises(TrainingPaused):
        trainer.train_epoch(StopAfter(_batches(0), 1, stop_file))

    assert trainer.state.batch_in_epoch == 2
    assert trainer.state.global_step == 1


def test_a_pause_writes_a_checkpoint_even_with_rolling_checkpoints_off(tmp_path, monkeypatch):
    stop_file = tmp_path / "STOP"
    monkeypatch.setenv(STOP_FILE_ENV, str(stop_file))
    trainer = _trainer(tmp_path, TinyModel(), every=0)

    with pytest.raises(TrainingPaused):
        trainer.train_epoch(StopAfter(_batches(0), 1, stop_file))

    payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert payload["provenance"]["kind"] == "paused"
    assert payload["state"]["batch_in_epoch"] == 1


def test_no_stop_file_configured_means_no_pause(tmp_path, monkeypatch):
    monkeypatch.delenv(STOP_FILE_ENV, raising=False)
    trainer = _trainer(tmp_path, TinyModel())

    trainer.train_epoch(_batches(0))

    assert not trainer.pause_requested
    assert trainer.state.batch_in_epoch == N_BATCHES


def test_a_checkpoint_is_renamed_into_place_not_written_in_place(tmp_path):
    """A kill mid-write must leave the previous last.pt, not a truncated one."""
    trainer = _trainer(tmp_path, TinyModel())

    trainer.train_epoch(_batches(0))

    assert (tmp_path / "last.pt").exists()
    assert not (tmp_path / "last.pt.tmp").exists()


def _refuse_renames(monkeypatch, times):
    """Make ``Path.replace`` raise the Windows sharing violation ``times`` times, then work.

    Returns:
        ``(attempts, waits)`` — lists filled as the code under test calls and sleeps.
    """
    from pathlib import Path

    from vulnloc.training import loop

    real_replace = Path.replace
    attempts: list[str] = []
    waits: list[float] = []

    def replace(self, target):
        attempts.append(Path(target).name)
        if len(attempts) <= times:
            raise PermissionError(5, "Access is denied", str(self), 5, str(target))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(loop.time, "sleep", waits.append)
    return attempts, waits


def test_a_locked_checkpoint_is_waited_for_not_died_on(tmp_path, monkeypatch):
    """§7 #56: a scanner holding last.pt for a moment cost a run two hours in."""
    trainer = _trainer(tmp_path, TinyModel())
    trainer.save_checkpoint(tmp_path / "last.pt", {})
    attempts, waits = _refuse_renames(monkeypatch, times=3)

    trainer.state.global_step = 7
    trainer.save_last("periodic")

    payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert payload["state"]["global_step"] == 7
    assert len(attempts) == 4
    assert waits == [1.0, 2.0, 4.0]
    assert not (tmp_path / "last.pt.tmp").exists()


def test_a_checkpoint_locked_for_good_still_fails_and_keeps_the_previous_one(tmp_path, monkeypatch):
    """Waiting is bounded, and what it gives up on is the new file, never the old one."""
    from vulnloc.training import loop

    trainer = _trainer(tmp_path, TinyModel())
    trainer.state.global_step = 3
    trainer.save_checkpoint(tmp_path / "last.pt", {})
    attempts, waits = _refuse_renames(monkeypatch, times=10_000)

    trainer.state.global_step = 9
    with pytest.raises(PermissionError):
        trainer.save_last("periodic")

    assert len(attempts) == len(loop._RENAME_RETRY_DELAYS_S) + 1
    assert waits == list(loop._RENAME_RETRY_DELAYS_S)
    payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert payload["state"]["global_step"] == 3


def test_the_rolling_checkpoint_carries_the_callers_provenance(tmp_path):
    trainer = _trainer(tmp_path, TinyModel())
    trainer.provenance = {"run_id": "abc", "seed": 7}

    trainer.train_epoch(_batches(0))
    payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)

    assert payload["provenance"]["run_id"] == "abc"
    assert payload["provenance"]["kind"] == "periodic"


def test_a_checkpoint_from_before_these_fields_existed_still_loads(tmp_path):
    """It resumes the old way — from the start of its epoch — rather than not at all."""
    trainer = _trainer(tmp_path, TinyModel())
    path = tmp_path / "old.pt"
    trainer.save_checkpoint(path, {})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["state"] = {"epoch": 3, "global_step": 40, "best_metric": 0.25}
    torch.save(payload, path)

    restored = _trainer(tmp_path, TinyModel())
    restored.load_checkpoint(path)

    assert restored.state.epoch == 3
    assert restored.state.batch_in_epoch == 0
    assert restored.state.best_metrics == {}


# --- the job: one run id, one learning curve, one best epoch --------------------------


class FakeDataset:
    """The three things `run_training` asks of a dataset."""

    flagged_beyond_cap = 0

    def __len__(self) -> int:
        return N_BATCHES * 2

    def positive_line_weight(self) -> float:
        return 1.0


class FakeLoader:
    """Stands in for a DataLoader whose batch sampler reshuffles per epoch."""

    def __init__(self, stop_at: tuple[int, int] | None = None, stop_file=None) -> None:
        self.batch_sampler = self
        self.epoch = 0
        self.stop_at = stop_at
        self.stop_file = stop_file

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        batches = _batches(self.epoch)
        if self.stop_at is not None and self.stop_at[0] == self.epoch:
            return iter(StopAfter(batches, self.stop_at[1], self.stop_file))
        return iter(batches)


# Best at epoch 1, then three evaluations without improvement: with patience 2 the run
# stops after epoch 3. A resume that forgot epoch 1 would call epoch 2 "best" and run on.
WATCHED = [0.5, 0.9, 0.3, 0.2, 0.1, 0.05]


@pytest.fixture
def job_cfg(isolated_cfg, tmp_path):
    cfg = isolated_cfg("train=semantic")
    with open_dict(cfg):
        cfg.train.checkpoint_dir = str(tmp_path / "checkpoints")
        cfg.train.epochs = len(WATCHED)
        cfg.train.patience = 2
        cfg.train.amp = False
        cfg.train.accumulation_steps = 2
        cfg.train.checkpoint_every_n_steps = 1
        cfg.train.loss = "bce"
        cfg.train.auto_pos_weight = False
        cfg.train.lr = 0.05
    return cfg


def _scripted_evaluate(monkeypatch):
    """Replace evaluation with the scripted curve, shared across processes of one run."""
    values = iter(WATCHED)

    def evaluate(*_args, **_kwargs):
        value = next(values)
        return {"line/top_5_accuracy": value, "loss": 1.0 - value}

    monkeypatch.setattr(job, "evaluate", evaluate)


def _run(cfg, model, loader):
    return job.run_training(
        cfg,
        model=model,
        train_set=FakeDataset(),
        eval_set=FakeDataset(),
        make_loader=lambda _dataset, shuffle: loader if shuffle else FakeLoader(),
        device=torch.device("cpu"),
    )


def _history(cfg, run_id, key="eval/line/top_5_accuracy"):
    client = mlflow.tracking.MlflowClient(tracking_uri=str(cfg.mlflow.tracking_uri))
    return sorted((point.step, point.value) for point in client.get_metric_history(run_id, key))


def _pause_then_resume(cfg, tmp_path, monkeypatch, stop_at):
    """Run until a pause at ``(epoch, batch)``, then resume in a fresh model."""
    stop_file = tmp_path / "STOP"
    monkeypatch.setenv(STOP_FILE_ENV, str(stop_file))
    _scripted_evaluate(monkeypatch)
    torch.manual_seed(0)
    with pytest.raises(SystemExit) as paused:
        _run(cfg, TinyModel(), FakeLoader(stop_at, stop_file))
    assert paused.value.code == PAUSED_EXIT_CODE
    stop_file.unlink()

    (run_dir,) = (tmp_path / "checkpoints").iterdir()
    with open_dict(cfg):
        cfg.train.resume_from = str(run_dir / "last.pt")
    torch.manual_seed(99)
    model = TinyModel()
    summary = _run(cfg, model, FakeLoader())
    return run_dir.name, summary, model


def test_a_resumed_job_is_the_same_run_with_the_same_result(job_cfg, tmp_path, monkeypatch):
    _scripted_evaluate(monkeypatch)
    torch.manual_seed(0)
    reference = TinyModel()
    whole = _run(job_cfg, reference, FakeLoader())
    for run_dir in (tmp_path / "checkpoints").iterdir():
        run_dir.rename(tmp_path / "reference_run")

    run_id, summary, model = _pause_then_resume(job_cfg, tmp_path, monkeypatch, stop_at=(2, 2))

    # One run id, so one learning curve: every epoch exactly once, none under another id.
    assert summary["run_id"] == run_id
    assert _history(job_cfg, run_id) == _history(job_cfg, whole["run_id"])
    assert [step for step, _ in _history(job_cfg, run_id)] == [0, 1, 2, 3]
    # The best epoch is the one from BEFORE the interruption, and patience was kept.
    assert summary["best"]["line/top_5_accuracy"] == 0.9
    assert summary["best"] == whole["best"]
    for expected, actual in zip(_weights(reference), _weights(model), strict=True):
        assert torch.equal(expected, actual)

    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["resumed_at"].startswith("epoch=2,batch=2,")
    assert run.data.params["train.resume_from"] == ""


def test_a_run_records_which_card_it_ran_on(job_cfg, tmp_path, monkeypatch):
    """§7 #49: "cuda:0" is every card, so two laptops' seeds would have been averaged."""
    _scripted_evaluate(monkeypatch)

    summary = _run(job_cfg, TinyModel(), FakeLoader())

    params = mlflow.get_run(summary["run_id"]).data.params
    assert params["resolved_device"] == "cpu"
    assert params["resolved_gpu"] == "none"


def test_runs_from_two_cards_are_not_replicates():
    """The guard this parameter exists for, exercised through the seed report's own check."""
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    script = Path(__file__).resolve().parents[2] / "scripts" / "report_seeds.py"
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_report_seeds_gpu", script)
    assert spec is not None and spec.loader is not None
    report_seeds = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = report_seeds
    spec.loader.exec_module(report_seeds)

    def run(run_id, seed, gpu):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, start_time=0, end_time=1),
            data=SimpleNamespace(
                params={
                    "seed": str(seed),
                    "normalize_whitespace": "True",
                    "train.name": "fused",
                    "resolved_device": "cuda:0",
                    "resolved_gpu": gpu,
                },
                tags={"config_hash": "c", "git_sha": "abc1234"},
            ),
        )

    first = run("a" * 32, 42, "NVIDIA GeForce RTX 3050 Ti Laptop GPU")
    second = run("b" * 32, 1, "NVIDIA GeForce RTX 4050 Laptop GPU")

    with pytest.raises(ConfigError, match="do not share a configuration"):
        report_seeds._check_comparable([first, second])


def test_a_paused_run_is_not_finished(job_cfg, tmp_path, monkeypatch):
    """report_seeds.py only reads FINISHED runs, so half a run can never be aggregated."""
    stop_file = tmp_path / "STOP"
    monkeypatch.setenv(STOP_FILE_ENV, str(stop_file))
    _scripted_evaluate(monkeypatch)

    with pytest.raises(SystemExit):
        _run(job_cfg, TinyModel(), FakeLoader((1, 2), stop_file))

    (run_dir,) = (tmp_path / "checkpoints").iterdir()
    run = mlflow.get_run(run_dir.name)
    assert run.info.status != "FINISHED"
    assert "paused_at" in run.data.tags
    assert not (run_dir / "summary.json").exists()


def test_a_pause_request_during_evaluation_waits_for_the_epoch_boundary(
    job_cfg, tmp_path, monkeypatch
):
    stop_file = tmp_path / "STOP"
    monkeypatch.setenv(STOP_FILE_ENV, str(stop_file))
    values = iter(WATCHED)

    def evaluate(*_args, **_kwargs):
        stop_file.write_text("stop", encoding="utf-8")
        value = next(values)
        return {"line/top_5_accuracy": value, "loss": 1.0 - value}

    monkeypatch.setattr(job, "evaluate", evaluate)

    with pytest.raises(SystemExit):
        _run(job_cfg, TinyModel(), FakeLoader())

    (run_dir,) = (tmp_path / "checkpoints").iterdir()
    payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    state = TrainState(**payload["state"])
    assert (state.epoch, state.batch_in_epoch) == (1, 0)
    assert _history(job_cfg, run_dir.name) == [(0, 0.5)]


def test_resuming_under_a_different_config_is_refused(job_cfg, tmp_path, monkeypatch):
    stop_file = tmp_path / "STOP"
    monkeypatch.setenv(STOP_FILE_ENV, str(stop_file))
    _scripted_evaluate(monkeypatch)
    with pytest.raises(SystemExit):
        _run(job_cfg, TinyModel(), FakeLoader((1, 2), stop_file))
    stop_file.unlink()

    (run_dir,) = (tmp_path / "checkpoints").iterdir()
    with open_dict(job_cfg):
        job_cfg.train.resume_from = str(run_dir / "last.pt")
        job_cfg.train.lr = 0.5

    with pytest.raises(ConfigError, match="not the one the run was started with"):
        _run(job_cfg, TinyModel(), FakeLoader())


def test_resuming_a_run_the_tracking_store_never_saw_is_refused(job_cfg, tmp_path):
    with open_dict(job_cfg):
        job_cfg.train.resume_from = str(tmp_path / "checkpoints" / ("0" * 32) / "last.pt")

    with pytest.raises(ConfigError, match="no such run"):
        _run(job_cfg, TinyModel(), FakeLoader())


def test_an_epoch_checkpointed_but_never_logged_is_logged_on_resume(job_cfg, tmp_path, monkeypatch):
    """The checkpoint is written before the tracking store; a kill between them is real."""
    stop_file = tmp_path / "STOP"
    monkeypatch.setenv(STOP_FILE_ENV, str(stop_file))
    _scripted_evaluate(monkeypatch)
    with pytest.raises(SystemExit):
        _run(job_cfg, TinyModel(), FakeLoader((1, 2), stop_file))
    stop_file.unlink()
    (run_dir,) = (tmp_path / "checkpoints").iterdir()
    run_id = run_dir.name
    assert _history(job_cfg, run_id) == [(0, 0.5)]

    with mlflow.start_run(run_id=run_id):
        job._log_epoch_if_missing(run_id, {"step": 0, "metrics": {"eval/loss": 9.0}})
        job._log_epoch_if_missing(
            run_id, {"step": 1, "metrics": {"eval/loss": 0.1, "eval/line/top_5_accuracy": 0.9}}
        )

    # Step 0 was already there and is not logged twice; step 1 was missing and now is not.
    assert _history(job_cfg, run_id, "eval/loss") == [(0, 0.5), (1, 0.1)]


def test_best_pt_wins_when_the_run_died_before_last_pt_caught_up(tmp_path):
    """Killed between the two writes at an epoch boundary: best.pt is the later state."""
    torch.manual_seed(0)
    writer = _trainer(tmp_path, TinyModel())
    writer.state = TrainState(epoch=1, global_step=6, batch_in_epoch=3)
    writer.save_checkpoint(tmp_path / "last.pt", {})
    writer.train_epoch(_batches(1)[:1])
    writer.state = TrainState(epoch=2, global_step=8, best_metric=0.9, best_metrics={"m": 0.9})
    writer.save_checkpoint(tmp_path / "best.pt", {"m": 0.9})
    later = (tmp_path / "last.pt").stat().st_mtime + 5
    import os

    os.utime(tmp_path / "best.pt", (later, later))

    restored = TinyModel()
    trainer = _trainer(tmp_path, restored)
    job._resume(trainer, tmp_path / "last.pt", tmp_path / "best.pt")

    assert (trainer.state.epoch, trainer.state.batch_in_epoch) == (2, 0)
    assert math.isclose(trainer.state.best_metric, 0.9)
    for expected, actual in zip(_weights(writer.model), _weights(restored), strict=True):
        assert torch.equal(expected, actual)
