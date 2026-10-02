"""The training loop: AMP, gradient accumulation, early stopping, checkpoints.

Defaults come from measurement, not convention (``Memory.md`` §1.9): AMP plus gradient
checkpointing at a 16-window batch is simultaneously the cheapest and the fastest
trainable configuration on the 4 GiB card this project runs on.

The loop is deliberately boring except in three places, each of which corresponds to a
trap this project has already been bitten by or has pre-recorded:

**Memory is asserted, not assumed.** Peak allocation is compared against the card's
physical memory after the first optimizer step, and a configuration that only survives
because Windows paged to host RAM is *warned about loudly*. ``Memory.md`` §7 #29: three of
24 probe configurations exceeded VRAM and ran anyway, 10–25× slower, with no exception. A
loop that waits for ``OutOfMemoryError`` never learns it is thrashing.

**Early stopping watches a line metric, not the loss.** The loss is dominated by the
0.795% class imbalance and improves happily while line localization does not.

**Checkpoints carry their provenance.** Config hash, git sha, run id and the epoch's
metrics travel in the file, so a checkpoint found later can be traced to the run that made
it — ``Rules.md`` §1.3, mechanically rather than by discipline.

**An interrupted run continues; it does not approximately restart.** A fine-tuned run is
hours and the machine is not free for hours at a stretch, so every run that matters is
interrupted at least once. ``last.pt`` therefore carries the position *inside* the epoch,
the early-stopping counters and the best metrics so far, and a resume replays the same
shuffle and skips exactly the batches already trained on. What it does not carry is the
dropout RNG: a resumed run is a replicate of the uninterrupted one, not a bit-identical
copy of it — which CUDA never offered in the first place (``Memory.md`` §7 #50).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from vulnloc.training.metrics import MetricAccumulator
from vulnloc.utils.exceptions import ModelError
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vulnloc.training.losses import LineLoss

__all__ = [
    "PAUSED_EXIT_CODE",
    "STOP_FILE_ENV",
    "EarlyStopping",
    "TrainState",
    "Trainer",
    "TrainingPaused",
    "evaluate",
]

_LOGGER = get_logger(__name__)

STOP_FILE_ENV = "VULNLOC_STOP_FILE"
"""Environment variable naming a file whose existence asks a run to pause.

An environment variable rather than a config key on purpose: a key would be logged as a
parameter and change ``config_hash`` for every run of every branch (``Memory.md`` §7 #38),
and whether the laptop is needed at eight in the morning is not part of an experiment.
"""

PAUSED_EXIT_CODE = 75
"""Exit status of a run that paused on request (``EX_TEMPFAIL``): not done, not broken."""


class TrainingPaused(Exception):  # noqa: N818 - a request honoured, not an error
    """Raised once a pause request has been honoured and ``last.pt`` is on disk."""


@dataclass
class EarlyStopping:
    """Stops training when the watched metric stops improving.

    Args:
        patience: Evaluations without improvement before stopping.
        min_delta: Improvement below this does not count.
        mode: ``"max"`` for metrics where higher is better.
    """

    patience: int = 3
    min_delta: float = 1e-4
    mode: str = "max"
    best: float = field(init=False)
    waited: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.best = -math.inf if self.mode == "max" else math.inf

    def update(self, value: float) -> bool:
        """Record a new value.

        Args:
            value: The watched metric.

        Returns:
            ``True`` if this is the best so far.
        """
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.waited = 0
        else:
            self.waited += 1
        return improved

    @property
    def should_stop(self) -> bool:
        """Whether patience has run out."""
        return self.waited >= self.patience


@dataclass
class TrainState:
    """What a checkpoint restores."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = -math.inf
    # Everything below exists so a resume is a continuation. Each has a default, so a
    # checkpoint written before these fields existed still loads — and resumes the old
    # way, from the start of its epoch.
    batch_in_epoch: int = 0
    """Batches of ``epoch`` already trained on. A resume skips exactly this many."""
    epoch_totals: dict[str, float] = field(default_factory=dict)
    """Loss sums over those batches, so the epoch's logged train loss covers all of it."""
    epoch_seconds: float = 0.0
    waited: int = 0
    """Early stopping's count of evaluations without improvement."""
    best_metrics: dict[str, float] = field(default_factory=dict)
    """The best epoch's full metric set — what ``summary.json`` reports."""
    last_epoch_log: dict[str, Any] = field(default_factory=dict)
    """``{"step", "metrics"}`` of the last finished epoch. The checkpoint is written
    before the tracking store is, so a resume can log an epoch a kill left unlogged."""


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a collated batch's tensors to ``device``, leaving the rest alone."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion: LineLoss,
    device: torch.device,
    *,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> dict[str, float]:
    """Run one evaluation pass.

    ``model.eval()`` settles dropout and normalisation; it does **not** stop autograd from
    recording the forward pass. Without ``no_grad`` every activation needed for a backward
    that never happens is retained until the next batch replaces it — roughly doubling the
    peak memory of evaluation for nothing, since neither the metrics nor the reported loss
    read a gradient.

    That is not a tidiness point on a 4 GiB card — see ``Memory.md`` §1.17 for what it
    was measured to cost on the largest `val` unit of the fused model, and §7 #29 for why
    the cost arrives as a silent slowdown rather than as an error on Windows. Adding this
    changes no number, because no number here was ever computed from a gradient.

    Args:
        model: The model.
        loader: Batches to evaluate on.
        criterion: The loss, for reporting.
        device: Where to run.
        threshold: Decision threshold for the metrics.
        use_amp: Run under autocast.

    Returns:
        Metrics plus ``loss``.
    """
    model.eval()
    accumulator = MetricAccumulator(threshold=threshold)
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = _move(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(batch)
        line_logits = output["line_logits"].float()
        function_logits = output["function_logits"].float()
        scoreable = output["line_mask"] & batch["line_valid"]
        losses = criterion(
            line_logits,
            batch["line_labels"],
            scoreable,
            function_logits,
            batch["targets"],
        )
        total_loss += float(losses["loss"])
        n_batches += 1
        accumulator.update(
            line_logits,
            batch["line_labels"],
            output["line_mask"],
            batch["line_valid"],
            function_logits,
            batch["targets"],
        )

    results = accumulator.result()
    results["loss"] = total_loss / max(n_batches, 1)
    return results


class Trainer:
    """Trains a line-scoring model with the memory strategy §1.9 measured.

    Args:
        model: The model to train.
        criterion: The loss.
        optimizer: The optimizer.
        device: Where to run.
        checkpoint_dir: Where checkpoints are written.
        use_amp: Autocast forward passes. Ignored on CPU.
        accumulation_steps: Optimizer steps every N batches, for an effective batch
            larger than memory allows.
        max_grad_norm: Gradient clipping. ``0`` disables it.
        watch_metric: The metric early stopping and checkpoint selection follow.
        early_stopping: Optional early stopper.
        scheduler: Optional LR scheduler, stepped once per optimizer step.
        checkpoint_every_n_steps: Write a resumable checkpoint this often, in optimizer
            steps. ``0`` disables it and reverts to epoch boundaries only — which on this
            hardware means an interrupted run loses everything, because an epoch is
            longer than most interruptions (``Memory.md`` §7 #31).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: LineLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        checkpoint_dir: Path | str = "artifacts/checkpoints",
        use_amp: bool = True,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        watch_metric: str = "line/top_5_accuracy",
        early_stopping: EarlyStopping | None = None,
        scheduler: Any = None,
        checkpoint_every_n_steps: int = 0,
    ) -> None:
        if accumulation_steps < 1:
            raise ModelError("accumulation_steps must be >= 1", value=accumulation_steps)
        if checkpoint_every_n_steps < 0:
            raise ModelError(
                "checkpoint_every_n_steps must be >= 0", value=checkpoint_every_n_steps
            )
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_amp = use_amp and device.type == "cuda"
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.watch_metric = watch_metric
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)
        self.state = TrainState()
        self.provenance: dict[str, Any] = {}
        """Written into every rolling checkpoint. The caller puts the run id here."""
        self._memory_checked = False
        self._cache_releases = 0
        stop_file = os.environ.get(STOP_FILE_ENV, "")
        self._stop_file = Path(stop_file) if stop_file else None

    def _check_memory(self) -> None:
        """Warn if the step only fits because the driver paged to host RAM.

        ``Memory.md`` §7 #29. Runs once, after the first real step, because that is when
        peak allocation is representative and there is still time to act on it.
        """
        if self._memory_checked or self.device.type != "cuda":
            return
        self._memory_checked = True
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**2
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
        if peak > total:
            _LOGGER.warning(
                "peak allocation exceeds physical VRAM; the driver is paging to host "
                "memory and this run will be 10-25x slower than it looks",
                extra={"peak_mib": round(peak, 1), "total_mib": round(total, 1)},
            )
        else:
            _LOGGER.info(
                "Memory headroom measured",
                extra={
                    "peak_mib": round(peak, 1),
                    "total_mib": round(total, 1),
                    "used_pct": round(100 * peak / total, 1),
                },
            )

    def _release_cache_if_spilling(self) -> None:
        """Hand cached blocks back to the driver once PyTorch holds more than the card has.

        ``_check_memory`` looks once, at the first step, at what is *allocated*. What pages
        to host RAM on Windows is what is *reserved* — allocated plus the allocator's cache
        — and that grows later: after the one oversized unit an epoch contains
        (``Memory.md`` §1.17), or after a checkpoint is loaded. Releasing the cache costs
        a few re-allocations and changes no number; not releasing it costs every later
        step, silently.
        """
        if self.device.type != "cuda":
            return
        total = torch.cuda.get_device_properties(self.device).total_memory
        reserved = torch.cuda.memory_reserved(self.device)
        if reserved <= total:
            return
        torch.cuda.empty_cache()
        self._cache_releases += 1
        if self._cache_releases == 1 or self._cache_releases % 50 == 0:
            _LOGGER.warning(
                "Reserved VRAM exceeded the card; cache released",
                extra={
                    "reserved_mib": round(reserved / 1024**2, 1),
                    "total_mib": round(total / 1024**2, 1),
                    "releases_so_far": self._cache_releases,
                    "global_step": self.state.global_step,
                },
            )

    def _maybe_checkpoint_step(self) -> None:
        """Write ``last.pt`` if enough optimizer steps have passed.

        Deliberately a *single rolling file* rather than one per interval: the point is to
        make an interruption cheap, not to keep a history, and 500 MB per snapshot would
        fill the disk long before it became useful. ``best.pt`` still holds the best
        model by the watched metric; this one only answers "where was I".
        """
        if not self.checkpoint_every_n_steps:
            return
        if self.state.global_step % self.checkpoint_every_n_steps:
            return
        self.save_last("periodic")

    def save_last(self, kind: str) -> None:
        """Write the rolling ``last.pt`` — the one file a resume reads.

        Args:
            kind: Why it was written: ``periodic``, ``epoch`` or ``paused``.
        """
        self.save_checkpoint(
            self.checkpoint_dir / "last.pt",
            {},
            **self.provenance,
            kind=kind,
            global_step=self.state.global_step,
            epoch=self.state.epoch,
        )

    @property
    def pause_requested(self) -> bool:
        """Whether the stop file named by :data:`STOP_FILE_ENV` exists."""
        return self._stop_file is not None and self._stop_file.exists()

    def _maybe_pause(self) -> None:
        """Honour a pause request at an optimizer-step boundary.

        Only here, where no gradient is half-accumulated: the checkpoint written is then
        a state the uninterrupted run also passed through.

        Raises:
            TrainingPaused: After ``last.pt`` is written.
        """
        if not self.pause_requested:
            return
        self.save_last("paused")
        raise TrainingPaused(
            f"paused at epoch {self.state.epoch}, batch {self.state.batch_in_epoch}, "
            f"step {self.state.global_step}"
        )

    def train_epoch(self, loader: DataLoader[Any], *, resume: bool = False) -> dict[str, float]:
        """Run one training epoch, or the rest of an interrupted one.

        Args:
            loader: Training batches, in the order the interrupted epoch saw them. The
                batch samplers shuffle from ``seed + epoch`` alone, which is what makes
                that order recoverable.
            resume: Continue from ``state.batch_in_epoch`` instead of starting the epoch.
                The batches before it are drawn and discarded, never trained on twice,
                and the loss totals pick up from the checkpoint so the returned averages
                still cover the whole epoch.

        Returns:
            ``{"loss": ..., "line_loss": ..., "function_loss": ..., "seconds": ...}``.

        Raises:
            TrainingPaused: If a pause was requested; ``last.pt`` is written first.
        """
        self.model.train()
        totals = {"loss": 0.0, "line_loss": 0.0, "function_loss": 0.0}
        skip = 0
        seconds_before = 0.0
        if resume:
            skip = self.state.batch_in_epoch
            totals.update(self.state.epoch_totals)
            seconds_before = self.state.epoch_seconds
        n_batches = skip
        started = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)
        for index, raw in enumerate(loader):
            if index < skip:
                continue
            batch = _move(raw, self.device)
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                output = self.model(batch)
                scoreable = output["line_mask"] & batch["line_valid"]
                losses = self.criterion(
                    output["line_logits"].float(),
                    batch["line_labels"],
                    scoreable,
                    output["function_logits"].float(),
                    batch["targets"],
                )
                loss = losses["loss"] / self.accumulation_steps

            self.scaler.scale(loss).backward()

            # Before the step, not after it: a checkpoint written by the step below has
            # to count this batch as done, or a resume would train on it a second time.
            for key in totals:
                totals[key] += float(losses[key].detach())
            n_batches += 1
            self.state.batch_in_epoch = n_batches
            self.state.epoch_totals = dict(totals)
            self.state.epoch_seconds = seconds_before + time.perf_counter() - started

            if (index + 1) % self.accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.state.global_step += 1
                self._check_memory()
                self._release_cache_if_spilling()
                self._maybe_checkpoint_step()
                self._maybe_pause()

        return {
            **{key: value / max(n_batches, 1) for key, value in totals.items()},
            "seconds": round(seconds_before + time.perf_counter() - started, 2),
        }

    def save_checkpoint(self, path: Path, metrics: Mapping[str, float], **provenance: Any) -> None:
        """Write a checkpoint with the provenance needed to trace it.

        Args:
            path: Destination file.
            metrics: The epoch's metrics.
            **provenance: Run id, git sha, config hash — whatever the caller can supply.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and renamed over it. A fine-tuned checkpoint is over a
        # gigabyte and takes seconds to write; a process killed during those seconds would
        # otherwise leave a truncated last.pt — the one file an interruption needs.
        partial = path.with_name(path.name + ".tmp")
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "state": vars(self.state),
                "metrics": dict(metrics),
                "provenance": provenance,
            },
            partial,
        )
        partial.replace(path)
        _LOGGER.info("Checkpoint written", extra={"path": str(path), **dict(provenance)})

    def load_checkpoint(self, path: Path, *, weights_only: bool = False) -> dict[str, Any]:
        """Restore from a checkpoint.

        Args:
            path: The checkpoint file.
            weights_only: Restore only the model weights, leaving optimizer and step
                counters alone. What evaluation wants; resuming wants the opposite.

        Returns:
            The loaded payload.

        Raises:
            ModelError: If the file does not exist.
        """
        if not path.exists():
            raise ModelError("checkpoint not found", path=str(path))
        # To host memory, never straight to the card. MEASURED 2026-10-02 on run
        # `cb2b1e99…`: loading a fine-tuned checkpoint onto the GPU put a second copy of
        # the weights and the optimizer state beside the live ones, PyTorch kept the freed
        # blocks cached, and the resumed process sat at 3,692 MiB dedicated plus 1,872 MiB
        # *shared* — paging, 5.0 s/step against 2.0 before the pause, with nothing raised
        # (Memory.md §7 #24). `load_state_dict` copies onto each parameter's own device.
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        if not weights_only:
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.scaler.load_state_dict(payload["scaler_state"])
            self.state = TrainState(**payload["state"])
        return payload
