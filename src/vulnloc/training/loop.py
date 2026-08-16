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
"""

from __future__ import annotations

import math
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

__all__ = ["EarlyStopping", "TrainState", "Trainer", "evaluate"]

_LOGGER = get_logger(__name__)


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


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a collated batch's tensors to ``device``, leaving the rest alone."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


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
    ) -> None:
        if accumulation_steps < 1:
            raise ModelError("accumulation_steps must be >= 1", value=accumulation_steps)
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
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)
        self.state = TrainState()
        self._memory_checked = False

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

    def train_epoch(self, loader: DataLoader[Any]) -> dict[str, float]:
        """Run one training epoch.

        Args:
            loader: Training batches.

        Returns:
            ``{"loss": ..., "line_loss": ..., "function_loss": ..., "seconds": ...}``.
        """
        self.model.train()
        totals = {"loss": 0.0, "line_loss": 0.0, "function_loss": 0.0}
        n_batches = 0
        started = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)
        for index, raw in enumerate(loader):
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

            for key in totals:
                totals[key] += float(losses[key].detach())
            n_batches += 1

        return {
            **{key: value / max(n_batches, 1) for key, value in totals.items()},
            "seconds": round(time.perf_counter() - started, 2),
        }

    def save_checkpoint(self, path: Path, metrics: Mapping[str, float], **provenance: Any) -> None:
        """Write a checkpoint with the provenance needed to trace it.

        Args:
            path: Destination file.
            metrics: The epoch's metrics.
            **provenance: Run id, git sha, config hash — whatever the caller can supply.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "state": vars(self.state),
                "metrics": dict(metrics),
                "provenance": provenance,
            },
            path,
        )
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
        payload: dict[str, Any] = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        if not weights_only:
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.scaler.load_state_dict(payload["scaler_state"])
            self.state = TrainState(**payload["state"])
        return payload
