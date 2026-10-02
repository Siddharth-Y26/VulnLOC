"""The branch-agnostic half of a training run (L3).

Everything from the loss down — optimizer, MLflow run, resume, the epoch loop, early
stopping, checkpointing, the summary artefact — is identical whether the encoder reads
tokens or graphs. Only four things differ between the two branches: which dataset is
built, which model is composed, how a batch is assembled, and a handful of logged
parameters. Those four are arguments here; the rest lives once.

**Why that matters more than tidiness.** Phase 4's GNN-only baseline is the second arm of
experiment A1, and A1 is the thing this project exists to answer. If the two arms ran
through two separately-maintained loops, a difference between them could be early
stopping, or the resume path, or which epoch got checkpointed — and the write-up would
still call it architecture. One loop makes that class of explanation unavailable.

Both entrypoints call this. ``scripts/train_semantic.py`` deliberately did not, while
Phase 3's seed sweep was in flight: rewriting the path that produced a logged seed —
however carefully — would have put "was it the refactor?" between those seeds and their
comparison. All three landed on 2026-09-14 (``Memory.md`` §1.14), and the duplication was
removed the same day.

One rule survives that move and is easy to break by accident: **do not add a logged
parameter here or in either caller without checking who becomes incomparable.** A fourth
semantic seed has to match the three that closed Phase 3, and the comparability guard in
``scripts/report_seeds.py`` compares the whole parameter set (§7 #39).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

import mlflow
import torch
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from vulnloc.training.loop import (
    PAUSED_EXIT_CODE,
    EarlyStopping,
    Trainer,
    TrainingPaused,
    TrainState,
    evaluate,
)
from vulnloc.training.losses import LineLoss
from vulnloc.training.tracking import start_run
from vulnloc.utils.config import git_sha
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch.utils.data import DataLoader

__all__ = ["LineDataset", "run_training"]

_LOGGER = get_logger(__name__)


def _resume(trainer: Trainer, last_path: Path, best_path: Path) -> dict[str, Any]:
    """Restore the trainer from wherever the interrupted run got furthest.

    That is ``last.pt`` in every case but one. At an epoch boundary ``best.pt`` is written
    before ``last.pt``, and both carry the same post-epoch state — so a run killed in the
    seconds between the two leaves a ``best.pt`` that is *ahead* of ``last.pt``. Resuming
    from ``last.pt`` then would replay the end of that epoch, evaluate it a second time,
    and could keep a ``best.pt`` the summary no longer describes. The file's timestamp
    says which case this is without reading a gigabyte to find out.

    Args:
        trainer: The trainer to restore.
        last_path: The rolling checkpoint named by ``train.resume_from``.
        best_path: The same run's best checkpoint, which may not exist.

    Returns:
        The payload that was loaded.
    """
    payload = trainer.load_checkpoint(last_path)
    if best_path.exists() and best_path.stat().st_mtime > last_path.stat().st_mtime:
        candidate = torch.load(best_path, map_location="cpu", weights_only=False)
        ahead = TrainState(**candidate["state"])
        here = trainer.state
        if (ahead.epoch, ahead.batch_in_epoch) > (here.epoch, here.batch_in_epoch):
            _LOGGER.warning(
                "best.pt is ahead of last.pt; resuming from best.pt",
                extra={"best_epoch": ahead.epoch, "last_epoch": here.epoch},
            )
            del candidate
            payload = trainer.load_checkpoint(best_path)
    return payload


def _pause(reason: str) -> NoReturn:
    """Leave the run for the next process to continue, and exit.

    ``last.pt`` is already on disk by the time this is called.

    Args:
        reason: Where the run stopped.

    Raises:
        SystemExit: Always, with :data:`PAUSED_EXIT_CODE` — which is how the queue tells
            a run that stopped because it was asked to from one that broke.
    """
    mlflow.set_tag("paused_at", reason)
    _LOGGER.info("Training paused on request", extra={"at": reason})
    raise SystemExit(PAUSED_EXIT_CODE)


def _log_epoch_if_missing(run_id: str, epoch_log: Mapping[str, Any]) -> None:
    """Log a finished epoch the interrupted process did not get to log.

    The checkpoint is written before the tracking store, so a kill between the two leaves
    an epoch that was trained, evaluated and checkpointed but is absent from the learning
    curve — and if it was the best epoch, ``summary.json`` and MLflow would then disagree
    about which one that was.

    Args:
        run_id: The run being resumed.
        epoch_log: ``TrainState.last_epoch_log``; empty before the first epoch ends.
    """
    if not epoch_log:
        return
    step = int(epoch_log["step"])
    metrics: dict[str, float] = dict(epoch_log["metrics"])
    history = MlflowClient().get_metric_history(run_id, "eval/loss")
    if any(point.step == step for point in history):
        return
    mlflow.log_metrics(metrics, step=step)
    _LOGGER.warning("Logged an epoch the interrupted run had not", extra={"epoch": step})


class LineDataset(Protocol):
    """What the loop needs of a dataset, regardless of what it encodes.

    Both :class:`~vulnloc.training.datasets.LineLabelledDataset` and
    :class:`~vulnloc.training.graph_datasets.GraphLineDataset` satisfy this. Stated as a
    protocol rather than a base class so neither branch has to inherit from the other.
    """

    flagged_beyond_cap: int

    def __len__(self) -> int:
        """Number of units."""
        ...

    def positive_line_weight(self) -> float:
        """``negatives / positives`` over scoreable lines."""
        ...


def run_training(
    cfg: DictConfig,
    *,
    model: torch.nn.Module,
    train_set: LineDataset,
    eval_set: LineDataset,
    make_loader: Callable[[LineDataset, bool], DataLoader[Any]],
    device: torch.device,
    extra_params: Mapping[str, Any] | None = None,
    param_groups: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train ``model`` under ``cfg`` and return the run summary.

    Args:
        cfg: Composed Hydra config. Reads ``train.*``, ``seed`` and
            ``representation.normalize_whitespace``.
        model: Already on ``device``. Its forward must return ``line_logits``,
            ``function_logits`` and ``line_mask``.
        train_set: Training units.
        eval_set: Evaluation units. The same object as ``train_set`` under
            ``train.overfit``, which is the point of that config.
        make_loader: ``(dataset, shuffle) -> DataLoader``. The branch supplies its own
            sampler and collation here — windows on one side, nodes on the other — and
            nothing else in this function knows the difference.
        device: Where the model lives.
        extra_params: Branch-specific values to log beside the shared ones. Logged, not
            merged into the config, so a run's parameters say what produced it.
        param_groups: Optimizer parameter groups, when one learning rate is not enough.
            ``None`` — the default, and what both single-branch entrypoints pass — keeps
            the previous behaviour exactly: every trainable parameter at ``cfg.train.lr``.
            Phase 5 needs the split because a fused model holds a 124M-parameter
            pretrained transformer that wants 2e-5 next to a from-scratch GNN that wants
            1e-3, and a single rate has to be wrong for one of them. Groups without their
            own ``lr`` fall back to ``cfg.train.lr``.

    Returns:
        ``{"run_id", "seed", "best", "checkpoint", "config"}``, also written to
        ``<checkpoint_dir>/<run_id>/summary.json`` and logged as an artefact.
    """
    seed = int(cfg.seed)
    criterion = LineLoss(
        kind=str(cfg.train.loss),
        pos_weight=(
            train_set.positive_line_weight()
            if bool(cfg.train.auto_pos_weight)
            else float(cfg.train.pos_weight)
        ),
        alpha=float(cfg.train.focal_alpha),
        gamma=float(cfg.train.focal_gamma),
        function_weight=float(cfg.train.function_weight),
    )
    trainable: Any = (
        [dict(group) for group in param_groups]
        if param_groups is not None
        else [p for p in model.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    resume_from = str(cfg.train.resume_from)
    # The checkpoint directory is named after the run, so the path says which run to
    # continue before the gigabyte behind it is read.
    resume_run_id = Path(resume_from).parent.name if resume_from else None

    with start_run(cfg, run_name=str(cfg.train.name), resume_run_id=resume_run_id) as run_id:
        # A reopened run already carries its parameters; they are not logged a second time.
        if not resume_run_id:
            mlflow.log_params(
                {
                    "seed": seed,
                    "loss": str(cfg.train.loss),
                    "pos_weight": round(criterion.pos_weight, 3),
                    "train_units": len(train_set),
                    "eval_units": len(eval_set),
                    "flagged_beyond_line_cap": train_set.flagged_beyond_cap,
                    # Logged as a parameter because it is the difference between a real
                    # number and the ADR-006 leak; a run without it is not comparable.
                    "normalize_whitespace": bool(cfg.representation.normalize_whitespace),
                    # The device actually used, not the one requested. `auto` and `cuda`
                    # resolve to the same card, so comparing the request refuses
                    # replicates that differ in nothing; comparing this compares what
                    # happened.
                    "resolved_device": str(device),
                    **dict(extra_params or {}),
                }
            )
        checkpoint_dir = Path(cfg.train.checkpoint_dir) / run_id
        early_stopping = EarlyStopping(
            patience=int(cfg.train.patience), mode=str(cfg.train.watch_mode)
        )
        trainer = Trainer(
            model,
            criterion,
            optimizer,
            device,
            checkpoint_dir=checkpoint_dir,
            use_amp=bool(cfg.train.amp),
            accumulation_steps=int(cfg.train.accumulation_steps),
            max_grad_norm=float(cfg.train.max_grad_norm),
            watch_metric=str(cfg.train.watch_metric),
            early_stopping=early_stopping,
            checkpoint_every_n_steps=int(cfg.train.checkpoint_every_n_steps),
        )
        trainer.provenance = {"run_id": run_id, "git_sha": git_sha(), "seed": seed}
        # A rolling checkpoint per epoch is a gigabyte a time for a fine-tuned model. For
        # a 150-epoch overfit run of minutes that is all cost: a restart is cheaper.
        rolling = bool(trainer.checkpoint_every_n_steps) and not bool(cfg.train.overfit)

        best: dict[str, float] = {}
        best_path = checkpoint_dir / "best.pt"
        epochs = int(cfg.train.epochs)
        start_epoch = 0
        mid_epoch = False
        if resume_from:
            _resume(trainer, Path(resume_from), best_path)
            state = trainer.state
            start_epoch = state.epoch
            mid_epoch = state.batch_in_epoch > 0
            if state.best_metrics:
                # Without these a resumed run forgets what it had already achieved: its
                # first evaluation becomes "the best" whatever it scores, and patience
                # starts over.
                early_stopping.best = state.best_metric
                early_stopping.waited = state.waited
                best = dict(state.best_metrics)
            _log_epoch_if_missing(run_id, state.last_epoch_log)
            resumed_at = (
                f"epoch={start_epoch},batch={state.batch_in_epoch},step={state.global_step}"
            )
            earlier = mlflow.get_run(run_id).data.tags.get("resumed_at", "")
            mlflow.set_tag("resumed_at", f"{earlier};{resumed_at}" if earlier else resumed_at)
            _LOGGER.info(
                "Resumed from checkpoint",
                extra={
                    "path": resume_from,
                    "resumed_epoch": start_epoch,
                    "batches_already_trained": state.batch_in_epoch,
                    "global_step": state.global_step,
                    "best_so_far": state.best_metric if best else None,
                },
            )

        train_loader = make_loader(train_set, True)
        eval_loader = make_loader(eval_set, False)

        for epoch in range(start_epoch, epochs):
            # True only when the run was interrupted between an early stop and its
            # summary; the loop below breaks before it could come round again.
            if early_stopping.should_stop:
                break
            trainer.state.epoch = epoch
            train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
            try:
                train_metrics = trainer.train_epoch(
                    train_loader, resume=mid_epoch and epoch == start_epoch
                )
            except TrainingPaused as paused:
                _pause(str(paused))
            eval_metrics = evaluate(
                model,
                eval_loader,
                criterion,
                device,
                threshold=float(cfg.train.threshold),
                use_amp=trainer.use_amp,
            )
            epoch_log = {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"eval/{k}": v for k, v in eval_metrics.items()},
            }

            # State first, files second, tracking store last. Every checkpoint written
            # from here on describes the run as it stands *after* this epoch, so a
            # resume from either file starts the next epoch and repeats nothing — and
            # logs this epoch itself if the process died before the last line did.
            watched = eval_metrics.get(trainer.watch_metric, 0.0)
            improved = early_stopping.update(watched)
            state = trainer.state
            state.epoch = epoch + 1
            state.batch_in_epoch = 0
            state.epoch_totals = {}
            state.epoch_seconds = 0.0
            state.waited = early_stopping.waited
            state.last_epoch_log = {"step": epoch, "metrics": epoch_log}
            if improved:
                best = dict(eval_metrics)
                state.best_metric = watched
                state.best_metrics = dict(best)
                trainer.save_checkpoint(
                    best_path,
                    eval_metrics,
                    run_id=run_id,
                    git_sha=git_sha(),
                    epoch=epoch,
                    seed=seed,
                )
            if rolling:
                trainer.save_last("epoch")
            mlflow.log_metrics(epoch_log, step=epoch)
            _LOGGER.info(
                "Epoch complete",
                extra={
                    "epoch": epoch,
                    "train_loss": round(train_metrics["loss"], 5),
                    "eval_loss": round(eval_metrics["loss"], 5),
                    "watched": round(watched, 4),
                    "seconds": train_metrics["seconds"],
                },
            )

            if early_stopping.should_stop:
                _LOGGER.info("Early stopping", extra={"epoch": epoch})
                break
            if trainer.pause_requested and epoch + 1 < epochs:
                if not rolling:
                    trainer.save_last("paused")
                _pause(f"epoch {epoch + 1}, batch 0, step {trainer.state.global_step}")

        summary = {
            "run_id": run_id,
            "seed": seed,
            "best": best,
            "checkpoint": str(best_path) if best else None,
            "config": OmegaConf.to_container(cfg.train, resolve=True),
        }
        summary_path = checkpoint_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(summary_path), artifact_path="training")

    _LOGGER.info("Training finished", extra={"run_id": run_id, "checkpoint": str(best_path)})
    return summary
