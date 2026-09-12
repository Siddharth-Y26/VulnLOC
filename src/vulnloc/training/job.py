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

``scripts/train_semantic.py`` does **not** call this yet, and the duplication is
deliberate rather than overlooked: Phase 3's seed sweep is unfinished (seed 42 logged,
seeds 1 and 2 not), and rewriting the path that produced the logged seed — however
carefully — would put "was it the refactor?" between those seeds and their comparison.

**TODO(decision):** fold ``scripts/train_semantic.py`` onto this function once Phase 3's
three seeds are logged. Until then, a change made here must be made there too, and the
tests that pin the loop's behaviour live in ``tests/unit/test_training_pipeline.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from vulnloc.training.loop import EarlyStopping, Trainer, evaluate
from vulnloc.training.losses import LineLoss
from vulnloc.training.tracking import start_run
from vulnloc.utils.config import git_sha
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch.utils.data import DataLoader

__all__ = ["LineDataset", "run_training"]

_LOGGER = get_logger(__name__)


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
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    with start_run(cfg, run_name=str(cfg.train.name)) as run_id:
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
                **dict(extra_params or {}),
            }
        )
        checkpoint_dir = Path(cfg.train.checkpoint_dir) / run_id
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
            early_stopping=EarlyStopping(
                patience=int(cfg.train.patience), mode=str(cfg.train.watch_mode)
            ),
            checkpoint_every_n_steps=int(cfg.train.checkpoint_every_n_steps),
        )

        start_epoch = 0
        if str(cfg.train.resume_from):
            payload = trainer.load_checkpoint(Path(str(cfg.train.resume_from)))
            # The interrupted epoch is redone from its start: the sampler's position
            # within it is not part of the checkpoint, and silently skipping the units it
            # had already seen would be worse than seeing a few of them twice.
            start_epoch = trainer.state.epoch
            _LOGGER.info(
                "Resumed from checkpoint",
                extra={
                    "path": str(cfg.train.resume_from),
                    "resumed_epoch": start_epoch,
                    "global_step": trainer.state.global_step,
                    "from_run": payload.get("provenance", {}).get("run_id", "?"),
                },
            )

        train_loader = make_loader(train_set, True)
        eval_loader = make_loader(eval_set, False)
        best: dict[str, float] = {}
        best_path = checkpoint_dir / "best.pt"

        for epoch in range(start_epoch, int(cfg.train.epochs)):
            trainer.state.epoch = epoch
            train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
            train_metrics = trainer.train_epoch(train_loader)
            eval_metrics = evaluate(
                model,
                eval_loader,
                criterion,
                device,
                threshold=float(cfg.train.threshold),
                use_amp=trainer.use_amp,
            )

            mlflow.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, step=epoch)
            mlflow.log_metrics({f"eval/{k}": v for k, v in eval_metrics.items()}, step=epoch)
            _LOGGER.info(
                "Epoch complete",
                extra={
                    "epoch": epoch,
                    "train_loss": round(train_metrics["loss"], 5),
                    "eval_loss": round(eval_metrics["loss"], 5),
                    "watched": round(eval_metrics.get(trainer.watch_metric, 0.0), 4),
                    "seconds": train_metrics["seconds"],
                },
            )

            watched = eval_metrics.get(trainer.watch_metric, 0.0)
            assert trainer.early_stopping is not None
            if trainer.early_stopping.update(watched):
                best = dict(eval_metrics)
                trainer.state.best_metric = watched
                trainer.save_checkpoint(
                    best_path,
                    eval_metrics,
                    run_id=run_id,
                    git_sha=git_sha(),
                    epoch=epoch,
                    seed=seed,
                )
            if trainer.early_stopping.should_stop:
                _LOGGER.info("Early stopping", extra={"epoch": epoch})
                break

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
