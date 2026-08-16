"""Train the LLM-only baseline — Phase 3's deliverable.

Produces the first number this project is allowed to report, and one arm of experiment A1
(cross-attention vs. concatenation). Everything is composed from Hydra config, so an
ablation is a command line and never an edit to this file (``Rules.md``: config over code).

The **overfit test** is not a nicety, it is a Phase 3 exit criterion: the model must reach
~100% on a 32-sample subset. A model that cannot memorise 32 functions has a broken head,
a broken loss, or a broken line alignment, and finding that out after a six-hour training
run is the expensive way to learn it. Run it first::

    python scripts/train_semantic.py train=overfit

Full training (``train=semantic`` is required — the root default is Phase 0's no-op stub,
which ``scripts/train.py`` still depends on)::

    python scripts/train_semantic.py train=semantic
    python scripts/train_semantic.py train=semantic llm.frozen=true   # frozen variant
    python scripts/train_semantic.py train=semantic seed=1            # one seed per run
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import mlflow
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vulnloc.models.heads.line_head import LineHead
from vulnloc.models.semantic.baseline import SemanticLineModel
from vulnloc.models.semantic.encoder import PretrainedCodeEncoder
from vulnloc.representation.tokenization import LineTokenizer
from vulnloc.training.datasets import LineLabelledDataset, WindowBatchSampler, collate_units
from vulnloc.training.device import resolve_device
from vulnloc.training.loop import EarlyStopping, Trainer, evaluate
from vulnloc.training.losses import LineLoss
from vulnloc.training.seeding import seed_everything
from vulnloc.training.tracking import start_run
from vulnloc.utils.config import git_sha
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)


def build_loader(
    dataset: LineLabelledDataset, cfg: DictConfig, *, shuffle: bool, seed: int
) -> DataLoader[Any]:
    """Wrap a dataset in a window-budgeted loader.

    Args:
        dataset: The dataset.
        cfg: Composed config.
        shuffle: Shuffle unit order.
        seed: Base seed for the sampler.

    Returns:
        A loader yielding collated window batches.
    """
    sampler = WindowBatchSampler(
        dataset,
        max_windows=int(cfg.train.max_windows_per_batch),
        shuffle=shuffle,
        seed=seed,
        max_units=int(cfg.train.max_units_per_batch),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_units,
        num_workers=0,  # The dataset is already in memory; workers would only re-pickle it.
    )


def build_model(cfg: DictConfig, device: torch.device) -> SemanticLineModel:
    """Compose encoder + head from config.

    Args:
        cfg: Composed config.
        device: Where the model lives.

    Returns:
        The model, on ``device``.
    """
    encoder = PretrainedCodeEncoder(
        str(cfg.llm.model_name),
        frozen=bool(cfg.llm.frozen),
        gradient_checkpointing=bool(cfg.llm.gradient_checkpointing),
        dropout=float(cfg.llm.dropout),
        pretrained=bool(cfg.llm.pretrained),
    )
    head = LineHead(
        encoder.hidden_size,
        hidden_dim=int(cfg.head.hidden_dim),
        dropout=float(cfg.head.dropout),
    )
    return SemanticLineModel(encoder, head, n_lines=int(cfg.llm.n_lines)).to(device)


def train(cfg: DictConfig) -> dict[str, Any]:
    """Run one training job.

    Args:
        cfg: Composed Hydra config.

    Returns:
        A summary with the best metrics and the checkpoint path.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    transformers.logging.set_verbosity_error()
    seed = int(cfg.seed)
    seed_everything(seed)
    device = resolve_device(str(cfg.device.kind), index=int(cfg.device.index))

    tokenizer = LineTokenizer(
        AutoTokenizer.from_pretrained(str(cfg.llm.tokenizer_name)),
        max_length=int(cfg.llm.max_length),
        stride=int(cfg.llm.stride),
        windowing=bool(cfg.llm.windowing),
    )
    cache_dir = Path(cfg.data.cache.output_dir)
    n_lines = int(cfg.llm.n_lines)

    train_set = LineLabelledDataset(
        cache_dir,
        str(cfg.train.train_split),
        tokenizer,
        n_lines=n_lines,
        limit=int(cfg.train.limit),
        line_labelled_only=bool(cfg.train.line_labelled_only),
    )
    eval_split = str(cfg.train.eval_split)
    eval_set = (
        train_set
        if cfg.train.overfit
        else LineLabelledDataset(
            cache_dir,
            eval_split,
            tokenizer,
            n_lines=n_lines,
            limit=int(cfg.train.eval_limit),
            line_labelled_only=bool(cfg.train.line_labelled_only),
        )
    )

    model = build_model(cfg, device)
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
                "model_name": str(cfg.llm.model_name),
                "frozen": bool(cfg.llm.frozen),
                "windowing": bool(cfg.llm.windowing),
                "loss": str(cfg.train.loss),
                "pos_weight": round(criterion.pos_weight, 3),
                "max_windows_per_batch": int(cfg.train.max_windows_per_batch),
                "train_units": len(train_set),
                "eval_units": len(eval_set),
                "flagged_beyond_line_cap": train_set.flagged_beyond_cap,
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
        )

        train_loader = build_loader(train_set, cfg, shuffle=True, seed=seed)
        eval_loader = build_loader(eval_set, cfg, shuffle=False, seed=seed)
        best: dict[str, float] = {}
        best_path = checkpoint_dir / "best.pt"

        for epoch in range(int(cfg.train.epochs)):
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    train(cfg)


if __name__ == "__main__":
    main()
