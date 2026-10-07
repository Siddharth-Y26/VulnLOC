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

from pathlib import Path
from typing import Any

import hydra
import torch
import transformers
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vulnloc.models.heads.line_head import LineHead
from vulnloc.models.semantic.baseline import SemanticLineModel
from vulnloc.models.semantic.encoder import PretrainedCodeEncoder
from vulnloc.representation.tokenization import LineTokenizer
from vulnloc.training.datasets import LineLabelledDataset, WindowBatchSampler, collate_units
from vulnloc.training.device import resolve_device
from vulnloc.training.job import run_training
from vulnloc.training.seeding import seed_everything
from vulnloc.utils.config import strip_comments_enabled
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
        normalize_whitespace=bool(cfg.representation.normalize_whitespace),
        strip_comments=strip_comments_enabled(cfg),
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
            normalize_whitespace=bool(cfg.representation.normalize_whitespace),
            strip_comments=strip_comments_enabled(cfg),
        )
    )

    model = build_model(cfg, device)

    return run_training(
        cfg,
        model=model,
        train_set=train_set,
        eval_set=eval_set,
        make_loader=lambda dataset, shuffle: build_loader(dataset, cfg, shuffle=shuffle, seed=seed),
        device=device,
        # Exactly the semantic-specific parameters this script logged before it moved onto
        # the shared loop, and no others. `run_training` already logs seed, loss,
        # pos_weight, the unit counts, flagged_beyond_line_cap, normalize_whitespace and
        # resolved_device. Adding anything new here would make a fourth seed incomparable
        # with the three that closed Phase 3 (Memory.md §7 #39).
        extra_params={
            "model_name": str(cfg.llm.model_name),
            "frozen": bool(cfg.llm.frozen),
            "windowing": bool(cfg.llm.windowing),
            "max_windows_per_batch": int(cfg.train.max_windows_per_batch),
        },
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    train(cfg)


if __name__ == "__main__":
    main()
