"""Train the GNN-only baseline — Phase 4's result, and A1's second arm.

The structural mirror of ``scripts/train_semantic.py``. Shards → graphs → node-budget
batches → ``StructuralEncoder`` → the shared line axis → the shared ``LineHead``. Every
step after the encoder is the same object the semantic baseline uses, which is what makes
the two comparable at all.

Run the overfit sanity check **first**. It is 32 graphs and a few minutes, and it proves
the message passing carries gradient and the node→line pooling points where it claims to.
Finding out otherwise after a multi-hour sweep is the expensive way to learn it::

    python scripts/train_structural.py +gnn=gcn train=overfit_structural
    python scripts/train_structural.py +gnn=gcn train=structural seed=42
    python scripts/train_structural.py +gnn=gat train=structural seed=42
    python scripts/train_structural.py +gnn=ggnn train=structural gnn.depth=6 seed=42

``+gnn=`` takes a ``+`` because the group is deliberately absent from ``config.yaml``'s
defaults while Phase 3's seed sweep is unfinished — adding it would change every semantic
run's ``config_hash`` and make the logged seed incomparable with the ones still to run.
See ``configs/gnn/README.md``.

**Three things must travel with any number this produces.** The ~30% line-label noise
(``Memory.md`` §1.6) and the 512-line cap (§7 #26), which the semantic branch shares; and
**7.84% of flagged lines carry no graph node at all** (§7 #17), which is a ceiling on this
branch specifically. The dataset logs the last one per split, measured rather than quoted.

And one more, stated because it is easy to forget: node features are **node-type
embeddings only**. ADR-002 §D2.5's subtoken half is not built, so this model cannot tell
two identifiers apart. A4 must not be read as measuring the full D2.5 feature set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from vulnloc.models.heads.line_head import LineHead
from vulnloc.models.structural import build_structural_encoder
from vulnloc.models.structural.baseline import StructuralLineModel
from vulnloc.training.device import resolve_device
from vulnloc.training.graph_datasets import (
    GraphLineDataset,
    NodeBudgetBatchSampler,
    collate_graphs,
    load_node_vocab,
)
from vulnloc.training.job import run_training
from vulnloc.training.seeding import seed_everything
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)


def build_loader(dataset: GraphLineDataset, cfg: DictConfig, *, shuffle: bool, seed: int) -> Any:
    """Wrap a dataset in a node-budgeted loader.

    Args:
        dataset: The dataset.
        cfg: Composed config.
        shuffle: Shuffle unit order.
        seed: Base seed for the sampler.

    Returns:
        A loader yielding collated graph batches.
    """
    sampler = NodeBudgetBatchSampler(
        dataset,
        max_nodes=int(cfg.train.max_nodes_per_batch),
        shuffle=shuffle,
        seed=seed,
        max_units=int(cfg.train.max_units_per_batch),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_graphs,
        num_workers=0,  # The dataset is already in memory; workers would only re-pickle it.
    )


def build_model(cfg: DictConfig, vocab_size: int, device: torch.device) -> StructuralLineModel:
    """Compose encoder + head from config.

    Args:
        cfg: Composed config.
        vocab_size: Node-type slots, from the vocabulary artefact rather than from the
            config — a size written down twice is a size that goes stale once.
        device: Where the model lives.

    Returns:
        The model, on ``device``.
    """
    # An overfit test is trying to defeat regularisation, so `train.zero_dropout` wins over
    # whichever `gnn`/`head` group was selected. Applied here rather than expected as two
    # CLI overrides, because group selection is order-dependent and a forgotten override
    # makes a healthy model look broken (measured: Top-1 0.80 with dropout, 1.00 without).
    zero = bool(OmegaConf.select(cfg, "train.zero_dropout"))
    encoder = build_structural_encoder(
        str(cfg.gnn.kind),
        vocab_size=vocab_size,
        hidden_dim=int(cfg.gnn.hidden_dim),
        depth=int(cfg.gnn.depth),
        heads=int(cfg.gnn.heads),
        dropout=0.0 if zero else float(cfg.gnn.dropout),
        residual=bool(cfg.gnn.residual),
        layer_norm=bool(cfg.gnn.layer_norm),
        edge_handling=str(cfg.gnn.edge_handling),
    )
    head = LineHead(
        encoder.output_dim,
        hidden_dim=int(cfg.head.hidden_dim),
        dropout=0.0 if zero else float(cfg.head.dropout),
    )
    return StructuralLineModel(encoder, head, n_lines=int(cfg.llm.n_lines)).to(device)


def train(cfg: DictConfig) -> dict[str, Any]:
    """Run one GNN-only training job.

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
    seed = int(cfg.seed)
    seed_everything(seed)
    device = resolve_device(str(cfg.device.kind), index=int(cfg.device.index))

    vocab, vocab_meta = load_node_vocab(
        str(cfg.gnn.vocab_path), expect_digest=str(cfg.gnn.expect_vocab_digest) or None
    )
    _LOGGER.info(
        "Node-type vocabulary loaded",
        extra={
            "digest": vocab_meta["digest"][:12],
            "size": vocab_meta["size"],
            "from_run": vocab_meta["run_id"],
            "builder": vocab_meta["builder"],
            "language_policy": vocab_meta["language_policy"],
        },
    )

    cache_dir = Path(cfg.data.cache.output_dir)
    # The line cap is `llm.n_lines` for both branches on purpose. It is a property of the
    # shared line axis, not of the encoder, and two branches capping differently would put
    # a silent difference inside the A1 comparison.
    n_lines = int(cfg.llm.n_lines)

    def make_dataset(split: str, limit: int) -> GraphLineDataset:
        return GraphLineDataset(
            cache_dir,
            split,
            vocab,
            builder=str(cfg.gnn.builder),
            language=str(cfg.gnn.language),
            n_lines=n_lines,
            limit=limit,
            line_labelled_only=bool(cfg.train.line_labelled_only),
            normalize_whitespace=bool(cfg.representation.normalize_whitespace),
        )

    train_set = make_dataset(str(cfg.train.train_split), int(cfg.train.limit))
    eval_set = (
        train_set
        if cfg.train.overfit
        else make_dataset(str(cfg.train.eval_split), int(cfg.train.eval_limit))
    )

    model = build_model(cfg, vocab_meta["size"], device)

    return run_training(
        cfg,
        model=model,
        train_set=train_set,
        eval_set=eval_set,
        make_loader=lambda dataset, shuffle: build_loader(dataset, cfg, shuffle=shuffle, seed=seed),
        device=device,
        extra_params={
            "branch": "gnn_only",
            "gnn_kind": str(cfg.gnn.kind),
            "gnn_depth": int(cfg.gnn.depth),
            "gnn_hidden_dim": int(cfg.gnn.hidden_dim),
            "edge_handling": str(cfg.gnn.edge_handling),
            "graph_builder": str(cfg.gnn.builder),
            "language_policy": str(cfg.gnn.language),
            "max_nodes_per_batch": int(cfg.train.max_nodes_per_batch),
            # The vocabulary is an artefact with a digest (Memory.md §1.11). Logging it
            # is what makes two checkpoints comparable — or provably not.
            "node_vocab_digest": vocab_meta["digest"][:12],
            "node_vocab_size": vocab_meta["size"],
            # This branch's own ceiling (§7 #17), measured on this split rather than
            # quoted from the corpus average.
            "flagged_without_a_graph_node": train_set.flagged_without_node,
            "graph_build_failures": train_set.build_failures,
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
