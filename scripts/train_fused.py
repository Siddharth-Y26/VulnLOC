"""Train the fused model — experiment A1, the question this project exists to answer.

Both branches, one line axis, one head, one training loop. Everything below the fusion is
the same object the two baselines used: ``PretrainedCodeEncoder`` from Phase 3,
``StructuralEncoder`` from Phase 4, ``align_*_to_lines``, ``LineHead``, and
``training/job.py``. Only ``+fusion=`` changes between the arms::

    python scripts/train_fused.py +fusion=cross_attention train=overfit_fused
    python scripts/train_fused.py +fusion=concat           train=overfit_fused
    python scripts/train_fused.py +fusion=cross_attention train=fused seed=42
    python scripts/train_fused.py +fusion=concat           train=fused seed=42

**Run the overfit config first, for every arm you intend to compare.** It is 32 functions
and proves the thing A1 cannot be run without: that gradient reaches *both* branches and
that the two line axes actually line up. An arm with a broken branch trains, converges,
and looks exactly like a result.

``+fusion=`` takes a ``+`` for the same reason ``+gnn=`` does, and the reason is
``Memory.md`` §7 #38: adding a group to ``config.yaml``'s ``defaults`` changes the
resolved config — and therefore the ``config_hash`` — of **every** run, including the
three semantic seeds and the three structural seeds that closed Phases 3 and 4. That would
make the baselines this experiment is measured against incomparable with each other, which
is a far worse outcome than typing a plus sign.

**Two learning rates, deliberately.** ``train.encoder_lr`` for the pretrained transformer,
``train.lr`` for the GNN, the fusion and the head. A single rate has to be wrong for one of
them. Both are fixed across every arm, so A1 stays controlled.

**What must travel with any number this produces**, all of it unchanged from the arms it
will be compared against: the ~30% line-label noise ceiling (``Memory.md`` §1.6), the
512-line cap (§7 #26), and the 7.84% of flagged lines with no graph node (§7 #17) — which
the fused model can score and the structural arm cannot, because the fused line mask is
the union of the two branches' coverage. That last one is a real advantage of fusion and
it is *not* the cross-attention mechanism: ``concat`` has it too. Do not let it be read as
A1's result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vulnloc.models.fused import FusedLineModel
from vulnloc.models.fusion import build_fusion
from vulnloc.models.heads.line_head import LineHead
from vulnloc.models.semantic.encoder import PretrainedCodeEncoder
from vulnloc.models.structural import build_structural_encoder
from vulnloc.representation.tokenization import LineTokenizer
from vulnloc.training.device import resolve_device
from vulnloc.training.fused_datasets import (
    FusedBudgetBatchSampler,
    FusedLineDataset,
    collate_fused,
)
from vulnloc.training.graph_datasets import load_node_vocab, load_subtoken_vocab
from vulnloc.training.job import run_training
from vulnloc.training.seeding import seed_everything
from vulnloc.training.semantic_cache import (
    SemanticFeatureCache,
    cache_inputs,
    corpus_fingerprints,
    encoder_revision,
)
from vulnloc.utils.config import strip_comments_enabled
from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)


def build_loader(
    dataset: FusedLineDataset, cfg: DictConfig, *, shuffle: bool, seed: int
) -> DataLoader[Any]:
    """Wrap a dataset in a loader bounded by both budgets.

    Args:
        dataset: The dataset.
        cfg: Composed config.
        shuffle: Shuffle unit order.
        seed: Base seed for the sampler.

    Returns:
        A loader yielding collated fused batches.
    """
    sampler = FusedBudgetBatchSampler(
        dataset,
        max_windows=int(cfg.train.max_windows_per_batch),
        max_nodes=int(cfg.train.max_nodes_per_batch),
        shuffle=shuffle,
        seed=seed,
        max_units=int(cfg.train.max_units_per_batch),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fused,
        num_workers=0,  # The dataset is already in memory; workers would only re-pickle it.
    )


def build_model(
    cfg: DictConfig, *, vocab_size: int, device: torch.device, subtoken_vocab_size: int = 0
) -> FusedLineModel:
    """Compose both encoders, the fusion and the head from config.

    Args:
        cfg: Composed config. Requires the ``fusion`` and ``gnn`` groups.
        vocab_size: Node-type slots, from the vocabulary artefact rather than the config.
        device: Where the model lives.
        subtoken_vocab_size: Slots for ADR-002 §D2.5's second half. ``0`` leaves node
            features as node-type only.

    Returns:
        The model, on ``device``.

    Raises:
        ConfigError: If the ``fusion`` group was not selected.
    """
    fusion_cfg = require_group(cfg, "fusion", hint="+fusion=cross_attention")
    gnn_cfg = require_group(cfg, "gnn", hint="+gnn=gcn")

    # An overfit test is trying to defeat regularisation, so `train.zero_dropout` wins over
    # whichever group was selected — applied here rather than as three CLI overrides,
    # because group selection is order-dependent (Memory.md §7 #36).
    zero = bool(OmegaConf.select(cfg, "train.zero_dropout"))

    semantic = PretrainedCodeEncoder(
        str(cfg.llm.model_name),
        frozen=bool(cfg.llm.frozen),
        gradient_checkpointing=bool(cfg.llm.gradient_checkpointing),
        dropout=0.0 if zero else float(cfg.llm.dropout),
        pretrained=bool(cfg.llm.pretrained),
    )
    structural = build_structural_encoder(
        str(gnn_cfg.kind),
        vocab_size=vocab_size,
        hidden_dim=int(gnn_cfg.hidden_dim),
        depth=int(gnn_cfg.depth),
        heads=int(gnn_cfg.heads),
        dropout=0.0 if zero else float(gnn_cfg.dropout),
        residual=bool(gnn_cfg.residual),
        layer_norm=bool(gnn_cfg.layer_norm),
        edge_handling=str(gnn_cfg.edge_handling),
        subtoken_vocab_size=subtoken_vocab_size,
    )

    args = dict(OmegaConf.to_container(fusion_cfg.args, resolve=True))  # type: ignore[arg-type]
    if zero:
        args["dropout"] = 0.0
    fusion = build_fusion(
        str(fusion_cfg.kind),
        sem_dim=semantic.hidden_size,
        str_dim=structural.output_dim,
        **args,
    )
    head = LineHead(
        fusion.output_dim,
        hidden_dim=int(cfg.head.hidden_dim),
        dropout=0.0 if zero else float(cfg.head.dropout),
    )
    return FusedLineModel(
        semantic,
        structural,
        fusion,
        head,
        n_lines=int(cfg.llm.n_lines),
        trim_line_axis=bool(fusion_cfg.trim_line_axis),
    ).to(device)


def semantic_cache_enabled(cfg: DictConfig) -> bool:
    """Whether this run reads cached semantic features, refusing an unfrozen encoder.

    ADR-009 §D9.5: only a frozen encoder is a pure function of its input. A fine-tuned
    run reading features from a frozen one would train a head against representations its
    own encoder never produces, and nothing would raise.

    Args:
        cfg: Composed config.

    Returns:
        ``True`` when ``train.semantic_cache.enabled`` is set.

    Raises:
        ConfigError: If the cache is enabled while ``llm.frozen`` is false.
    """
    enabled = bool(OmegaConf.select(cfg, "train.semantic_cache.enabled"))
    if enabled and not bool(cfg.llm.frozen):
        raise ConfigError(
            "the semantic feature cache requires a frozen encoder",
            hint=(
                "Add 'llm.frozen=true' (ADR-009), or 'train.semantic_cache.enabled=false' "
                "for a fine-tuned run on the live path."
            ),
        )
    return enabled


def require_group(cfg: DictConfig, group: str, *, hint: str) -> DictConfig:
    """Return a config group, failing loudly when it was not selected.

    Args:
        cfg: Composed config.
        group: Group name.
        hint: The override that would select it.

    Returns:
        The group.

    Raises:
        ConfigError: If the group is absent.
    """
    selected = OmegaConf.select(cfg, group)
    if selected is None:
        raise ConfigError(
            "config group was not selected",
            group=group,
            hint=f"Add '{hint}' — the leading '+' is required (Memory.md §7 #38).",
        )
    return selected  # type: ignore[return-value]


def parameter_groups(model: FusedLineModel, cfg: DictConfig) -> list[dict[str, Any]]:
    """Split the model into a pretrained-encoder group and everything else.

    The two halves were trained at rates two orders of magnitude apart in Phases 3 and 4,
    and putting them under one rate makes the fused model worse than either baseline for a
    reason that has nothing to do with fusion.

    Args:
        model: The composed model.
        cfg: Composed config. Reads ``train.encoder_lr`` and ``train.lr``.

    Returns:
        Two AdamW parameter groups, each tagged with a ``name`` for the log line. Empty
        groups are dropped — a frozen encoder contributes no trainable parameters, and
        AdamW rejects an empty group.
    """
    encoder = [p for p in model.semantic.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    groups = [
        {"name": "semantic_encoder", "params": encoder, "lr": float(cfg.train.encoder_lr)},
        {"name": "structural_fusion_head", "params": rest, "lr": float(cfg.train.lr)},
    ]
    return [group for group in groups if group["params"]]


def train(cfg: DictConfig) -> dict[str, Any]:
    """Run one fused training job.

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

    fusion_cfg = require_group(cfg, "fusion", hint="+fusion=cross_attention")
    gnn_cfg = require_group(cfg, "gnn", hint="+gnn=gcn")

    vocab, vocab_meta = load_node_vocab(
        str(gnn_cfg.vocab_path), expect_digest=str(gnn_cfg.expect_vocab_digest) or None
    )
    subtoken_vocab: dict[str, int] | None = None
    subtoken_meta: dict[str, Any] = {}
    if bool(OmegaConf.select(cfg, "gnn.use_subtokens")):
        subtoken_vocab, subtoken_meta = load_subtoken_vocab(
            str(gnn_cfg.subtoken_vocab_path),
            expect_digest=str(gnn_cfg.expect_subtoken_digest) or None,
        )

    tokenizer = LineTokenizer(
        AutoTokenizer.from_pretrained(str(cfg.llm.tokenizer_name)),
        max_length=int(cfg.llm.max_length),
        stride=int(cfg.llm.stride),
        windowing=bool(cfg.llm.windowing),
    )

    cache_dir = Path(cfg.data.cache.output_dir)
    # One cap for both branches: it is a property of the shared line axis, not of either
    # encoder, and two branches capping differently would put a silent difference inside
    # the A1 comparison.
    n_lines = int(cfg.llm.n_lines)

    use_cache = semantic_cache_enabled(cfg) and bool(fusion_cfg.kind != "gnn_only")
    revision = encoder_revision(str(cfg.llm.model_name)) if use_cache else ""

    def load_cache(split: str) -> SemanticFeatureCache | None:
        if not use_cache:
            return None
        fingerprint, shard_digest = corpus_fingerprints(cfg, split)
        return SemanticFeatureCache.load(
            str(cfg.train.semantic_cache.dir),
            split,
            cache_inputs(
                cfg,
                split=split,
                split_fingerprint=fingerprint,
                shard_digest=shard_digest,
                encoder_revision=revision,
            ),
        )

    def make_dataset(split: str, limit: int) -> FusedLineDataset:
        return FusedLineDataset(
            cache_dir,
            split,
            tokenizer,
            vocab,
            subtoken_vocab=subtoken_vocab,
            builder=str(gnn_cfg.builder),
            language=str(gnn_cfg.language),
            n_lines=n_lines,
            limit=limit,
            line_labelled_only=bool(cfg.train.line_labelled_only),
            normalize_whitespace=bool(cfg.representation.normalize_whitespace),
            strip_comments=strip_comments_enabled(cfg),
            semantic_cache=load_cache(split),
        )

    train_set = make_dataset(str(cfg.train.train_split), int(cfg.train.limit))
    eval_set = (
        train_set
        if cfg.train.overfit
        else make_dataset(str(cfg.train.eval_split), int(cfg.train.eval_limit))
    )

    model = build_model(
        cfg,
        vocab_size=vocab_meta["size"],
        device=device,
        subtoken_vocab_size=int(subtoken_meta.get("size", 0)),
    )
    groups = parameter_groups(model, cfg)
    _LOGGER.info(
        "Fused model composed",
        extra={
            "fusion": str(fusion_cfg.kind),
            "gnn": str(gnn_cfg.kind),
            "sem_dim": model.fusion.sem_dim,
            "str_dim": model.fusion.str_dim,
            "fused_dim": model.fusion.output_dim,
            "produces_attention": model.fusion.produces_attention,
            "uses_semantic": model.fusion.uses_semantic,
            "uses_structural": model.fusion.uses_structural,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_groups": {
                str(group["name"]): sum(int(p.numel()) for p in group["params"]) for group in groups
            },
        },
    )

    return run_training(
        cfg,
        model=model,
        train_set=train_set,
        eval_set=eval_set,
        make_loader=lambda dataset, shuffle: build_loader(dataset, cfg, shuffle=shuffle, seed=seed),
        device=device,
        param_groups=groups,
        extra_params={
            "branch": "fused",
            # A1's independent variable. Everything else in this block is held fixed
            # across the arms, and a report that groups on anything else is grouping on a
            # constant.
            "fusion_kind": str(fusion_cfg.kind),
            "fusion_hidden_dim": int(fusion_cfg.args.hidden_dim),
            "fusion_heads": int(OmegaConf.select(cfg, "fusion.args.heads") or 0),
            "fusion_n_layers": int(OmegaConf.select(cfg, "fusion.args.n_layers") or 0),
            # Ablation A6. `both` is the design and what every cross-attention run before
            # this parameter existed was; arms without attention have no direction.
            "fusion_direction": str(
                OmegaConf.select(cfg, "fusion.args.direction")
                or ("both" if str(fusion_cfg.kind) == "cross_attention" else "none")
            ),
            "trim_line_axis": bool(fusion_cfg.trim_line_axis),
            "model_name": str(cfg.llm.model_name),
            "frozen": bool(cfg.llm.frozen),
            # ADR-009. Which semantic path produced this run, and exactly which cache.
            "semantic_cache": use_cache,
            "semantic_cache_digest": train_set.semantic_cache_digest[:12],
            "windowing": bool(cfg.llm.windowing),
            "gnn_kind": str(gnn_cfg.kind),
            "gnn_depth": int(gnn_cfg.depth),
            "gnn_hidden_dim": int(gnn_cfg.hidden_dim),
            "edge_handling": str(gnn_cfg.edge_handling),
            "graph_builder": str(gnn_cfg.builder),
            "language_policy": str(gnn_cfg.language),
            "use_subtokens": bool(subtoken_vocab),
            "node_vocab_digest": vocab_meta["digest"][:12],
            "node_vocab_size": vocab_meta["size"],
            "subtoken_vocab_digest": (subtoken_meta.get("digest") or "")[:12],
            "max_windows_per_batch": int(cfg.train.max_windows_per_batch),
            "max_nodes_per_batch": int(cfg.train.max_nodes_per_batch),
            "encoder_lr": float(cfg.train.encoder_lr),
            # The population this arm actually trained on, against each branch's own set.
            # Non-zero here means the fused arm and the baselines it is compared with saw
            # different functions, which is a confound and has to be reported as one.
            "units_dropped_semantic_only": train_set.semantic_only,
            "units_dropped_structural_only": train_set.structural_only,
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
