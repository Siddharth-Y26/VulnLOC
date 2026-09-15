"""What the fused model actually costs on 4 GiB — landmine #44, measured.

``configs/train/fused.yaml`` ships batch budgets that are **starting points, not results**,
and `Memory.md` §7 #44 says so in as many words. This script is how that stops being true.
It is the Phase 5 counterpart of ``scripts/probe_vram.py`` and inherits its one hard-won
rule:

**On Windows an over-budget configuration does not raise.** The WDDM driver pages past
physical VRAM into host RAM, so an over-sized batch returns correct numbers at 10–25× the
cost (``Memory.md`` §7 #29, measured in Phase 3). Catching ``OutOfMemoryError`` therefore
finds almost nothing. Every result is compared against the card's physical memory, and
``within_physical_vram`` — not the absence of an exception — is what "fits" means.

Three things make this probe different from Phase 3's, and each of them is the difference
between a number and a guess:

**It uses real corpus batches, not synthetic tensors.** Phase 3 could synthesise a batch
because one window is one window. Here a unit costs *windows* on one side and *nodes* on
the other and **the two are uncorrelated** — a long flat function is expensive for the
encoder and cheap for the GNN, a short deeply-nested one the reverse. Synthesising that
relationship would be assuming the exact thing being measured.

**It probes the worst batch the sampler can produce, not an average one.** Phase 4's
§7 #18 is the rule: *a batch is bounded by its worst graph, not the mean.* For every
budget the probe finds the batch that maximises windows and the batch that maximises
nodes, and reports the worse of the two.

**It probes the batch the sampler cannot avoid.** A unit larger than the budget on its own
still gets its own batch — both samplers document this, because dropping it would discard
the largest functions, which carry the most flagged lines (§1.8). So the corpus's largest
single unit is a floor on peak memory that no budget setting can lower, and if *that* does
not fit, the budget grid is answering the wrong question. It is probed separately, on both
partitions, and reported as ``worst_unit``.

It also records whether the loss and the gradients stayed **finite**, which is a Phase 5
question and not a Phase 3 one: ``configs/train/structural.yaml`` turns AMP off on the
grounds that fp16 message passing over graphs of wildly varying degree is a needless
source of non-finite values, while ``configs/train/semantic.yaml`` turns it on because it
is the measured best configuration for the encoder. The fused model contains both halves
under one ``train.amp``, so which way that setting should go is a measurement, not a
preference.

Usage::

    python scripts/probe_vram_fused.py +fusion=cross_attention +gnn=gcn train=fused
    python scripts/probe_vram_fused.py +fusion=concat           +gnn=gcn train=fused
    python scripts/probe_vram_fused.py +fusion=cross_attention +gnn=gcn train=fused train.amp=false

One arm per invocation, because the arm is a config group and its constructor arguments
differ (``heads`` and ``n_layers`` exist for ``cross_attention`` alone — see
``configs/fusion/README.md``). Each writes ``artifacts/fusion/vram_probe_<arm>.json``.

**This is a memory measurement, not a training run.** It logs no line-level metric, writes
no checkpoint, and says nothing about whether cross-attention beats concatenation. It does
not touch the Phase 3 / Phase 4 gates (``Memory.md`` §6 item 1).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
import torch
import transformers
from omegaconf import DictConfig
from transformers import AutoTokenizer

from vulnloc.representation.tokenization import LineTokenizer
from vulnloc.training.device import resolve_device
from vulnloc.training.fused_datasets import (
    FusedBudgetBatchSampler,
    FusedLineDataset,
    collate_fused,
)
from vulnloc.training.graph_datasets import load_node_vocab, load_subtoken_vocab
from vulnloc.training.losses import LineLoss
from vulnloc.training.seeding import seed_everything
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import ModelError
from vulnloc.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vulnloc.training.fused_datasets import FusedUnit

# `scripts/` is not a package, so the sibling entrypoint is imported by path. It is
# imported rather than reimplemented on purpose: a probe that composed the model its own
# way would measure a model nobody trains, which is the one thing a VRAM probe must not
# do. `train_structural.py` and `train_semantic.py` each own their `build_model` the same
# way, so there is no package-side function to share instead.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fused import build_model, require_group  # noqa: E402

_LOGGER = get_logger(__name__)


def _peak_mib(device: torch.device) -> float:
    """Peak allocated MiB since the last reset; 0 on CPU."""
    if device.type != "cuda":
        return 0.0
    return round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)


def _total_mib(device: torch.device) -> float:
    """Physical memory of the device in MiB; infinite on CPU, which cannot spill."""
    if device.type != "cuda":
        return float("inf")
    return round(torch.cuda.get_device_properties(device).total_memory / 1024**2, 1)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a batch's tensors, leaving everything else alone.

    The fused batch is flat precisely so this walk reaches every tensor — ``Memory.md``
    §7 #45.
    """
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def worst_batches(
    dataset: FusedLineDataset, *, max_windows: int, max_nodes: int
) -> list[tuple[str, list[int]]]:
    """Find the most expensive batches a budget can produce, on each cost axis.

    Sizing a probe from an average batch measures the average and reassures you about the
    wrong thing: ``Memory.md`` §7 #18 — *a batch is bounded by its worst graph, not the
    mean*. The two axes are separated because they are uncorrelated, so the batch that
    maximises one is usually not the batch that maximises the other.

    Args:
        dataset: The joined dataset.
        max_windows: Window budget.
        max_nodes: Node budget.

    Returns:
        ``[(label, indices), …]`` — at most two entries, deduplicated when one batch
        happens to be worst on both axes.
    """
    sampler = FusedBudgetBatchSampler(
        dataset, max_windows=max_windows, max_nodes=max_nodes, shuffle=False
    )
    by_windows: tuple[int, list[int]] = (-1, [])
    by_nodes: tuple[int, list[int]] = (-1, [])
    for batch in sampler:
        windows = sum(dataset[index].n_windows for index in batch)
        nodes = sum(dataset[index].n_nodes for index in batch)
        if windows > by_windows[0]:
            by_windows = (windows, list(batch))
        if nodes > by_nodes[0]:
            by_nodes = (nodes, list(batch))

    found = [("max_windows", by_windows[1]), ("max_nodes", by_nodes[1])]
    if by_windows[1] == by_nodes[1]:
        return [("max_windows_and_nodes", by_windows[1])]
    return [entry for entry in found if entry[1]]


def worst_units(dataset: FusedLineDataset) -> list[tuple[str, list[int]]]:
    """Find the single units that no budget setting can make cheaper.

    A unit over budget on its own still gets its own batch — both samplers say so, because
    dropping it would discard the largest functions, which carry the most flagged lines
    (``Memory.md`` §1.8). These are therefore a **floor** on peak memory, and if the floor
    does not fit then the budget grid is answering the wrong question.

    Args:
        dataset: The joined dataset.

    Returns:
        ``[(label, [index]), …]``, deduplicated when one unit is largest on both axes.
    """
    widest = max(range(len(dataset)), key=lambda i: dataset[i].n_windows)
    biggest = max(range(len(dataset)), key=lambda i: dataset[i].n_nodes)
    if widest == biggest:
        return [("worst_unit_windows_and_nodes", [widest])]
    return [("worst_unit_windows", [widest]), ("worst_unit_nodes", [biggest])]


def probe_one(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: LineLoss,
    units: Sequence[FusedUnit],
    device: torch.device,
    *,
    use_amp: bool,
    steps: int,
    step: str = "train",
) -> dict[str, Any]:
    """Run one batch repeatedly and report what it cost.

    The step is the whole Phase 5 shape — both encoders, the line alignment, the fusion,
    the head and the per-line loss — because all of it has to be resident at once, and a
    probe of the encoder alone is the number Phase 3 already has.

    Args:
        model: The fused model, already on ``device``.
        optimizer: Its optimizer, already warmed so that Adam's moments are allocated.
            They are real memory and they exist during training, so a probe that excluded
            them would under-report by roughly twice the trainable parameter count.
        criterion: The training loss.
        units: The units of this batch.
        device: Where to run.
        use_amp: Autocast to fp16, as ``train.amp`` asks.
        steps: Timed steps after one warm-up.
        step: What to measure.

            ``train``
                A full training step — forward, loss, backward, optimizer.
            ``eval``
                What :func:`~vulnloc.training.loop.evaluate` does: ``model.eval()`` under
                ``no_grad``, forward and loss only. The eval split is never trained on, so
                probing it with a training step would measure a batch that never happens.
            ``eval_grad``
                The same forward with autograd still **recording** — what ``evaluate`` did
                before 2026-09-14. Kept as a permanent witness to what the ``no_grad``
                guard buys, because the cost is wildly non-obvious: HuggingFace gates
                gradient checkpointing on ``self.training``
                (``transformers/modeling_layers.py``), so in eval mode checkpointing is
                **off** and every activation of the forward is retained. Evaluation was
                therefore the most expensive pass in the loop, not the cheapest.

    Returns:
        A record with ``oom``, ``within_physical_vram``, ``finite``, peak memory and
        seconds per step.
    """
    if step not in {"train", "eval", "eval_grad"}:
        raise ModelError("unknown probe step", step=step, accepted=["train", "eval", "eval_grad"])

    batch = _to_device(collate_fused(units), device)
    windows = int(batch["sem_input_ids"].shape[0])
    nodes = int(batch["str_node_type"].shape[0])
    shape = {
        "units": int(batch["n_units"]),
        "windows": windows,
        "nodes": nodes,
        "edges": int(batch["str_edge_index"].shape[1]),
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    finite = True
    train_step = step == "train"
    model.train(train_step)

    def forward_loss() -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(batch)
            return criterion(
                output["line_logits"],
                batch["line_labels"],
                output["line_mask"] & batch["line_valid"],
                output["function_logits"],
                batch["targets"],
            )["loss"]

    def one_step() -> float:
        nonlocal finite
        if not train_step:
            with torch.no_grad() if step == "eval" else torch.enable_grad():
                loss = forward_loss()
            if not torch.isfinite(loss):
                finite = False
            return float(loss)

        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss()
        loss.backward()
        # Recorded rather than raised: `structural.yaml` turns AMP off on the grounds that
        # fp16 message passing over graphs of varying degree produces non-finite values,
        # and `semantic.yaml` turns it on. The fused model has one switch for both halves,
        # so this is the field that decides which way it should point.
        if not torch.isfinite(loss):
            finite = False
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                finite = False
                break
        optimizer.step()
        return float(loss.detach())

    try:
        one_step()  # Warm-up: allocator growth and autotuning are not the measurement.
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = _peak_mib(device)
        record: dict[str, Any] = {
            "oom": False,
            "peak_mib": peak,
            "peak_pct_of_vram": (
                round(100.0 * peak / _total_mib(device), 1) if device.type == "cuda" else None
            ),
            "seconds_per_step": round(elapsed / max(steps, 1), 4),
            "units_per_second": round(steps * shape["units"] / max(elapsed, 1e-9), 2),
            "within_physical_vram": peak <= _total_mib(device),
            "finite": finite,
        }
    except torch.OutOfMemoryError:
        # One of two ways to fail; the other does not raise — see the module docstring.
        # The peak reached before the allocator gave up is reported rather than discarded:
        # "it OOM'd" and "it OOM'd needing 2.6x the card" are different findings.
        peak = _peak_mib(device)
        record = {
            "oom": True,
            "peak_mib": peak,
            "peak_pct_of_vram": (
                round(100.0 * peak / _total_mib(device), 1) if device.type == "cuda" else None
            ),
            "seconds_per_step": None,
            "units_per_second": None,
            "within_physical_vram": False,
            "finite": None,
        }
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return record | shape | {"step": step}


def probe(cfg: DictConfig) -> dict[str, Any]:
    """Run the whole grid for one fusion arm and write the report.

    Args:
        cfg: Composed Hydra config. Requires ``+fusion=<arm>`` and ``+gnn=<kind>``.

    Returns:
        The report, as written to disk.

    Raises:
        ModelError: If no budget in the grid fits in physical VRAM.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    transformers.logging.set_verbosity_error()
    seed_everything(int(cfg.seed))

    fusion_cfg = require_group(cfg, "fusion", hint="+fusion=cross_attention")
    gnn_cfg = require_group(cfg, "gnn", hint="+gnn=gcn")
    arm = str(fusion_cfg.kind)
    device = resolve_device(str(cfg.device.kind), index=int(cfg.device.index))
    settings = fusion_cfg.probe
    steps = int(settings.steps)
    budgets = [(int(windows), int(nodes)) for windows, nodes in settings.budgets]
    use_amp = bool(cfg.train.amp) and device.type == "cuda"

    vocab, vocab_meta = load_node_vocab(
        str(gnn_cfg.vocab_path), expect_digest=str(gnn_cfg.expect_vocab_digest) or None
    )
    subtoken_vocab: dict[str, int] | None = None
    subtoken_meta: dict[str, Any] = {}
    if bool(gnn_cfg.use_subtokens):
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

    def make_dataset(split: str) -> FusedLineDataset:
        return FusedLineDataset(
            Path(cfg.data.cache.output_dir),
            split,
            tokenizer,
            vocab,
            subtoken_vocab=subtoken_vocab,
            builder=str(gnn_cfg.builder),
            language=str(gnn_cfg.language),
            n_lines=int(cfg.llm.n_lines),
            limit=int(settings.limit),
            line_labelled_only=bool(cfg.train.line_labelled_only),
            normalize_whitespace=bool(cfg.representation.normalize_whitespace),
        )

    splits = {
        name: make_dataset(name) for name in (str(cfg.train.train_split), str(cfg.train.eval_split))
    }
    train_set = splits[str(cfg.train.train_split)]

    results: list[dict[str, Any]] = []
    with start_run(cfg, run_name=f"vram-probe-fused-{arm}") as run_id:
        mlflow.log_params(
            {
                "probe": "fused_vram",
                "fusion_kind": arm,
                "gnn_kind": str(gnn_cfg.kind),
                "gnn_depth": int(gnn_cfg.depth),
                "amp": use_amp,
                "gradient_checkpointing": bool(cfg.llm.gradient_checkpointing),
                "trim_line_axis": bool(fusion_cfg.trim_line_axis),
                "use_subtokens": bool(subtoken_vocab),
                "resolved_device": str(device),
            }
        )
        model = build_model(
            cfg,
            vocab_size=vocab_meta["size"],
            device=device,
            subtoken_vocab_size=int(subtoken_meta.get("size", 0)),
        )
        model.train()
        criterion = LineLoss(
            kind=str(cfg.train.loss),
            pos_weight=train_set.positive_line_weight(),
            alpha=float(cfg.train.focal_alpha),
            gamma=float(cfg.train.focal_gamma),
            function_weight=float(cfg.train.function_weight),
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # The budget grid, on the training split — the batches the sampler will really
        # produce, at their worst rather than at their average.
        for max_windows, max_nodes in budgets:
            for label, indices in worst_batches(
                train_set, max_windows=max_windows, max_nodes=max_nodes
            ):
                record = probe_one(
                    model,
                    optimizer,
                    criterion,
                    [train_set[index] for index in indices],
                    device,
                    use_amp=use_amp,
                    steps=steps,
                ) | {
                    "kind": "budget",
                    "split": train_set.split,
                    "label": label,
                    "max_windows": max_windows,
                    "max_nodes": max_nodes,
                }
                results.append(record)
                _LOGGER.info("Probed", extra=dict(record))
                if not record["oom"] and record["peak_mib"] is not None:
                    mlflow.log_metric(
                        f"vram/{arm}/w{max_windows}_n{max_nodes}/{label}/peak_mib",
                        record["peak_mib"],
                    )

        # The floor no budget can lower, on both partitions. `val` is probed too because
        # its functions are larger (Memory.md §7 #12) and eval batches must fit as well.
        for name, dataset in splits.items():
            # The eval split is never trained on, so probing it with a training step would
            # measure a batch that never happens. It also gets `eval_grad` — the
            # pre-2026-09-14 behaviour — so the report carries what `evaluate`'s `no_grad`
            # guard is worth rather than asserting it.
            probe_steps = ["train"] if name == train_set.split else ["eval", "eval_grad"]
            for label, indices in worst_units(dataset):
                for probe_step in probe_steps:
                    record = probe_one(
                        model,
                        optimizer,
                        criterion,
                        [dataset[index] for index in indices],
                        device,
                        use_amp=use_amp,
                        steps=steps,
                        step=probe_step,
                    ) | {
                        "kind": "worst_unit",
                        "split": name,
                        "label": label,
                        "unit_id": dataset[indices[0]].unit_id,
                        "max_windows": None,
                        "max_nodes": None,
                    }
                    results.append(record)
                    _LOGGER.info("Probed", extra=dict(record))

        # "Fits" means fits in PHYSICAL memory. A budget that only ran because the driver
        # paged to host RAM has not fitted; it has degraded (Memory.md §7 #29).
        budget_rows = [r for r in results if r["kind"] == "budget"]
        fitting = [
            (r["max_windows"], r["max_nodes"])
            for r in budget_rows
            if r["within_physical_vram"] and r["finite"]
        ]
        failing = {
            (r["max_windows"], r["max_nodes"])
            for r in budget_rows
            if not (r["within_physical_vram"] and r["finite"])
        }
        # A budget only counts if EVERY worst-case batch at that budget fitted.
        largest = max(
            (pair for pair in set(fitting) if pair not in failing),
            key=lambda pair: (pair[0], pair[1]),
            default=None,
        )
        # `eval_grad` rows deliberately measure behaviour the code no longer has, so they
        # are held out of the verdict — counting them would make the report say the fused
        # arm spills when it does not.
        current = [r for r in results if r["step"] != "eval_grad"]
        spilled = [
            {
                "kind": r["kind"],
                "label": r["label"],
                "split": r["split"],
                "step": r["step"],
                "max_windows": r["max_windows"],
                "max_nodes": r["max_nodes"],
                "peak_mib": r["peak_mib"],
                "units": r["units"],
                "windows": r["windows"],
                "nodes": r["nodes"],
            }
            for r in current
            if not r["oom"] and not r["within_physical_vram"]
        ]
        # What `evaluate`'s `no_grad` guard is worth, measured rather than asserted: the
        # same evaluation forward with autograd recording, which is what the loop did
        # before 2026-09-14. In eval mode HuggingFace bypasses gradient checkpointing
        # (`transformers/modeling_layers.py` gates it on `self.training`), so every
        # activation is retained and evaluation becomes the most expensive pass, not the
        # cheapest.
        no_grad_guard = [
            {
                "split": r["split"],
                "unit_id": r.get("unit_id"),
                "windows": r["windows"],
                "nodes": r["nodes"],
                "with_no_grad_mib": next(
                    (
                        q["peak_mib"]
                        for q in results
                        if q["step"] == "eval"
                        and q["label"] == r["label"]
                        and q["split"] == r["split"]
                    ),
                    None,
                ),
                "grad_enabled_mib": r["peak_mib"],
                "grad_enabled_oom": r["oom"],
            }
            for r in results
            if r["step"] == "eval_grad"
        ]
        non_finite = [
            {"kind": r["kind"], "label": r["label"], "split": r["split"]}
            for r in current
            if r["finite"] is False
        ]

        payload = {
            "run_id": run_id,
            "fusion_kind": arm,
            "gnn_kind": str(gnn_cfg.kind),
            "graph_builder": str(gnn_cfg.builder),
            "use_subtokens": bool(subtoken_vocab),
            "amp": use_amp,
            "gradient_checkpointing": bool(cfg.llm.gradient_checkpointing),
            "trim_line_axis": bool(fusion_cfg.trim_line_axis),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "total_vram_mib": _total_mib(device) if device.type == "cuda" else None,
            "trainable_parameters": trainable,
            "n_lines_cap": int(cfg.llm.n_lines),
            "units": {name: len(dataset) for name, dataset in splits.items()},
            "results": results,
            "largest_budget_that_fits": (
                {"max_windows": largest[0], "max_nodes": largest[1]} if largest else None
            ),
            "spilled_to_host_memory": spilled,
            "no_grad_guard": no_grad_guard,
            "non_finite": non_finite,
        }
        output_path = Path(settings.output_dir) / f"vram_probe_{arm}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        mlflow.log_artifact(str(output_path), artifact_path="fusion")

    if largest is None:
        raise ModelError(
            "no budget in the grid fits in physical VRAM; the fused arm cannot train here "
            "at these settings",
            device=str(device),
            arm=arm,
            budgets=budgets,
            report=str(output_path),
        )
    _LOGGER.info(
        "Fused VRAM probe written",
        extra={
            "path": str(output_path),
            "arm": arm,
            "largest_budget_that_fits": largest,
            "spilled": len(spilled),
            "non_finite": len(non_finite),
        },
    )
    return payload


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    probe(cfg)


if __name__ == "__main__":
    main()
