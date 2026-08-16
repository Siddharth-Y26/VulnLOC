"""What actually fits in 4 GiB — the Phase 3 training-strategy decision, measured.

``Memory.md`` §7 #7 has carried "4 GiB VRAM ceiling" as an open design input since Phase 0,
with the instruction **"do not discover this during a sweep."** This script is how it stops
being a guess: it runs real forward and backward passes at a grid of batch sizes and
configurations on the real encoder, and reports peak allocated memory and throughput for
each.

Four configurations, in increasing order of what they give up:

``finetune``
    Everything trainable, full precision. The most expressive and the most expensive.
``amp``
    Everything trainable under autocast — activations in half precision, master weights
    in float32. Usually the cheapest real win.
``checkpoint``
    Gradient checkpointing on top of AMP: activations are recomputed in the backward pass
    instead of being stored. Trades roughly a third more time for a large memory drop.
``frozen``
    Encoder in ``eval`` mode under ``no_grad``; only the head above it trains. Cheapest
    by far and the fallback if nothing else fits — at the cost of the pretrained
    representation never adapting to C code.

An OOM is a **result**, not a crash: it is caught, recorded as ``oom=true``, and the grid
continues. That is the whole point of probing before a sweep rather than during one.

**On Windows an OOM may never arrive.** The WDDM driver spills past physical VRAM into
host RAM rather than failing, so an over-sized batch returns correct numbers at a fraction
of the speed — measured here at **5.6× slower for 2× the work**. Catching
``OutOfMemoryError`` is therefore not enough to find the ceiling: every result is also
compared against the card's physical memory, and ``within_physical_vram`` is the field
that decides what actually fits. A sweep that trusted the absence of a crash would simply
run ten times slower and report valid-looking results.

Usage::

    python scripts/probe_vram.py
    python scripts/probe_vram.py 'llm.probe.batch_sizes=[1,2,4]'
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import hydra
import mlflow
import torch
import transformers
from omegaconf import DictConfig
from transformers import AutoModel

from vulnloc.models.alignment import align_tokens_to_lines
from vulnloc.training.device import resolve_device
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import ModelError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

MODES = ("finetune", "amp", "checkpoint", "frozen")


def _peak_bytes(device: torch.device) -> int:
    """Peak allocated bytes since the last reset, 0 on CPU."""
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0


def _total_mib(device: torch.device) -> float:
    """Physical memory of the device in MiB; infinite on CPU, which cannot spill."""
    if device.type != "cuda":
        return float("inf")
    return torch.cuda.get_device_properties(device).total_memory / 1024**2


def probe_one(
    model: Any,
    device: torch.device,
    *,
    mode: str,
    batch_size: int,
    seq_length: int,
    n_lines: int,
    steps: int,
) -> dict[str, Any]:
    """Run one configuration and report what it cost.

    The step is deliberately the *whole* Phase 3 shape, not just the encoder: tokens are
    pooled onto the line axis and a per-line loss is taken, because the line pooling and
    the retained ``[B, L, D]`` tensor are part of what has to fit.

    Args:
        model: The encoder.
        device: Where to run.
        mode: One of :data:`MODES`.
        batch_size: Windows per step.
        seq_length: Tokens per window.
        n_lines: Line slots to pool onto.
        steps: Timed steps after one warm-up.

    Returns:
        A record with ``oom``, ``within_physical_vram``, peak memory and seconds per step.
    """
    frozen = mode == "frozen"
    use_amp = mode in {"amp", "checkpoint"} and device.type == "cuda"

    if mode == "checkpoint":
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    model.train(not frozen)

    hidden = model.config.hidden_size
    head = torch.nn.Linear(hidden, 1).to(device)
    parameters = list(head.parameters()) + ([] if frozen else list(model.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)

    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    # Spread tokens over lines the way real code does, rather than one line per token.
    token_to_line = torch.randint(1, n_lines + 1, (batch_size, seq_length), device=device)
    targets = torch.randint(0, 2, (batch_size, n_lines), device=device).float()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if frozen:
                with torch.no_grad():
                    states = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                states = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled, mask = align_tokens_to_lines(
                states.last_hidden_state.float(), token_to_line, n_lines=n_lines
            )
            logits = head(pooled).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], targets[mask])
        loss.backward()
        optimizer.step()

    try:
        one_step()  # Warm-up: allocator growth and cuDNN autotuning are not the measurement.
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_mib = round(_peak_bytes(device) / 1024**2, 1)
        record = {
            "oom": False,
            "peak_mib": peak_mib,
            "seconds_per_step": round(elapsed / max(steps, 1), 4),
            "windows_per_second": round(steps * batch_size / max(elapsed, 1e-9), 1),
            "within_physical_vram": peak_mib <= _total_mib(device),
        }
    except torch.OutOfMemoryError:
        # One of two ways to fail. The other one does not raise — see the module docstring.
        record = {
            "oom": True,
            "peak_mib": None,
            "seconds_per_step": None,
            "within_physical_vram": False,
        }
    finally:
        # The head and optimizer state go out of scope with this frame; freeing the
        # cache here is what makes the *next* grid point start from a clean allocator.
        if device.type == "cuda":
            torch.cuda.empty_cache()

    record |= {"mode": mode, "batch_size": batch_size, "seq_length": seq_length}
    return record


def probe(cfg: DictConfig) -> dict[str, Any]:
    """Run the whole grid and write the report.

    Args:
        cfg: Composed Hydra config. Requires ``llm.probe``.

    Returns:
        The report, as written to disk.

    Raises:
        ModelError: If no configuration in the grid fits.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    transformers.logging.set_verbosity_error()
    settings = cfg.llm.probe
    device = resolve_device(str(cfg.device.kind), index=int(cfg.device.index))
    seq_length = int(cfg.llm.max_length)
    n_lines = int(settings.n_lines)
    steps = int(settings.steps)
    batch_sizes = [int(size) for size in settings.batch_sizes]
    modes = [str(mode) for mode in settings.modes]

    results: list[dict[str, Any]] = []
    with start_run(cfg, run_name="vram-probe") as run_id:
        model = AutoModel.from_pretrained(str(cfg.llm.model_name)).to(device)
        total_params = sum(parameter.numel() for parameter in model.parameters())

        for mode in modes:
            for batch_size in batch_sizes:
                record = probe_one(
                    model,
                    device,
                    mode=mode,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    n_lines=n_lines,
                    steps=steps,
                )
                results.append(record)
                _LOGGER.info("Probed", extra={k: v for k, v in record.items()})
                if not record["oom"]:
                    mlflow.log_metric(f"vram/{mode}/bs{batch_size}/peak_mib", record["peak_mib"])
                elif device.type == "cuda":
                    # Larger batches in the same mode cannot fit either.
                    break

        # "Largest that fits" means fits in *physical* memory. A configuration that only
        # ran because the driver paged to host RAM has not fitted; it has degraded.
        largest = {
            mode: max(
                (
                    r["batch_size"]
                    for r in results
                    if r["mode"] == mode and r["within_physical_vram"]
                ),
                default=0,
            )
            for mode in modes
        }
        spilled = [
            {"mode": r["mode"], "batch_size": r["batch_size"], "peak_mib": r["peak_mib"]}
            for r in results
            if not r["oom"] and not r["within_physical_vram"]
        ]
        if not any(largest.values()):
            raise ModelError(
                "no configuration in the grid fits; Phase 3 cannot train on this device",
                device=str(device),
                modes=modes,
                batch_sizes=batch_sizes,
            )

        payload = {
            "run_id": run_id,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "total_vram_mib": (
                round(torch.cuda.get_device_properties(device).total_memory / 1024**2, 1)
                if device.type == "cuda"
                else None
            ),
            "model_name": str(cfg.llm.model_name),
            "encoder_parameters": total_params,
            "seq_length": seq_length,
            "n_lines": n_lines,
            "results": results,
            "largest_batch_that_fits": largest,
            "spilled_to_host_memory": spilled,
        }
        output_path = Path(cfg.llm.report.output_dir) / "vram_probe.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        mlflow.log_artifact(str(output_path), artifact_path="llm")

    _LOGGER.info("VRAM probe written", extra={"path": str(output_path), **largest})
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
