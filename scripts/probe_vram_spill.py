r"""Measure where this card starts paging to host RAM, and what avoiding it costs.

Two questions, both asked of the hardware rather than of the documentation:

**At what pool size does PyTorch start spilling into shared GPU memory?** On Windows a
process that outgrows its dedicated-VRAM budget does not fail — the driver pages the excess
to host RAM and every later step is slower (``Memory.md`` §7 #24, #55, #57). The budget is
not the card's total: this probe grows the allocator's pool in fixed steps and, at each one,
reads this process's own ``GPU Process Memory`` counters, so the first row whose *shared*
usage rises above its baseline is the answer. It also prints what the driver reports as
free, which is the signal ``vulnloc.training.loop.cache_release_reason`` acts on.

**What does releasing the allocator's cache every step cost?** That is the worst case of
the guard firing on every optimizer step: the same working set allocated and freed with and
without ``torch.cuda.empty_cache()``.

It trains nothing, reads no corpus and writes nothing. It is **not** an MLflow run: its
numbers describe a machine on a day, so quote them with the date and what else was on the
card. The counters exist only on Windows; elsewhere the first table is skipped.

Usage::

    python scripts/probe_vram_spill.py
    python scripts/probe_vram_spill.py --step-mib 64 --max-mib 4608
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import time

import torch

MIB = 1024**2
BYTES_PER_FLOAT32 = 4

SPILL_MARGIN_MIB = 16.0
"""Shared usage has to rise this far above its context-only baseline to count as a spill.

The baseline itself moves by a couple of MiB once the first block is allocated; a spill
arrives in steps the size of the block.
"""


def process_gpu_memory(pid: int) -> tuple[float, float] | None:
    """This process's dedicated and shared GPU memory, as Windows accounts for it.

    Args:
        pid: The process to look up.

    Returns:
        ``(dedicated_mib, shared_mib)`` summed over adapters, or ``None`` when the counters
        cannot be read (not Windows, or PowerShell refused).
    """
    if platform.system() != "Windows":
        return None
    command = (
        f"$c = (Get-Counter '\\GPU Process Memory(pid_{pid}*)\\Dedicated Usage',"
        f"'\\GPU Process Memory(pid_{pid}*)\\Shared Usage' -ErrorAction SilentlyContinue)"
        ".CounterSamples;"
        "$d = ($c | Where-Object { $_.Path -like '*dedicated*' } | "
        "Measure-Object CookedValue -Sum).Sum;"
        "$s = ($c | Where-Object { $_.Path -like '*shared*' } | "
        "Measure-Object CookedValue -Sum).Sum;"
        '"$d $s"'
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    fields = completed.stdout.split()
    if completed.returncode != 0 or len(fields) != 2:
        return None
    return float(fields[0]) / MIB, float(fields[1]) / MIB


def probe_spill_onset(device: torch.device, *, step_mib: int, max_mib: int) -> None:
    """Grow the pool and print, per step, where the memory actually went.

    Args:
        device: The CUDA device.
        step_mib: Size of each added block.
        max_mib: Stop once this much is reserved.
    """
    baseline = process_gpu_memory(os.getpid())
    if baseline is None:
        print("  GPU process counters unavailable on this platform; skipping the onset table.")
        return
    print(f"  context only: dedicated {baseline[0]:.1f} MiB, shared {baseline[1]:.1f} MiB")
    print(f"  {'reserved':>9} {'dedicated':>10} {'shared':>8} {'driver_free':>12}   spilling")
    blocks: list[torch.Tensor] = []
    while torch.cuda.memory_reserved(device) / MIB < max_mib:
        try:
            blocks.append(
                torch.ones(step_mib * MIB // BYTES_PER_FLOAT32, dtype=torch.float32, device=device)
            )
        except torch.OutOfMemoryError as error:
            print(f"  out of memory: {str(error).splitlines()[0]}")
            break
        torch.cuda.synchronize(device)
        usage = process_gpu_memory(os.getpid())
        if usage is None:
            print("  counters stopped answering; stopping.")
            break
        reserved = torch.cuda.memory_reserved(device) / MIB
        free = torch.cuda.mem_get_info(device)[0] / MIB
        spilling = "yes" if usage[1] > baseline[1] + SPILL_MARGIN_MIB else ""
        print(f"  {reserved:9.1f} {usage[0]:10.1f} {usage[1]:8.1f} {free:12.1f}   {spilling}")
    del blocks
    torch.cuda.empty_cache()


def _cycle(device: torch.device, *, total_mib: int, block_mib: int, release: bool) -> float:
    """Allocate, touch and free one step's worth of memory; return the seconds it took."""
    started = time.perf_counter()
    blocks = [
        torch.empty(block_mib * MIB // BYTES_PER_FLOAT32, device=device)
        for _ in range(total_mib // block_mib)
    ]
    for block in blocks:
        block.fill_(1.0)
    torch.cuda.synchronize(device)
    del blocks
    if release:
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def probe_release_cost(device: torch.device, *, sizes_mib: list[int], repeats: int) -> None:
    """Print the median cost of a step with and without a cache release.

    Args:
        device: The CUDA device.
        sizes_mib: Working-set sizes to try.
        repeats: Cycles per measurement; the median is reported.
    """
    block_mib = 64
    for total_mib in sizes_mib:
        _cycle(device, total_mib=total_mib, block_mib=block_mib, release=False)
        kept = sorted(
            _cycle(device, total_mib=total_mib, block_mib=block_mib, release=False)
            for _ in range(repeats)
        )[repeats // 2]
        released = sorted(
            _cycle(device, total_mib=total_mib, block_mib=block_mib, release=True)
            for _ in range(repeats)
        )[repeats // 2]
        print(
            f"  {total_mib:5d} MiB per step: cache kept {1000 * kept:7.1f} ms, "
            f"released every step {1000 * released:7.1f} ms, "
            f"extra {1000 * (released - kept):7.1f} ms"
        )
        torch.cuda.empty_cache()


def main() -> int:
    """Entry point.

    Returns:
        0 on success, 1 when there is no CUDA device to probe.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--step-mib", type=int, default=128, help="Pool growth per row.")
    parser.add_argument("--max-mib", type=int, default=4352, help="Stop at this much reserved.")
    parser.add_argument(
        "--release-sizes-mib",
        type=int,
        nargs="+",
        default=[1024, 2048, 3072],
        help="Working sets for the release-cost table.",
    )
    parser.add_argument("--repeats", type=int, default=9, help="Cycles per cost measurement.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device; nothing to probe.")
        return 1
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    free, total = (value / MIB for value in torch.cuda.mem_get_info(device))
    print(
        f"{properties.name}: total {total:.1f} MiB, driver free {free:.1f} MiB, "
        f"torch {torch.__version__}"
    )

    print("\nWhere the pool starts spilling:")
    probe_spill_onset(device, step_mib=args.step_mib, max_mib=args.max_mib)
    print("\nWhat releasing the cache costs:")
    probe_release_cost(device, sizes_mib=args.release_sizes_mib, repeats=args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
