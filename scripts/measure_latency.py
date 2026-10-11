r"""How long one function takes on a CPU, measured (``PRD.md`` NFR-4; Phase 9 exit criterion).

NFR-4 recommends a p95 under two seconds for a single function on a CPU. This script
measures it instead of asserting it: a bundle is loaded through the serving path, a sample
of ``val`` functions drawn by content hash is analyzed one at a time — parse, both
branches, calibration, explanation, verification, the whole request — and the percentiles
are logged as an MLflow run with the machine they were taken on.

Three things a latency number hides unless they are printed beside it:

* **what else the machine was doing.** A figure taken while a training run holds the CPU
  is a figure about the training run. The script records the load it saw and
  ``+latency.require_idle=true`` (the default) refuses to start above a CPU threshold;
* **the length of the functions.** Cost grows with windows and nodes, so the percentiles
  are also given by length band — the corpus average would hide the long ones, and
  ``clean_eval`` functions are four times the average length (``CLAUDE.md`` #12);
* **the first request.** Loading the bundle is reported separately and warm-up requests
  are not counted.

``val`` only: latency does not need a held-out function.

Usage::

    python scripts/measure_latency.py +latency.bundle=./artifacts/serving/ba7dcc5f
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from vulnloc.data.cache import iter_shard
from vulnloc.serving.inference import Analyzer
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import ConfigError, VulnLocError
from vulnloc.utils.hashing import stable_sample
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

SPLIT = "val"
TARGET_P95_SECONDS = 2.0
"""NFR-4's recommended bound. Reported against, never tuned towards."""

LENGTH_BANDS = ((1, 25), (26, 50), (51, 100), (101, 250), (251, 10**9))
"""Function length bands, in source lines, the percentiles are also reported for."""

IDLE_CPU_PERCENT = 25.0
"""Above this average CPU load the machine is not idle enough for the number to mean much."""

CONFIG_KEYS = frozenset({"bundle", "n", "seed", "warmup", "threads", "require_idle"})
"""What a caller may pass as ``+latency.<key>``. ``start_run`` logs each one as a parameter,
exactly as given — so nothing measured may be logged under the same name."""


def measured_params(report: dict[str, Any]) -> dict[str, Any]:
    """The facts logged as parameters beside the config, under keys the config does not use.

    MLflow refuses a parameter whose value changes. The first run of record logged the
    bundle path a second time, normalized, as ``latency.bundle``: the run failed after
    every request had been timed and before anything was written (2026-10-11,
    ``8b74b485…``). The path as given is already a parameter; the resolved one is in the
    report.

    Args:
        report: The report ``measure`` assembled.

    Returns:
        Parameter name to value. No name is ``latency.<key>`` for a key in ``CONFIG_KEYS``.
    """
    return {
        "latency.requests": report["requests"]["n"],
        "latency.device": report["device"],
        "latency.torch_threads": report["torch_threads"],
        "latency.processor": report["processor"][:240],
    }


def cpu_load_percent(samples: int = 5, interval: float = 0.4) -> float | None:
    """Average CPU load over a short window, or ``None`` where it cannot be read.

    Args:
        samples: Readings to average.
        interval: Seconds between readings.

    Returns:
        Percent busy, over every core.
    """
    if platform.system() != "Windows":
        if not hasattr(os, "getloadavg"):
            return None
        return float(100.0 * os.getloadavg()[0] / (os.cpu_count() or 1))

    import ctypes  # noqa: PLC0415 - Windows only

    class _FileTime(ctypes.Structure):
        _fields_ = (("low", ctypes.c_uint32), ("high", ctypes.c_uint32))

    def times() -> tuple[int, int]:
        """Idle time, and kernel + user time (the kernel figure includes idle)."""
        idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
        ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        )
        as_int = [(t.high << 32) | t.low for t in (idle, kernel, user)]
        return as_int[0], as_int[1] + as_int[2]

    readings = []
    idle_before, total_before = times()
    for _ in range(samples):
        time.sleep(interval)
        idle_now, total_now = times()
        spent = total_now - total_before
        if spent > 0:
            readings.append(100.0 * (1.0 - (idle_now - idle_before) / spent))
        idle_before, total_before = idle_now, total_now
    return float(np.mean(readings)) if readings else None


def percentiles(seconds: list[float]) -> dict[str, float]:
    """The summary of one set of timings.

    Args:
        seconds: One duration per request.

    Returns:
        ``{"n", "p50", "p95", "max", "mean"}``.
    """
    values = np.asarray(seconds, dtype=np.float64)
    return {
        "n": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def measure(cfg: DictConfig) -> dict[str, Any]:
    """Time a sample of validation functions through one bundle and log the result.

    Args:
        cfg: Composed config. Requires ``latency.bundle``; reads ``latency.n`` (default
            200), ``latency.seed``, ``latency.warmup`` (default 3), ``latency.threads``
            and ``latency.require_idle`` (default ``true``).

    Returns:
        The report, as logged.

    Raises:
        ConfigError: If no bundle is named, or the machine is busy and ``require_idle``
            is on.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )

    def option(name: str, default: Any = None) -> Any:
        value = OmegaConf.select(cfg, f"latency.{name}")
        return default if value is None else value

    bundle_dir = option("bundle")
    if not bundle_dir:
        raise ConfigError("no bundle to time", hint="Pass '+latency.bundle=<directory>'.")
    threads = option("threads")
    if threads is not None:
        torch.set_num_threads(int(threads))

    load_before = cpu_load_percent()
    if (
        bool(option("require_idle", True))
        and load_before is not None
        and load_before > IDLE_CPU_PERCENT
    ):
        raise ConfigError(
            "the machine is busy; a latency taken now would measure the other job",
            cpu_load_percent=round(load_before, 1),
            hint="Wait for training to finish, or pass '+latency.require_idle=false' and "
            "say so beside the number.",
        )

    started = time.perf_counter()
    analyzer = Analyzer.load(str(bundle_dir))
    load_seconds = time.perf_counter() - started

    labelled = {
        unit.unit_id: unit
        for unit in iter_shard(Path(cfg.data.cache.output_dir), SPLIT)
        if unit.vulnerable_lines
    }
    chosen = sorted(
        stable_sample(
            sorted(labelled),
            min(int(option("n", 200)), len(labelled)),
            seed=int(option("seed", 1337)),
        )
    )
    for unit_id in chosen[: int(option("warmup", 3))]:
        analyzer.analyze(labelled[unit_id].source, language=labelled[unit_id].language)

    timings: list[tuple[int, float]] = []
    failed: list[str] = []
    for unit_id in chosen:
        unit = labelled[unit_id]
        begun = time.perf_counter()
        try:
            analyzer.analyze(unit.source, language=unit.language, unit_id=unit_id)
        except VulnLocError as error:
            # A request that fails is a request; it is counted, named and left out of the
            # percentiles, which describe answers.
            _LOGGER.warning(
                "A timed request failed", extra={"unit_id": unit_id, "error": str(error)}
            )
            failed.append(unit_id)
            continue
        timings.append((unit.n_lines, time.perf_counter() - begun))
    load_after = cpu_load_percent()

    overall = percentiles([seconds for _, seconds in timings])
    bands = {}
    for low, high in LENGTH_BANDS:
        inside = [seconds for lines, seconds in timings if low <= lines <= high]
        if inside:
            bands[f"{low}-{high if high < 10**9 else 'up'}"] = percentiles(inside)
    report: dict[str, Any] = {
        "bundle": Path(str(bundle_dir)).as_posix(),
        "source_run": analyzer.bundle.run_id,
        "split": SPLIT,
        "device": str(analyzer.device),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cpu_load_percent_before": load_before,
        "cpu_load_percent_after": load_after,
        "bundle_load_seconds": load_seconds,
        "requests": overall,
        "by_length": bands,
        "failed": failed,
        "target_p95_seconds": TARGET_P95_SECONDS,
        "meets_target": overall["p95"] < TARGET_P95_SECONDS,
    }
    with start_run(cfg, run_name="serving-latency") as run_id:
        report["run_id"] = run_id
        mlflow.set_tags(
            {"artefact": "latency", "split": SPLIT, "source_run": analyzer.bundle.run_id}
        )
        mlflow.log_params(measured_params(report))
        mlflow.log_metrics(
            {
                "p50_seconds": overall["p50"],
                "p95_seconds": overall["p95"],
                "max_seconds": overall["max"],
                "bundle_load_seconds": load_seconds,
                "failed_requests": float(len(failed)),
                "cpu_load_percent_before": float(load_before or 0.0),
            }
        )
        mlflow.log_dict(report, "latency/report.json")
    _LOGGER.info(
        "Latency measured",
        extra={
            "run_id": run_id,
            "requests": overall["n"],
            "p50": round(overall["p50"], 3),
            "p95": round(overall["p95"], 3),
            "max": round(overall["max"], 3),
            "meets_target": report["meets_target"],
            "cpu_load_before": None if load_before is None else round(load_before, 1),
        },
    )
    return report


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    measure(cfg)


if __name__ == "__main__":
    main()
