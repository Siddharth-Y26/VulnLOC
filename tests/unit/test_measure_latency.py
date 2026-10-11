"""Guards the latency measurement's bookkeeping.

The timing itself needs a bundle and is exercised by running the script. What is pinned
here is what lost the first run of record: a measured fact logged under a config key.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "measure_latency.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_measure_latency", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure_latency = _load()


def _report() -> dict[str, Any]:
    return {
        "bundle": "artifacts/serving/ba7dcc5f",
        "device": "cpu",
        "processor": "x" * 400,
        "torch_threads": 6,
        "requests": {"n": 200, "p50": 0.2, "p95": 1.1, "max": 3.0, "mean": 0.4},
    }


def test_no_measured_fact_is_logged_under_a_config_key() -> None:
    """MLflow refuses a parameter that changes; `start_run` logged the config's first.

    `+latency.bundle=./artifacts/serving/ba7dcc5f` and the same path normalized are two
    values. Logging the second under the first's name failed the run after every request
    had been timed.
    """
    logged = measure_latency.measured_params(_report())

    taken = {f"latency.{key}" for key in measure_latency.CONFIG_KEYS}
    assert not set(logged) & taken
    assert "latency.bundle" not in logged


def test_the_config_keys_are_the_ones_the_script_reads() -> None:
    """A key the script starts reading must be added here, or the guard above is blind."""
    source = SCRIPT.read_text(encoding="utf-8")

    for key in measure_latency.CONFIG_KEYS:
        assert f'option("{key}"' in source, key
    read = {part.split('"')[0] for part in source.split('option("')[1:]}
    assert read == set(measure_latency.CONFIG_KEYS)


def test_the_logged_facts_say_what_was_measured_and_on_what() -> None:
    logged = measure_latency.measured_params(_report())

    assert logged["latency.requests"] == 200
    assert logged["latency.device"] == "cpu"
    assert logged["latency.torch_threads"] == 6
    assert len(logged["latency.processor"]) == 240


def test_percentiles_describe_the_timings_they_were_given() -> None:
    summary = measure_latency.percentiles([0.1, 0.2, 0.3, 0.4, 2.0])

    assert summary["n"] == 5
    assert summary["p50"] == pytest.approx(0.3)
    assert summary["max"] == pytest.approx(2.0)
    assert summary["p50"] <= summary["p95"] <= summary["max"]
    assert summary["mean"] == pytest.approx(0.6)
