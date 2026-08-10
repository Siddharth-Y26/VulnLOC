"""Phase 0 end-to-end smoke test.

Covers the exit criterion "``python scripts/train.py train=debug`` runs a no-op loop
and logs to MLflow" without shelling out, so failures produce a real traceback.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import mlflow
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def _load_train_script() -> ModuleType:
    """Import ``scripts/train.py`` by path — ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_train", TRAIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_script() -> ModuleType:
    return _load_train_script()


def test_train_script_exists():
    assert TRAIN_SCRIPT.is_file()


def test_debug_run_logs_a_metric_to_mlflow(train_script, isolated_cfg, tmp_path):
    cfg = isolated_cfg("train=debug", "train.max_steps=3")

    run_id = train_script.run_training(cfg)

    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics["noop/completed_steps"] == 3.0

    client = mlflow.tracking.MlflowClient(tracking_uri=str(cfg.mlflow.tracking_uri))
    history = client.get_metric_history(run_id, "noop/loss")
    assert [point.step for point in history] == [0, 1, 2]


def test_debug_run_writes_structured_logs(train_script, isolated_cfg, tmp_path):
    cfg = isolated_cfg("train=debug")

    run_id = train_script.run_training(cfg)

    log_file = tmp_path / "logs" / "vulnloc.log"
    records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    messages = {record["message"] for record in records}
    assert {"Seeded run", "MLflow run started", "No-op training complete"} <= messages

    # Architecture.md §8: every line emitted inside the run carries its run id.
    inside = [r for r in records if r["message"] == "No-op training complete"]
    assert [r["run_id"] for r in inside] == [run_id]


def test_debug_run_respects_the_seed_override(train_script, isolated_cfg):
    cfg = isolated_cfg("train=debug", "seed=1234")

    run_id = train_script.run_training(cfg)

    assert mlflow.get_run(run_id).data.params["seed"] == "1234"


def test_debug_run_records_the_accelerator(train_script, isolated_cfg):
    cfg = isolated_cfg("train=debug")

    run_id = train_script.run_training(cfg)

    tags = mlflow.get_run(run_id).data.tags
    assert tags["device"] in {"cpu", "cuda:0"}
    assert tags["torch_version"]
    assert tags["cuda_build"]


def test_debug_run_honours_a_forced_cpu_device(train_script, isolated_cfg):
    cfg = isolated_cfg("train=debug", "device.kind=cpu")

    run_id = train_script.run_training(cfg)

    assert mlflow.get_run(run_id).data.tags["device"] == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_debug_run_uses_the_gpu_when_available(train_script, isolated_cfg):
    # The loop performs a real matmul on the resolved device, so a passing run here is
    # evidence that the GPU computes — not merely that torch reports one.
    cfg = isolated_cfg("train=debug", "device.kind=cuda")

    run_id = train_script.run_training(cfg)

    tags = mlflow.get_run(run_id).data.tags
    assert tags["device"] == "cuda:0"
    assert tags["gpu_name"]
    assert float(tags["gpu_total_gib"]) > 0


def test_run_is_traceable_to_code_and_config(train_script, isolated_cfg):
    # Rules.md §1.3 / §5.8: a run that cannot be traced back to a commit and a config
    # cannot support a claim in the paper.
    cfg = isolated_cfg("train=debug")

    run_id = train_script.run_training(cfg)

    tags = mlflow.get_run(run_id).data.tags
    assert tags["git_sha"]
    assert len(tags["config_hash"]) == 12
    assert tags["vulnloc_version"]
