"""Guards the second machine's export: results come back as files, never as a merge.

What has to survive the trip is what a write-up needs to cite a number without lying about
where it came from: which machine, which GPU, which commit, which run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "export_results.py"
RUN_ID = "0123456789abcdef0123456789abcdef"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_export_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


export_results = _load()


class Step:
    """The two attributes of a queue step the export reads."""

    def __init__(self, name: str, group: str) -> None:
        self.name = name
        self.group = group


@pytest.fixture
def machine(tmp_path):
    """A working tree with one finished run, one pending step and a step of another group."""
    repo = tmp_path / "repo"
    run_dir = repo / "artifacts" / "checkpoints" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "best": {
                    "line/top_5_accuracy": 0.61,
                    "line/top_1_accuracy": 0.35,
                    "line/f1": 0.36,
                    "line/ifa_mean": 10.5,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "best.pt").write_bytes(b"weights")
    logs = repo / "artifacts" / "queue" / "logs"
    logs.mkdir(parents=True)
    (logs / "m2-reference-seed42.log").write_text("log", encoding="utf-8")
    (repo / "artifacts" / "mlflow.db").write_bytes(b"sqlite")
    reports = repo / "artifacts" / "m2"
    reports.mkdir()
    (reports / "seed_report_fused_m2_reference.json").write_text("{}", encoding="utf-8")
    plan = [
        Step("m2-reference-seed42", "m2"),
        Step("m2-gat-seed42", "m2"),
        Step("d97-concat-seed1", "d97"),
    ]
    state = {
        "steps": {
            "m2-reference-seed42": {"status": "done", "run_id": RUN_ID, "minutes": 14.2},
            "d97-concat-seed1": {"status": "paused", "run_id": "f" * 32},
        }
    }
    return repo, plan, state, reports


def _export(machine, out, **kwargs):
    repo, plan, state, reports = machine
    return export_results.export(
        kwargs.pop("groups", ["m2"]),
        out,
        repo=repo,
        plan=plan,
        state=state,
        reports_dir=reports,
        gpu="Test GPU, 6144 MiB",
        commit="abc1234",
        **kwargs,
    )


def test_the_export_names_the_machine_the_gpu_and_the_commit(machine, tmp_path):
    folder = _export(machine, tmp_path / "out")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["gpu"] == "Test GPU, 6144 MiB"
    assert manifest["commit"] == "abc1234"
    assert manifest["host"]
    assert "never pooled" in manifest["note"]


def test_only_the_requested_groups_are_exported(machine, tmp_path):
    folder = _export(machine, tmp_path / "out")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))

    assert [step["name"] for step in manifest["steps"]] == ["m2-reference-seed42", "m2-gat-seed42"]
    assert "d97-concat-seed1" not in (folder / "queue_state.json").read_text(encoding="utf-8")


def test_a_finished_run_carries_its_summary_and_headline_metrics(machine, tmp_path):
    folder = _export(machine, tmp_path / "out")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    finished, pending = manifest["steps"]

    assert finished["top5"] == pytest.approx(0.61)
    assert finished["ifa"] == pytest.approx(10.5)
    assert (folder / "runs" / RUN_ID / "summary.json").is_file()
    assert pending["status"] == "pending"
    assert "top5" not in pending


def test_checkpoints_travel_only_when_asked_for(machine, tmp_path):
    without = _export(machine, tmp_path / "a")
    with_them = _export(machine, tmp_path / "b", with_checkpoints=True)

    assert not (without / "runs" / RUN_ID / "best.pt").exists()
    assert (with_them / "runs" / RUN_ID / "best.pt").read_bytes() == b"weights"


def test_logs_reports_and_a_readable_copy_of_the_store_come_along(machine, tmp_path):
    folder = _export(machine, tmp_path / "out")

    assert (folder / "logs" / "m2-reference-seed42.log").is_file()
    assert (folder / "reports" / "seed_report_fused_m2_reference.json").is_file()
    stores = list(folder.glob("mlflow_*.db"))
    assert len(stores) == 1
    # Named after the host, so it can never be mistaken for the importing machine's store.
    assert stores[0].name != "mlflow.db"


def test_a_second_export_never_overwrites_the_first(machine, tmp_path, monkeypatch):
    first = _export(machine, tmp_path / "out")
    real_datetime = export_results.datetime

    class Later(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2099, 12, 31, 23, 59, 59)

    monkeypatch.setattr(export_results, "datetime", Later)
    second = _export(machine, tmp_path / "out")

    assert first != second
    assert first.exists() and second.exists()


def test_an_unknown_group_is_refused(machine, tmp_path):
    with pytest.raises(ValueError, match="no queue step"):
        _export(machine, tmp_path / "out", groups=["nonexistent"])


def test_the_table_is_readable_without_the_json(machine, tmp_path):
    folder = _export(machine, tmp_path / "out")
    table = (folder / "RESULTS.md").read_text(encoding="utf-8")

    assert "Test GPU, 6144 MiB" in table
    assert "| m2-reference-seed42 | done | 01234567 | 14.2 |" in table


def test_a_missing_nvidia_smi_is_reported_not_hidden(monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(export_results.subprocess, "run", missing)

    assert export_results.describe_gpu().startswith("unavailable: nvidia-smi could not run")
