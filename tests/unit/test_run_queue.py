"""Guards the queue's handling of runs that are longer than the machine is available.

The queue decides, unattended, whether an interrupted run is continued or started over,
and it is the only thing between a three-hour run and the laptop being picked up at eight
in the morning. A wrong decision is silent either way: restarting wastes the night, and
resuming the wrong checkpoint produces a number.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from vulnloc.training import loop

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_queue.py"

RUN_ID = "0123456789abcdef0123456789abcdef"


def _load() -> Any:
    """Import ``scripts/run_queue.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_run_queue", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_queue = _load()


@pytest.fixture
def queue(tmp_path, monkeypatch):
    """The queue module with every path it writes moved under ``tmp_path``."""
    monkeypatch.setattr(run_queue, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(run_queue, "STATE_PATH", tmp_path / "queue" / "state.json")
    monkeypatch.setattr(run_queue, "LOG_DIR", tmp_path / "queue" / "logs")
    monkeypatch.setattr(run_queue, "STOP_PATH", tmp_path / "queue" / "STOP")
    monkeypatch.setattr(run_queue, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(run_queue, "REPO", tmp_path)
    monkeypatch.setattr(run_queue, "POLL_SECONDS", 0.2)
    return run_queue


def _step(tmp_path: Path, body: str, *, resumable: bool = True) -> Any:
    """A step whose script is ``body``, standing in for a training entrypoint."""
    script = tmp_path / "fake_step.py"
    script.write_text(body, encoding="utf-8")
    return run_queue.Step(
        name="fake", group="test", why="test", args=[], script=str(script), resumable=resumable
    )


def _leave_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoints" / RUN_ID / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    return checkpoint


# --- the two mirrored constants ------------------------------------------------------


def test_the_queue_and_the_trainer_agree_on_how_a_pause_is_signalled():
    """They are copies, not imports, so that `--status` does not have to load torch."""
    assert run_queue.STOP_FILE_ENV == loop.STOP_FILE_ENV
    assert run_queue.PAUSED_EXIT_CODE == loop.PAUSED_EXIT_CODE


# --- the D9.7 slate ---------------------------------------------------------------------


def _group(name: str) -> list[Any]:
    return [step for step in run_queue.build_plan() if step.group == name]


def test_every_fine_tuned_run_trains_the_encoder_off_the_live_path():
    """The cache is refused for an unfrozen encoder; both overrides must travel together."""
    training = [step for step in _group("d97") if step.script.endswith("train_fused.py")]

    assert len(training) == 9
    for step in training:
        assert "llm.frozen=false" in step.args
        assert "train.semantic_cache.enabled=false" in step.args
        assert "train=fused" in step.args
        assert step.resumable


def test_the_two_arms_d97_names_run_seed_by_seed():
    """Wherever the week stops, treatment and control are at equal seed counts."""
    names = [step.name for step in _group("d97")][:6]

    assert names == [
        "d97-cross_attention-seed42",
        "d97-concat-seed42",
        "d97-cross_attention-seed1",
        "d97-concat-seed1",
        "d97-cross_attention-seed2",
        "d97-concat-seed2",
    ]


def test_the_fine_tuned_seed_reports_cannot_pick_up_the_frozen_screen():
    reports = [step for step in _group("d97") if step.script.endswith("report_seeds.py")]

    assert len(reports) == 3
    for step in reports:
        assert any("frozen: False" in argument for argument in step.args)


def test_the_overfit_check_differs_from_the_real_runs_only_in_the_train_config():
    check = {step.name.removeprefix("d97-overfit-"): step for step in _group("d97check")}
    real = {
        step.name.removeprefix("d97-").removesuffix("-seed42"): step
        for step in _group("d97")
        if step.name.endswith("-seed42")
    }

    assert set(check) == set(real) == {"cross_attention", "concat", "llm_only"}
    for arm, step in check.items():
        ignoring = {"train=fused", "train=overfit_fused", "seed=42"}
        assert set(step.args) - ignoring == set(real[arm].args) - ignoring
        assert not step.resumable


def test_step_names_are_unique():
    names = [step.name for step in run_queue.build_plan()]

    assert len(names) == len(set(names))


# --- when is a step continued rather than restarted ------------------------------------


def test_a_paused_step_is_continued_from_its_checkpoint(queue, tmp_path):
    checkpoint = _leave_checkpoint(tmp_path)
    step = _step(tmp_path, "")

    found = queue.resume_checkpoint(step, {"status": "paused", "run_id": RUN_ID})

    assert found == checkpoint


def test_the_run_id_is_recovered_from_the_log_when_the_queue_itself_was_killed(queue, tmp_path):
    """A killed queue never wrote the run id into its state; the log still has it."""
    checkpoint = _leave_checkpoint(tmp_path)
    log = tmp_path / "fake.log"
    log.write_text(f"Checkpoint written path=artifacts/checkpoints/{RUN_ID}/last.pt\n")
    step = _step(tmp_path, "")

    found = queue.resume_checkpoint(step, {"status": "running", "log": str(log)})

    assert found == checkpoint


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"status": "done", "run_id": RUN_ID},
        {"status": "paused"},
        {"status": "paused", "run_id": "f" * 32},
    ],
    ids=["never ran", "already done", "no run id", "no checkpoint on disk"],
)
def test_a_step_with_nothing_to_continue_starts_over(queue, tmp_path, record):
    _leave_checkpoint(tmp_path)

    assert queue.resume_checkpoint(_step(tmp_path, ""), record) is None


def test_a_step_not_marked_resumable_is_never_continued(queue, tmp_path):
    _leave_checkpoint(tmp_path)
    step = _step(tmp_path, "", resumable=False)

    assert queue.resume_checkpoint(step, {"status": "paused", "run_id": RUN_ID}) is None


# --- --until ----------------------------------------------------------------------------


def test_until_means_the_next_time_the_clock_reads_it():
    evening = datetime(2026, 10, 2, 22, 0)

    assert run_queue.parse_until("23:30", evening) == datetime(2026, 10, 2, 23, 30)
    assert run_queue.parse_until("08:30", evening) == datetime(2026, 10, 3, 8, 30)


def test_until_rejects_something_that_is_not_a_time():
    with pytest.raises(ValueError):
        run_queue.parse_until("tomorrow", datetime(2026, 10, 2, 22, 0))


# --- running a step ---------------------------------------------------------------------

EXIT_WITH = "import sys\nsys.exit({code})\n"

WAIT_FOR_STOP = """
import os, sys, time
from pathlib import Path
stop = Path(os.environ["VULNLOC_STOP_FILE"])
for _ in range(400):
    if stop.exists():
        sys.exit(75)
    time.sleep(0.05)
sys.exit(1)
"""

RECORD_ARGV = """
import sys
from pathlib import Path
Path(sys.argv[0]).with_suffix(".argv").write_text("\\n".join(sys.argv[1:]), encoding="utf-8")
"""


@pytest.mark.parametrize(("code", "status"), [(0, "done"), (75, "paused"), (3, "failed")])
def test_the_exit_code_decides_what_the_step_is(queue, tmp_path, code, status):
    state: dict[str, Any] = {"steps": {}}

    outcome = queue.run_step(_step(tmp_path, EXIT_WITH.format(code=code)), state)

    assert outcome == status
    assert state["steps"]["fake"]["status"] == status
    assert json.loads(queue.STATE_PATH.read_text(encoding="utf-8")) == state


def test_a_deadline_pauses_the_running_step_instead_of_killing_it(queue, tmp_path):
    state: dict[str, Any] = {"steps": {}}
    passed = datetime.now() - timedelta(minutes=1)

    outcome = queue.run_step(_step(tmp_path, WAIT_FOR_STOP), state, deadline=passed)

    assert outcome == "paused"
    assert queue.STOP_PATH.exists()


def test_a_step_that_cannot_resume_is_not_told_how_to_stop(queue, tmp_path, monkeypatch):
    """Otherwise a deadline would pause a run that the next start throws away."""
    monkeypatch.delenv("VULNLOC_STOP_FILE", raising=False)
    body = 'import os, sys\nsys.exit(4 if "VULNLOC_STOP_FILE" in os.environ else 0)\n'
    state: dict[str, Any] = {"steps": {}}

    assert queue.run_step(_step(tmp_path, body, resumable=False), state) == "done"


def test_a_continued_step_is_handed_its_checkpoint_and_keeps_its_history(queue, tmp_path):
    _leave_checkpoint(tmp_path)
    step = _step(tmp_path, RECORD_ARGV)
    queue.LOG_DIR.mkdir(parents=True)
    (queue.LOG_DIR / "fake.log").write_text("first attempt\n", encoding="utf-8")
    state: dict[str, Any] = {
        "steps": {"fake": {"status": "paused", "run_id": RUN_ID, "minutes": 90.0, "attempts": 1}}
    }

    assert queue.run_step(step, state) == "done"

    argv = (tmp_path / "fake_step.argv").read_text(encoding="utf-8").splitlines()
    assert argv == [f"train.resume_from=./checkpoints/{RUN_ID}/last.pt"]
    record = state["steps"]["fake"]
    assert record["attempts"] == 2
    assert record["minutes"] >= 90.0
    # The first attempt's log is the first half of the same run, not something to overwrite.
    assert (queue.LOG_DIR / "fake.log").read_text(encoding="utf-8").startswith("first attempt")


def test_a_first_attempt_gets_no_resume_argument(queue, tmp_path):
    step = _step(tmp_path, RECORD_ARGV)

    queue.run_step(step, {"steps": {}})

    assert (tmp_path / "fake_step.argv").read_text(encoding="utf-8") == ""


def test_status_reports_a_finished_runs_result(queue, tmp_path):
    summary = tmp_path / "checkpoints" / RUN_ID / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps({"best": {"line/top_5_accuracy": 0.54081, "line/top_1_accuracy": 0.3}}),
        encoding="utf-8",
    )

    assert queue.result_line(RUN_ID) == "  Top-5 0.5408  Top-1 0.3000"
    assert queue.result_line(None) == ""
    assert queue.result_line("f" * 32) == ""


# --- machine 2: whole groups, each with its own reference arm (Memory.md §7 #49) --------


def _overrides(step: Any) -> set[str]:
    return {argument for argument in step.args if not argument.startswith("seed=")}


def test_every_machine_two_run_is_frozen_and_reads_the_cache():
    for step in _group("m2") + _group("m2check"):
        assert "llm.frozen=true" in step.args
        assert not any("semantic_cache.enabled=false" in argument for argument in step.args)
        assert step.why.startswith("MACHINE 2 ONLY")


def test_machine_two_carries_its_own_reference_arm_at_every_seed():
    """Without it, a variant would be read against this laptop's runs — §7 #49."""
    names = {step.name for step in _group("m2")}

    assert {f"m2-reference-seed{seed}" for seed in (42, 1, 2)} <= names


def test_each_machine_two_variant_changes_one_thing_against_its_reference():
    steps = {step.name: step for step in _group("m2")}
    reference = _overrides(steps["m2-reference-seed42"])
    expected = {
        "gat": ({"+gnn=gat"}, {"+gnn=gcn"}),
        "ggnn": ({"+gnn=ggnn"}, {"+gnn=gcn"}),
        "bce_weighted": ({"train.loss=bce", "train.auto_pos_weight=true"}, set()),
        "bce_plain": (
            {"train.loss=bce", "train.auto_pos_weight=false", "train.pos_weight=1.0"},
            set(),
        ),
        "line_only": ({"train.function_weight=0"}, set()),
        "graphcodebert": (
            {
                "llm.model_name=microsoft/graphcodebert-base",
                "llm.tokenizer_name=microsoft/graphcodebert-base",
                "train.semantic_cache.dir=./artifacts/semantic_cache_graphcodebert",
            },
            set(),
        ),
    }
    for name, (added, removed) in expected.items():
        for seed in (42, 1, 2):
            variant = _overrides(steps[f"m2-{name}-seed{seed}"])
            assert variant - reference == added, name
            assert reference - variant == removed, name


def test_the_reference_arm_is_the_a1_arm_it_is_compared_with():
    """Machine 2's reference arm is this laptop's frozen A1 cross_attention arm.

    So the two can be set side by side as a reproducibility check — and never pooled.
    """
    a1 = _overrides(next(s for s in _group("phase5") if s.name == "a1-cross_attention-seed42"))
    m2 = _overrides(next(s for s in _group("m2") if s.name == "m2-reference-seed42"))

    assert a1 == m2


def test_machine_two_runs_seed_by_seed():
    """A night cut short leaves every variant at the same seed count."""
    seeds = [step.name.rsplit("seed", 1)[1] for step in _group("m2")]

    assert seeds == ["42"] * 7 + ["1"] * 7 + ["2"] * 7


def test_the_graphcodebert_cache_is_built_before_anything_reads_it():
    check = [step.name for step in _group("m2check")]

    assert check[0] == "m2-build-graphcodebert-cache"
    build = _group("m2check")[0]
    reader = next(s for s in _group("m2") if s.name == "m2-graphcodebert-seed42")
    cache_dir = "train.semantic_cache.dir=./artifacts/semantic_cache_graphcodebert"
    assert cache_dir in build.args
    assert cache_dir in reader.args


# --- machine 2, fine-tuned: A5 with both arms on the same machine ------------------------


def test_the_fine_tuned_codebert_arm_is_d97s_cross_attention_config():
    """So it can be read beside D9.7 as a reproduction — and never pooled with it."""
    d97 = _overrides(next(s for s in _group("d97") if s.name == "d97-cross_attention-seed42"))
    m2ft = _overrides(next(s for s in _group("m2ft") if s.name == "m2ft-codebert-seed42"))

    assert d97 == m2ft


def test_the_fine_tuned_arms_differ_only_in_the_encoder():
    steps = {step.name: step for step in _group("m2ft")}
    for seed in (42, 1, 2):
        codebert = _overrides(steps[f"m2ft-codebert-seed{seed}"])
        graphcodebert = _overrides(steps[f"m2ft-graphcodebert-seed{seed}"])
        assert graphcodebert - codebert == {
            "llm.model_name=microsoft/graphcodebert-base",
            "llm.tokenizer_name=microsoft/graphcodebert-base",
        }
        assert codebert - graphcodebert == set()
        assert steps[f"m2ft-codebert-seed{seed}"].resumable


def test_the_fine_tuned_machine_two_runs_go_seed_by_seed_and_are_checked_first():
    names = [step.name for step in _group("m2ft")]

    assert names == [
        f"m2ft-{arm}-seed{seed}" for seed in (42, 1, 2) for arm in ("codebert", "graphcodebert")
    ]
    for step in _group("m2ftcheck"):
        assert "train=overfit_fused" in step.args
        assert "llm.frozen=false" in step.args
        assert not step.resumable


# --- a fresh machine must name its groups -----------------------------------------------


def test_a_fresh_machine_refuses_to_start_the_whole_slate(queue, monkeypatch):
    """The second laptop starts with an empty state file; unscoped, it would run Phase 3."""
    monkeypatch.setattr(sys, "argv", ["run_queue.py"])

    assert queue.main() == 2
    assert not queue.LOG_DIR.exists()


def test_several_groups_can_be_named_at_once(queue, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["run_queue.py", "--only", "m2check", "m2ftcheck", "--dry-run"]
    )

    assert queue.main() == 0
    printed = capsys.readouterr().out
    assert printed.count("train=overfit_fused") == 5
    assert "build_semantic_cache.py" in printed
