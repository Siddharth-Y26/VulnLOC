"""The scoring script's refusals — the parts that decide what may be scored at all.

Whether the scores are right is checked where it can be: against the metrics the run itself
logged, every time the script runs. What is pinned here is what it must refuse without
being asked, because each of these would otherwise produce a clean-looking file: a held-out
partition scored before its time, a probe scored as if it were a result, a run id that
matches two checkpoints.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vulnloc.utils.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "score_checkpoint.py"


def _load() -> Any:
    """Import ``scripts/score_checkpoint.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_score_checkpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


score_checkpoint = _load()


# --- the held-out partitions are touched once, at the end (Rules.md §5.3) -------------


@pytest.mark.parametrize("split", ["test", "clean_eval"])
def test_a_held_out_partition_is_refused_until_the_final_evaluation(split):
    with pytest.raises(ConfigError, match="evaluated once, at the end"):
        score_checkpoint.require_scoreable_split(split, final=False)


@pytest.mark.parametrize("split", ["test", "clean_eval"])
def test_the_final_evaluation_may_score_a_held_out_partition(split):
    score_checkpoint.require_scoreable_split(split, final=True)


@pytest.mark.parametrize("split", ["val", "train"])
def test_validation_and_train_need_no_flag(split):
    score_checkpoint.require_scoreable_split(split, final=False)


def test_the_held_out_set_is_exactly_the_two_reported_partitions():
    """Adding a partition to the split without adding it here would leave it unguarded."""
    assert {"test", "clean_eval"} == score_checkpoint.HELD_OUT_SPLITS


# --- which run ------------------------------------------------------------------------


def test_a_prefix_expands_to_the_one_run_it_names(tmp_path):
    (tmp_path / ("80fc5238" + "a" * 24)).mkdir()
    (tmp_path / ("c193e3bc" + "b" * 24)).mkdir()

    assert score_checkpoint.resolve_run_id("80fc", tmp_path) == "80fc5238" + "a" * 24


def test_a_prefix_matching_two_runs_is_refused_not_guessed(tmp_path):
    (tmp_path / ("80fc5238" + "a" * 24)).mkdir()
    (tmp_path / ("80fc9999" + "b" * 24)).mkdir()

    with pytest.raises(ConfigError, match="exactly one checkpoint directory"):
        score_checkpoint.resolve_run_id("80fc", tmp_path)


def test_a_run_with_no_checkpoint_directory_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="exactly one checkpoint directory"):
        score_checkpoint.resolve_run_id("deadbeef", tmp_path)
    with pytest.raises(ConfigError, match="exactly one checkpoint directory"):
        score_checkpoint.resolve_run_id("deadbeef", tmp_path / "absent")


# --- is the run a result --------------------------------------------------------------


def _run(*, status="FINISHED", params=None, tags=None):
    return SimpleNamespace(
        info=SimpleNamespace(run_id="r" * 32, status=status),
        data=SimpleNamespace(
            params={"train.limit": "0", "train.overfit": "False", **(params or {})},
            tags=dict(tags or {}),
        ),
    )


def test_a_finished_full_corpus_run_may_be_scored():
    score_checkpoint.refuse_unusable_run(_run())


@pytest.mark.parametrize(
    ("run", "why"),
    [
        (_run(status="FAILED"), "status is FAILED"),
        (_run(tags={"void": "probe"}), "tagged void"),
        (_run(params={"train.limit": "3000"}), "a subsample"),
        (_run(params={"train.overfit": "True"}), "a debugging instrument"),
    ],
    ids=["paused or crashed", "void", "subsample probe", "overfit check"],
)
def test_a_run_that_is_not_a_result_is_not_scored(run, why):
    with pytest.raises(ConfigError, match="not a result to score") as refusal:
        score_checkpoint.refuse_unusable_run(run)

    assert any(why in reason for reason in refusal.value.details["why"])


def test_every_reason_is_reported_not_just_the_first():
    run = _run(status="FAILED", tags={"void": "x"}, params={"train.limit": "600"})

    with pytest.raises(ConfigError) as refusal:
        score_checkpoint.refuse_unusable_run(run)

    assert len(refusal.value.details["why"]) == 3


# --- scoring on comment-stripped text (Memory.md §1.25) -------------------------------


def _cfg(**representation: bool):
    from omegaconf import OmegaConf

    return OmegaConf.create({"representation": {"normalize_whitespace": True, **representation}})


def test_a_run_is_scored_on_the_text_it_was_trained_on_unless_asked_otherwise():
    trained = _cfg()

    config, text_changed = score_checkpoint.evaluation_config(trained, _cfg())

    assert config is trained
    assert text_changed is False


def test_stripping_at_evaluation_is_flagged_as_a_change_of_text():
    trained = _cfg()

    config, text_changed = score_checkpoint.evaluation_config(trained, _cfg(strip_comments=True))

    assert text_changed is True
    assert config.representation.strip_comments is True
    assert "strip_comments" not in trained.representation, "the run's own config is not edited"


def test_a_run_trained_on_stripped_text_is_not_a_change_of_text():
    trained = _cfg(strip_comments=True)

    config, text_changed = score_checkpoint.evaluation_config(trained, _cfg(strip_comments=True))

    assert config is trained
    assert text_changed is False


def test_a_run_trained_on_stripped_text_is_never_scored_with_comments_put_back():
    trained = _cfg(strip_comments=True)

    config, text_changed = score_checkpoint.evaluation_config(trained, _cfg())

    assert config.representation.strip_comments is True
    assert text_changed is False


def test_stripping_at_evaluation_reads_the_stripped_semantic_cache_not_the_one_of_record():
    from omegaconf import OmegaConf

    trained = OmegaConf.create(
        {
            "representation": {"normalize_whitespace": True},
            "train": {"semantic_cache": {"enabled": True, "dir": "./artifacts/semantic_cache"}},
        }
    )

    config, text_changed = score_checkpoint.evaluation_config(trained, _cfg(strip_comments=True))

    assert text_changed is True
    assert config.train.semantic_cache.dir == "./artifacts/semantic_cache_stripped"
    assert config.train.semantic_cache.enabled is True
    assert trained.train.semantic_cache.dir == "./artifacts/semantic_cache"
