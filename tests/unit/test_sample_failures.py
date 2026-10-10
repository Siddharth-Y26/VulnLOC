"""The failure cases are drawn by rule, and what is counted about them is counted right.

``scripts/sample_failures.py`` decides which functions a reader inspects for Phase 8's
failure taxonomy. The tests pin the three things that would quietly bias that reading: the
definition of a failure (every seed, the evaluation's own tie order), the counts printed
beside each case, and the refusal of anything that is not validation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vulnloc.data.cache import write_shards
from vulnloc.data.labeling import derive_line_labels
from vulnloc.evaluation.evaluator import PartitionUnit
from vulnloc.evaluation.metrics.localization import initial_false_alarms, ranked_labels
from vulnloc.training.line_scores import UnitScores, save_scores
from vulnloc.utils.exceptions import ConfigError, DataError

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "sample_failures.py"


def _load() -> Any:
    """Import ``scripts/sample_failures.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_sample_failures", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


failures = _load()


def _unit(
    unit_id: str, logits: list[float], flagged: list[int], scoreable: list[bool] | None = None
) -> UnitScores:
    labels = np.zeros(len(logits), dtype=np.int8)
    labels[[number - 1 for number in flagged]] = 1
    mask = np.ones(len(logits), dtype=np.bool_) if scoreable is None else np.array(scoreable)
    return UnitScores(
        unit_id=unit_id, logits=np.array(logits, dtype=np.float32), labels=labels, scoreable=mask
    )


def test_ranks_break_ties_in_source_order_as_the_evaluation_does() -> None:
    logits = np.array([0.5, 2.0, 2.0, -1.0, 2.0], dtype=np.float32)
    labels = np.array([0, 0, 1, 0, 1])
    ranks = failures.line_ranks(logits)
    assert ranks.tolist() == [4, 1, 2, 5, 3]
    in_rank_order = labels[np.argsort(ranks)] > 0
    assert in_rank_order.tolist() == ranked_labels(logits, labels)["source"].tolist()


def test_the_first_flagged_rank_is_the_evaluations_ifa_plus_one() -> None:
    unit = _unit("u", [3.0, 1.0, 2.0, 0.0], flagged=[2, 4])
    ranked = ranked_labels(unit.logits, unit.labels)["source"]
    assert failures.first_flagged_rank(unit) == int(initial_false_alarms(ranked)) + 1 == 3
    assert failures.first_flagged_rank(_unit("v", [1.0, 2.0], flagged=[])) is None


def test_a_failure_is_a_miss_on_every_seed_not_on_one() -> None:
    ranks = {
        42: {"always": 30, "once": 30, "never": 2},
        1: {"always": 12, "once": 4, "never": 1},
        2: {"always": 11, "once": 25, "never": 3},
    }
    assert failures.consistent_misses(ranks, 10) == ["always"]
    assert failures.consistent_misses(ranks, 11) == []


def test_seeds_that_ranked_different_units_are_not_one_partition() -> None:
    with pytest.raises(DataError, match="did not rank the same units"):
        failures.consistent_misses({42: {"a": 1}, 1: {"b": 1}}, 10)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("", "blank"),
        ("}", "syntax alone"),
        ("} else {", "syntax alone"),
        ("return;", "syntax alone"),
        ("#endif", "preprocessor"),
        ("# define NOENV(x)", "preprocessor"),
        ("if (!offset) {", "condition or loop header"),
        ("} else if (Z_TYPE(key) == IS_LONG) {", "condition or loop header"),
        ("for (i = 0; i < 4 && out->data[i]; i++) {", "condition or loop header"),
        ("return ret;", "return or jump"),
        ("goto outexcept;", "return or jump"),
        ("unsigned *p;", "declaration"),
        ("uint16 bitspersample, samplesperpixel;", "declaration"),
        ("int ws[BMAX + 1];", "declaration"),
        ("zval key, *data, d, *old_data;", "declaration"),
        ("p = (unsigned *) b;", "assignment"),
        ("int p = 0;", "assignment"),
        ("j += *p++;", "assignment"),
        ("memcpy (p, der, ti.length);", "call"),
        ("dest_filename.sprintf_cat( condor_basename(filename) );", "call"),
        ("clearenv();", "call"),
        ("data_pa,", "other"),
        ('"errors. This is done automatically by default.\\n"', "other"),
    ],
)
def test_a_line_is_sorted_by_the_first_rule_that_matches(text: str, kind: str) -> None:
    assert failures.line_kind(text) == kind
    assert failures.line_kind(f"   {text}  ") == kind


def test_what_is_counted_about_a_units_flagged_lines() -> None:
    lines = ("int n;", "", "}", "copy(a, b);", "copy(a, b);", "n = 1;", "past the cap")
    text = PartitionUnit(lines=lines, flagged=frozenset({2, 3, 4, 6, 7}))
    # Six scored lines; line 6 has no features; the model's top line is the blank one.
    unit = _unit(
        "u", [0.1, 9.0, 0.3, 0.2, 0.4, -50.0], [2, 3, 4, 6], [True, True, True, True, True, False]
    )
    seen = failures.observe(unit, text, k=1)
    assert seen == {
        "scored_lines": 6,
        "text_lines": 7,
        "flagged_inside_cap": 4,
        "flagged_past_cap": 1,
        "flagged_blank": 1,
        "flagged_unscoreable": 1,
        "flagged_syntax_only": 1,
        "flagged_text_repeats": 1,
        "every_target_degenerate": 0,
        "top_ranked_blank": 1,
        "top_k_near_flagged": 1,
    }
    nothing_to_find = PartitionUnit(lines=("x = 1;", "", "}"), flagged=frozenset({2, 3}))
    seen = failures.observe(_unit("v", [1.0, 0.5, 0.2], [2, 3]), nothing_to_find, k=1)
    assert seen["every_target_degenerate"] == 1


@pytest.mark.parametrize(("top_line", "near"), [(10, 1), (7, 1), (13, 1), (6, 0), (14, 0)])
def test_near_a_flagged_line_means_within_three_lines_of_it(top_line: int, near: int) -> None:
    text = PartitionUnit(lines=tuple(f"s{i}();" for i in range(20)), flagged=frozenset({10}))
    logits = [0.0] * 20
    logits[top_line - 1] = 5.0
    assert failures.observe(_unit("u", logits, [10]), text, k=1)["top_k_near_flagged"] == near


def test_a_kinds_rate_says_how_few_units_it_rests_on() -> None:
    partition = {
        "big": PartitionUnit(
            lines=("#if A", "#else", "#endif", "x = 1;"), flagged=frozenset({1, 2, 3})
        ),
        "small": PartitionUnit(lines=("#if B", "y = 2;", "z = 3;"), flagged=frozenset({2})),
    }
    scores = {
        seed: {
            "big": _unit("big", [0.1, 0.2, 0.3, 9.0], [1, 2, 3]),
            "small": _unit("small", top, [2]),
        }
        for seed, top in ((42, [0.0, 5.0, 1.0]), (1, [5.0, 0.0, 1.0]))
    }
    table = failures.kind_table(scores, partition, ["big", "small"], k=1)
    preprocessor = table["preprocessor"]
    assert (preprocessor["lines"], preprocessor["flagged"], preprocessor["flagged_units"]) == (
        4,
        3,
        1,
    )
    assert preprocessor["largest_unit_share"] == 1.0
    assert preprocessor["found_in_top_k"] == 0.0
    # One flagged assignment, top-ranked under seed 42 and not under seed 1.
    assert table["assignment"]["flagged"] == 1
    assert table["assignment"]["found_in_top_k"] == 0.5


# --- end to end, on a partition of twelve-line functions --------------------------------------

SOURCE = "\n".join(
    ["void f(char *n) {", "char buf[8];", "int i;", "strcpy(buf, n);"]
    + [f"g{i}();" for i in range(7)]
    + ["}"]
)
UNIT_IDS = [f"unit-{index}" for index in range(6)]


def _build(
    tmp_path: Path, *, found_by: dict[int, set[str]], strip: bool = False, held_out: bool = False
) -> dict[str, Path]:
    """Write a shard and one cross-attention score file per seed.

    The fourth line of every unit is flagged. A seed ranks it first in the units named in
    ``found_by[seed]`` and last in the others.
    """
    cache_dir, scores_dir = tmp_path / "cache", tmp_path / "scores"
    units = [
        derive_line_labels(
            unit_id=unit_id,
            source=SOURCE,
            target=1,
            flaw_line_index="3",
            flaw_line="strcpy(buf, n);",
            metadata={"project": "p", "CWE ID": "CWE-120"},
        )
        for unit_id in UNIT_IDS
    ]
    write_shards(units, dict.fromkeys(UNIT_IDS, "val"), cache_dir)
    scores_dir.mkdir()
    for seed, found in found_by.items():
        run_id = f"{seed:032x}"
        scored = []
        for unit_id in UNIT_IDS:
            logits = np.linspace(1.0, 2.0, 12)
            logits[3] = 9.0 if unit_id in found else -9.0
            scored.append(_unit(unit_id, logits.tolist(), [4]))
        save_scores(scores_dir / f"{run_id}_val.npz", scored)
        manifest = {
            "source_run": run_id,
            "seed": seed,
            "fusion_kind": "cross_attention",
            "fusion_direction": "",
            "frozen": False,
            "split": "val",
            "held_out": held_out,
            "strip_comments": strip,
            "strip_comments_at_training": strip,
        }
        (scores_dir / f"{run_id}_val.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return {"cache": cache_dir, "scores": scores_dir}


def _draw(paths: dict[str, Path], **options: Any) -> dict[str, Any]:
    arm = "cross_attention_retrained" if options.pop("strip", False) else "cross_attention"
    payload: dict[str, Any] = failures.draw_cases(
        paths["scores"],
        paths["cache"],
        arm=arm,
        strip=arm.endswith("retrained"),
        normalize=True,
        **options,
    )
    return payload


def test_only_units_every_seed_misses_are_drawn_and_the_sheet_shows_them(tmp_path: Path) -> None:
    found_by = {42: {"unit-0", "unit-1"}, 1: {"unit-0", "unit-2"}, 2: {"unit-0"}}
    payload = _draw(_build(tmp_path, found_by=found_by), n_cases=2)
    assert payload["ranked_units"] == 6
    assert payload["failure_units"] == ["unit-3", "unit-4", "unit-5"]
    assert len(payload["cases"]) == 2
    assert {case["unit_id"] for case in payload["cases"]} <= {"unit-3", "unit-4", "unit-5"}
    case = payload["cases"][0]
    assert case["first_flagged_rank"] == {"1": 12, "2": 12, "42": 12}
    assert case["flagged"] == [
        {
            "line": 4,
            "ranks": {"1": 12, "2": 12, "42": 12},
            "kind": "call",
            "text": "strcpy(buf, n);",
        }
    ]
    assert (case["project"], case["cwe"]) == ("p", "CWE-120")
    payload["run_id"] = "r"
    sheet = failures.render_markdown(payload)
    assert "**3 of 6 ranked units.**" in sheet
    assert "L   4   12 /  12 /  12  [call]  strcpy(buf, n);" in sheet
    assert sheet.count("\ncategory:\n") == 2


def test_the_draw_is_a_function_of_the_unit_ids_and_the_seed(tmp_path: Path) -> None:
    paths = _build(tmp_path, found_by={42: set(), 1: set(), 2: set()})
    first = [case["unit_id"] for case in _draw(paths, n_cases=3)["cases"]]
    assert first == [case["unit_id"] for case in _draw(paths, n_cases=3)["cases"]]
    others = {
        tuple(case["unit_id"] for case in _draw(paths, n_cases=3, sample_seed=seed)["cases"])
        for seed in range(8)
    }
    assert len(others) > 1


def test_an_arm_short_of_a_seed_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="required seeds"):
        _draw(_build(tmp_path, found_by={42: set(), 1: set()}))


def test_held_out_scores_and_scores_of_the_other_text_are_refused(tmp_path: Path) -> None:
    held_out = _build(tmp_path / "a", found_by={42: set(), 1: set(), 2: set()}, held_out=True)
    with pytest.raises(ConfigError, match="held-out"):
        _draw(held_out)
    stripped = _build(tmp_path / "b", found_by={42: set(), 1: set(), 2: set()}, strip=True)
    with pytest.raises(ConfigError, match="other text variant"):
        failures.draw_cases(
            stripped["scores"],
            stripped["cache"],
            arm="cross_attention_retrained",
            strip=False,
            normalize=True,
        )


def test_no_partition_but_validation_can_be_asked_for(isolated_cfg: Any) -> None:
    cfg = isolated_cfg("+failures.split=test")
    with pytest.raises(ConfigError, match="validation only"):
        failures.sample(cfg)
