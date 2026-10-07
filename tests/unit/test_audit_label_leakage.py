"""Guards the formatting-leak audit's tells and its two verdicts.

The audit is the instrument that found both formatting leaks in this corpus (ADR-006, and
the comment terminator that lost its slash, 2026-10-07). An instrument that misreads is
worse than none, so what is pinned here is what each tell fires on, and the case the
audit's first version could not see at all: a tell that is *never* right, which precision
reports as zero and which is just as readable as one that is always right.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "audit_label_leakage.py"


def _load() -> Any:
    """Import ``scripts/audit_label_leakage.py`` by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_audit_label_leakage", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load()


def unit(lines: list[str], flagged: set[int]) -> SimpleNamespace:
    return SimpleNamespace(
        source="\n".join(lines), n_lines=len(lines), vulnerable_lines=frozenset(flagged)
    )


@pytest.mark.parametrize(
    "line",
    [
        "qio_channel_wait(ioc);  /* needed by nbd_reply_ready.  *",
        "* and consider using that in this branch *",
        "*ch = '\\0';  /* and replace with null-termination *",
        "   x = 1; /* note *   ",
    ],
)
def test_a_comment_that_ends_with_a_bare_star_is_the_tell(line):
    assert audit.comment_end_lost_slash(line)


@pytest.mark.parametrize(
    "line",
    [
        "x = 1;  /* note */",  # intact
        "/**",  # a doc-comment opener ends in `*` in any corpus
        "/*",
        "/* **** *****",  # a banner
        "foo(char *",  # code that ends in `*`, no comment on the line
        "y = a *",
        "*",
        "",
    ],
)
def test_lines_that_merely_end_in_a_star_are_not(line):
    assert not audit.comment_end_lost_slash(line)


def test_the_two_comment_tells_are_registered_and_disjoint():
    lost, intact = audit.TELLS["comment_end_lost_slash"], audit.TELLS["comment_end_intact"]

    assert lost("f();  /* note *") and not intact("f();  /* note *")
    assert intact("f();  /* note */") and not lost("f();  /* note */")


def test_score_split_counts_the_units_a_tell_can_answer():
    units = [
        unit(["a();", "b();  /* why *", "c();  /* also *"], {2, 3}),
        unit(["d();", "e();  /* fine */"], {1}),
    ]

    scores, judged, flagged = audit.score_split(units, normalize=True, n_lines=512)

    lost = scores["comment_end_lost_slash"]
    assert (judged, flagged) == (5, 3)
    assert (lost.tp, lost.fp, lost.fn) == (2, 0, 1)
    assert lost.units_with_true_positive == 1, "two hits in one unit are one unit"
    intact = scores["comment_end_intact"]
    assert (intact.tp, intact.fp) == (0, 1)


def test_lines_past_the_line_cap_are_not_judged():
    units = [unit(["a();", "b();  /* why *"], {2})]

    scores, judged, _ = audit.score_split(units, normalize=True, n_lines=1)

    assert judged == 1
    assert scores["comment_end_lost_slash"].tp == 0


def test_a_tell_that_is_always_right_leaks():
    score = audit.TellScore(tp=40, fp=0, fn=100)

    report = score.to_dict(base_rate=6.0)

    assert report["leaks"] is True
    assert report["excludes_label"] is False
    assert report["precision"] == 1.0


def test_a_tell_that_is_never_right_is_reported_as_excluding_the_label():
    # 2,444 lines at a 6.07% base rate should hold about 148 flagged ones. Finding none is
    # not "found nothing": every one of those lines is readable as unflagged.
    score = audit.TellScore(tp=0, fp=2444, fn=3386)

    report = score.to_dict(base_rate=6.07)

    assert report["leaks"] is False
    assert report["excludes_label"] is True
    assert report["expected_true_positives_at_base_rate"] == pytest.approx(148.4, abs=0.1)


def test_a_tell_that_fires_too_rarely_to_judge_is_neither():
    score = audit.TellScore(tp=0, fp=50, fn=10)

    report = score.to_dict(base_rate=6.0)

    assert report["leaks"] is False
    assert report["excludes_label"] is False, "3 expected hits is not enough to call zero odd"


def test_a_tell_at_the_base_rate_is_neither():
    score = audit.TellScore(tp=60, fp=940, fn=0)

    report = score.to_dict(base_rate=6.0)

    assert report["leaks"] is False
    assert report["excludes_label"] is False
