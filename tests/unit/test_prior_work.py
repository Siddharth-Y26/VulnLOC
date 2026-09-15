"""The quarantine around published numbers (``docs/prior_work.yaml``).

``Rules.md`` §5.7 says baselines are reproduced, not quoted, and that anything copied from
a paper is labelled ``(reported)`` and never compared directly to ours as if it had been
run under our splits. That is a rule about *presentation*, which is the kind of rule that
decays quietly: a caveat gets trimmed to fit a slide, a value is moved into a chart because
the chart looked empty, and the deck now claims a comparison the data does not support.

So the invariants are pinned here rather than trusted. The load-bearing one is the last
test: **a quoted number must never appear inside a figure.** A table can carry the reason a
row does not line up; a bar on a shared axis cannot, and the axis makes the claim anyway.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "make_review3_figures.py"
CATALOGUE = REPO_ROOT / "docs" / "prior_work.yaml"
FIGURES = REPO_ROOT / "artifacts" / "review3" / "figures"


def _load() -> Any:
    """Import ``scripts/make_review3_figures.py`` by path — ``scripts/`` is not a package."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_make_review3_figures", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


make_review3_figures = _load()


@pytest.fixture(scope="module")
def catalogue() -> dict[str, Any]:
    return make_review3_figures._prior_work(CATALOGUE)


def _all_values(catalogue: dict[str, Any]) -> list[str]:
    """Every quoted number in the file, as it is written there."""
    out = []
    for group in ("systems", "independent_remeasurement"):
        for entry in catalogue[group]:
            out.extend(str(metric["value"]) for metric in entry["metrics"])
    return out


def _rendered_text(svg: str) -> list[str]:
    """Every string a reader actually sees in a figure.

    Deliberately *not* the raw SVG: geometry is full of numbers, and a coordinate that
    happens to read ``193.7`` is not a claim about anyone's results. Axis ticks are skipped
    for the same reason — a tick at 0.65 is our scale, not LineVul's Top-10.
    """
    shown = [m.group(1) for m in re.finditer(r"<(?:title|desc)[^>]*>([^<]*)<", svg)]
    for match in re.finditer(r"<text([^>]*)>([^<]*)</text>", svg):
        attrs, body = match.groups()
        if 'class="tick' in attrs:
            continue
        shown.append(body)
    return shown


# --- the catalogue itself -------------------------------------------------------------


def test_every_quoted_system_says_where_it_came_from_and_why_it_does_not_line_up(
    catalogue: dict[str, Any],
) -> None:
    """A number without a source is a number from memory, which §0.1 forbids outright."""
    for entry in catalogue["systems"]:
        for field in ("citation", "url", "approach", "split", "source_of_values"):
            assert entry.get(field), f"{entry['system']} is missing {field}"
        assert entry["not_comparable_because"].strip(), entry["system"]
        assert entry["metrics"], entry["system"]
        for metric in entry["metrics"]:
            assert metric.get("name") and metric.get("value") is not None


def test_quoted_values_are_strings_so_published_precision_survives(
    catalogue: dict[str, Any],
) -> None:
    """LineVD reports 0.360, not 0.36.

    YAML parses a bare ``0.360`` as a float and renders it back as ``0.36``, silently
    dropping a significant figure someone else chose to publish. These values are never
    arithmetic — they are quotations — so they are stored as text.
    """
    for value in _all_values(catalogue):
        assert isinstance(value, str), f"{value!r} must be quoted in the YAML"


def test_the_hypothesis_is_stored_as_a_hypothesis(catalogue: dict[str, Any]) -> None:
    """ADR-006's claim about published baselines is untested and must stay labelled so."""
    hypothesis = catalogue["hypothesis"]
    assert hypothesis["status"] == "UNTESTED"
    assert hypothesis["what_would_settle_it"].strip()
    assert hypothesis["how_to_present_it_until_then"].strip()


# --- how it renders -------------------------------------------------------------------


def test_every_quoted_value_renders_labelled_reported(catalogue: dict[str, Any]) -> None:
    """§5.7's label is applied by the renderer, so it cannot be forgotten per row."""
    semantic = {
        "headline": {"eval/line/f1": {"mean": 0.3692}, "eval/line/ifa_mean": {"mean": 11.93}}
    }

    markdown = make_review3_figures._prior_work_markdown(catalogue, semantic)

    for entry in catalogue["systems"]:
        for metric in entry["metrics"]:
            assert f"**{metric['value']}** (reported)" in markdown, metric["name"]


def test_every_caveat_travels_with_its_row(catalogue: dict[str, Any]) -> None:
    """A two-column table of "them" and "us" is a comparison whatever the prose says."""
    semantic = {
        "headline": {"eval/line/f1": {"mean": 0.3692}, "eval/line/ifa_mean": {"mean": 11.93}}
    }

    markdown = make_review3_figures._prior_work_markdown(catalogue, semantic)

    for entry in catalogue["systems"]:
        assert entry["not_comparable_because"].split(".")[0].strip() in markdown
    for caveat in catalogue["shared_caveats"]:
        assert caveat.split(".")[0].strip() in markdown


def test_our_own_numbers_in_that_section_come_from_the_seed_report(
    catalogue: dict[str, Any],
) -> None:
    """The half of the section that is ours still obeys §0.3 — it is read, not typed."""
    semantic = {
        "headline": {"eval/line/f1": {"mean": 0.1234}, "eval/line/ifa_mean": {"mean": 99.5}}
    }

    markdown = make_review3_figures._prior_work_markdown(catalogue, semantic)

    assert "0.1234" in markdown
    assert "99.50" in markdown


# --- the one that matters -------------------------------------------------------------


@pytest.mark.skipif(not FIGURES.is_dir(), reason="figures not generated in this checkout")
def test_no_quoted_number_ever_reaches_a_figure(catalogue: dict[str, Any]) -> None:
    """The rule a chart cannot express.

    A caveat column can say "different split, different denominator, possibly different
    inputs". A shared axis says "these are the same kind of thing" and has no room to take
    it back. ``docs/prior_work.yaml`` itself carries the evidence for why that matters:
    LineVul is reported at Top-10 0.65 in its own paper and re-measured at 0.740 by a
    later one, a spread wider than most improvements in this literature.
    """
    values = _all_values(catalogue)
    assert list(FIGURES.glob("*.svg")), "no figures to check"

    for svg in sorted(FIGURES.glob("*.svg")):
        for shown in _rendered_text(svg.read_text(encoding="utf-8")):
            for value in values:
                assert not re.search(
                    rf"(?<![\d.]){re.escape(value)}(?![\d.])", shown
                ), f"{svg.name} shows the reported value {value} in {shown!r}"
