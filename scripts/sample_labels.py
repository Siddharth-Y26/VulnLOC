"""Label-audit sheet generator — Phase 1 exit criterion (human task).

*20 randomly sampled line labels manually inspected by a human and judged correct.*
This script produces the sheet; **a person fills in the verdicts.** Nothing here
judges a label, and no downstream code reads the verdicts — they are evidence for
`DATA_CARD.md` and `Memory.md`, written by whoever did the reading.

The sample is drawn by content hash (see
:func:`vulnloc.data.labeling.sample_for_review`), so it is fixed before anyone looks at
the data and a second reviewer receives exactly the same functions.

Usage::

    python scripts/sample_labels.py
    python scripts/sample_labels.py data.label_audit.k=40 data.label_audit.seed=7

Outputs ``label_audit.md`` (the sheet) and ``label_audit.json`` (the sample's identity,
so the review can be tied to a specific draw, plus the corpus-wide label-derivation
counters measured on the same pass).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from vulnloc.data.labeling import (
    LabelledFunction,
    LabellingStats,
    sample_for_review,
    split_source_lines,
)
from vulnloc.data.registry import DATASET_REGISTRY
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

# Long functions are windowed around their flagged lines: a reviewer needs the
# surrounding statements to judge a label, not the whole 400-line body.
_FULL_LISTING_MAX_LINES = 120
_CONTEXT_LINES = 12


def _windows(flagged: list[int], n_lines: int) -> list[tuple[int, int]]:
    """Merge per-label context windows into non-overlapping ranges.

    Args:
        flagged: 1-indexed flagged line numbers.
        n_lines: Total lines in the function.

    Returns:
        Inclusive ``(start, end)`` line ranges, in order.
    """
    merged: list[tuple[int, int]] = []
    for line in sorted(flagged):
        start, end = max(1, line - _CONTEXT_LINES), min(n_lines, line + _CONTEXT_LINES)
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def render_listing(unit: LabelledFunction) -> str:
    """Render the function with flagged lines marked.

    Args:
        unit: The function under review.

    Returns:
        A fenced code block. Flagged lines are prefixed ``>>``; elisions are marked.
    """
    lines = split_source_lines(unit.source)
    flagged = sorted(unit.vulnerable_lines)
    ranges = (
        [(1, len(lines))]
        if len(lines) <= _FULL_LISTING_MAX_LINES
        else _windows(flagged, len(lines))
    )

    width = len(str(len(lines)))
    out: list[str] = []
    previous_end = 0
    for start, end in ranges:
        if start > previous_end + 1:
            out.append(f"   {'.' * width}  ... {start - previous_end - 1} lines omitted ...")
        for number in range(start, end + 1):
            marker = ">>" if number in unit.vulnerable_lines else "  "
            out.append(f"{marker} {number:>{width}} | {lines[number - 1]}")
        previous_end = end
    if previous_end < len(lines):
        out.append(f"   {'.' * width}  ... {len(lines) - previous_end} lines omitted ...")

    return "```c\n" + "\n".join(out) + "\n```"


def render_entry(index: int, unit: LabelledFunction) -> str:
    """Render one review item.

    Args:
        index: 1-based position in the sheet.
        unit: The function under review.

    Returns:
        A Markdown section with provenance, the listing, and a blank verdict.
    """
    meta = unit.metadata
    provenance = " · ".join(
        f"**{label}:** {meta.get(key) or '—'}"
        for label, key in (
            ("project", "project"),
            ("CVE", "CVE ID"),
            ("CWE", "CWE ID"),
            ("date", "Publish Date"),
            ("commit", "commit_id"),
        )
    )
    issues = ", ".join(unit.issues) if unit.issues else "none"
    return "\n".join(
        [
            f"### {index}. `{unit.unit_id}`",
            "",
            provenance,
            "",
            f"**Flagged lines:** {', '.join(str(n) for n in sorted(unit.vulnerable_lines))} "
            f"of {unit.n_lines} · **derivation issues:** {issues}",
            "",
            render_listing(unit),
            "",
            "| | |",
            "|---|---|",
            "| **Verdict** | `correct` / `wrong` / `unsure` — *replace this* |",
            "| **Reason** | *e.g. the flagged line is the overflow; or: this is a "
            "refactor swept in by the fix commit* |",
            "",
        ]
    )


def render_sheet(units: list[LabelledFunction], *, seed: int, dataset: str) -> str:
    """Render the full review sheet.

    Args:
        units: The sampled functions.
        seed: Selection seed, recorded so the draw can be reproduced.
        dataset: Dataset the sample came from.

    Returns:
        The Markdown document.
    """
    header = [
        "# Line-label audit sheet",
        "",
        f"**Sample:** {len(units)} line-labelled functions from `{dataset}`, "
        f"drawn by content hash at seed `{seed}` "
        "(`vulnloc.data.labeling.sample_for_review`).",
        "",
        "**This is a human task** — `Phases.md` Phase 1 requires sampled line labels to "
        "be inspected by a person and judged correct. Nothing in the codebase fills this in.",
        "",
        "## How to judge one",
        "",
        "The `labels_v1` claim is: *the lines deleted or modified by the "
        "vulnerability-fixing commit are the vulnerable lines.* For each function ask "
        "whether the flagged lines are where the vulnerability actually lives.",
        "",
        "- `correct` — the flagged lines contain the flaw, or the check/assignment whose "
        "absence is the flaw.",
        "- `wrong` — the flagged lines are a refactor, a rename, a comment, or unrelated "
        "cleanup swept into the fix commit; or the flaw is plainly elsewhere.",
        "- `unsure` — insufficient context. Say what was missing; do not guess.",
        "",
        "A `wrong` verdict is a *result*, not a failure — the measured Big-Vul label-noise "
        "rate is 19.2% (ADR-003), so roughly four of twenty landing on `wrong` would be "
        "consistent with what we already know. Record the count honestly in "
        "`docs/DATA_CARD.md` and `Memory.md`.",
        "",
        "## Tally (fill in when done)",
        "",
        "| Reviewer | Date | correct | wrong | unsure |",
        "|---|---|---|---|---|",
        "| | | | | |",
        "",
        "---",
        "",
    ]
    return "\n".join(header) + "\n".join(
        render_entry(i, unit) for i, unit in enumerate(units, start=1)
    )


def _counting(
    units: Iterator[LabelledFunction], stats: LabellingStats
) -> Iterator[LabelledFunction]:
    """Pass functions through, folding each into ``stats`` on the way.

    Sampling already streams the whole corpus, and the corpus is 11 GB. Deriving the
    quality counters in the same pass makes them free; running them separately would
    cost a second traversal for numbers that fall out of work already being done.

    Args:
        units: The stream to observe.
        stats: Counters to update.

    Yields:
        Each function, unchanged.
    """
    for unit in units:
        stats.record(unit)
        yield unit


def sample(cfg: DictConfig) -> dict[str, Any]:
    """Draw the sample, measure label-derivation quality, and write the review sheet.

    Args:
        cfg: Composed Hydra config. Requires ``data.raw_dir`` and ``data.label_audit``.

    Returns:
        The sample's identity record and the corpus-wide quality counters.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    settings = cfg.data.label_audit
    dataset = str(settings.dataset)

    loader = DATASET_REGISTRY.get(dataset)(
        cfg.data.raw_dir, splits=tuple(str(s) for s in settings.source_splits)
    )
    stats = LabellingStats()
    units = sample_for_review(
        _counting(loader.iter_functions(), stats), k=int(settings.k), seed=int(settings.seed)
    )

    output = Path(settings.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "label_audit.md").write_text(
        render_sheet(units, seed=int(settings.seed), dataset=dataset), encoding="utf-8"
    )

    record = {
        "dataset": dataset,
        "seed": int(settings.seed),
        "k": int(settings.k),
        # Corpus-wide derivation quality. `flagged_lines` here counts *distinct,
        # in-range, verified* line numbers; the audit's figure counts raw index tokens.
        # The two differing is a finding about the corpus, not a bug — see Memory.md §7.
        "labelling": stats.as_dict(),
        "sampled": [
            {
                "unit_id": u.unit_id,
                "project": u.metadata.get("project"),
                "cve": u.metadata.get("CVE ID"),
                "cwe": u.metadata.get("CWE ID"),
                "n_lines": u.n_lines,
                "vulnerable_lines": sorted(u.vulnerable_lines),
                "issues": list(u.issues),
            }
            for u in units
        ],
    }
    (output / "label_audit.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _LOGGER.info(
        "Label audit sheet written",
        extra={
            "sampled": len(units),
            "path": str(output / "label_audit.md"),
            **stats.as_dict(),
        },
    )
    return record


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    sample(cfg)


if __name__ == "__main__":
    main()
