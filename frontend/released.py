"""The Findings tab's model: the released bundle, run exactly as the packaged API runs it.

**Decided by the project owner on 2026-10-12**: one site, in this tool's layout, holding
everything — so Findings is no longer the Phase 3 semantic model alone but the model of
record (the fused cross-attention checkpoint, comments stripped, calibrated), loaded from
the serving bundle through :class:`vulnloc.serving.inference.Analyzer`. Nothing is
re-implemented here: the scores, the cut-off, the explanations and their verification are
the service's own, and this module only arranges them for the page.

Two things the owner asked of that arrangement, the same day:

- **No count of flagged lines as a headline.** The model's cut-off is reached by most lines
  of a short function whether or not anything is wrong with it (``CLAUDE.md`` landmine
  #74), so a count reads as a verdict the model cannot give.
- **A reason is shown only for a line a known weakness pattern matches.** A line above the
  cut-off that no pattern matches gets no card. It keeps its rank in the whole-function
  view, where the model's ordering is still the thing to read.

What that leaves unsaid is said on the page from a measured record, never typed in
(:func:`pattern_record`): a pattern is a text match, and it matches harmless functions
nearly as often as vulnerable ones.

The weights are loaded on the CPU, as the service loads them (``PRD.md`` FR-24).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from frontend.scoring import IMPORT_LOCK

__all__ = [
    "BUNDLE",
    "BUNDLE_ENV",
    "CONTEXT_LINES",
    "MIN_ATTENTION_WEIGHT",
    "PATTERN_RECORD",
    "bundle_dir",
    "pattern_record",
    "released_findings",
    "version",
]

_REPO = Path(__file__).resolve().parent.parent

BUNDLE = _REPO / "artifacts" / "serving" / "ba7dcc5f-r2"
"""The bundle of record (``docs/Memory.md`` §1.33): export ``ef031969…`` of run ``ba7dcc5f…``."""

BUNDLE_ENV = "VULNLOC_BUNDLE"
"""Names another bundle directory to load — the variable the packaged service reads."""

PATTERN_RECORD = _REPO / "artifacts" / "screening" / "verdicts_val.json"
"""How often a pattern matched on vulnerable and on harmless ``val`` functions
(``docs/VerdictComparison.md``, run ``6c942e6c…``) — an experiment, not a number of record."""

CONTEXT_LINES = 2
"""Source lines shown either side of a card's line."""

MIN_ATTENTION_WEIGHT = 0.005
"""An attention weight that rounds to 0.00 says nothing and is not listed."""

_ANALYZER: Any = None


def bundle_dir() -> Path:
    """The bundle this tool loads.

    Returns:
        ``VULNLOC_BUNDLE`` when set, otherwise the bundle of record.
    """
    named = os.environ.get(BUNDLE_ENV)
    return Path(named) if named else BUNDLE


def _load() -> Any:
    """Load and verify the bundle once.

    Returns:
        The service's ``Analyzer``, on the CPU.

    Raises:
        FileNotFoundError: If the bundle directory does not exist.
    """
    global _ANALYZER
    with IMPORT_LOCK:
        if _ANALYZER is None:
            root = bundle_dir()
            if not root.is_dir():
                raise FileNotFoundError(
                    f"Serving bundle missing: {root}. Build it with "
                    "scripts/export_serving_bundle.py, or set VULNLOC_BUNDLE."
                )
            from vulnloc.serving.inference import Analyzer

            _ANALYZER = Analyzer.load(root)
        return _ANALYZER


def version() -> dict[str, Any]:
    """Which checkpoint answers the Findings tab.

    Returns:
        The bundle's provenance, read from its manifest.
    """
    manifest = _load().bundle.manifest
    return {
        "name": "Released model (CodeBERT + GCN, cross-attention, fine-tuned)",
        "bundle": bundle_dir().name,
        "run_id": str(manifest["run_id"]),
        "git_sha": str(manifest["git_sha"]),
        "fusion": str(manifest["fusion"]["kind"]),
        "seed": int(manifest.get("seed", 0)),
        "threshold": float(manifest["calibration"]["threshold"]),
        "calibrator": str(manifest["calibration"]["kind"]),
        "strip_comments": bool(manifest["text"]["strip_comments"]),
        "line_cap": int(manifest["llm"]["n_lines"]),
        "exported_at": str(manifest.get("exported_at", "")),
        "device": "cpu",
    }


def pattern_record() -> dict[str, Any] | None:
    """How often a weakness pattern matched, on functions of both kinds — read from disk.

    The record is quoted only for the bundle it was measured with: another checkpoint
    reaches its cut-off on other lines, so its patterns match on other functions.

    Returns:
        ``{run_id, split, vulnerable: {n, raised}, not_vulnerable: {n, raised}}``, or
        ``None`` when the record has not been generated or belongs to another bundle.
    """
    if not PATTERN_RECORD.exists():
        return None
    record = json.loads(PATTERN_RECORD.read_text(encoding="utf-8"))
    if Path(str(record.get("bundle", ""))).name != bundle_dir().name:
        return None
    pattern = record["headlines"]["pattern"]
    return {
        "run_id": str(record["run_id"]),
        "split": "val",
        "held_out": bool(record.get("held_out", False)),
        "vulnerable": {
            "n": int(pattern["vulnerable"]["n"]),
            "raised": int(pattern["vulnerable"]["raised"]),
        },
        "not_vulnerable": {
            "n": int(pattern["not_vulnerable"]["n"]),
            "raised": int(pattern["not_vulnerable"]["raised"]),
        },
    }


def _line_scores(analyzer: Any, analysis: Any, lines: list[str]) -> list[dict[str, Any]]:
    """Every line's calibrated score and rank, as the service computed them.

    The service's report lists only the five highest-ranked lines and the flagged ones; the
    whole-function view needs all of them. They are rebuilt from the analysis' own logits
    with the service's calibrator and its rule for which lines are offered, and the result
    is checked against the report before it is returned.

    Args:
        analyzer: The loaded analyzer.
        analysis: The analysis of this paste.
        lines: The unit's lines as the model saw them.

    Returns:
        One ``{line, score, rank, flagged}`` per source line; ``score`` and ``rank`` are
        ``None`` for a line the model was not offered.

    Raises:
        RuntimeError: If the rebuilt ranking disagrees with the service's report.
    """
    import torch

    from vulnloc.explanation.types import Prediction

    logits = torch.from_numpy(analysis.logits)  # [n]
    n = int(logits.shape[0])
    offered = [
        bool(usable) and bool(text.strip())
        for usable, text in zip(analysis.scoreable.tolist(), lines[:n], strict=True)
    ]
    probabilities = analyzer.calibrator.probabilities(logits.double()).tolist()
    pred = Prediction(
        unit_id="paste",
        probabilities=tuple(
            float(value) if usable else 0.0
            for value, usable in zip(probabilities, offered, strict=True)
        ),
        scoreable=tuple(offered),
        threshold=analyzer.threshold,
    )
    reported = [(entry["line"], entry["rank"]) for entry in analysis.report["ranked_lines"]]
    rebuilt = [(line, rank) for rank, line in enumerate(pred.ranking[: len(reported)], start=1)]
    if reported != rebuilt:
        # Two orderings of one function on one page would be worse than no page.
        raise RuntimeError(f"the rebuilt ranking {rebuilt} is not the service's ranking {reported}")
    scored = set(pred.ranking)
    return [
        {
            "line": line,
            "score": pred.score(line) if line in scored else None,
            "rank": pred.rank(line) if line in scored else None,
            "flagged": pred.is_flagged(line),
        }
        for line in range(1, len(lines) + 1)
    ]


def released_findings(code: str) -> dict[str, Any]:
    """The Findings tab: the released model's ranking, and a reason where a pattern matches.

    Args:
        code: The pasted source.

    Returns:
        A JSON-serializable report. ``cards`` holds one entry per line that is above the
        model's cut-off **and** matched by a known weakness pattern, most suspicious first;
        ``scores`` holds every line's score and rank; ``cwe_659`` holds MITRE's CWE-659 view
        as reference data, which the page does not draw (:mod:`frontend.cwe_659`).

    Raises:
        ValueError: If ``code`` is empty.
    """
    from frontend.cwe_659 import reference as cwe_659_reference
    from frontend.fix_templates import advice_for
    from vulnloc.representation.units import CodeUnit

    if not code.strip():
        raise ValueError("Paste a function first — there is nothing to analyse.")

    analyzer = _load()
    analysis = analyzer.analyze(code, language="C", unit_id="paste")
    report = analysis.report
    # The text the model saw: edge whitespace and comments removed, line count kept.
    unit = CodeUnit.from_source(
        "paste",
        code,
        language="C",
        normalize_whitespace=analyzer.normalize_whitespace,
        strip_comments=analyzer.strip_comments,
    )
    lines = list(unit.lines)

    def context(line: int) -> list[dict[str, Any]]:
        first = max(1, line - CONTEXT_LINES)
        last = min(len(lines), line + CONTEXT_LINES)
        return [{"line": n, "text": lines[n - 1]} for n in range(first, last + 1)]

    cards = [
        {
            "line": finding["line"],
            "rank": finding["rank"],
            "score": finding["score"],
            "text": finding["code"],
            "context": context(finding["line"]),
            "pattern": finding["pattern"],
            "label": finding["label"],
            "cwe": finding["cwe"],
            # The service's own sentence, verified against the prediction. Shown whole.
            "explanation": finding["explanation"],
            "advice": advice_for(finding["pattern"]),
            "attended": [
                attended
                for attended in finding["attended_lines"]
                if attended["weight"] >= MIN_ATTENTION_WEIGHT
            ],
            "tokens": finding["attention_top_tokens"],
            "dataflow_path": finding["dataflow_path"],
            # One name per hop; a name carried over two hops is said once, in path order.
            "dataflow_symbols": list(dict.fromkeys(finding["dataflow_symbols"])),
        }
        for finding in report["findings"]
        if finding["pattern"]
    ]
    card_lines = {card["line"] for card in cards}
    return {
        "lines": lines,
        "scores": _line_scores(analyzer, analysis, lines),
        "cards": cards,
        "top": [
            {**entry, "has_card": entry["line"] in card_lines} for entry in report["ranked_lines"]
        ],
        "threshold": report["threshold"],
        "patterns": report["patterns"],
        "coverage": report["coverage"],
        "measured": report["measured"],
        "pattern_record": pattern_record(),
        "model": version(),
        "seconds": analysis.diagnostics["seconds"],
        # Reference data only: no pattern comes from it, and the page does not draw it.
        "cwe_659": cwe_659_reference(report["patterns"]["checked"], cards),
    }
