r"""The cases for the failure taxonomy — drawn by rule, from stored validation scores.

Phase 8 asks for a failure taxonomy over at least twenty *inspected* cases. Inspecting is a
person's work; what a script can guarantee is that the cases were not picked. So:

* a **failure** is defined before any is read: a unit whose first flagged line the model
  of record ranks outside the top ``k`` on **every** seed — the same ranking the
  evaluation uses (every line a candidate, equal scores in source order). One seed missing
  a function is noise; three seeds missing it is something about the function;
* the cases are drawn from all such units **by content hash**, so the draw is a function
  of the unit ids and a recorded seed and nothing else;
* beside each case go the things that can be *measured* about it — how many of its flagged
  lines are blank after comment stripping, have no features, sit past the line cap,
  consist of syntax alone, or repeat the text of an unflagged line — counted over **all**
  failures, not only the drawn ones. A category a reader assigns covers the drawn cases;
  a count covers every failure. The sheet keeps the two apart.

The sheet shows each function with every line's rank, the flagged lines marked, and two
empty fields, ``category:`` and ``notes:``. It names no category itself.

Validation only. The taxonomy describes the partition everything was selected on; a
held-out partition is refused outright, not gated — nothing about ``test`` or
``clean_eval`` is to be read function by function.

Usage::

    python scripts/sample_failures.py +representation.strip_comments=true \
        +failures.scores_dir=./artifacts/phase6/scores_retrained
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf

from vulnloc.data.cache import iter_shard
from vulnloc.evaluation.evaluator import PartitionUnit, arm_name, load_partition
from vulnloc.training.line_scores import UnitScores, load_scores
from vulnloc.training.tracking import start_run
from vulnloc.utils.config import strip_comments_enabled
from vulnloc.utils.exceptions import ConfigError, DataError
from vulnloc.utils.hashing import stable_sample
from vulnloc.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_LOGGER = get_logger(__name__)

SPLIT = "val"
"""The only partition a failure case is drawn from."""

SYNTAX_ONLY = re.compile(
    r"^(?:[{}()\[\];,\s]|else|do|try|break\s*;|continue\s*;|return\s*;|default\s*:)+$"
)
"""A line that is punctuation or a bare keyword: nothing on it can be wrong by itself."""

OBSERVATIONS = {
    "flagged_past_cap": "has a flagged line past the line cap",
    "flagged_blank": "has a flagged line that is blank after comment stripping",
    "flagged_unscoreable": "has a flagged line the model had no features for",
    "flagged_syntax_only": "has a flagged line that is syntax alone (a brace, a bare keyword)",
    "flagged_text_repeats": "has a flagged line whose text also stands on an unflagged line",
    "every_target_degenerate": (
        "every flagged line inside the cap is blank, unscoreable or syntax alone"
    ),
    "top_ranked_blank": "has a blank line as the model's top-ranked line (the display seed)",
    "top_k_near_flagged": (
        "has one of the model's top-k lines on, or within three lines of, a flagged line "
        "(the display seed)"
    ),
}

NEAR_LINES = 3
"""How far from a flagged line a top-ranked line may be and still count as *near* it."""

_NAME = r"\**[A-Za-z_]\w*(?:\s*\[[^\]]*\])*"
LINE_KINDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("blank", re.compile(r"^$")),
    ("syntax alone", SYNTAX_ONLY),
    ("preprocessor", re.compile(r"^#")),
    (
        "condition or loop header",
        re.compile(r"^(?:\}\s*)?(?:else\s+)?(?:if|for|while|switch)\s*\("),
    ),
    ("return or jump", re.compile(r"^(?:return|goto|break|continue)\b")),
    ("declaration", re.compile(rf"^(?:[A-Za-z_]\w*[\s\*]+)+{_NAME}(?:\s*,\s*{_NAME})*\s*;$")),
    ("assignment", re.compile(r"(?<![=!<>])=(?!=)")),
    ("call", re.compile(r"^[A-Za-z_][\w:.\->\[\]]*\s*\(")),
)
"""What a line *is*, by rule, first match wins; anything else is ``other`` (a continuation
of a statement split over lines, a string literal, a label). The rules read one stripped
line and nothing around it: they are a coarse syntactic sorting, not a parse."""

OTHER_KIND = "other"


def line_kind(text: str) -> str:
    """The syntactic kind of one line, by the rules of :data:`LINE_KINDS`.

    Args:
        text: One source line.

    Returns:
        The first kind whose rule matches the stripped line, or ``other``.
    """
    stripped = text.strip()
    for name, pattern in LINE_KINDS:
        if pattern.search(stripped):
            return name
    return OTHER_KIND


def line_ranks(logits: np.ndarray) -> np.ndarray:
    """Each line's 1-indexed rank, most suspicious first, equal scores in source order.

    Args:
        logits: ``[n_lines]`` line scores.

    Returns:
        ``[n_lines]`` int — the order ``evaluation.metrics.localization`` reports.
    """
    order = np.argsort(-np.asarray(logits, dtype=np.float64), kind="stable")
    ranks = np.empty(order.shape[0], dtype=np.int64)
    ranks[order] = np.arange(1, order.shape[0] + 1)
    return ranks


def first_flagged_rank(unit: UnitScores) -> int | None:
    """Rank of the unit's best-ranked flagged line, or ``None`` if none is inside the cap.

    Args:
        unit: One unit's scores.

    Returns:
        The 1-indexed rank; ``IFA + 1`` for this unit.
    """
    flagged = unit.labels > 0
    if not flagged.any():
        return None
    return int(line_ranks(unit.logits)[flagged].min())


def consistent_misses(ranks: Mapping[int, Mapping[str, int]], k: int) -> list[str]:
    """Units whose first flagged line is outside the top ``k`` on every seed.

    Args:
        ranks: ``seed -> unit_id -> first flagged rank``.
        k: The cut-off.

    Returns:
        Sorted unit ids.

    Raises:
        DataError: If the seeds did not rank the same units — the checkpoints would then
            not be scores of one partition.
    """
    populations = {frozenset(by_unit) for by_unit in ranks.values()}
    if len(populations) != 1:
        raise DataError("the seeds did not rank the same units", seeds=sorted(ranks))
    units = next(iter(populations))
    return sorted(unit for unit in units if all(by_unit[unit] > k for by_unit in ranks.values()))


def observe(unit: UnitScores, text: PartitionUnit, *, k: int) -> dict[str, int]:
    """What can be counted about one unit's flagged lines without judging anything.

    Args:
        unit: The unit's scores under one seed. Only ``top_ranked_blank`` and
            ``top_k_near_flagged`` depend on which.
        text: The unit's lines and every flagged line number.
        k: The cut-off a failure is defined by; ``top_k_near_flagged`` reads the top ``k``.

    Returns:
        A count for each key of :data:`OBSERVATIONS` — 0 or 1 for the last three — plus
        ``scored_lines``, ``text_lines`` and ``flagged_inside_cap``.
    """
    n = int(unit.logits.shape[0])
    inside = sorted(number for number in text.flagged if number <= n)
    unflagged = {text.lines[i].strip() for i in range(n) if (i + 1) not in text.flagged}
    blank = [number for number in inside if not text.lines[number - 1].strip()]
    unscoreable = [number for number in inside if not bool(unit.scoreable[number - 1])]
    syntax = [number for number in inside if SYNTAX_ONLY.match(text.lines[number - 1].strip())]
    repeats = [
        number
        for number in inside
        if text.lines[number - 1].strip() and text.lines[number - 1].strip() in unflagged
    ]
    degenerate = set(blank) | set(unscoreable) | set(syntax)
    ranks = line_ranks(unit.logits)
    top_line = int(np.argmin(ranks))
    top_k = np.flatnonzero(ranks <= k) + 1
    near = any(abs(int(line) - number) <= NEAR_LINES for line in top_k for number in inside)
    return {
        "scored_lines": n,
        "text_lines": len(text.lines),
        "flagged_inside_cap": len(inside),
        "flagged_past_cap": len(text.flagged) - len(inside),
        "flagged_blank": len(blank),
        "flagged_unscoreable": len(unscoreable),
        "flagged_syntax_only": len(syntax),
        "flagged_text_repeats": len(repeats),
        "every_target_degenerate": int(bool(inside) and degenerate == set(inside)),
        "top_ranked_blank": int(not text.lines[top_line].strip()),
        "top_k_near_flagged": int(near),
    }


def kind_table(
    scores: Mapping[int, Mapping[str, UnitScores]],
    partition: Mapping[str, PartitionUnit],
    units: Sequence[str],
    k: int,
) -> dict[str, dict[str, Any]]:
    """Per syntactic kind: how many lines, how many flagged, and how often a flagged one is found.

    Args:
        scores: ``seed -> unit_id -> scores``.
        partition: The partition's text.
        units: The units to count over.
        k: A flagged line is *found* when it is among its unit's top ``k`` lines.

    Returns:
        ``kind -> {"lines", "flagged", "flagged_units", "largest_unit_share",
        "flagged_rate", "found_in_top_k", "found_by_chance"}`` — ``found_in_top_k`` is the
        share of the kind's flagged lines ranked in the top ``k`` of their unit, averaged
        over the seeds (``None`` for a kind with no flagged line); ``found_by_chance`` is
        what a ranking drawn at random would score on the same lines, ``min(1, k / n)`` for
        a line of an ``n``-line unit, averaged — the figure a *found* rate has to beat;
        ``flagged_units`` is how many units hold those flagged lines and
        ``largest_unit_share`` the share of them in the one unit with the most — a rate
        that rests on a handful of functions says so. Counted over scored lines only.
    """
    lines: dict[str, int] = {}
    flagged: dict[str, dict[str, int]] = {}
    found: dict[str, list[int]] = {}
    chance: dict[str, list[float]] = {}
    seeds = sorted(scores)
    for unit_id in units:
        text = partition[unit_id]
        labels = scores[seeds[0]][unit_id].labels > 0
        ranks = [line_ranks(scores[seed][unit_id].logits) for seed in seeds]
        n = int(labels.shape[0])
        for index in range(n):
            kind = line_kind(text.lines[index])
            lines[kind] = lines.get(kind, 0) + 1
            if labels[index]:
                by_unit = flagged.setdefault(kind, {})
                by_unit[unit_id] = by_unit.get(unit_id, 0) + 1
                found.setdefault(kind, []).extend(int(rank[index] <= k) for rank in ranks)
                chance.setdefault(kind, []).append(min(1.0, k / n))
    table: dict[str, dict[str, Any]] = {}
    for kind, count in sorted(lines.items(), key=lambda item: -item[1]):
        by_unit = flagged.get(kind, {})
        n_flagged = sum(by_unit.values())
        table[kind] = {
            "lines": count,
            "flagged": n_flagged,
            "flagged_units": len(by_unit),
            "largest_unit_share": (max(by_unit.values()) / n_flagged) if by_unit else None,
            "flagged_rate": n_flagged / count,
            "found_in_top_k": (sum(found[kind]) / len(found[kind])) if by_unit else None,
            "found_by_chance": (sum(chance[kind]) / n_flagged) if by_unit else None,
        }
    return table


def build_case(
    unit_id: str,
    scores: Mapping[int, UnitScores],
    text: PartitionUnit,
    metadata: Mapping[str, Any],
    *,
    display_seed: int,
    top: int,
    k: int,
) -> dict[str, Any]:
    """Everything the sheet shows for one failure.

    Args:
        unit_id: The unit.
        scores: ``seed -> the unit's scores``.
        text: The unit's lines and flagged line numbers.
        metadata: The unit's provenance, from the shard.
        display_seed: The seed whose ranking is printed line by line.
        top: How many of the model's top-ranked lines to list.
        k: The cut-off a failure is defined by.

    Returns:
        The case: observations, per-seed first ranks, the flagged lines with their rank
        under every seed, the display seed's top lines, and every scored line's rank.
    """
    ranks = {seed: line_ranks(unit.logits) for seed, unit in scores.items()}
    shown = scores[display_seed]
    n = int(shown.logits.shape[0])
    flagged = [number for number in sorted(text.flagged) if number <= n]
    order = np.argsort(ranks[display_seed], kind="stable")[:top]
    return {
        "unit_id": unit_id,
        "project": metadata.get("project"),
        "cwe": metadata.get("CWE ID"),
        "observations": observe(shown, text, k=k),
        "first_flagged_rank": {
            str(seed): int(ranks[seed][shown.labels > 0].min()) for seed in sorted(ranks)
        },
        "flagged": [
            {
                "line": number,
                "ranks": {str(seed): int(ranks[seed][number - 1]) for seed in sorted(ranks)},
                "kind": line_kind(text.lines[number - 1]),
                "text": text.lines[number - 1],
            }
            for number in flagged
        ],
        "top": [
            {
                "rank": int(ranks[display_seed][index]),
                "line": int(index) + 1,
                "flagged": bool(shown.labels[index] > 0),
                "text": text.lines[index],
            }
            for index in order
        ],
        "lines": [
            {
                "line": index + 1,
                "rank": int(ranks[display_seed][index]),
                "flagged": (index + 1) in text.flagged,
                "text": text.lines[index],
            }
            for index in range(n)
        ],
    }


def tally(observations: Sequence[Mapping[str, int]]) -> dict[str, int]:
    """How many units show each observation at least once.

    Args:
        observations: One :func:`observe` result per unit.

    Returns:
        ``observation -> units``, for the keys of :data:`OBSERVATIONS`.
    """
    return {key: sum(1 for entry in observations if entry[key] > 0) for key in OBSERVATIONS}


def _block(rows: Sequence[str]) -> list[str]:
    fence = "````"
    return [f"{fence}text", *rows, fence]


def _mark(entry: Mapping[str, Any]) -> str:
    return ">>" if entry["flagged"] else "  "


def _render_case(
    index: int, case: Mapping[str, Any], seeds: Sequence[int], display_seed: int
) -> list[str]:
    """One case of the sheet."""
    seen = case["observations"]
    by_seed = " · ".join(f"seed {seed}: {case['first_flagged_rank'][str(seed)]}" for seed in seeds)
    noted = [text for key, text in OBSERVATIONS.items() if seen[key] > 0]
    lines = [
        f"## Case {index:02d} — `{case['unit_id']}`",
        "",
        f"Project: {case['project'] or 'not recorded'} · CWE: {case['cwe'] or 'not recorded'} · "
        f"{seen['scored_lines']} scored lines of {seen['text_lines']} · "
        f"{seen['flagged_inside_cap']} flagged inside the cap, {seen['flagged_past_cap']} past it",
        "",
        f"Rank of the first flagged line — {by_seed}",
        "",
        "Measured: " + ("; ".join(noted) if noted else "none of the listed observations") + ".",
        "",
        "Flagged lines — line, rank under seed "
        + " / ".join(str(seed) for seed in seeds)
        + ", kind by rule, text:",
        "",
        *_block(
            [
                f"L{entry['line']:>4}  "
                + " / ".join(f"{entry['ranks'][str(seed)]:>3}" for seed in seeds)
                + f"  [{entry['kind']}]  {entry['text']}"
                for entry in case["flagged"]
            ]
        ),
        "",
        f"What seed {display_seed} ranked first — rank, line, text (`>>` marks a flagged line):",
        "",
        *_block(
            [
                f"#{entry['rank']:<3} L{entry['line']:>4} {_mark(entry)} {entry['text']}"
                for entry in case["top"]
            ]
        ),
        "",
        "<details><summary>The function, every scored line with its rank under seed "
        f"{display_seed}</summary>",
        "",
        *_block(
            [
                f"L{entry['line']:>4} #{entry['rank']:<4}{_mark(entry)} {entry['text']}"
                for entry in case["lines"]
            ]
        ),
        "",
        "</details>",
        "",
        "category:",
        "",
        "notes:",
        "",
    ]
    return lines


def render_markdown(payload: Mapping[str, Any]) -> str:
    """The review sheet.

    Args:
        payload: From :func:`sample`.

    Returns:
        The document.
    """
    seeds = payload["seeds"]
    population, ranked = payload["failures"], payload["ranked_units"]
    lines = [
        "# Failure cases for the taxonomy — `val`, the model of record",
        "",
        f"Generated by `scripts/sample_failures.py`, MLflow run `{payload['run_id']}`. Arm "
        f"`{payload['arm']}`, seeds {', '.join(str(seed) for seed in seeds)} (runs "
        + ", ".join(f"`{run[:8]}`" for run in payload["source_runs"])
        + "). Comment-stripped text.",
        "",
        "**A failure here** is a unit whose first flagged line is ranked outside the top "
        f"{payload['k']} on every one of the {len(seeds)} seeds: **{population} of {ranked} "
        f"ranked units.** {len(payload['cases'])} of them are below, drawn by content hash "
        f"(seed {payload['sample_seed']}) — not chosen.",
        "",
        "## What can be counted, over every failure and over every ranked unit",
        "",
        f"| A unit that… | Of the {population} failures | Of the {ranked} ranked units |",
        "|---|---|---|",
    ]
    for key, text in OBSERVATIONS.items():
        lines.append(
            f"| {text} | {payload['tally_failures'][key]} | {payload['tally_ranked'][key]} |"
        )
    quartiles = payload["scored_lines_quartiles"]
    lines += [
        "",
        "Scored lines per unit, minimum / quartiles / maximum — failures: "
        + " / ".join(str(value) for value in quartiles["failures"])
        + "; every ranked unit: "
        + " / ".join(str(value) for value in quartiles["ranked"])
        + ".",
        "",
        "## What kind of line is flagged, and how often the model finds it",
        "",
        f"Every scored line of the {ranked} ranked units, sorted into kinds by rule (first match "
        "wins; the rules read one line and nothing around it). *Found* is the share of "
        f"a kind's flagged lines ranked in the top {payload['k']} of their own function, averaged "
        f"over the {len(seeds)} seeds. *By chance* is what a ranking drawn at random would score "
        "on the same lines — a function's length sets how hard a line is to find, so a *Found* "
        "is read against it, not against 1. **A rate that rests on few units is about those "
        "functions, not about the kind** — the two columns after *Flagged* say how few.",
        "",
        "| Kind of line | Lines | Flagged | In how many units | Largest unit's share | "
        "Flagged, % of the kind | Found | By chance | Flagged, in the failures |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    kinds, in_failures = payload["kinds"]["ranked"], payload["kinds"]["failures"]

    def cell(value: float | None, pattern: str) -> str:
        return "—" if value is None else pattern.format(value)

    for kind, row in kinds.items():
        share = None if row["largest_unit_share"] is None else 100 * row["largest_unit_share"]
        lines.append(
            f"| {kind} | {row['lines']} | {row['flagged']} | {row['flagged_units']} | "
            f"{cell(share, '{:.0f}%')} | {100 * row['flagged_rate']:.1f} | "
            f"{cell(row['found_in_top_k'], '{:.3f}')} | {cell(row['found_by_chance'], '{:.3f}')} | "
            f"{in_failures.get(kind, {}).get('flagged', 0)} |"
        )
    total, flagged = (sum(row[key] for row in kinds.values()) for key in ("lines", "flagged"))
    overall, by_chance = (
        sum(row["flagged"] * (row[key] or 0.0) for row in kinds.values()) / flagged
        for key in ("found_in_top_k", "found_by_chance")
    )
    lines += [
        f"| **every kind** | {total} | {flagged} | {payload['ranked_units']} | | "
        f"{100 * flagged / total:.1f} | {overall:.3f} | {by_chance:.3f} | "
        f"{sum(row['flagged'] for row in in_failures.values())} |",
        "",
        "These are counts, not causes. About 30% of the corpus's line labels are wrong "
        "(`Memory.md` §1.6); a case below may be a wrong label, and only a reader can say so.",
        "",
        "For each case, fill in `category:` and `notes:`. Nothing here names a category.",
        "",
    ]
    for index, case in enumerate(payload["cases"], start=1):
        lines += _render_case(index, case, seeds, payload["display_seed"])
    return "\n".join(lines).rstrip() + "\n"


def _quartiles(values: Sequence[int]) -> list[int]:
    return [int(value) for value in np.percentile(np.asarray(values), [0, 25, 50, 75, 100])]


def draw_cases(
    scores_dir: Path,
    cache_dir: Path,
    *,
    arm: str,
    strip: bool,
    normalize: bool,
    k: int = 10,
    n_cases: int = 30,
    top: int = 10,
    sample_seed: int = 1337,
    display_seed: int = 42,
    require_seeds: int = 3,
) -> dict[str, Any]:
    """Find every failure of one arm on validation and draw the cases. Writes nothing.

    Args:
        scores_dir: Directory of ``<run>_val.npz`` score files and their manifests.
        cache_dir: Directory of the preprocessed shards.
        arm: The arm, as :func:`vulnloc.evaluation.evaluator.arm_name` names it.
        strip: Whether the scores were made from comment-stripped text.
        normalize: Apply ADR-006 whitespace normalization, as every reported run did.
        k: A unit fails when its first flagged line is outside the top ``k`` on every seed.
        n_cases: How many failures to draw.
        top: How many of the model's top-ranked lines each case lists.
        sample_seed: Seed of the content-hash draw.
        display_seed: The seed whose ranking is printed line by line.
        require_seeds: How many seeds the arm must have.

    Returns:
        The payload ``render_markdown`` reads, without a run id.

    Raises:
        ConfigError: For scores that are held-out or of the other text variant, or an arm
            without the required seeds.
        DataError: If the seeds did not rank the same units.
    """
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(scores_dir.glob(f"*_{SPLIT}.manifest.json"))
    ]
    manifests = [m for m in manifests if arm_name(m) == arm]
    if any(
        bool(m.get("held_out")) or bool(m.get("strip_comments", False)) != strip for m in manifests
    ):
        raise ConfigError(
            "these scores are held-out, or were made from the other text variant",
            scores_dir=str(scores_dir),
            evaluation_strips_comments=strip,
        )
    by_seed = {int(m["seed"]): str(m["source_run"]) for m in manifests}
    if (
        len(by_seed) != require_seeds
        or len(manifests) != require_seeds
        or display_seed not in by_seed
    ):
        raise ConfigError(
            "the arm does not have the required seeds",
            arm=arm,
            seeds=sorted(by_seed),
            required=require_seeds,
            display_seed=display_seed,
        )

    scores: dict[int, dict[str, UnitScores]] = {
        seed: {unit.unit_id: unit for unit in load_scores(scores_dir / f"{run}_{SPLIT}.npz")}
        for seed, run in by_seed.items()
    }
    ranks = {
        seed: {
            unit_id: rank
            for unit_id, unit in units.items()
            if (rank := first_flagged_rank(unit)) is not None
        }
        for seed, units in scores.items()
    }
    failures = consistent_misses(ranks, k)
    ranked = sorted(ranks[display_seed])
    partition = load_partition(cache_dir, SPLIT, normalize=normalize, strip=strip)
    seen = {unit: observe(scores[display_seed][unit], partition[unit], k=k) for unit in ranked}
    drawn = sorted(stable_sample(failures, min(n_cases, len(failures)), seed=sample_seed))
    wanted = set(drawn)
    metadata = {
        unit.unit_id: dict(unit.metadata)
        for unit in iter_shard(cache_dir, SPLIT)
        if unit.unit_id in wanted
    }
    seeds = sorted(by_seed)
    return {
        "generated_by": "scripts/sample_failures.py",
        "split": SPLIT,
        "arm": arm,
        "seeds": seeds,
        "source_runs": [by_seed[seed] for seed in seeds],
        "strip_comments": strip,
        "k": k,
        "sample_seed": sample_seed,
        "display_seed": display_seed,
        "ranked_units": len(ranked),
        "failures": len(failures),
        "failure_units": failures,
        "tally_failures": tally([seen[unit] for unit in failures]),
        "tally_ranked": tally([seen[unit] for unit in ranked]),
        "scored_lines_quartiles": {
            "failures": _quartiles([seen[unit]["scored_lines"] for unit in failures]),
            "ranked": _quartiles([seen[unit]["scored_lines"] for unit in ranked]),
        },
        "kinds": {
            "ranked": kind_table(scores, partition, ranked, k),
            "failures": kind_table(scores, partition, failures, k),
        },
        "cases": [
            build_case(
                unit,
                {seed: scores[seed][unit] for seed in seeds},
                partition[unit],
                metadata.get(unit, {}),
                display_seed=display_seed,
                top=top,
                k=k,
            )
            for unit in drawn
        ],
    }


def sample(cfg: DictConfig) -> dict[str, Any]:
    """Draw the failure cases, write the review sheet and log the run that carries it.

    Args:
        cfg: Composed config. Reads ``failures.scores_dir``, ``failures.arm`` (default
            ``cross_attention_retrained``), ``failures.k`` (10), ``failures.n`` (30),
            ``failures.seed`` (1337), ``failures.display_seed`` (42), ``failures.top`` (10),
            ``failures.require_seeds`` (3) and ``failures.output_dir``.

    Returns:
        The payload, as written to ``<output_dir>/failure_cases.json``.

    Raises:
        ConfigError: For a partition other than ``val``; and from :func:`draw_cases`.
        DataError: From :func:`draw_cases`.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )

    def option(name: str, default: Any) -> Any:
        value = OmegaConf.select(cfg, f"failures.{name}")
        return default if value is None else value

    if str(option("split", SPLIT)) != SPLIT:
        raise ConfigError(
            "failure cases are drawn on validation only", split=str(option("split", SPLIT))
        )
    output_dir = Path(str(option("output_dir", "./artifacts/phase8")))
    payload = draw_cases(
        Path(str(option("scores_dir", "./artifacts/phase6/scores_retrained"))),
        Path(cfg.data.cache.output_dir),
        arm=str(option("arm", "cross_attention_retrained")),
        strip=strip_comments_enabled(cfg),
        normalize=bool(cfg.representation.normalize_whitespace),
        k=int(option("k", 10)),
        n_cases=int(option("n", 30)),
        top=int(option("top", 10)),
        sample_seed=int(option("seed", 1337)),
        display_seed=int(option("display_seed", 42)),
        require_seeds=int(option("require_seeds", 3)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = output_dir / "failure_cases.json", output_dir / "failure_cases.md"
    with start_run(cfg, run_name="failure-cases") as run_id:
        payload["run_id"] = run_id
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        markdown_path.write_text(render_markdown(payload), encoding="utf-8")
        mlflow.set_tags({"artefact": "failure-cases", "split": SPLIT})
        mlflow.log_params(
            {
                "failures.arm": payload["arm"],
                "failures.k": payload["k"],
                "failures.sample_seed": payload["sample_seed"],
                "failures.source_runs": ",".join(run[:8] for run in payload["source_runs"]),
            }
        )
        mlflow.log_metrics(
            {
                "ranked_units": float(payload["ranked_units"]),
                "failures": float(payload["failures"]),
                "cases": float(len(payload["cases"])),
            }
        )
        mlflow.log_artifact(str(json_path), artifact_path="failures")
        mlflow.log_artifact(str(markdown_path), artifact_path="failures")
    _LOGGER.info(
        "Failure cases written",
        extra={
            "sheet": str(markdown_path),
            "failures": payload["failures"],
            "cases": len(payload["cases"]),
            "run_id": run_id,
        },
    )
    return payload


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    sample(cfg)


if __name__ == "__main__":
    main()
