r"""What comment stripping does to the corpus, measured — the evidence for ADR-010.

``utils.text.strip_comments`` exists to remove a label leak (``Memory.md`` §1.25). A
transform that rewrites every function's text has to be shown to do exactly that and
nothing else, on the corpus rather than on fixtures. For each partition's line-labelled
units this script reports:

* **Agreement with the grammar.** On units that parse cleanly and carry no lost-slash
  line, Tree-sitter knows where every comment is. Removing those byte ranges and removing
  comments with the state machine must leave the same non-whitespace characters on every
  line. Anything less than 100% is a bug in the stripper.
* **What comments held.** How many flagged and unflagged lines carried a comment marker
  before; how many lines, and how many *flagged* lines, are blank after — the labelled
  lines that held nothing but a comment, which no localizer can find once it is gone.
* **Whether the tells are gone.** How often each comment tell still fires after stripping.
* **Parse damage.** Clean parses before and after, overall and among units that had a
  lost-slash line; and the share of flagged lines sitting in units that do not parse
  cleanly, before and after.

Usage::

    python scripts/report_comment_stripping.py
    python scripts/report_comment_stripping.py 'graph.report.splits=[train,val]'
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig

from vulnloc.data.cache import iter_shard, shard_path
from vulnloc.representation import CodeUnit, ParseResult, TreeSitterParser, iter_nodes
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger, setup_logging
from vulnloc.utils.text import ends_comment_without_slash, strip_comments

_LOGGER = get_logger(__name__)

_WHITESPACE = re.compile(r"\s+")
_NEWLINE = 0x0A


def squeezed(text: str) -> list[str]:
    """Each line of ``text`` with all whitespace removed.

    Args:
        text: Source text.

    Returns:
        One string per line — what two comment strippers must agree on, whatever spacing
        each leaves behind.
    """
    return [_WHITESPACE.sub("", line) for line in text.split("\n")]


def strip_with_grammar(parsed: ParseResult) -> str:
    """Remove the byte ranges Tree-sitter itself calls comments, keeping every newline.

    Args:
        parsed: A parse of the unit.

    Returns:
        The source without its comment nodes, line count unchanged.
    """
    data = bytearray(parsed.source.encode("utf-8"))
    spans = sorted(
        (
            (node.start_byte, node.end_byte)
            for node in iter_nodes(parsed.tree.root_node)
            if node.type == "comment"
        ),
        reverse=True,
    )
    for start, end in spans:
        newlines = bytes(byte for byte in data[start:end] if byte == _NEWLINE)
        data[start:end] = b" " + newlines
    return data.decode("utf-8")


def has_comment_marker(line: str) -> bool:
    """Whether a line visibly carries a comment — a marker anywhere on it.

    Args:
        line: One source line.

    Returns:
        ``True`` for a line containing ``/*``, ``*/`` or ``//``. A marker inside a string
        literal counts; this is a count of what a reader sees, not a parse.
    """
    return "/*" in line or "*/" in line or "//" in line


def pct(part: int, whole: int) -> float:
    """``part`` as a percentage of ``whole``, two decimals, zero when ``whole`` is zero."""
    return round(100.0 * part / whole, 2) if whole else 0.0


def report_split(units: list[CodeUnit], parser: TreeSitterParser, *, policy: str) -> dict[str, Any]:
    """Measure one partition.

    Args:
        units: The partition's line-labelled units, whitespace-normalized, comments in place.
        parser: A parser to reuse.
        policy: Grammar-selection policy (ADR-005).

    Returns:
        The partition's section of the report.
    """
    counts = dict.fromkeys(
        (
            "lines",
            "flagged_lines",
            "lines_with_comment_marker",
            "flagged_lines_with_comment_marker",
            "blank_before",
            "blank_after",
            "flagged_blank_before",
            "flagged_blank_after",
            "units_whose_flagged_lines_are_all_blank_after",
            "lost_slash_before",
            "lost_slash_after",
            "intact_end_before",
            "intact_end_after",
            "clean_before",
            "clean_after",
            "units_with_lost_slash",
            "lost_slash_units_clean_before",
            "lost_slash_units_clean_after",
            "flagged_in_damaged_units_before",
            "flagged_in_damaged_units_after",
            "compared_with_grammar",
            "identical_to_grammar",
        ),
        0,
    )
    disagreements: list[dict[str, Any]] = []
    for unit in units:
        stripped = strip_comments(unit.source)
        after = stripped.split("\n")
        if len(after) != unit.n_lines:
            raise DataError(
                "comment stripping changed a unit's line count",
                unit_id=unit.unit_id,
                before=unit.n_lines,
                after=len(after),
            )
        flagged_after_code = 0
        for number, (old, new) in enumerate(zip(unit.lines, after, strict=True), start=1):
            flagged = number in unit.vulnerable_lines
            counts["lines"] += 1
            counts["flagged_lines"] += flagged
            marker = has_comment_marker(old)
            counts["lines_with_comment_marker"] += marker
            counts["flagged_lines_with_comment_marker"] += flagged and marker
            counts["blank_before"] += not old
            counts["blank_after"] += not new
            counts["flagged_blank_before"] += flagged and not old
            counts["flagged_blank_after"] += flagged and not new
            flagged_after_code += flagged and bool(new)
            counts["lost_slash_before"] += ends_comment_without_slash(old)
            counts["lost_slash_after"] += ends_comment_without_slash(new)
            counts["intact_end_before"] += old.endswith("*/")
            counts["intact_end_after"] += new.endswith("*/")
        counts["units_whose_flagged_lines_are_all_blank_after"] += not flagged_after_code

        before_parse = parser.parse_with_policy(
            unit.source, policy=policy, declared=unit.language, unit_id=unit.unit_id
        )
        after_parse = parser.parse_with_policy(
            stripped, policy=policy, declared=unit.language, unit_id=unit.unit_id
        )
        n_flagged = len(unit.vulnerable_lines)
        counts["clean_before"] += before_parse.is_clean
        counts["clean_after"] += after_parse.is_clean
        counts["flagged_in_damaged_units_before"] += 0 if before_parse.is_clean else n_flagged
        counts["flagged_in_damaged_units_after"] += 0 if after_parse.is_clean else n_flagged
        lost = any(ends_comment_without_slash(line) for line in unit.lines)
        if lost:
            counts["units_with_lost_slash"] += 1
            counts["lost_slash_units_clean_before"] += before_parse.is_clean
            counts["lost_slash_units_clean_after"] += after_parse.is_clean
        elif before_parse.is_clean:
            counts["compared_with_grammar"] += 1
            mine, theirs = squeezed(stripped), squeezed(strip_with_grammar(before_parse))
            if mine == theirs:
                counts["identical_to_grammar"] += 1
            elif len(disagreements) < 5:
                line = next(i for i, (a, b) in enumerate(zip(mine, theirs, strict=True)) if a != b)
                disagreements.append(
                    {
                        "unit_id": unit.unit_id,
                        "line": line + 1,
                        "source": unit.lines[line][:120],
                        "state_machine": mine[line][:80],
                        "grammar": theirs[line][:80],
                    }
                )

    n_units = len(units)
    flagged = counts["flagged_lines"]
    unflagged = counts["lines"] - flagged
    unflagged_with_marker = (
        counts["lines_with_comment_marker"] - counts["flagged_lines_with_comment_marker"]
    )
    return {
        "line_labelled_units": n_units,
        **counts,
        "flagged_lines_with_comment_marker_pct": pct(
            counts["flagged_lines_with_comment_marker"], flagged
        ),
        "unflagged_lines_with_comment_marker_pct": pct(unflagged_with_marker, unflagged),
        "flagged_blank_after_pct": pct(counts["flagged_blank_after"], flagged),
        "clean_before_pct": pct(counts["clean_before"], n_units),
        "clean_after_pct": pct(counts["clean_after"], n_units),
        "flagged_in_damaged_units_before_pct": pct(
            counts["flagged_in_damaged_units_before"], flagged
        ),
        "flagged_in_damaged_units_after_pct": pct(
            counts["flagged_in_damaged_units_after"], flagged
        ),
        "identical_to_grammar_pct": pct(
            counts["identical_to_grammar"], counts["compared_with_grammar"]
        ),
        "disagreements_with_grammar": disagreements,
    }


def report(cfg: DictConfig) -> dict[str, Any]:
    """Measure comment stripping on every requested partition.

    Args:
        cfg: Composed config. Uses ``graph.report`` for paths and splits, and
            ``graph.language`` for the grammar policy.

    Returns:
        The report, as written to disk.

    Raises:
        DataError: If the cache holds no shards, a requested shard is missing, or the
            transform ever changes a unit's line count.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )
    settings = cfg.graph.report
    cache_dir = Path(settings.cache_dir)
    requested = [str(split) for split in settings.splits]
    splits = sorted(requested) if requested else sorted(p.stem for p in cache_dir.glob("*.jsonl"))
    if not splits:
        raise DataError(f"No shards in {cache_dir}. Run scripts/build_cache.py first.")
    policy = str(cfg.graph.language)
    parser = TreeSitterParser()

    shards: dict[str, Any] = {}
    with start_run(cfg, run_name="comment-stripping-report") as run_id:
        mlflow.set_tag("artefact", "data")
        for split in splits:
            if not shard_path(cache_dir, split).exists():
                raise DataError(f"Shard not found for {split}", split=split)
            units = [
                CodeUnit.from_labelled_function(labelled)
                for labelled in iter_shard(cache_dir, split)
                if labelled.vulnerable_lines
            ]
            if not units:
                continue
            section = report_split(units, parser, policy=policy)
            shards[split] = section
            for key in (
                "identical_to_grammar_pct",
                "flagged_blank_after_pct",
                "clean_before_pct",
                "clean_after_pct",
                "flagged_lines_with_comment_marker_pct",
                "unflagged_lines_with_comment_marker_pct",
            ):
                mlflow.log_metric(f"{split}/{key}", float(section[key]))
            _LOGGER.info(
                "Comment stripping measured",
                extra={
                    "split": split,
                    "units": section["line_labelled_units"],
                    "identical_to_grammar": f"{section['identical_to_grammar']}/"
                    f"{section['compared_with_grammar']}",
                    "flagged_blank_after_pct": section["flagged_blank_after_pct"],
                    "clean_pct": f"{section['clean_before_pct']} -> {section['clean_after_pct']}",
                    "lost_slash_after": section["lost_slash_after"],
                },
            )
        payload = {
            "generated_by": "scripts/report_comment_stripping.py",
            "run_id": run_id,
            "language_policy": policy,
            "shards": shards,
        }
        output_path = Path(settings.output_path).with_name("comment_stripping.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(output_path))
        _LOGGER.info("Report written", extra={"path": str(output_path), "run_id": run_id})
    return payload


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    report(cfg)


if __name__ == "__main__":
    main()
