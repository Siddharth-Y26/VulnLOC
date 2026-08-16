"""Characterise the units that no grammar parses cleanly — evidence for ADR-005.

``Memory.md`` §1.7 measured that 88.24% of Big-Vul parses cleanly as C++ against 37.01%
under the corpus's own ``lang`` column. That comparison decides *which* grammar to use.
It says nothing about the **20,480 functions (10.86%) that parse cleanly under neither**,
and a decision that quietly absorbs a tenth of the corpus into "recovered anyway" is not
a decision, it is an omission.

So this script asks the only question that matters downstream: **what does a damaged
parse actually cost?** A tree with ERROR nodes in it is still a tree, and the structural
branch does not consume syntactic correctness — it consumes nodes attached to lines. The
report therefore measures the damaged cohort against the clean one on the terms Phase 6
will be graded on:

- how much of the corpus, and how much of the *labelled* corpus, is in the cohort;
- how much of each unit's source sits inside an ERROR span (severity, not incidence);
- line coverage, and **flagged**-line coverage, damaged vs clean;
- which projects it concentrates in.

It also draws a deterministic sample of first-error contexts to
``artifacts/representation/parse_failure_sample.md`` so the causes get named by a human
rather than guessed at by a heuristic — the same discipline as ``sample_labels.py``.

Usage::

    python scripts/characterize_parse_failures.py
    python scripts/characterize_parse_failures.py graph.report.limit=2000
"""

from __future__ import annotations

import heapq
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig
from tree_sitter import Node

from vulnloc.data.cache import iter_shard, shard_path
from vulnloc.representation import GRAPH_REGISTRY, ParseResult, TreeSitterParser
from vulnloc.representation.units import CodeUnit
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.hashing import unit_interval
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

GRAMMARS = ("C", "CPP")
SAMPLE_SIZE = 40
SAMPLE_SEED = 1337
CONTEXT_LINES = 2


def error_byte_span(result: ParseResult) -> int:
    """Return how many source bytes lie inside an ERROR node.

    Counts *outermost* ERROR nodes only. Tree-sitter nests error recovery, so summing
    every ERROR node's width would double-count the same bytes and can exceed the length
    of the file — a severity metric that reports 140% damage is worse than none.

    Args:
        result: A parse result.

    Returns:
        Total bytes covered by top-level ERROR nodes.
    """
    total = 0
    stack: list[Node] = [result.root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR":
            total += node.end_byte - node.start_byte
            continue  # Do not descend: nested ERRORs cover bytes already counted.
        stack.extend(node.children)
    return total


def failure_category(result: ParseResult, unit: CodeUnit) -> str:
    """Categorise a damaged parse by where the damage starts.

    Kept identical to ``report_representation._error_bucket`` so the two artefacts can
    be read side by side.

    Args:
        result: The damaged parse.
        unit: The unit it came from.

    Returns:
        A short category label.
    """
    if result.missing_nodes and not result.error_nodes:
        return "missing_token_only"
    if result.first_error_line == 1:
        return "error_on_first_line"
    if result.first_error_line is not None and result.first_error_line >= unit.n_lines - 1:
        return "error_at_end_of_function"
    return "error_mid_function"


@dataclass
class Cohort:
    """Running totals for one cohort of units — clean, or clean under no grammar."""

    name: str
    units: int = 0
    vulnerable: int = 0
    line_labelled: int = 0
    flagged_lines: int = 0
    flagged_lines_covered: int = 0
    source_lines: int = 0
    covered_lines: int = 0
    source_bytes: int = 0
    error_bytes: int = 0
    error_nodes: int = 0
    missing_nodes: int = 0
    nodes: int = 0
    categories: Counter[str] = field(default_factory=Counter)
    projects: Counter[str] = field(default_factory=Counter)
    declared: Counter[str] = field(default_factory=Counter)

    def record(
        self, unit: CodeUnit, result: ParseResult, covered: frozenset[int], nodes: int
    ) -> None:
        """Fold one unit into the cohort.

        Args:
            unit: The unit.
            result: Its parse under the C++ grammar (ADR-005's policy).
            covered: Lines carrying at least one graph node.
            nodes: Node count of its graph.
        """
        self.units += 1
        self.vulnerable += unit.target
        self.line_labelled += bool(unit.vulnerable_lines)
        self.flagged_lines += len(unit.vulnerable_lines)
        self.flagged_lines_covered += len(unit.vulnerable_lines & covered)
        self.source_lines += unit.n_lines
        self.covered_lines += len(covered)
        self.source_bytes += len(result.source.encode("utf-8"))
        self.error_bytes += error_byte_span(result)
        self.error_nodes += result.error_nodes
        self.missing_nodes += result.missing_nodes
        self.nodes += nodes
        self.projects[str(unit.metadata.get("project", "?"))] += 1
        self.declared[unit.language] += 1
        if not result.is_clean:
            self.categories[failure_category(result, unit)] += 1

    def to_dict(self) -> dict[str, Any]:
        """Return the cohort's section of the report.

        Returns:
            A JSON-serializable mapping.
        """
        units = max(self.units, 1)
        return {
            "cohort": self.name,
            "units": self.units,
            "vulnerable": self.vulnerable,
            "line_labelled": self.line_labelled,
            "flagged_lines": self.flagged_lines,
            "mean_lines_per_unit": round(self.source_lines / units, 2),
            "mean_nodes_per_unit": round(self.nodes / units, 2),
            "damage": {
                "error_nodes": self.error_nodes,
                "missing_nodes": self.missing_nodes,
                "source_bytes": self.source_bytes,
                "bytes_inside_an_error_node": self.error_bytes,
                "bytes_inside_an_error_node_pct": round(
                    100 * self.error_bytes / max(self.source_bytes, 1), 2
                ),
                "failure_categories": dict(self.categories.most_common()),
            },
            "line_coverage": {
                "source_lines": self.source_lines,
                "lines_with_a_node_pct": round(
                    100 * self.covered_lines / max(self.source_lines, 1), 2
                ),
                "flagged_lines": self.flagged_lines,
                "flagged_lines_with_a_node": self.flagged_lines_covered,
                "flagged_lines_with_a_node_pct": round(
                    100 * self.flagged_lines_covered / max(self.flagged_lines, 1), 2
                ),
            },
            "declared_language": dict(self.declared.most_common()),
            "top_projects": dict(self.projects.most_common(10)),
        }


class FailureSample:
    """The ``k`` lowest-hashed damaged units, kept while streaming the corpus.

    Equivalent to :func:`~vulnloc.utils.hashing.stable_sample` over the whole damaged
    population, but without holding the population: a bounded max-heap on the hash keeps
    exactly the ``k`` smallest. The sample is therefore a pure function of *which units
    are damaged*, not of the order the shards were read in.
    """

    def __init__(self, k: int, seed: int) -> None:
        self.k = k
        self.seed = seed
        self._heap: list[tuple[float, str, dict[str, Any]]] = []

    def offer(self, unit: CodeUnit, result: ParseResult) -> None:
        """Consider one damaged unit for the sample.

        Args:
            unit: The unit.
            result: Its C++ parse.
        """
        # Negated so the heap's root is the *largest* hash — the one to evict.
        rank = -unit_interval(self.seed, unit.unit_id)
        if len(self._heap) >= self.k and rank < self._heap[0][0]:
            return
        line = result.first_error_line or 1
        start = max(1, line - CONTEXT_LINES)
        end = min(unit.n_lines, line + CONTEXT_LINES)
        entry = {
            "unit_id": unit.unit_id,
            "project": str(unit.metadata.get("project", "?")),
            "cwe": str(unit.metadata.get("CWE ID", "?")),
            "declared_language": unit.language,
            "n_lines": unit.n_lines,
            "first_error_line": result.first_error_line,
            "error_nodes": result.error_nodes,
            "missing_nodes": result.missing_nodes,
            "category": failure_category(result, unit),
            "context": [(n, unit.lines[n - 1]) for n in range(start, end + 1)],
        }
        heapq.heappush(self._heap, (rank, unit.unit_id, entry))
        if len(self._heap) > self.k:
            heapq.heappop(self._heap)

    def entries(self) -> list[dict[str, Any]]:
        """Return the sample, ordered as ``stable_sample`` would order it.

        Returns:
            The drawn units, smallest hash first.
        """
        return [entry for _, _, entry in sorted(self._heap, key=lambda item: (-item[0], item[1]))]


def render_sample(entries: list[dict[str, Any]], path: Path) -> None:
    """Write the sample as a markdown sheet for human categorisation.

    Args:
        entries: The drawn units.
        path: Destination file.
    """
    lines = [
        "# Parse failures under both grammars — sample for human categorisation",
        "",
        f"{len(entries)} units drawn by content hash at seed {SAMPLE_SEED} from the units that",
        "parse cleanly under **neither** the C nor the C++ grammar. Drawn before anyone looked",
        "at them (`Rules.md` §5.7). Each block shows the first line Tree-sitter reported an",
        f"error on, with {CONTEXT_LINES} lines of context.",
        "",
        "Fill in `cause:` with one of `truncated_signature`, `macro`, `genuine_syntax`,",
        "`cpp_construct`, `other` — and say why in a clause.",
        "",
        "---",
        "",
    ]
    for index, entry in enumerate(entries, start=1):
        lines += [
            f"## {index}. `{entry['unit_id']}` — {entry['project']} ({entry['cwe']})",
            "",
            f"- declared `{entry['declared_language']}` · {entry['n_lines']} lines · "
            f"category `{entry['category']}`",
            f"- first error on line {entry['first_error_line']} · "
            f"{entry['error_nodes']} ERROR / {entry['missing_nodes']} MISSING",
            "",
            "```c",
        ]
        for number, text in entry["context"]:
            marker = ">>" if number == entry["first_error_line"] else "  "
            lines.append(f"{marker} {number:>4} | {text}")
        lines += ["```", "", "cause: ", "", "---", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def characterize(cfg: DictConfig) -> dict[str, Any]:
    """Measure the damaged-parse cohort against the clean one.

    Args:
        cfg: Composed Hydra config. Uses ``graph.report`` for the cache location.

    Returns:
        The report, as written to disk.

    Raises:
        DataError: If a requested shard is missing.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    settings = cfg.graph.report
    cache_dir = Path(settings.cache_dir)
    limit = int(settings.limit)
    requested = [str(s) for s in settings.splits]
    splits = sorted(requested) if requested else sorted(p.stem for p in cache_dir.glob("*.jsonl"))
    if not splits:
        raise DataError(
            f"No shards in {cache_dir}. Run scripts/build_cache.py first.",
            path=str(cache_dir),
        )

    parser = TreeSitterParser()
    builder = GRAPH_REGISTRY.get("ast")()
    clean = Cohort("clean_under_either_grammar")
    damaged = Cohort("clean_under_neither_grammar")
    sample = FailureSample(SAMPLE_SIZE, SAMPLE_SEED)
    per_split: dict[str, dict[str, int]] = {}

    with start_run(cfg, run_name="parse-failure-characterization") as run_id:
        for split in splits:
            path = shard_path(cache_dir, split)
            if not path.exists():
                raise DataError(f"Shard not found: {path}", path=str(path))

            split_units = 0
            split_damaged = 0
            for index, labelled in enumerate(iter_shard(cache_dir, split)):
                if limit and index >= limit:
                    break
                unit = CodeUnit.from_labelled_function(labelled)
                results = {
                    grammar: parser.parse(unit.source, language=grammar, unit_id=unit.unit_id)
                    for grammar in GRAMMARS
                }
                # ADR-005 parses everything as C++, so the cost of the damage has to be
                # measured on the C++ tree — the one the project will actually build on.
                policy_result = results["CPP"]
                graph = builder.build(policy_result)
                covered = graph.lines_covered
                is_damaged = not any(result.is_clean for result in results.values())
                cohort = damaged if is_damaged else clean
                cohort.record(unit, policy_result, covered, len(graph.nodes))
                if is_damaged:
                    sample.offer(unit, policy_result)
                    split_damaged += 1
                split_units += 1

            per_split[split] = {
                "units": split_units,
                "clean_under_neither": split_damaged,
                "clean_under_neither_pct": round(100 * split_damaged / max(split_units, 1), 2),
            }
            _LOGGER.info("Shard characterised", extra={"split": split, **per_split[split]})

        payload = {
            "run_id": run_id,
            "limit": limit,
            "sample_seed": SAMPLE_SEED,
            "per_split": per_split,
            "cohorts": {
                "clean_under_either_grammar": clean.to_dict(),
                "clean_under_neither_grammar": damaged.to_dict(),
            },
        }
        output_dir = Path(settings.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        # A capped run is a smoke test, not a measurement. Naming it apart keeps it from
        # overwriting the corpus-wide artefact that ADR-005 cites.
        suffix = f"_limit{limit}" if limit else ""
        report_path = output_dir / f"parse_failures{suffix}.json"
        report_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        sample_path = output_dir / f"parse_failure_sample{suffix}.md"
        render_sample(sample.entries(), sample_path)

        mlflow.log_metrics(
            {
                "parse_failures/units": damaged.units,
                "parse_failures/flagged_lines": damaged.flagged_lines,
                "parse_failures/flagged_line_coverage_pct": damaged.to_dict()["line_coverage"][
                    "flagged_lines_with_a_node_pct"
                ],
                "parse_failures/error_byte_pct": damaged.to_dict()["damage"][
                    "bytes_inside_an_error_node_pct"
                ],
            }
        )
        mlflow.log_artifact(str(report_path), artifact_path="representation")
        mlflow.log_artifact(str(sample_path), artifact_path="representation")

    _LOGGER.info("Parse-failure characterization written", extra={"path": str(report_path)})
    return payload


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    characterize(cfg)


if __name__ == "__main__":
    main()
