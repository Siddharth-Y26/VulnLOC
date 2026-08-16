"""Corpus-wide parse and graph report — Phase 2 exit criteria.

``Phases.md`` Phase 2 requires two numbers before the phase can close: the **parse
success rate on the corpus, with failures categorised rather than hidden**, and the
**graph statistics** — mean nodes and edges, and the edge-type distribution. Both come
from here, and both are written to an artefact so that a later claim can cite a file
rather than a memory.

The report also carries the evidence for open decision Q8 (``Memory.md`` §5): each unit
is parsed under *both* grammars and both rates are reported. The corpus's own ``lang``
column says ``C`` for 98.6% of functions, and that turns out not to be the grammar most
of them parse cleanly under. Nothing here decides Q8 — it measures it.

Reads the preprocessed shards only; it never opens a Big-Vul CSV.

Usage::

    python scripts/report_representation.py
    python scripts/report_representation.py graph.report.limit=500
    python scripts/report_representation.py 'graph.report.splits=[clean_eval]'
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig

from vulnloc.data.cache import iter_shard, shard_path
from vulnloc.representation import TreeSitterParser, build_ast_graph
from vulnloc.representation.units import CodeUnit
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

GRAMMARS = ("C", "CPP")


def discover_splits(cache_dir: Path, requested: list[str]) -> list[str]:
    """Return the shards to report on.

    Args:
        cache_dir: Directory holding the shards.
        requested: Explicit shard names; empty means every shard present.

    Returns:
        Shard names, sorted.

    Raises:
        DataError: If the cache directory holds no shards.
    """
    if requested:
        return sorted(requested)
    found = sorted(path.stem for path in cache_dir.glob("*.jsonl"))
    if not found:
        raise DataError(
            f"No shards in {cache_dir}. Run scripts/build_cache.py first.",
            path=str(cache_dir),
        )
    return found


class ShardReport:
    """Accumulates parse and graph statistics for one shard.

    Kept as running totals rather than a list of per-unit records: the corpus is
    188,636 functions and the report should not need more memory than the pass itself.
    """

    def __init__(self, split: str) -> None:
        self.split = split
        self.units = 0
        self.clean: Counter[str] = Counter()
        self.clean_either = 0
        self.clean_neither = 0
        self.declared_language: Counter[str] = Counter()
        self.declared_clean = 0
        self.error_nodes = 0
        self.missing_nodes = 0
        self.first_error_lines: Counter[str] = Counter()
        self.nodes = 0
        self.edges = 0
        self.named_nodes = 0
        self.source_lines = 0
        self.covered_lines = 0
        self.edge_types: Counter[str] = Counter()
        self.node_types: Counter[str] = Counter()
        self.max_nodes = 0
        self.max_nodes_unit = ""
        self.labelled_lines = 0
        self.labelled_lines_covered = 0

    def record(
        self,
        unit: CodeUnit,
        results: dict[str, Any],
        graph: Any,
    ) -> None:
        """Fold one unit into the totals.

        Args:
            unit: The unit that was parsed.
            results: Parse result per grammar name.
            graph: The AST graph built from the declared-language parse.
        """
        self.units += 1
        self.declared_language[unit.language] += 1

        for name, result in results.items():
            if result.is_clean:
                self.clean[name] += 1
        if len(results) > 1:
            any_clean = any(result.is_clean for result in results.values())
            self.clean_either += any_clean
            self.clean_neither += not any_clean

        declared = results["declared"]
        self.declared_clean += declared.is_clean
        self.error_nodes += declared.error_nodes
        self.missing_nodes += declared.missing_nodes
        if not declared.is_clean:
            self.first_error_lines[_error_bucket(declared, unit)] += 1

        self.nodes += len(graph.nodes)
        self.edges += len(graph.edges)
        self.named_nodes += sum(1 for node in graph.nodes if node.is_named)
        self.source_lines += unit.n_lines
        covered = graph.lines_covered
        self.covered_lines += len(covered)
        self.edge_types.update(graph.edge_type_counts())
        self.node_types.update(node.node_type for node in graph.nodes)
        if len(graph.nodes) > self.max_nodes:
            self.max_nodes = len(graph.nodes)
            self.max_nodes_unit = unit.unit_id

        # The number that actually matters downstream: a flagged line with no node on
        # it cannot receive a structural score at all.
        self.labelled_lines += len(unit.vulnerable_lines)
        self.labelled_lines_covered += len(unit.vulnerable_lines & covered)

    def to_dict(self) -> dict[str, Any]:
        """Return the shard's section of the report.

        Returns:
            A JSON-serializable mapping.
        """
        units = max(self.units, 1)
        report: dict[str, Any] = {
            "split": self.split,
            "units": self.units,
            "declared_language": dict(self.declared_language.most_common()),
            "parse": {
                "clean_as_declared": self.declared_clean,
                "clean_as_declared_pct": round(100 * self.declared_clean / units, 2),
                "error_nodes": self.error_nodes,
                "missing_nodes": self.missing_nodes,
                "failure_categories": dict(self.first_error_lines.most_common()),
            },
            "graph": {
                "graph_type": "ast",
                "total_nodes": self.nodes,
                "total_edges": self.edges,
                "mean_nodes_per_unit": round(self.nodes / units, 2),
                "mean_edges_per_unit": round(self.edges / units, 2),
                "named_node_pct": round(100 * self.named_nodes / max(self.nodes, 1), 2),
                "max_nodes": self.max_nodes,
                "max_nodes_unit": self.max_nodes_unit,
                "edge_type_distribution": dict(self.edge_types.most_common()),
                "top_node_types": dict(self.node_types.most_common(20)),
                "distinct_node_types": len(self.node_types),
            },
            "line_coverage": {
                "source_lines": self.source_lines,
                "lines_with_a_node": self.covered_lines,
                "lines_with_a_node_pct": round(
                    100 * self.covered_lines / max(self.source_lines, 1), 2
                ),
                "labelled_lines": self.labelled_lines,
                "labelled_lines_with_a_node": self.labelled_lines_covered,
                "labelled_lines_with_a_node_pct": round(
                    100 * self.labelled_lines_covered / max(self.labelled_lines, 1), 2
                ),
            },
        }
        if self.clean:
            report["grammar_comparison"] = {
                "clean_as_c": self.clean.get("C", 0),
                "clean_as_cpp": self.clean.get("CPP", 0),
                "clean_as_c_pct": round(100 * self.clean.get("C", 0) / units, 2),
                "clean_as_cpp_pct": round(100 * self.clean.get("CPP", 0) / units, 2),
                "clean_under_either": self.clean_either,
                "clean_under_neither": self.clean_neither,
            }
        return report


def _error_bucket(result: Any, unit: CodeUnit) -> str:
    """Categorise a failed parse, so failures are reported rather than counted.

    Args:
        result: The declared-language parse result.
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


def report(cfg: DictConfig) -> dict[str, Any]:
    """Build the corpus-wide representation report.

    Args:
        cfg: Composed Hydra config. Requires ``graph.report``.

    Returns:
        The report, as written to disk.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    settings = cfg.graph.report
    cache_dir = Path(settings.cache_dir)
    splits = discover_splits(cache_dir, [str(s) for s in settings.splits])
    limit = int(settings.limit)
    compare = bool(settings.compare_grammars)

    parser = TreeSitterParser()
    shards: dict[str, Any] = {}

    with start_run(cfg, run_name="representation-report") as run_id:
        for split in splits:
            path = shard_path(cache_dir, split)
            if not path.exists():
                raise DataError(f"Shard not found: {path}", path=str(path))

            accumulator = ShardReport(split)
            for index, labelled in enumerate(iter_shard(cache_dir, split)):
                if limit and index >= limit:
                    break
                unit = CodeUnit.from_labelled_function(labelled)
                results = {
                    "declared": parser.parse(
                        unit.source, language=unit.language, unit_id=unit.unit_id
                    )
                }
                if compare:
                    for grammar in GRAMMARS:
                        results[grammar] = parser.parse(
                            unit.source, language=grammar, unit_id=unit.unit_id
                        )
                graph = build_ast_graph(results["declared"])
                accumulator.record(unit, results, graph)

            shards[split] = accumulator.to_dict()
            _LOGGER.info(
                "Shard reported",
                extra={
                    "split": split,
                    "units": accumulator.units,
                    "clean_pct": shards[split]["parse"]["clean_as_declared_pct"],
                    "mean_nodes": shards[split]["graph"]["mean_nodes_per_unit"],
                },
            )
            mlflow.log_metrics(
                {
                    f"parse/{split}/clean_pct": shards[split]["parse"]["clean_as_declared_pct"],
                    f"graph/{split}/mean_nodes": shards[split]["graph"]["mean_nodes_per_unit"],
                    f"graph/{split}/mean_edges": shards[split]["graph"]["mean_edges_per_unit"],
                }
            )

        payload = {
            "run_id": run_id,
            "builder": str(cfg.graph.builder),
            "language_policy": str(cfg.graph.language),
            "limit": limit,
            "compare_grammars": compare,
            "shards": shards,
            "totals": _totals(shards),
        }
        output_path = Path(settings.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        mlflow.log_artifact(str(output_path), artifact_path="representation")

    _LOGGER.info("Representation report written", extra={"path": str(output_path)})
    return payload


def _totals(shards: dict[str, Any]) -> dict[str, Any]:
    """Sum the per-shard sections into a corpus-wide row.

    Args:
        shards: Per-shard report sections.

    Returns:
        The corpus totals.
    """
    units = sum(shard["units"] for shard in shards.values())
    clean = sum(shard["parse"]["clean_as_declared"] for shard in shards.values())
    nodes = sum(shard["graph"]["total_nodes"] for shard in shards.values())
    edges = sum(shard["graph"]["total_edges"] for shard in shards.values())
    source_lines = sum(shard["line_coverage"]["source_lines"] for shard in shards.values())
    covered = sum(shard["line_coverage"]["lines_with_a_node"] for shard in shards.values())
    labelled = sum(shard["line_coverage"]["labelled_lines"] for shard in shards.values())
    labelled_covered = sum(
        shard["line_coverage"]["labelled_lines_with_a_node"] for shard in shards.values()
    )
    categories: Counter[str] = Counter()
    for shard in shards.values():
        categories.update(shard["parse"]["failure_categories"])

    totals = {
        "units": units,
        "clean_as_declared": clean,
        "clean_as_declared_pct": round(100 * clean / max(units, 1), 2),
        "failure_categories": dict(categories.most_common()),
        "total_nodes": nodes,
        "total_edges": edges,
        "mean_nodes_per_unit": round(nodes / max(units, 1), 2),
        "mean_edges_per_unit": round(edges / max(units, 1), 2),
        "source_lines": source_lines,
        "lines_with_a_node_pct": round(100 * covered / max(source_lines, 1), 2),
        "labelled_lines": labelled,
        "labelled_lines_with_a_node": labelled_covered,
        "labelled_lines_with_a_node_pct": round(100 * labelled_covered / max(labelled, 1), 2),
    }
    if any("grammar_comparison" in shard for shard in shards.values()):
        totals["grammar_comparison"] = {
            key: sum(shard["grammar_comparison"][key] for shard in shards.values())
            for key in ("clean_as_c", "clean_as_cpp", "clean_under_either", "clean_under_neither")
        }
    return totals


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    report(cfg)


if __name__ == "__main__":
    main()
