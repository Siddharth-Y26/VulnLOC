"""Render the four graph types for one real corpus unit, as inline SVG.

``Phases.md`` Phase 2 lists a rendered graph as a deliverable, and
``representation/visualize.py`` emits Graphviz DOT for it — but DOT needs the ``dot``
binary, which is not installed on this machine, and a spring-layout picture of 250 nodes
tells a reader nothing anyway.

This script lays the graph out on **the line axis instead of a spring layout**: y is the
node's 1-indexed line, x is its column within that line. That is the project's actual
claim made visible — the semantic branch and the structural branch meet because every
node knows its line — and a line-mapping bug shows up as a node sitting on the wrong row
next to the source text, which is the whole point of the renderer per
``visualize.py``'s docstring.

Reads the preprocessed shards only; it never opens a Big-Vul CSV. Writes
``artifacts/representation/figures.json`` — the SVG strings plus the unit's provenance,
so the figure in a slide can cite the unit it came from.

Usage::

    python scripts/make_review_figures.py
    python scripts/make_review_figures.py --split clean_eval --max-lines 40
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

from vulnloc.data.cache import iter_shard
from vulnloc.representation import GRAPH_REGISTRY, TreeSitterParser
from vulnloc.representation.units import CodeUnit
from vulnloc.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from vulnloc.representation.graphs.base import CodeGraph

_LOGGER = get_logger(__name__)

BUILDERS = ("ast", "cfg", "dfg", "cpg")

# docs/Design.md §2.5 — the project's categorical graph palette. Not re-picked here.
EDGE_COLOUR = {
    "ast": "#5B8FF9",
    "cfg": "#61DDAA",
    "dfg": "#F6BD16",
    "call": "#7262FD",
}
FLAGGED = "#D1495B"
NODE_FILL = "#8C97A3"

ROW_H = 17.0
PAD_L = 54.0
PAD_T = 26.0
PAD_R = 18.0
COL_W = 3.1
MAX_COL = 108


def pick_unit(
    cache_dir: Path,
    split: str,
    *,
    max_lines: int,
    min_lines: int,
    max_flag_frac: float,
    scan: int,
) -> tuple[CodeUnit, dict[str, CodeGraph]]:
    """Find a small, cleanly-parsing, line-labelled vulnerable unit and build its graphs.

    A figure is only honest if the unit behind it is ordinary. The filter is therefore
    on size and parse cleanliness alone — the first unit that fits is taken, rather than
    the prettiest of many, so the choice is not a silent cherry-pick of an easy example.

    Args:
        cache_dir: Directory holding the shards.
        split: Partition to scan.
        max_lines: Reject units longer than this — a figure past ~40 rows is unreadable.
        min_lines: Reject units shorter than this; a 5-line stub shows no control flow.
        max_flag_frac: Reject units where more than this fraction of lines is flagged.
            Big-Vul fix commits routinely rewrite a whole function, so a large minority
            of positives have nearly every line flagged — a real property of the corpus
            (``Memory.md`` §1.6), but one that makes the localization figure unreadable
            because there is nothing left un-flagged to contrast against.
        scan: Stop after this many units so the scan stays under a few seconds.

    Returns:
        The chosen unit and its four graphs, keyed by builder name.

    Raises:
        SystemExit: If no unit in the scanned prefix satisfies the filter.
    """
    parser = TreeSitterParser()
    for index, labelled in enumerate(iter_shard(cache_dir, split)):
        if index >= scan:
            break
        unit = CodeUnit.from_labelled_function(labelled)
        if unit.target != 1 or not unit.vulnerable_lines:
            continue
        if not min_lines <= unit.n_lines <= max_lines:
            continue
        if len(unit.vulnerable_lines) > max_flag_frac * unit.n_lines:
            continue
        parsed = parser.parse_with_policy(
            unit.source,
            policy="best_of_two",
            declared=unit.language,
            unit_id=unit.unit_id,
        )
        if parsed.error_nodes or parsed.missing_nodes:
            continue
        graphs = {name: GRAPH_REGISTRY.get(name)().build(parsed) for name in BUILDERS}
        # A figure that shows no data flow does not illustrate the structural branch.
        if not any(e.edge_type == "dfg" for e in graphs["cpg"].edges):
            continue
        if not any(e.edge_type == "cfg" for e in graphs["cpg"].edges):
            continue
        _LOGGER.info(
            "Unit selected",
            extra={
                "unit": unit.unit_id,
                "lines": unit.n_lines,
                "grammar": parsed.language,
                "ast_nodes": len(graphs["ast"].nodes),
            },
        )
        return unit, graphs
    raise SystemExit(
        f"No unit in the first {scan} of {split!r} matched "
        f"({min_lines}-{max_lines} lines, clean parse, cfg+dfg present). Widen the filter."
    )


def _columns(unit: CodeUnit, graph: CodeGraph) -> dict[int, float]:
    """Map each node to an x offset from the start of its own line.

    Derived from ``start_byte`` and the line's byte span, never from ``node.start_point``
    — that segfaults the interpreter on real corpus input (landmine #14).

    Args:
        unit: The source unit, for its line map.
        graph: The graph whose nodes are being placed.

    Returns:
        node_id to x pixel offset.
    """
    placed: dict[int, float] = {}
    for node in graph.nodes:
        start, _ = unit.line_map.line_byte_span(node.line_no)
        col = min(node.start_byte - start, MAX_COL)
        placed[node.node_id] = PAD_L + col * COL_W
    return placed


def render_svg(
    unit: CodeUnit,
    graph: CodeGraph,
    *,
    edge_types: tuple[str, ...],
) -> str:
    """Draw one graph laid out on the line axis.

    Args:
        unit: The source unit; supplies the row count and the flagged lines.
        graph: The graph to draw.
        edge_types: Which edge types to draw. The CPG is a union, so restricting this
            is exactly how ablation A3 selects a sub-graph — the figure and the ablation
            use the same knob.

    Returns:
        A standalone ``<svg>`` element. Theme-neutral: it uses only the palette colours
        and ``currentColor``, so it reads on either ground.
    """
    xs = _columns(unit, graph)
    height = PAD_T + unit.n_lines * ROW_H + 10
    width = PAD_L + MAX_COL * COL_W + PAD_R

    def y_of(line: int) -> float:
        return PAD_T + (line - 0.5) * ROW_H

    parts: list[str] = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" '
        f'role="img" aria-label="{html.escape(graph.graph_type)} graph of '
        f'{html.escape(unit.unit_id)} laid out on the line axis" '
        'xmlns="http://www.w3.org/2000/svg" class="gfig">'
    ]

    # Row bands: the line axis itself. Flagged lines get the risk colour from Design.md.
    for line in range(1, unit.n_lines + 1):
        top = PAD_T + (line - 1) * ROW_H
        flagged = line in unit.vulnerable_lines
        if flagged:
            parts.append(
                f'<rect x="0" y="{top:.1f}" width="{width:.0f}" height="{ROW_H:.1f}" '
                f'fill="{FLAGGED}" opacity="0.13"/>'
                f'<rect x="0" y="{top:.1f}" width="3" height="{ROW_H:.1f}" fill="{FLAGGED}"/>'
            )
        parts.append(
            f'<text x="{PAD_L - 12:.0f}" y="{y_of(line) + 3.5:.1f}" text-anchor="end" '
            f'class="gnum">{line}</text>'
        )

    for edge_type in edge_types:
        colour = EDGE_COLOUR[edge_type]
        thin = edge_type == "ast"
        for edge in graph.edges:
            if edge.edge_type != edge_type:
                continue
            x1, x2 = xs[edge.source], xs[edge.target]
            y1 = y_of(graph.nodes[edge.source].line_no)
            y2 = y_of(graph.nodes[edge.target].line_no)
            if y1 == y2 and abs(x1 - x2) < 0.5:
                continue
            # Bow same-row edges outward, or they collapse into the row band.
            bow = 9.0 if abs(y1 - y2) < 1 else 0.0
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 - bow
            parts.append(
                f'<path d="M{x1:.1f} {y1:.1f} Q{cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}" '
                f'fill="none" stroke="{colour}" '
                f'stroke-width="{0.5 if thin else 1.15}" '
                f'opacity="{0.42 if thin else 0.85}"/>'
            )

    for node in graph.nodes:
        flagged = node.line_no in unit.vulnerable_lines
        parts.append(
            f'<circle cx="{xs[node.node_id]:.1f}" cy="{y_of(node.line_no):.1f}" '
            f'r="{2.4 if node.is_named else 1.5:.1f}" '
            f'fill="{FLAGGED if flagged else NODE_FILL}" '
            f'opacity="{0.95 if node.is_named else 0.55}"><title>'
            f"{html.escape(node.node_type)} · line {node.line_no}</title></circle>"
        )

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    """Select a unit, render the four figures, and write them to an artefact."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="./artifacts/data/cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-lines", type=int, default=12)
    ap.add_argument("--max-lines", type=int, default=34)
    ap.add_argument("--max-flag-frac", type=float, default=0.35)
    ap.add_argument("--scan", type=int, default=20000)
    ap.add_argument("--out", default="./artifacts/representation/figures.json")
    args = ap.parse_args()

    setup_logging(json_to_file=False)
    cache_dir = Path(args.cache_dir)
    unit, graphs = pick_unit(
        cache_dir,
        args.split,
        max_lines=args.max_lines,
        min_lines=args.min_lines,
        max_flag_frac=args.max_flag_frac,
        scan=args.scan,
    )

    subsets: dict[str, tuple[str, ...]] = {
        "ast": ("ast",),
        "cfg": ("cfg",),
        "dfg": ("dfg",),
        "cpg": ("ast", "cfg", "dfg", "call"),
    }
    figures = {
        name: {
            "svg": render_svg(unit, graphs[name], edge_types=subsets[name]),
            "nodes": len(graphs[name].nodes),
            "edges": sum(1 for e in graphs[name].edges if e.edge_type in subsets[name]),
        }
        for name in BUILDERS
    }

    payload = {
        "unit_id": unit.unit_id,
        "split": args.split,
        "declared_language": unit.language,
        "n_lines": unit.n_lines,
        "target": unit.target,
        "vulnerable_lines": sorted(unit.vulnerable_lines),
        "metadata": {k: str(v) for k, v in unit.metadata.items()},
        "source_lines": list(unit.lines),
        "figures": figures,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOGGER.info("Figures written", extra={"path": str(out), "unit": unit.unit_id})


if __name__ == "__main__":
    main()
