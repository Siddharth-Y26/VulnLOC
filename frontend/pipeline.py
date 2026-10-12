"""Run the real VulnLoc pipeline over one pasted function, and report what each stage did.

This is the analysis half of the inspector; ``server.py`` is only HTTP around it.

**Everything here is computed live from `src/vulnloc`.** No stage is simulated, no number
is invented. Where a phase is not built, this module returns nothing for it rather than a
plausible-looking placeholder — `Rules.md` forbids reporting a figure that did not come
from a real run, and a demo is not an exemption.

**Scores come from one place only: :func:`findings`**, which calls ``scoring.py`` (the
Phase 3 semantic model, seed 42) — added 2026-09-15 by the owner's decision. Everything
else in this file is the detailed report and computes no score. Run ``1843412a…`` remains
void (ADR-006): it learned the corpus's indentation artefact rather than the vulnerability,
which is why :func:`findings` never scores un-normalized text.

This is **not** the Phase 9 serving API. Phase 9 is unstarted, and `Phases.md` forbids
building ahead; this is a development instrument that lives outside `src/vulnloc`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from vulnloc.representation import GRAPH_REGISTRY, TreeSitterParser
from vulnloc.representation.graphs.base import EDGE_TYPES
from vulnloc.representation.line_map import split_lines
from vulnloc.representation.units import CodeUnit
from vulnloc.utils.text import normalize_source_whitespace

__all__ = [
    "AnalysisOptions",
    "analyse",
    "findings",
    "measured_results",
    "sample_functions",
    "structural_findings",
]

# A pasted function is small, but a pathological paste is not. The AST of a 500-line
# function reaches a few thousand nodes, and shipping all of them to a browser that draws
# each as an SVG element is how a viewer becomes unusable. The cap is reported alongside
# the full count so the reader can tell a complete picture from a partial one.
MAX_RENDER_NODES = 900
MAX_RENDER_EDGES = 2400

# The tokenizer is heavy to construct and identical between requests.
_TOKENIZER_CACHE: dict[tuple[str, int, int, bool], Any] = {}
_PARSER: TreeSitterParser | None = None


def _parser() -> TreeSitterParser:
    """Return the process-wide parser, built on first use."""
    global _PARSER
    if _PARSER is None:
        _PARSER = TreeSitterParser()
    return _PARSER


@dataclass(frozen=True)
class AnalysisOptions:
    """What the inspector was asked to run.

    Attributes:
        policy: Grammar-selection policy (ADR-005) — ``best_of_two``, ``c``, ``cpp`` or
            ``declared``.
        declared: The language a corpus would have claimed, consulted only under
            ``declared``. Pasted code has no corpus, so this is the user's assertion.
        graph: Which builder to run — ``ast``, ``cfg``, ``dfg`` or ``cpg``.
        edge_types: CPG edge subset (ablation A3). Empty means all of them.
        normalize: Apply the ADR-006 whitespace fix.
        windowing: Window long functions instead of truncating them.
        max_length: Encoder window length in tokens.
        stride: Overlap between consecutive windows, in body tokens.
        tokenizer_name: HuggingFace tokenizer id.
    """

    policy: str = "best_of_two"
    declared: str = "C"
    graph: str = "cpg"
    edge_types: tuple[str, ...] = ()
    normalize: bool = True
    windowing: bool = True
    max_length: int = 512
    stride: int = 128
    tokenizer_name: str = "microsoft/codebert-base"


def _leading_whitespace_tell(lines: list[str]) -> list[int]:
    """Which lines the ADR-006 leak rule would flag, 1-indexed.

    The rule reads no code and no label. On Big-Vul's shipped text it recovers the flaw
    label at precision 1.0000 with zero false positives on all six partitions, which is
    why the corpus is normalized before anything sees it. Run on pasted code it is a
    demonstration, not a prediction: it says "these are the lines that would have been
    marked", and on ordinary indented code that is most of them.

    Args:
        lines: Source lines.

    Returns:
        1-indexed line numbers beginning with a space or tab.
    """
    return [i for i, line in enumerate(lines, start=1) if line[:1] in (" ", "\t")]


def _normalization_report(raw: str, options: AnalysisOptions) -> dict[str, Any]:
    """Report what the ADR-006 whitespace fix changes about this paste.

    Args:
        raw: The source exactly as pasted.
        options: Analysis options.

    Returns:
        A JSON-serializable section.
    """
    raw_lines = split_lines(raw)
    fixed_lines = split_lines(normalize_source_whitespace(raw))
    changed = [
        {"line": i, "before": before, "after": after}
        for i, (before, after) in enumerate(zip(raw_lines, fixed_lines, strict=True), start=1)
        if before != after
    ]
    tell = _leading_whitespace_tell(raw_lines)
    return {
        "applied": options.normalize,
        "n_lines": len(raw_lines),
        "n_lines_changed": len(changed),
        # Capped: a heavily indented paste changes every line and the list adds nothing
        # after the first screenful.
        "changed": changed[:60],
        "changed_truncated": len(changed) > 60,
        "line_count_preserved": len(raw_lines) == len(fixed_lines),
        "leak_tell": {
            "lines_flagged_before": tell,
            "lines_flagged_after": _leading_whitespace_tell(fixed_lines),
        },
    }


def _parse_report(result: Any, options: AnalysisOptions) -> dict[str, Any]:
    """Summarise one parse, damage included.

    Args:
        result: The :class:`ParseResult`.
        options: Analysis options.

    Returns:
        A JSON-serializable section.
    """
    return {
        "policy": options.policy,
        "language_used": result.language,
        "is_clean": result.is_clean,
        "error_nodes": result.error_nodes,
        "missing_nodes": result.missing_nodes,
        "first_error_line": result.first_error_line,
    }


def _grammar_comparison(source: str, options: AnalysisOptions) -> list[dict[str, Any]]:
    """Parse under each policy so ADR-005's choice is visible rather than asserted.

    Args:
        source: The text to parse.
        options: Analysis options, for the declared language.

    Returns:
        One row per policy, each with its damage counts.
    """
    rows = []
    for policy in ("c", "cpp", "declared", "best_of_two"):
        try:
            parsed = _parser().parse_with_policy(
                source, policy=policy, declared=options.declared, unit_id="paste"
            )
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never swallowed
            rows.append({"policy": policy, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(
            {
                "policy": policy,
                "language_used": parsed.language,
                "is_clean": parsed.is_clean,
                "error_nodes": parsed.error_nodes,
                "missing_nodes": parsed.missing_nodes,
                "first_error_line": parsed.first_error_line,
            }
        )
    return rows


def _graph_report(result: Any, options: AnalysisOptions) -> dict[str, Any]:
    """Build the requested graph and report it, node-by-node.

    Args:
        result: The :class:`ParseResult`.
        options: Analysis options.

    Returns:
        A JSON-serializable section carrying nodes and edges for rendering.

    Raises:
        RegistryError: If the builder name is not registered.
    """
    builder = GRAPH_REGISTRY.get(options.graph)()
    graph = builder.build(result)

    edges = graph.edges
    if options.graph == "cpg" and options.edge_types:
        wanted = set(options.edge_types)
        edges = tuple(edge for edge in edges if edge.edge_type in wanted)

    # Nodes are sent in id order, so a truncated graph is a prefix of the real one rather
    # than an arbitrary sample. Edges are then filtered to the surviving node set: an
    # edge pointing at a node the browser never received would draw from nowhere.
    kept = graph.nodes[:MAX_RENDER_NODES]
    kept_ids = {node.node_id for node in kept}
    drawable = [e for e in edges if e.source in kept_ids and e.target in kept_ids]
    drawable_capped = drawable[:MAX_RENDER_EDGES]

    covered = graph.lines_covered
    return {
        "type": options.graph,
        "n_nodes": len(graph.nodes),
        "n_edges": len(edges),
        "edge_types": dict(sorted(graph.edge_type_counts().items())),
        "available_edge_types": list(EDGE_TYPES),
        "n_lines": graph.n_lines,
        "lines_covered": sorted(covered),
        "line_coverage_pct": (
            round(100 * len(covered) / graph.n_lines, 2) if graph.n_lines else 0.0
        ),
        "lines_uncovered": sorted(set(range(1, graph.n_lines + 1)) - covered),
        "nodes": [
            {
                "id": n.node_id,
                "type": n.node_type,
                "line": n.line_no,
                "named": n.is_named,
                "text": n.text,
            }
            for n in kept
        ],
        "edges": [{"s": e.source, "t": e.target, "type": e.edge_type} for e in drawable_capped],
        "nodes_truncated": len(graph.nodes) > len(kept),
        "edges_truncated": len(drawable) > len(drawable_capped),
        "render_caps": {"nodes": MAX_RENDER_NODES, "edges": MAX_RENDER_EDGES},
    }


def _tokenizer(options: AnalysisOptions) -> Any:
    """Return a cached :class:`LineTokenizer` for these options.

    Args:
        options: Analysis options.

    Returns:
        The tokenizer.
    """
    from frontend.scoring import IMPORT_LOCK

    with IMPORT_LOCK:
        from transformers import AutoTokenizer

    from vulnloc.representation.tokenization import LineTokenizer

    key = (options.tokenizer_name, options.max_length, options.stride, options.windowing)
    if key not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[key] = LineTokenizer(
            AutoTokenizer.from_pretrained(options.tokenizer_name),
            max_length=options.max_length,
            stride=options.stride,
            windowing=options.windowing,
        )
    return _TOKENIZER_CACHE[key]


def _token_report(unit: CodeUnit, options: AnalysisOptions) -> dict[str, Any]:
    """Tokenize, window, and report what a truncating encoder would have lost.

    The truncation comparison is the point rather than a footnote: measured corpus-wide,
    truncating at 512 tokens destroys 38.1% of all flagged lines and 44.5% on
    `clean_eval` (run ``b7ccc9ca…``). Windowing keeps 100%.

    Args:
        unit: The unit to encode.
        options: Analysis options.

    Returns:
        A JSON-serializable section.
    """
    tokenizer = _tokenizer(options)
    encoded = tokenizer.encode(unit)

    # Tokens per line, taken as the MAXIMUM across windows rather than the sum. Windows
    # overlap by the stride, so a line near a boundary appears in two of them; summing
    # would report the same tokens twice and make the busiest-looking lines the ones that
    # merely straddle a seam. The max is the count from whichever window holds the line
    # most completely, which is the number a reader means by "how long is this line".
    tokens_per_line: dict[int, int] = {}
    for window in encoded.windows:
        counts: dict[int, int] = {}
        for line in window.token_to_line:
            if line >= 1:
                counts[line] = counts.get(line, 0) + 1
        for line, count in counts.items():
            tokens_per_line[line] = max(tokens_per_line.get(line, 0), count)

    windows = [
        {
            "index": window.window_index,
            "n_real_tokens": window.n_real_tokens,
            "lines": sorted(window.lines_covered),
        }
        for window in encoded.windows
    ]
    covered = encoded.lines_covered
    first_window = sorted(encoded.windows[0].lines_covered) if encoded.windows else []
    all_lines = set(range(1, unit.n_lines + 1))
    return {
        "n_body_tokens": encoded.n_body_tokens,
        "n_windows": encoded.n_windows,
        "max_length": options.max_length,
        "stride": options.stride,
        "windowing": options.windowing,
        "windows": windows,
        "tokens_per_line": {str(k): v for k, v in sorted(tokens_per_line.items())},
        "lines_covered": sorted(covered),
        "truncation": {
            "lines_in_first_window": first_window,
            "lines_beyond_first_window": sorted(all_lines - set(first_window)),
            "would_lose_lines": len(all_lines) - len(first_window),
        },
    }


def _line_axis(
    unit: CodeUnit, graph_section: dict[str, Any], token_section: dict[str, Any]
) -> dict[str, Any]:
    """Collapse every representation onto the shared line axis.

    This is the project's central construction (`Architecture.md` §4.3): the semantic
    branch pools tokens onto lines, the structural branch pools nodes onto lines, and
    fusion happens on that common index. Rendering it per line is what makes a line
    mapping bug visible to a person.

    Args:
        unit: The unit.
        graph_section: Output of :func:`_graph_report`.
        token_section: Output of :func:`_token_report`.

    Returns:
        A JSON-serializable section, one row per source line.
    """
    tokens_per_line = {int(k): v for k, v in token_section["tokens_per_line"].items()}

    nodes_per_line: dict[int, int] = {}
    types_per_line: dict[int, list[str]] = {}
    for node in graph_section["nodes"]:
        line = node["line"]
        nodes_per_line[line] = nodes_per_line.get(line, 0) + 1
        if node["named"] and len(types_per_line.setdefault(line, [])) < 6:
            types_per_line[line].append(node["type"])

    windows_per_line: dict[int, list[int]] = {}
    for window in token_section["windows"]:
        for line in window["lines"]:
            windows_per_line.setdefault(line, []).append(window["index"])

    covered_by_tokens = set(token_section["lines_covered"])
    covered_by_graph = set(graph_section["lines_covered"])

    rows = []
    for number, text in enumerate(unit.lines, start=1):
        rows.append(
            {
                "line": number,
                "text": text,
                "n_tokens": tokens_per_line.get(number, 0),
                "n_nodes": nodes_per_line.get(number, 0),
                "node_types": types_per_line.get(number, []),
                "in_windows": windows_per_line.get(number, []),
                "has_tokens": number in covered_by_tokens,
                "has_nodes": number in covered_by_graph,
                "blank": not text.strip(),
            }
        )
    return {
        "n_lines": unit.n_lines,
        "rows": rows,
        # The two coverage ceilings, side by side, because they are different failures:
        # a line with no tokens is invisible to the semantic branch, a line with no node
        # is invisible to the structural one, and corpus-wide 7.84% of flagged lines fall
        # into the second (landmine #17).
        "lines_without_tokens": sorted(set(range(1, unit.n_lines + 1)) - covered_by_tokens),
        "lines_without_nodes": sorted(set(range(1, unit.n_lines + 1)) - covered_by_graph),
    }


def analyse(code: str, options: AnalysisOptions) -> dict[str, Any]:
    """Run every built stage of the pipeline over ``code``.

    Args:
        code: The pasted source.
        options: What to run.

    Returns:
        A JSON-serializable report, one section per stage.

    Raises:
        ValueError: If ``code`` is empty.
    """
    if not code.strip():
        raise ValueError("Paste a function first — there is nothing to analyse.")

    normalization = _normalization_report(code, options)

    # CodeUnit is the L0 -> L1 boundary and applies ADR-006 itself. Passing the flag
    # through rather than pre-normalizing keeps the inspector on the same code path the
    # training pipeline uses.
    unit = CodeUnit.from_source(
        "paste",
        code,
        language=options.declared,
        normalize_whitespace=options.normalize,
    )

    parsed = _parser().parse_with_policy(
        unit.source, policy=options.policy, declared=options.declared, unit_id=unit.unit_id
    )

    graph_section = _graph_report(parsed, options)
    token_section = _token_report(unit, options)

    return {
        "source": {
            "n_lines": unit.n_lines,
            "n_chars": len(unit.source),
            "lines": list(unit.lines),
        },
        "normalization": normalization,
        "parse": _parse_report(parsed, options),
        "grammars": _grammar_comparison(unit.source, options),
        "graph": graph_section,
        "tokens": token_section,
        "axis": _line_axis(unit, graph_section, token_section),
        "options": {
            "policy": options.policy,
            "graph": options.graph,
            "edge_types": list(options.edge_types),
            "normalize": options.normalize,
            "windowing": options.windowing,
            "max_length": options.max_length,
            "stride": options.stride,
        },
    }


CONTEXT_LINES = 2
"""Source lines shown either side of a finding."""

_PUNCTUATION_ONLY = re.compile(r"^[\s{}()\[\];,]*$")
"""Lines with no identifier, keyword or literal — braces, a lone `);`, blank."""


def findings(code: str) -> dict[str, Any]:
    """The Findings tab: which lines to look at first, and what a known pattern says about them.

    Two sources, kept separate all the way to the screen (see ``fix_templates.py``):
    the **model** ranks lines and gives no reason; a **template** matches line text and
    knows nothing about the model. Normalization is always on here — a score computed on
    un-normalized text would be the ADR-006 leak, not a finding.

    Args:
        code: The pasted source.

    Returns:
        A JSON-serializable report.

    Raises:
        ValueError: If ``code`` is empty.
    """
    from frontend.fix_templates import match_templates
    from frontend.scoring import score_lines

    if not code.strip():
        raise ValueError("Paste a function first — there is nothing to analyse.")

    unit = CodeUnit.from_source("paste", code, language="C", normalize_whitespace=True)
    lines = list(unit.lines)
    scored = score_lines(unit)
    patterns = match_templates(lines)
    by_line = {entry["line"]: entry for entry in scored["scores"]}

    def context(line: int) -> list[dict[str, Any]]:
        first = max(1, line - CONTEXT_LINES)
        last = min(len(lines), line + CONTEXT_LINES)
        return [{"line": n, "text": lines[n - 1]} for n in range(first, last + 1)]

    # Cards go to the highest-ranked lines that carry code. A lone `}` in the top five of a
    # six-line function is noise to a reader, not a finding. The model's own ranks are kept,
    # so a card can read "#4" after a skipped "#3". Removing lines that essentially never
    # hold the flaw can only raise the chance the real line is shown, so the measured Top-5
    # quoted beside these cards stays a lower bound rather than becoming an overstatement.
    ranked = sorted((e for e in scored["scores"] if e["rank"] is not None), key=lambda e: e["rank"])
    skipped = [e["line"] for e in ranked if _PUNCTUATION_ONLY.match(lines[e["line"] - 1])]
    chosen = [e for e in ranked if e["line"] not in skipped][: scored["top_k"]]
    top_lines = {entry["line"] for entry in chosen}
    top = [
        {
            **entry,
            "text": lines[entry["line"] - 1],
            "context": context(entry["line"]),
            "patterns": patterns.get(entry["line"], []),
        }
        for entry in chosen
    ]
    skipped_above = [
        line
        for line in skipped
        if by_line[line]["rank"] <= max((e["rank"] for e in chosen), default=0)
    ]
    elsewhere = [
        {
            "line": line,
            "text": lines[line - 1],
            "rank": by_line.get(line, {}).get("rank"),
            "score": by_line.get(line, {}).get("score"),
            "patterns": found,
        }
        for line, found in sorted(patterns.items())
        if line not in top_lines
    ]
    return {
        "lines": lines,
        "scores": scored["scores"],
        "top": top,
        "skipped_punctuation_lines": skipped_above,
        "patterns_elsewhere": elsewhere,
        "unscored_lines": scored["unscored_lines"],
        "beyond_line_cap": scored["beyond_line_cap"],
        "top_k": scored["top_k"],
        "model": scored["model"],
        "measured": scored["measured"],
    }


def structural_findings(code: str) -> dict[str, Any]:
    """The Detailed report's Structural branch panel: the GNN's top lines on this paste.

    Built like :func:`findings` — normalization always on, punctuation-only lines skipped
    with the model's own ranks kept — so the two rankings are comparable line for line.

    Args:
        code: The pasted source.

    Returns:
        A JSON-serializable report.

    Raises:
        ValueError: If ``code`` is empty.
    """
    from frontend.structural import score_lines

    if not code.strip():
        raise ValueError("Paste a function first — there is nothing to analyse.")

    unit = CodeUnit.from_source("paste", code, language="C", normalize_whitespace=True)
    lines = list(unit.lines)
    scored = score_lines(unit)
    ranked = sorted((e for e in scored["scores"] if e["rank"] is not None), key=lambda e: e["rank"])
    chosen = [e for e in ranked if not _PUNCTUATION_ONLY.match(lines[e["line"] - 1])]
    return {
        # Every line's rank, so another model's table can say where this one put a line.
        "scores": scored["scores"],
        "top": [{**e, "text": lines[e["line"] - 1]} for e in chosen[: scored["top_k"]]],
        # Only lines holding code: a blank line having no node is not worth a reader's time.
        "lines_without_node": [
            line for line in scored["lines_without_node"] if lines[line - 1].strip()
        ],
        "model": scored["model"],
    }


_SEED_REPORTS = {
    "semantic": "artifacts/phase3/seed_report_semantic.json",
    "structural": "artifacts/phase4/seed_report_structural.json",
    "cross_attention": "artifacts/phase5/seed_report_fused_cross_attention.json",
    "concat": "artifacts/phase5/seed_report_fused_concat.json",
    "sum": "artifacts/phase5/seed_report_fused_sum.json",
    "gated": "artifacts/phase5/seed_report_fused_gated.json",
    "llm_only": "artifacts/phase5/seed_report_fused_llm_only.json",
}
"""Where each trained model's held-out numbers live. Read, never typed in."""


def measured_results() -> dict[str, Any]:
    """Held-out Top-5/Top-1 for every trained model, read from the seed reports on disk.

    Returns:
        ``{name: {"top_5", "top_5_std", "top_1", "n_seeds", "eval_split", "run_ids"}}``, or
        ``{name: {"missing": path}}`` for a report that has not been generated.
    """
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    results: dict[str, Any] = {}
    for name, relative in _SEED_REPORTS.items():
        path = repo / relative
        if not path.exists():
            results[name] = {"missing": relative}
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        headline = report["headline"]
        results[name] = {
            "top_5": headline["eval/line/top_5_accuracy"]["mean"],
            "top_5_std": headline["eval/line/top_5_accuracy"]["std"],
            "top_1": headline["eval/line/top_1_accuracy"]["mean"],
            "n_seeds": report.get("n_seeds"),
            "eval_split": report.get("eval_split"),
            "run_ids": [run["run_id"] for run in report.get("runs", [])],
        }
    return results


def sample_functions() -> list[dict[str, Any]]:
    """Every sample the page offers: the four pipeline samples, then the examples.

    The pipeline samples come first so ``?sample=0`` keeps opening the one it always did.

    Returns:
        ``{name, note, code, group, lines}`` per sample (``frontend.samples``).
    """
    from frontend.samples import examples

    pipeline = [{**sample, "group": "pipeline", "lines": []} for sample in _pipeline_samples()]
    return pipeline + examples()


def _pipeline_samples() -> list[dict[str, str]]:
    """Short C/C++ functions that exercise specific parts of the pipeline.

    Written for this tool rather than taken from the corpus: pasting a real Big-Vul
    function into a demo would put corpus text — and its flaw labels — on screen next to
    an interface that cannot score it, which invites exactly the misreading ADR-006 came
    from.

    Returns:
        Named samples with their source.
    """
    return [
        {
            "name": "Unchecked copy",
            "note": "A classic bounds bug. Small enough to read the whole AST.",
            "code": (
                "void copy_name(char *dst, const char *src)\n"
                "{\n"
                "    char buf[32];\n"
                "    int n = strlen(src);\n"
                "    if (dst == NULL)\n"
                "        return;\n"
                "    memcpy(buf, src, n);\n"
                "    strcpy(dst, buf);\n"
                "}\n"
            ),
        },
        {
            "name": "Loop and branch",
            "note": "Exercises the CFG: a loop, an early return and a fallthrough.",
            "code": (
                "int scan(const int *values, int count)\n"
                "{\n"
                "    int total = 0;\n"
                "    for (int i = 0; i < count; i++) {\n"
                "        if (values[i] < 0)\n"
                "            continue;\n"
                "        if (values[i] > 1000)\n"
                "            return -1;\n"
                "        total += values[i];\n"
                "    }\n"
                "    return total;\n"
                "}\n"
            ),
        },
        {
            "name": "C++ only",
            "note": "Parses cleanly as C++ and not as C — the evidence for ADR-005.",
            "code": (
                "void Handler::~Handler()\n"
                "{\n"
                "    delete this->buffer;\n"
                "    for (auto &entry : this->entries) {\n"
                "        entry.release();\n"
                "    }\n"
                "}\n"
            ),
        },
        {
            "name": "Reassigned pointer",
            "note": "Exercises the DFG: several definitions reaching one use.",
            "code": (
                "char *pick(char *a, char *b, int flag)\n"
                "{\n"
                "    char *p = a;\n"
                "    if (flag)\n"
                "        p = b;\n"
                "    p = p + 1;\n"
                "    return p;\n"
                "}\n"
            ),
        },
    ]
