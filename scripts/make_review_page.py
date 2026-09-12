"""Assemble the Phase-review page from measured artefacts.

Every number on the page comes from a file in ``artifacts/`` or from a doc in ``docs/``;
nothing is typed in by hand here except prose. ``Rules.md`` §0.3 forbids hand-edited
figures, and the same reasoning applies to a review deck — so the page is generated,
and re-running this script after a new report picks the new numbers up.

Usage::

    python scripts/make_review_page.py --out artifacts/review/review.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

FIGURE_ORDER = ("ast", "cfg", "dfg", "cpg")

FIGURE_BLURB = {
    "ast": (
        "Abstract Syntax Tree",
        "Every token and every construct, one node each. Dense and complete — but it "
        "encodes how the code is <em>written</em>, not how it <em>runs</em>.",
    ),
    "cfg": (
        "Control Flow Graph",
        "Statement-to-statement execution order, including the branch at line 8. Sparse "
        "by design: it answers &ldquo;can this line reach that one?&rdquo;",
    ),
    "dfg": (
        "Data Flow Graph",
        "Definition-to-use edges over identifier names. The edge from the allocation on "
        "line 5 to its uses downstream is the one a CWE-200 leak lives on.",
    ),
    "cpg": (
        "Code Property Graph",
        "The literal union of the three above over one shared node set. Ablation A3 is "
        "an <code>edge_types</code> argument here, not a second construction.",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON artefact.

    Args:
        path: File to read.

    Returns:
        The decoded object.

    Raises:
        SystemExit: If the artefact is missing, naming the script that produces it.
    """
    if not path.exists():
        raise SystemExit(f"Missing artefact: {path}. Generate it before building the page.")
    return json.loads(path.read_text(encoding="utf-8"))


def code_block(lines: list[str], flagged: set[int]) -> str:
    """Render the source with a gutter, flagged rows marked by colour and by a label.

    Args:
        lines: 1-indexed via ``lines[i - 1]``.
        flagged: 1-indexed flagged line numbers.

    Returns:
        HTML for the code viewer.
    """
    rows = []
    for number, text in enumerate(lines, start=1):
        mark = " flag" if number in flagged else ""
        label = '<span class="sr">flagged: </span>' if number in flagged else ""
        rows.append(
            f'<div class="crow{mark}"><span class="cnum">{number}</span>'
            f'<code>{label}{html.escape(text) or "&nbsp;"}</code></div>'
        )
    return "".join(rows)


def figure_panel(name: str, figure: dict[str, Any]) -> str:
    """Render one graph figure with its caption and counts.

    Args:
        name: Builder name.
        figure: The ``figures.json`` entry — SVG string plus node and edge counts.

    Returns:
        HTML for one panel.
    """
    title, blurb = FIGURE_BLURB[name]
    return (
        f'<figure class="gpanel"><figcaption><h4>{title} '
        f'<span class="tag">{name}</span></h4>'
        f"<p>{blurb}</p>"
        f'<p class="counts">{figure["nodes"]} nodes · {figure["edges"]} edges</p>'
        f"</figcaption>{figure['svg']}</figure>"
    )


def main() -> None:
    """Build the page."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", default="./artifacts/representation/figures.json")
    ap.add_argument("--report", default="./artifacts/representation/report_ast.json")
    ap.add_argument("--body", default="./scripts/review_body.html")
    ap.add_argument("--out", default="./artifacts/review/review.html")
    args = ap.parse_args()

    figures = read_json(Path(args.figures))
    report = read_json(Path(args.report))
    body = Path(args.body).read_text(encoding="utf-8")

    meta = figures["metadata"]
    unit_caption = (
        f"{html.escape(figures['unit_id'])} · {html.escape(meta.get('project', '?'))} · "
        f"{html.escape(meta.get('CVE ID', '?'))} · {html.escape(meta.get('CWE ID', '?'))} · "
        f"split <code>{html.escape(figures['split'])}</code> · parsed as "
        f"<code>cpp</code> under ADR-005"
    )

    panels = "".join(figure_panel(name, figures["figures"][name]) for name in FIGURE_ORDER)
    code = code_block(figures["source_lines"], set(figures["vulnerable_lines"]))
    clean_eval = report["shards"]["clean_eval"]

    page = (
        body.replace("<!--CODE-->", code)
        .replace("<!--PANELS-->", panels)
        .replace("<!--UNITCAPTION-->", unit_caption)
        .replace("<!--REPORTRUN-->", html.escape(report["run_id"]))
        .replace("<!--CEMEANNODES-->", f"{clean_eval['graph']['mean_nodes_per_unit']:,.0f}")
        .replace(
            "<!--CECOVERAGE-->",
            f"{clean_eval['line_coverage']['labelled_lines_with_a_node_pct']:.2f}",
        )
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
