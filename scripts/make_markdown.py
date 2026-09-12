"""Convert the generated review pages to Markdown.

The HTML pages are the source of truth; this is a projection of them, for people who
want to paste sections into a report, a thesis chapter or a slide deck. It is a
converter for *our* markup specifically, not a general one — the pages use a small,
known set of structures (question cards, callouts, stat rows, the code viewer, the graph
figures), and each gets the Markdown shape that actually reads well rather than a
mechanical tag-for-tag translation.

The graph figures cannot live inline: Markdown has no SVG element and most renderers
strip raw SVG. They are written out as separate ``.svg`` files next to the document and
linked as images, which every renderer handles and which also makes them reusable in a
slide.

Usage::

    python scripts/make_markdown.py
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

VOID = {"br", "hr", "img", "input", "meta", "link", "col", "wbr"}
SKIP = {"style", "title", "script"}

CALLOUT_TITLE = {
    "plain": "In plain terms",
    "warnbox": "Caution",
    "note": "Note",
}


@dataclass
class Node:
    """One element or text run in the parsed page.

    Attributes:
        tag: Element name, or ``"#text"`` for a text run.
        attrs: Attribute mapping.
        children: Child nodes in document order.
        text: Text content, for ``#text`` nodes only.
        parent: Owning node, or ``None`` at the root.
    """

    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Node] = field(default_factory=list)
    text: str = ""
    parent: Node | None = None

    def cls(self) -> set[str]:
        """Return the node's CSS classes."""
        return set(self.attrs.get("class", "").split())


class TreeBuilder(HTMLParser):
    """Build a :class:`Node` tree, dropping style and script content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self._stack = [self.root]
        self._muted = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open an element, unless it is inside skipped content."""
        if tag in SKIP:
            self._muted += 1
            return
        if self._muted:
            return
        node = Node(tag, {k: v or "" for k, v in attrs}, parent=self._stack[-1])
        self._stack[-1].children.append(node)
        if tag not in VOID:
            self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        """Close the nearest matching open element, tolerating stray end tags."""
        if tag in SKIP:
            self._muted = max(0, self._muted - 1)
            return
        if self._muted or tag in VOID:
            return
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        """Append a text run, discarding whitespace-only data."""
        if self._muted or not data.strip():
            return
        self._stack[-1].children.append(Node("#text", text=data, parent=self._stack[-1]))


def escape(text: str) -> str:
    """Escape Markdown control characters that appear in our prose.

    Args:
        text: Raw text.

    Returns:
        Text safe to place in a Markdown paragraph.
    """
    return re.sub(r"([*_`\[\]])", r"\\\1", text)


def inline(node: Node) -> str:
    """Render a node's contents as inline Markdown.

    Args:
        node: The node whose children are rendered.

    Returns:
        A single line of Markdown, whitespace collapsed.
    """
    out: list[str] = []
    for child in node.children:
        # Screen-reader-only spans duplicate information the visual design carries some
        # other way. In text output that duplication is just noise.
        if "sr" in child.cls():
            continue
        if child.tag == "#text":
            out.append(escape(child.text))
        elif child.tag in {"strong", "b"}:
            out.append(f"**{inline(child).strip()}**")
        elif child.tag in {"em", "i"}:
            out.append(f"*{inline(child).strip()}*")
        elif child.tag == "code":
            out.append(f"`{_plain(child).strip()}`")
        elif child.tag == "a":
            label = inline(child).strip()
            href = child.attrs.get("href", "")
            out.append(f"[{label}]({href})" if href.startswith("http") else label)
        elif child.tag == "br":
            out.append("<br>")
        else:
            out.append(inline(child))
    return re.sub(r"[ \t\n]+", " ", "".join(out)).strip()


def _plain(node: Node) -> str:
    """Return a node's text with no Markdown escaping.

    Args:
        node: The node to flatten.

    Returns:
        Concatenated text.
    """
    if node.tag == "#text":
        return node.text
    return "".join(_plain(c) for c in node.children if "sr" not in c.cls())


def render_table(node: Node) -> list[str]:
    """Render a table as a GitHub-flavoured Markdown table.

    Args:
        node: The ``<table>`` node.

    Returns:
        Markdown lines.
    """
    rows: list[list[str]] = []
    header: list[str] = []
    for tr in _descend(node, "tr"):
        cells = [inline(c).replace("|", "\\|") for c in tr.children if c.tag in {"td", "th"}]
        if any(c.tag == "th" for c in tr.children) and not header:
            header = cells
        else:
            rows.append(cells)
    if not header and rows:
        header, rows = rows[0], rows[1:]
    width = max([len(header)] + [len(r) for r in rows], default=0)
    if not width:
        return []
    pad = lambda r: r + [""] * (width - len(r))  # noqa: E731
    lines = ["| " + " | ".join(pad(header)) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(pad(r)) + " |" for r in rows]
    return [*lines, ""]


def _descend(node: Node, tag: str) -> list[Node]:
    """Collect every descendant with the given tag.

    Args:
        node: Subtree root.
        tag: Tag to match.

    Returns:
        Matching nodes in document order.
    """
    found: list[Node] = []
    for child in node.children:
        if child.tag == tag:
            found.append(child)
        found.extend(_descend(child, tag))
    return found


def render_codeview(node: Node) -> list[str]:
    """Render the source viewer as a fenced code block.

    Flagged lines cannot be tinted in Markdown, so the marker becomes a text arrow —
    colour is never the only carrier of the information on the HTML page either.

    Args:
        node: The ``.codeview`` node.

    Returns:
        Markdown lines.
    """
    lines = ["```c"]
    for row in node.children[-1].children if node.children else []:
        if row.tag != "div":
            continue
        number = ""
        body = ""
        for cell in row.children:
            if "cnum" in cell.cls():
                number = _plain(cell).strip()
            elif cell.tag == "code":
                body = _plain(cell).rstrip()
        marker = ">>" if "flag" in row.cls() else "  "
        lines.append(f"{marker} {number:>3} | {body}")
    lines += ["```", "", "`>>` marks a line flagged as vulnerable in the ground truth.", ""]
    return lines


def render_figure(node: Node, figures: dict, out_dir: Path) -> list[str]:
    """Write a graph figure to its own SVG file and link it.

    Args:
        node: The ``.gpanel`` figure node.
        figures: The decoded ``figures.json``.
        out_dir: Directory the Markdown document lives in.

    Returns:
        Markdown lines.
    """
    caption = next((c for c in node.children if c.tag == "figcaption"), None)
    if caption is None:
        return []
    heading = next((c for c in caption.children if c.tag == "h4"), None)
    tag = next((t for t in _descend(caption, "span") if "tag" in t.cls()), None)
    name = _plain(tag).strip() if tag is not None else ""
    title = re.sub(r"\s+", " ", _plain(heading)).replace(name, "").strip() if heading else name

    svg = figures.get("figures", {}).get(name, {}).get("svg")
    if not svg:
        return []
    assets = out_dir / "figures"
    assets.mkdir(parents=True, exist_ok=True)
    path = assets / f"{name}.svg"
    path.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n{svg}\n', encoding="utf-8")

    body = [inline(p) for p in caption.children if p.tag == "p" and "counts" not in p.cls()]
    counts = next((inline(p) for p in caption.children if "counts" in p.cls()), "")
    return [
        f"#### {title} (`{name}`)",
        "",
        f"![{title} of the example function, laid out on the line axis](figures/{name}.svg)",
        "",
        *([" ".join(body), ""] if body else []),
        *([f"*{counts}*", ""] if counts else []),
    ]


def render(node: Node, figures: dict, out_dir: Path, depth: int = 0) -> list[str]:
    """Render a node and its subtree to Markdown lines.

    Args:
        node: Node to render.
        figures: The decoded ``figures.json``, for graph panels.
        out_dir: Directory the document is written to.
        depth: List nesting depth.

    Returns:
        Markdown lines.
    """
    classes = node.cls()
    tag = node.tag
    out: list[str] = []

    if tag == "svg" or "legend" in classes:
        return []

    if "codeview" in classes:
        return render_codeview(node)

    if "gpanel" in classes:
        return render_figure(node, figures, out_dir)

    if tag == "table":
        return render_table(node)

    if tag in {"h1", "h2", "h3", "h4"}:
        text = inline(node)
        return [f"{'#' * int(tag[1])} {text}", ""] if text else []

    if tag == "p":
        text = inline(node)
        if not text:
            return []
        if "q" in classes:
            hard = node.parent is not None and "hard" in node.parent.cls()
            return [f"> {'⚠️ ' if hard else ''}**Q. {text}**", ">"]
        if "a" in classes:
            return [f"> {text}", ">"]
        if "eyebrow" in classes:
            return [f"*{text}*", ""]
        return [text, ""]

    if tag in {"ul", "ol"}:
        bullet = (lambda i: f"{i}.") if tag == "ol" else (lambda i: "-")  # noqa: ARG005
        indent = "  " * depth
        for index, item in enumerate((c for c in node.children if c.tag == "li"), start=1):
            nested = [c for c in item.children if c.tag in {"ul", "ol"}]
            out.append(f"{indent}{bullet(index)} {inline(item)}")
            for sub in nested:
                out.extend(render(sub, figures, out_dir, depth + 1))
        out.append("")
        return out

    if classes & set(CALLOUT_TITLE):
        which = next(c for c in classes if c in CALLOUT_TITLE)
        inner: list[str] = []
        for child in node.children:
            inner.extend(render(child, figures, out_dir, depth))
        heading = inner[0].lstrip("# ") if inner and inner[0].startswith("#") else ""
        rest = inner[1:] if heading else inner
        while rest and rest[0] == "":
            rest.pop(0)
        title = heading or CALLOUT_TITLE[which]
        return [f"> **{title}**", ">", *[f"> {line}" if line else ">" for line in rest], ""]

    if "qa" in classes:
        for child in node.children:
            out.extend(render(child, figures, out_dir, depth))
        while out and out[-1] == ">":
            out.pop()
        return [*out, ""]

    if "numbers" in classes or "tiles" in classes:
        for item in node.children:
            value = next((_plain(c).strip() for c in item.children if c.tag in {"b", "div"}), "")
            label = next(
                (_plain(c).strip() for c in reversed(item.children) if c.tag in {"span", "div"}),
                "",
            )
            if value and label and value != label:
                out.append(f"- **{value}** — {label}")
        return [*out, ""] if out else []

    for child in node.children:
        out.extend(render(child, figures, out_dir, depth))
    return out


def convert(page_path: Path, figures_path: Path, out_path: Path) -> None:
    """Convert one page to Markdown.

    Args:
        page_path: The generated HTML page.
        figures_path: ``figures.json``, source of the SVG assets.
        out_path: Destination ``.md`` file.
    """
    builder = TreeBuilder()
    builder.feed(page_path.read_text(encoding="utf-8"))
    figures = json.loads(figures_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = render(builder.root, figures, out_path.parent)

    collapsed: list[str] = []
    for line in lines:
        if line == "" and collapsed and collapsed[-1] == "":
            continue
        collapsed.append(line)

    out_path.write_text("\n".join(collapsed).strip() + "\n", encoding="utf-8")
    print(f"{out_path}  ({len(collapsed):,} lines)")


def main() -> None:
    """Convert every generated page."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", default="./artifacts/representation/figures.json")
    ap.add_argument(
        "--pages",
        nargs="*",
        default=["./artifacts/review/review.html", "./artifacts/review/panel_prep.html"],
    )
    args = ap.parse_args()
    for name in args.pages:
        page = Path(name)
        convert(page, Path(args.figures), page.with_suffix(".md"))


if __name__ == "__main__":
    main()
