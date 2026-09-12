"""Render a Markdown document to a print-ready HTML page.

Written for the paper drafts in ``docs/paper/`` so they can go through the same headless
Chrome print step as the generated review pages (``scripts/make_pdfs.py``).

This handles the subset of Markdown those documents actually use: ATX headings,
paragraphs, blockquotes, ordered and unordered lists, pipe tables, horizontal rules,
fenced code, and inline emphasis, code spans, links and autolinks. It is not a general
Markdown implementation, and it is not trying to be — the alternative was adding a
dependency to a locked environment for one document, and the failure mode of a small
known-subset converter is a visibly wrong page rather than a silently different one.

Two ordering rules matter and are easy to get wrong:

* code spans and autolinks are extracted *before* HTML escaping, or their contents get
  escaped twice;
* ``**bold**`` is converted before ``*italic*``, or the italic pass eats the first
  asterisk of every bold run.

Usage::

    python scripts/md_to_html.py docs/paper/literature-review-and-methodology.md
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

STYLE = """
:root {
  --brand: #0B76B8; --brand-dark: #075C90; --surface-soft: #E8F4FC;
  --border-brand: #9CCDE8; --bg-page: #FFFFFF; --bg-elevated: #FBFCFD;
  --text-primary: #111418; --text-body: #23292F; --text-muted: #5C6874;
  --border-subtle: #D8DEE4;
  --serif: Georgia, "Times New Roman", serif;
  --sans: "Segoe UI", -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: Consolas, "Cascadia Mono", "SF Mono", monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --brand: #57B4E8; --brand-dark: #8FCDF2; --surface-soft: #12212C;
    --border-brand: #2A4A61; --bg-page: #0E1216; --bg-elevated: #161C22;
    --text-primary: #E6EBF0; --text-body: #C4CDD6; --text-muted: #8B98A5;
    --border-subtle: #262E36;
  }
}
:root[data-theme="dark"] {
  --brand: #57B4E8; --brand-dark: #8FCDF2; --surface-soft: #12212C;
  --border-brand: #2A4A61; --bg-page: #0E1216; --bg-elevated: #161C22;
  --text-primary: #E6EBF0; --text-body: #C4CDD6; --text-muted: #8B98A5;
  --border-subtle: #262E36;
}

body {
  background: var(--bg-page); color: var(--text-body);
  font-family: var(--serif); font-size: 17px; line-height: 1.62;
  margin: 0; padding: 48px 24px 96px;
}
main { max-width: 46em; margin: 0 auto; }

h1, h2, h3, h4 { font-family: var(--sans); color: var(--text-primary); text-wrap: balance; }
h1 { font-size: 32px; line-height: 1.2; margin: 0 0 8px; font-weight: 700; }
h2 {
  font-size: 24px; line-height: 1.25; margin: 46px 0 14px; font-weight: 700;
  padding-bottom: 8px; border-bottom: 2px solid var(--border-brand);
}
h3 { font-size: 18.5px; margin: 34px 0 10px; font-weight: 650; }
h4 { font-size: 16px; margin: 26px 0 8px; font-weight: 650; }
p { margin: 0 0 15px; }
ul, ol { margin: 0 0 15px; padding-left: 26px; }
li { margin-bottom: 8px; }
li > ul, li > ol { margin-top: 8px; }
strong { font-weight: 700; color: var(--text-primary); }
a { color: var(--brand-dark); text-underline-offset: 2px; word-break: break-word; }
a:focus-visible { outline: 2px solid var(--brand); outline-offset: 2px; }
code {
  font-family: var(--mono); font-size: .85em; background: var(--surface-soft);
  border: 1px solid var(--border-brand); border-radius: 3px; padding: 0 4px;
}
pre {
  font-family: var(--mono); font-size: 13px; background: var(--bg-elevated);
  border: 1px solid var(--border-subtle); border-radius: 6px;
  padding: 14px 16px; overflow-x: auto;
}
pre code { background: none; border: 0; padding: 0; font-size: inherit; }
hr { border: 0; border-top: 1px solid var(--border-subtle); margin: 38px 0; }

blockquote {
  margin: 20px 0; padding: 14px 20px; border-left: 4px solid var(--brand);
  background: var(--bg-elevated); border-radius: 0 6px 6px 0;
  font-family: var(--sans); font-size: 15.5px; color: var(--text-muted);
}
blockquote p:last-child { margin-bottom: 0; }

.tablewrap { overflow-x: auto; margin: 20px 0; }
table {
  border-collapse: collapse; width: 100%; font-family: var(--sans); font-size: 14.5px;
}
th, td {
  text-align: left; padding: 9px 12px; vertical-align: top;
  border-bottom: 1px solid var(--border-subtle);
}
th {
  background: var(--surface-soft); color: var(--text-primary); font-weight: 650;
  font-size: 12.5px; letter-spacing: .03em; text-transform: uppercase;
  border-bottom: 1px solid var(--border-brand);
}

/* The reference list: hanging indent, so the numbers form a readable column. */
h2#references ~ p {
  font-family: var(--sans); font-size: 14.5px; line-height: 1.5;
  padding-left: 2.6em; text-indent: -2.6em; margin-bottom: 11px;
}

@page { size: A4; margin: 20mm 18mm; }
@media print {
  :root, :root[data-theme="dark"] {
    --brand: #0B76B8; --brand-dark: #075C90; --surface-soft: #EDF5FB;
    --border-brand: #9CCDE8; --bg-page: #FFFFFF; --bg-elevated: #FAFBFC;
    --text-primary: #111418; --text-body: #1D2329; --text-muted: #55616D;
    --border-subtle: #D3DAE1;
  }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { font-size: 10.5pt; padding: 0; }
  main { max-width: none; }
  h1 { font-size: 20pt; }
  h2 { font-size: 14pt; margin-top: 22pt; }
  h3 { font-size: 11.5pt; }
  h1, h2, h3, h4 { break-after: avoid; page-page-break-after: avoid; }
  h2 { break-before: page; page-break-before: always; }
  h1 + p, h2:first-of-type { break-before: auto; page-break-before: auto; }
  p, li, blockquote, tr { break-inside: avoid; page-break-inside: avoid; }
  .tablewrap { overflow: visible; }
  thead { display: table-header-group; }
  table { font-size: 9pt; }
  a { text-decoration: none; }
}
"""

INLINE_CODE = re.compile(r"`([^`]+)`")
AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
ITALIC = re.compile(r"(?<!\*)\*(?!\s)([^*]+?)(?<!\s)\*(?!\*)")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
ULI = re.compile(r"^(\s*)[-*]\s+(.*)$")
OLI = re.compile(r"^(\s*)\d+\.\s+(.*)$")


def inline(text: str) -> str:
    """Convert inline Markdown to HTML.

    Args:
        text: One logical line of Markdown, already joined across soft wraps.

    Returns:
        HTML for that text.
    """
    held: list[str] = []

    def hold(markup: str) -> str:
        held.append(markup)
        return f"\x00{len(held) - 1}\x00"

    # Before escaping, or the contents would be escaped and then escaped again.
    text = INLINE_CODE.sub(lambda m: hold(f"<code>{html.escape(m.group(1))}</code>"), text)

    def autolink(match: re.Match[str]) -> str:
        url = match.group(1)
        return hold(f'<a href="{html.escape(url, quote=True)}">{html.escape(url)}</a>')

    text = AUTOLINK.sub(autolink, text)
    text = LINK.sub(
        lambda m: hold(
            f'<a href="{html.escape(m.group(2), quote=True)}">{html.escape(m.group(1))}</a>'
        ),
        text,
    )

    text = html.escape(text)
    text = BOLD.sub(r"<strong>\1</strong>", text)
    text = ITALIC.sub(r"<em>\1</em>", text)

    return re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], text)


def slug(text: str) -> str:
    """Build an id for a heading.

    Args:
        text: Heading text, already rendered to HTML.

    Returns:
        A lowercase, hyphenated identifier.
    """
    bare = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[^a-z0-9]+", "-", bare.lower()).strip("-")


def render_table(rows: list[str]) -> list[str]:
    """Render a pipe table.

    Args:
        rows: The table's raw lines, including the alignment row.

    Returns:
        HTML lines.
    """

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    out = ['<div class="tablewrap"><table>', "<thead><tr>"]
    out += [f"<th>{inline(c)}</th>" for c in header]
    out += ["</tr></thead>", "<tbody>"]
    for row in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
    out += ["</tbody></table></div>"]
    return out


def convert(markdown: str) -> str:
    """Convert a Markdown document body to HTML.

    Args:
        markdown: The document text.

    Returns:
        HTML for the document body.
    """
    lines = markdown.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            block: list[str] = []
            index += 1
            while index < total and not lines[index].strip().startswith("```"):
                block.append(html.escape(lines[index]))
                index += 1
            index += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            continue

        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            out.append("<hr>")
            index += 1
            continue

        heading = HEADING.match(stripped)
        if heading:
            level = len(heading.group(1))
            text = inline(heading.group(2).strip())
            out.append(f'<h{level} id="{slug(text)}">{text}</h{level}>')
            index += 1
            continue

        # A table needs its alignment row to be recognised as one.
        if (
            stripped.startswith("|")
            and index + 1 < total
            and re.fullmatch(r"\|[\s:|-]+\|?", lines[index + 1].strip())
        ):
            rows = []
            while index < total and lines[index].strip().startswith("|"):
                rows.append(lines[index])
                index += 1
            out.extend(render_table(rows))
            continue

        if stripped.startswith(">"):
            quoted: list[str] = []
            while index < total and lines[index].strip().startswith(">"):
                quoted.append(re.sub(r"^\s*>\s?", "", lines[index]))
                index += 1
            out.append("<blockquote>" + convert("\n".join(quoted)) + "</blockquote>")
            continue

        if ULI.match(line) or OLI.match(line):
            ordered = bool(OLI.match(line))
            pattern = OLI if ordered else ULI
            items: list[str] = []
            while index < total:
                match = pattern.match(lines[index])
                if not match:
                    # A wrapped continuation line belongs to the item above it.
                    if items and lines[index].strip() and not lines[index].startswith("|"):
                        if HEADING.match(lines[index].strip()) or lines[index].strip() == "---":
                            break
                        items[-1] += " " + lines[index].strip()
                        index += 1
                        continue
                    break
                items.append(match.group(2).strip())
                index += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>")
            out += [f"<li>{inline(item)}</li>" for item in items]
            out.append(f"</{tag}>")
            continue

        paragraph: list[str] = []
        while index < total and lines[index].strip():
            current = lines[index]
            if (
                HEADING.match(current.strip())
                or current.strip().startswith(("|", ">", "```"))
                or ULI.match(current)
                or OLI.match(current)
                or re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", current.strip())
            ):
                break
            paragraph.append(current.strip())
            index += 1
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")

    return "\n".join(out)


def main() -> None:
    """Convert each given Markdown file to an HTML page beside it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="+", help="Markdown files to convert.")
    ap.add_argument("--out-dir", default=None, help="Directory for the HTML (default: alongside).")
    args = ap.parse_args()

    for name in args.sources:
        source = Path(name)
        if not source.exists():
            raise SystemExit(f"Missing file: {source}")
        text = source.read_text(encoding="utf-8")

        title = next(
            (line.lstrip("# ").strip() for line in text.split("\n") if line.startswith("# ")),
            source.stem,
        )
        page = (
            f"<title>{html.escape(title)}</title>\n<style>{STYLE}</style>\n"
            f"<main>\n{convert(text)}\n</main>\n"
        )

        out_dir = Path(args.out_dir) if args.out_dir else source.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{source.stem}.html"
        out.write_text(page, encoding="utf-8")
        print(f"{out}  ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
