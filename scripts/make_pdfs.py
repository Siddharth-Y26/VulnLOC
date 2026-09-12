"""Print the generated review pages to PDF with headless Chrome.

The pages are published as Artifacts, where the platform supplies the ``<!doctype>``
wrapper. A file printed locally gets no such wrapper, and a bare fragment puts Chrome in
quirks mode — which changes box sizing and silently shifts the layout. So the wrapper is
added here, to a copy, rather than to the source page.

No Python PDF library is installed and none is added: Chrome is already present, it is
the same engine the Artifact renders in, and it honours the ``@media print`` rules the
page already carries. WeasyPrint would render a different page.

Usage::

    python scripts/make_pdfs.py
    python scripts/make_pdfs.py --pages artifacts/review/review.html
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

WRAPPER = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    "{head}\n</head>\n<body>\n{body}\n</body>\n</html>\n"
)


def find_browser() -> str:
    """Locate a Chromium-family browser.

    Returns:
        Path to the executable.

    Raises:
        SystemExit: If neither Chrome nor Edge is installed.
    """
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise SystemExit(
        "No Chrome or Edge found. Install one, or open the .html and use " "Ctrl+P → Save as PDF."
    )


def wrap(page: str) -> str:
    """Wrap a page fragment in a full HTML document.

    The fragment starts with ``<title>`` and ``<style>``, which belong in the head; the
    rest is body content.

    Args:
        page: The generated page fragment.

    Returns:
        A complete HTML document.
    """
    marker = "</style>"
    split = page.index(marker) + len(marker)
    return WRAPPER.format(head=page[:split], body=page[split:])


def to_pdf(browser: str, page_path: Path, out_path: Path) -> None:
    """Render one page to PDF.

    Args:
        browser: Path to the Chrome or Edge executable.
        page_path: The generated HTML page.
        out_path: Destination PDF.

    Raises:
        SystemExit: If the browser exits non-zero or writes nothing.
    """
    document = wrap(page_path.read_text(encoding="utf-8"))
    # Chrome needs a real file URL, and a throwaway profile so an already-running
    # instance does not hand the job to the existing process and exit immediately.
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / page_path.name
        staged.write_text(document, encoding="utf-8")
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                f"--user-data-dir={tmp}/profile",
                "--no-pdf-header-footer",
                "--virtual-time-budget=10000",
                f"--print-to-pdf={out_path.resolve()}",
                staged.resolve().as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    if result.returncode != 0 or not out_path.exists():
        raise SystemExit(
            f"Browser failed for {page_path} (exit {result.returncode}):\n"
            f"{result.stderr.strip()[:2000]}"
        )
    print(f"{out_path}  ({out_path.stat().st_size / 1024:,.0f} KB)")


def main() -> None:
    """Print every requested page to PDF beside itself."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pages",
        nargs="*",
        default=["./artifacts/review/review.html", "./artifacts/review/panel_prep.html"],
    )
    args = ap.parse_args()

    browser = find_browser()
    print(f"using {shutil.which(browser) or browser}")
    for name in args.pages:
        page = Path(name)
        if not page.exists():
            raise SystemExit(f"Missing page: {page}. Run scripts/make_review_page.py first.")
        to_pdf(browser, page, page.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
