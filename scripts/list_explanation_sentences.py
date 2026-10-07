r"""Write every sentence the template explainer can emit, on one page, for a person to read.

Phase 7's third exit criterion is *"explanations reviewed for absence of attack
instructions"* (``Rules.md`` §7.2, §7.3). The explainer generates nothing: all it can say
is a template sentence from ``explanation/templates.py`` or one of the fixed forms in
``explanation/generator.py`` and ``explanation/contract.py``. This script lists them all,
so the review is reading one short page rather than sampling outputs — and so the page
cannot drift from the code, because it is printed from it.

It runs no model and logs no run. The automatic half of the criterion (a vocabulary scan
of every sentence) is a test; this is the half a person does.

Usage::

    python scripts/list_explanation_sentences.py
    python scripts/list_explanation_sentences.py --out artifacts/phase7/sentences_for_review.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vulnloc.explanation.contract import NOT_FLAGGED_TEXT
from vulnloc.explanation.generator import FIXED_SENTENCES
from vulnloc.explanation.templates import all_templates
from vulnloc.explanation.verifier import BANNED_PHRASES

DEFAULT_OUTPUT = Path("artifacts/phase7/sentences_for_review.md")


def render() -> str:
    """The review page.

    Returns:
        Markdown listing every template sentence and every fixed sentence form.
    """
    templates = all_templates()
    rows = "\n".join(
        f"| {index} | `{template.id}` | {template.label} | {template.cwe} | {template.reason} |"
        for index, template in enumerate(templates, start=1)
    )
    fixed = {**FIXED_SENTENCES, "a unit with no flagged line": NOT_FLAGGED_TEXT}
    fixed_rows = "\n".join(
        f"| {chr(ord('A') + index)} | {name} | {sentence} |"
        for index, (name, sentence) in enumerate(fixed.items())
    )
    banned = ", ".join(f'"{phrase}"' for phrase in BANNED_PHRASES)
    return f"""# Every sentence the template explainer can emit

Printed from the code by `scripts/list_explanation_sentences.py`. Nothing else is ever
shown to a user as an explanation: the explainer fills the `{{...}}` slots with names and
numbers taken from the function and the prediction, and generates no other text.

**What to check** (`Rules.md` §7.2, §7.3): each sentence says *why a construct is risky*.
None may say *how to trigger it or use it*. If one does, or could be read that way, note
its number.

## {len(templates)} template sentences

| # | Template | Category shown | CWE | Sentence |
|---|---|---|---|---|
{rows}

## {len(fixed)} fixed sentence forms

| | When | Sentence |
|---|---|---|
{fixed_rows}

## What is already checked automatically

Every sentence above, and every finished explanation, is scanned for this vocabulary and
rejected if it contains any of it: {banned}. That check cannot judge meaning; this page
is for the judgement.

---

Reviewed by: ______________________   Date: ____________   Sentences of concern: ____________
"""


def main() -> None:
    """Write the page and say where."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Where to write the page.")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(), encoding="utf-8")
    print(
        f"{len(all_templates())} template sentences and {len(FIXED_SENTENCES) + 1} fixed forms "
        f"written to {args.out}"
    )


if __name__ == "__main__":
    main()
