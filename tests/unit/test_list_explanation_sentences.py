"""The review page must list everything the explainer can say — or the review misses it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from vulnloc.explanation.contract import NOT_FLAGGED_TEXT
from vulnloc.explanation.generator import FIXED_SENTENCES
from vulnloc.explanation.templates import all_templates
from vulnloc.explanation.verifier import BANNED_PHRASES

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "list_explanation_sentences.py"


def _load() -> Any:
    """Import ``scripts/list_explanation_sentences.py`` by path."""
    spec = importlib.util.spec_from_file_location(
        "vulnloc_scripts_list_explanation_sentences", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


listing = _load()


def test_the_page_lists_every_template_sentence_and_every_fixed_form():
    page = listing.render()

    for template in all_templates():
        assert f"`{template.id}`" in page
        assert template.reason in page
    for sentence in FIXED_SENTENCES.values():
        assert sentence in page
    assert NOT_FLAGGED_TEXT in page


def test_no_fixed_sentence_form_contains_attack_vocabulary():
    # The templates are scanned in test_explanation_templates.py; these are the rest of
    # what a user can be shown.
    for name, sentence in {**FIXED_SENTENCES, "not flagged": NOT_FLAGGED_TEXT}.items():
        hits = [phrase for phrase in BANNED_PHRASES if phrase in sentence.lower()]
        assert not hits, f"{name}: {hits}"


def test_the_fixed_forms_are_the_ones_the_explainer_uses():
    # A sentence written inline in `explain()` would be shown to users and missing from
    # the review page. The generator's source may contain no other prose string literal.
    source = (REPO_ROOT / "src" / "vulnloc" / "explanation" / "generator.py").read_text(
        encoding="utf-8"
    )
    body = source[source.index("    def explain(") :]

    assert 'sentences.append(f"' not in body
    assert 'sentences.append("' not in body
