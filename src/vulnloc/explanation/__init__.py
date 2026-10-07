"""L5 — Explanation: evidence extraction, templates, generation, grounding verifier.

Explanations are produced from evidence the model actually used — its cross-attention —
and from static analysis of the function that is worded as such (template patterns, the
name-based data-flow path); never narrated after the fact (``docs/Architecture.md`` §1.5).
This layer re-runs no inference.

Phase 7a (2026-10-07) is the deterministic template explainer. A registry entry only
exists once its module is imported, so ``generator`` is imported here explicitly.
"""

from vulnloc.explanation.contract import output_contract
from vulnloc.explanation.evidence import EvidenceSettings, extract_evidence
from vulnloc.explanation.generator import TemplateExplainer
from vulnloc.explanation.registry import EXPLAINER_REGISTRY, register_explainer
from vulnloc.explanation.types import AttendedLine, Evidence, Explanation, Prediction
from vulnloc.explanation.verifier import GroundingReport, verify

__all__ = [
    "EXPLAINER_REGISTRY",
    "AttendedLine",
    "Evidence",
    "EvidenceSettings",
    "Explanation",
    "GroundingReport",
    "Prediction",
    "TemplateExplainer",
    "extract_evidence",
    "output_contract",
    "register_explainer",
    "verify",
]
