"""L5 — Explanation: evidence extraction, templates, generation, grounding verifier.

Explanations are produced from evidence the model actually used — attention weights,
node importances, dataflow paths — never narrated after the fact
(``docs/Architecture.md`` §1.5). This layer re-runs no inference. Populated in Phase 7.
"""

from vulnloc.explanation.registry import EXPLAINER_REGISTRY, register_explainer

__all__ = ["EXPLAINER_REGISTRY", "register_explainer"]
