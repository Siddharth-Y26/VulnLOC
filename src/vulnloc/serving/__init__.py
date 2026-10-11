"""L6 — Serving: FastAPI application, single-function inference, Pydantic schemas.

Must not import training code (``docs/Architecture.md`` §3; enforced by ``import-linter``).
Phase 9 (2026-10-10): a checkpoint is served from a *bundle* — one directory holding its
weights, tokenizer, vocabularies, calibrator and model card (:mod:`.bundle`); a request is
featurized from L1 pieces and answered with the ``PRD.md`` §5.4 object (:mod:`.inference`);
the API is :func:`~vulnloc.serving.app.create_app`.

The application is not imported here: importing ``vulnloc.serving`` should not pull in
FastAPI for a caller that only wants to load a bundle.
"""

from vulnloc.serving.bundle import Bundle, load_bundle
from vulnloc.serving.inference import Analysis, Analyzer, featurize

__all__ = ["Analysis", "Analyzer", "Bundle", "featurize", "load_bundle"]
