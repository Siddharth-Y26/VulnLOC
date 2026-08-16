"""Per-line prediction heads behind ``LocalizationHead``.

Phase 3 builds the semantic-only line head; Phase 6 extends the same interface to the
fused branch. Importing this package registers every head — see
``data/loaders/__init__.py`` for why the explicit import is required.
"""

from vulnloc.models.heads.line_head import LineHead

__all__ = ["LineHead"]
