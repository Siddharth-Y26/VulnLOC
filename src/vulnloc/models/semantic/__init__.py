"""Semantic encoders (CodeBERT, GraphCodeBERT) behind ``SemanticEncoder`` (Phase 3).

Importing this package registers every encoder. A ``@register_semantic`` decorator only
runs when its module is imported, so without this a config naming ``codebert`` would fail
with "unknown semantic encoder" despite the encoder existing — the same registry-discovery
trap ``data/loaders/__init__.py`` documents. Adding an encoder means adding one import
here.
"""

from vulnloc.models.semantic.baseline import SemanticLineModel
from vulnloc.models.semantic.encoder import PretrainedCodeEncoder

__all__ = ["PretrainedCodeEncoder", "SemanticLineModel"]
