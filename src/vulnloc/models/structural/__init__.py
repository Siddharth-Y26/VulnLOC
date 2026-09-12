"""Structural encoders (GCN, GAT, GGNN) behind ``StructuralEncoder`` (Phase 4).

Importing this package registers every encoder. A ``@register_structural`` decorator only
runs when its module is imported, so without this a config naming ``gat`` would fail with
"unknown structural encoder" despite the encoder existing — the same registry-discovery
trap ``data/loaders/__init__.py`` documents. Adding an encoder means adding one import
here.
"""

from vulnloc.models.structural.encoder import (
    CONVOLUTIONS,
    NodeFeatures,
    StructuralEncoder,
    build_structural_encoder,
)

__all__ = [
    "CONVOLUTIONS",
    "NodeFeatures",
    "StructuralEncoder",
    "build_structural_encoder",
]
