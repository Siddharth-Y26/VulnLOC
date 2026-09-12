"""The structural branch: message passing over a code graph (``Architecture.md`` §4.2).

Mirrors :mod:`vulnloc.models.semantic.encoder` deliberately. The contract is
``forward(batch) -> [N, D_str]`` — this module returns *node* states and stops there.
Projection onto lines is :func:`~vulnloc.models.alignment.align_nodes_to_lines`'s job,
shared with the semantic branch through one primitive, because a branch that pooled its
own way would let the two axes drift apart without anything failing.

**Three variants, one class per convolution, one shared scaffold.** GCN, GAT and GGNN
differ only in the layer they stack, so the embedding, the residual wiring, the norm and
the readout live once in :class:`StructuralEncoder` and each variant supplies its layers.
Phase 4's exit criteria require all three to be config-selectable and require an
over-smoothing check across depths; both are settings here rather than separate files, so
the comparison is between convolutions and not between three subtly different scaffolds.

**Node features are node-type embeddings (ADR-002 §D2.5, first half).** The second half —
a subtoken embedding over the node's own text — is **not built here**, and the reason is
recorded rather than quietly skipped: ``to_pyg`` emits ``node_type`` and ``line`` but no
subtoken ids, so adding it is an L1 change plus a vocabulary, not a model change. See
``TODO(decision)`` on :class:`NodeFeatures`. Until it lands, the branch sees grammar
structure and not identifier names, and any claim about what it "understands" has to say
so.

**Over-smoothing is the thing to watch.** Stacked message passing drives node states
toward a single vector, and on these graphs that risk is not hypothetical: the AST of a
corpus function reaches 56,266 nodes and averages 252 (``Memory.md`` §7), so a 6-layer
model has a receptive field covering most of a typical function. Residual connections and
layer norm are on by default for that reason, and ``depth`` is a config knob precisely so
the 2/4/6 comparison Phase 4 owes can be run without editing code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vulnloc.models.registry import register_structural
from vulnloc.representation.graphs.base import EDGE_TYPES
from vulnloc.utils.exceptions import ModelError
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["NodeFeatures", "StructuralEncoder", "build_structural_encoder"]

_LOGGER = get_logger(__name__)

# `to_pyg` reserves vocabulary slot 0 for grammar symbols unseen when the vocabulary was
# built, so an unknown type degrades one node instead of failing a batch.
UNKNOWN_NODE_TYPE = 0

CONVOLUTIONS = ("gcn", "gat", "ggnn")
"""The three variants Phase 4 requires. Keys, not classes, so config selects them."""

EDGE_HANDLING = ("auto", "ignore", "weighted")
"""How a relation is allowed to matter. See :class:`StructuralEncoder`."""

N_EDGE_TYPES = len(EDGE_TYPES)
"""How many relations a graph can carry — ``ast``, ``cfg``, ``dfg``, ``call``.

Taken from L1's own tuple rather than written down here. L2 may import L1, and a
hardcoded 4 would go stale the moment a fifth relation lands, sizing the table one row
short and folding the new relation into an existing one without raising.
"""


class NodeFeatures(torch.nn.Module):
    """Initial node features: a node-type embedding.

    ADR-002 §D2.5 fixes node features as **node-type embedding + subtoken embedding**.
    Only the first is implemented.

    TODO(decision): the subtoken embedding needs subtoken ids on the graph, which
    ``representation/to_pyg.py`` does not emit — it carries ``node_type`` and ``line``
    only. Adding it means choosing a subtoken vocabulary and extending the L1 tensor
    layout, which is an ADR-002 amendment rather than a model change. Until then this
    branch cannot distinguish two identifiers, and experiment A4 must not be read as
    measuring the full D2.5 feature set.

    Args:
        vocab_size: Number of node-type slots, slot 0 (unknown) included.
        dim: Embedding width, ``D_str``.
        dropout: Dropout on the embedding output.

    Raises:
        ModelError: If ``vocab_size`` or ``dim`` is not positive.
    """

    def __init__(self, vocab_size: int, dim: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ModelError("vocab_size must be positive", vocab_size=vocab_size)
        if dim <= 0:
            raise ModelError("dim must be positive", dim=dim)
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = torch.nn.Embedding(vocab_size, dim, padding_idx=UNKNOWN_NODE_TYPE)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, node_type: torch.Tensor) -> torch.Tensor:
        """Embed node types.

        Args:
            node_type: ``[N]`` vocabulary indices.

        Returns:
            ``[N, D_str]`` initial node states.

        Raises:
            ModelError: If an index falls outside the embedding table. Clamping instead
                would train slot 0 on whatever fell off the end and hide a vocabulary
                built from the wrong corpus — the failure ``Memory.md`` §7 records for
                grammar symbol lists.
        """
        if node_type.numel() and int(node_type.max()) >= self.vocab_size:
            raise ModelError(
                "node type index outside the vocabulary",
                observed_max=int(node_type.max()),
                vocab_size=self.vocab_size,
                hint="Rebuild with scripts/build_node_vocab.py, or check the builder matches.",
            )
        return torch.as_tensor(self.dropout(self.embedding(node_type)))


def _make_layers(
    kind: str, dim: int, depth: int, heads: int, *, typed_edges: bool
) -> tuple[torch.nn.ModuleList, str]:
    """Build ``depth`` convolutions of one kind.

    Args:
        kind: One of :data:`CONVOLUTIONS`.
        dim: Node state width, constant across layers so residuals need no projection.
        depth: How many layers to stack.
        heads: Attention heads, GAT only.
        typed_edges: Whether the layers must accept a per-edge relation signal. Only GAT
            needs it at construction time (``edge_dim``); GCN and GGNN take an
            ``edge_weight`` at call time and need nothing here.

    Returns:
        ``(layers, description)``.

    Raises:
        ModelError: If ``kind`` is not a known convolution.
    """
    from torch_geometric.nn import GATConv, GatedGraphConv, GCNConv

    if kind == "gcn":
        return torch.nn.ModuleList([GCNConv(dim, dim) for _ in range(depth)]), "GCNConv"
    if kind == "gat":
        if dim % heads:
            raise ModelError("hidden_dim must divide evenly by heads", hidden_dim=dim, heads=heads)
        # concat=True with dim//heads per head keeps the width constant, so depth is the
        # only thing that changes between the over-smoothing runs.
        return (
            torch.nn.ModuleList(
                [
                    GATConv(
                        dim,
                        dim // heads,
                        heads=heads,
                        concat=True,
                        edge_dim=1 if typed_edges else None,
                    )
                    for _ in range(depth)
                ]
            ),
            f"GATConv(heads={heads})",
        )
    if kind == "ggnn":
        # GatedGraphConv carries its own GRU across `num_layers` propagations, so one
        # module is the whole stack; wrapping it in a list keeps the forward loop uniform.
        return (
            torch.nn.ModuleList([GatedGraphConv(dim, num_layers=depth)]),
            f"GatedGraphConv(num_layers={depth})",
        )
    raise ModelError("unknown convolution", kind=kind, known=list(CONVOLUTIONS))


class StructuralEncoder(torch.nn.Module):
    """Message passing over one batch of code graphs.

    Args:
        vocab_size: Node-type vocabulary size, unknown slot included.
        hidden_dim: Node state width, ``D_str``.
        kind: Which convolution — see :data:`CONVOLUTIONS`.
        depth: Layers to stack. The 2/4/6 sweep Phase 4 owes runs off this.
        heads: Attention heads, GAT only.
        dropout: Dropout between layers.
        residual: Add a skip connection around each layer.
        layer_norm: Normalise after each layer.
        edge_handling: What a relation is allowed to mean — see :data:`EDGE_HANDLING`.

            * ``weighted`` learns one scalar per relation and scales that edge's message
              by it, uniformly across all three convolutions (``edge_weight`` for GCN and
              GGNN, a one-dimensional ``edge_attr`` for GAT). Uniform on purpose: a
              mechanism that differed per convolution would make the A2 comparison a
              comparison of edge handling rather than of convolutions.
            * ``ignore`` treats every edge alike. Correct for an AST, where there is only
              one relation, and it reproduces every pre-existing number.
            * ``auto`` (the default) picks ``ignore`` for a graph carrying one relation
              and **raises** for a graph carrying several. That refusal is the point:
              ``Memory.md`` §7 #22 records a whole phase in which ``graph.language`` sat
              in the config unread, and a CPG fed to an edge-blind encoder would fail the
              same way — silently, by scoring a union of four relations as if it were one.

    Raises:
        ModelError: If a dimension is invalid, or ``kind`` or ``edge_handling`` is
            unknown.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        hidden_dim: int = 256,
        kind: str = "gcn",
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
        layer_norm: bool = True,
        edge_handling: str = "auto",
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ModelError("depth must be positive", depth=depth)
        if edge_handling not in EDGE_HANDLING:
            raise ModelError(
                "unknown edge handling", edge_handling=edge_handling, known=list(EDGE_HANDLING)
            )
        self.kind = kind
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.edge_handling = edge_handling

        self.features = NodeFeatures(vocab_size, hidden_dim, dropout=dropout)
        # One learned scalar per relation, softplus-constrained positive at use. Built
        # whenever it *might* be needed, so switching `auto` to `weighted` on a loaded
        # checkpoint does not change the parameter set.
        self.edge_type_weight = torch.nn.Parameter(torch.zeros(N_EDGE_TYPES))
        self.layers, description = _make_layers(
            kind, hidden_dim, depth, heads, typed_edges=edge_handling == "weighted"
        )
        self.norms = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(hidden_dim) if layer_norm else torch.nn.Identity()
                for _ in self.layers
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)

        _LOGGER.info(
            "Structural encoder ready",
            extra={
                "kind": kind,
                "convolution": description,
                "depth": depth,
                "hidden_dim": hidden_dim,
                "vocab_size": vocab_size,
                "residual": residual,
                "edge_handling": edge_handling,
                "trainable_parameters": sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                ),
            },
        )

    @property
    def output_dim(self) -> int:
        """Width of the node states this encoder returns, ``D_str``."""
        return self.hidden_dim

    def _edge_signal(self, batch: Mapping[str, Any]) -> torch.Tensor | None:
        """Resolve ``edge_handling`` against the relations this batch actually carries.

        Args:
            batch: The batch being encoded.

        Returns:
            ``[E]`` positive per-edge weights, or ``None`` when every edge is alike.

        Raises:
            ModelError: Under ``auto``, if the batch carries more than one relation. The
                encoder will not quietly average a CPG's four relations into one.
        """
        if self.edge_handling == "ignore":
            return None

        edge_type = batch.get("edge_type")
        if edge_type is None or edge_type.numel() == 0:
            if self.edge_handling == "weighted":
                raise ModelError(
                    "edge_handling='weighted' needs 'edge_type' in the batch",
                    present=sorted(batch),
                    hint="collate_graphs emits it; a hand-built batch has to as well.",
                )
            return None

        if int(edge_type.max()) >= N_EDGE_TYPES:
            raise ModelError(
                "edge type index outside the relation table",
                observed_max=int(edge_type.max()),
                n_edge_types=N_EDGE_TYPES,
                known=list(EDGE_TYPES),
            )

        if self.edge_handling == "auto":
            distinct = int(torch.unique(edge_type).numel())
            if distinct > 1:
                raise ModelError(
                    "this graph carries several relations and edge_handling is 'auto'",
                    distinct_edge_types=distinct,
                    known=list(EDGE_TYPES),
                    hint=(
                        "Set edge_handling='weighted' to let relations differ, or "
                        "'ignore' to state on purpose that they should not. A CPG scored "
                        "as an untyped union is Memory.md §7 #22 happening again."
                    ),
                )
            return None

        # softplus keeps every weight positive: a learned negative would let one relation
        # subtract its neighbours' messages, which is not what "this edge matters less"
        # means and would make the learned weights unreadable as relation importance.
        return torch.nn.functional.softplus(self.edge_type_weight)[edge_type]

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Encode one batch of graphs.

        Args:
            batch: Must carry ``node_type`` ``[N]`` and ``edge_index`` ``[2, E]``. Under
                ``edge_handling='weighted'`` it must also carry ``edge_type`` ``[E]``.

        Returns:
            Node states, ``[N, D_str]``.

        Raises:
            ModelError: If a required key is missing, or the batch's relations and the
                configured ``edge_handling`` disagree.
        """
        for key in ("node_type", "edge_index"):
            if key not in batch:
                raise ModelError(
                    "batch is missing a required key", missing=key, present=sorted(batch)
                )

        states = self.features(batch["node_type"])
        edge_index = batch["edge_index"]
        edge_signal = self._edge_signal(batch)
        # GAT reads a per-edge feature vector, GCN and GGNN a per-edge scalar. Same
        # learned number either way, so the three stay comparable.
        edge_kwargs: dict[str, torch.Tensor] = {}
        if edge_signal is not None:
            key = "edge_attr" if self.kind == "gat" else "edge_weight"
            edge_kwargs[key] = edge_signal.unsqueeze(-1) if self.kind == "gat" else edge_signal

        for layer, norm in zip(self.layers, self.norms, strict=True):
            updated = layer(states, edge_index, **edge_kwargs)
            updated = self.dropout(torch.nn.functional.gelu(updated))
            # The residual is what keeps a 6-layer stack from collapsing every node onto
            # one vector. Phase 4 reports depth 2/4/6; without this the comparison would
            # measure the absence of skip connections rather than the depth.
            states = norm(updated + states if self.residual else updated)
        return torch.as_tensor(states)

    def relation_weights(self) -> dict[str, float]:
        """Return the learned per-relation weight, by name.

        Under ``ignore``, or ``auto`` on a single-relation graph, these are unused and
        the flat values say so. Under ``weighted`` they are readable as what the model
        made of each relation, which is the point of constraining them positive.

        Returns:
            ``{relation: weight}``, in ``EDGE_TYPES`` order.
        """
        values = torch.nn.functional.softplus(self.edge_type_weight).detach().tolist()
        return dict(zip(EDGE_TYPES, values, strict=True))

    def extra_repr(self) -> str:
        """Summarise the configuration."""
        return (
            f"kind={self.kind}, depth={self.depth}, hidden_dim={self.hidden_dim}, "
            f"edge_handling={self.edge_handling}"
        )


@register_structural("gcn")
def build_gcn(**kwargs: Any) -> StructuralEncoder:
    """Registry entry for the GCN variant.

    Args:
        **kwargs: Forwarded to :class:`StructuralEncoder`.

    Returns:
        The encoder.
    """
    return StructuralEncoder(kind="gcn", **kwargs)


@register_structural("gat")
def build_gat(**kwargs: Any) -> StructuralEncoder:
    """Registry entry for the GAT variant.

    Args:
        **kwargs: Forwarded to :class:`StructuralEncoder`.

    Returns:
        The encoder.
    """
    return StructuralEncoder(kind="gat", **kwargs)


@register_structural("ggnn")
def build_ggnn(**kwargs: Any) -> StructuralEncoder:
    """Registry entry for the GGNN variant.

    Args:
        **kwargs: Forwarded to :class:`StructuralEncoder`.

    Returns:
        The encoder.
    """
    return StructuralEncoder(kind="ggnn", **kwargs)


def build_structural_encoder(kind: str, **kwargs: Any) -> StructuralEncoder:
    """Build an encoder through the registry.

    Args:
        kind: Registered key — ``gcn``, ``gat`` or ``ggnn``.
        **kwargs: Forwarded to :class:`StructuralEncoder`.

    Returns:
        The encoder.

    Raises:
        RegistryError: If ``kind`` is not registered.
    """
    from vulnloc.models.registry import STRUCTURAL_REGISTRY

    encoder = STRUCTURAL_REGISTRY.get(kind)(**kwargs)
    if not isinstance(encoder, StructuralEncoder):  # pragma: no cover - registry misuse
        raise ModelError(
            "registered structural encoder returned the wrong type",
            kind=kind,
            returned=type(encoder).__name__,
        )
    return encoder
