"""The GNN-only baseline — Phase 4's deliverable and the second comparator for A1.

Encoder → line alignment → line head. Nothing else: no tokens, no fusion. The mirror of
:class:`~vulnloc.models.semantic.baseline.SemanticLineModel`, and deliberately the same
three steps in the same order, because A1 compares fusion strategies and any difference in
plumbing between the two arms shows up inside that comparison wearing the architecture's
clothes.

**One line axis, one head, one primitive.** The branches differ only in what they encode
and which alignment function projects it: ``align_tokens_to_lines`` there,
:func:`~vulnloc.models.alignment.align_nodes_to_lines` here. Both land on ``[B, L, D]`` and
both hand that to the same :class:`~vulnloc.models.heads.line_head.LineHead`. Phase 5 then
adds fusion *between* the alignment and the head rather than replacing either.

**Nodes, not units.** PyG concatenates graphs, so the encoder sees ``[N, D]`` across the
whole batch and ``batch`` is what ties each node back to its function. It is a required
key rather than an optional one for the same reason ``window_to_unit`` is on the semantic
side: without it every graph would be scored as if it were one function, and nothing would
raise.

**What this branch cannot see.** Node features are node-type embeddings only — the
ADR-002 §D2.5 subtoken half is not built (see ``NodeFeatures``), so two different
identifiers are the same token to this model. And 7.84% of flagged lines corpus-wide carry
no graph node at all (``Memory.md`` §7 #17), which is a ceiling on this branch's recall
specifically and not a ceiling the semantic branch shares. Neither belongs in a footnote:
a GNN-only number that does not mention them is overstated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vulnloc.models.alignment import align_nodes_to_lines
from vulnloc.models.heads.line_head import LineHead
from vulnloc.models.structural.encoder import StructuralEncoder
from vulnloc.utils.exceptions import ModelError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["StructuralLineModel"]


class StructuralLineModel(torch.nn.Module):
    """Structure-only per-line vulnerability scorer.

    Args:
        encoder: The graph encoder.
        head: The per-line head. Its ``input_dim`` must match the encoder's width.
        n_lines: Line slots, ``L``. Lines beyond this are dropped — see
            :func:`~vulnloc.models.alignment.scatter_mean_lines`.

    Raises:
        ModelError: If the head's width does not match the encoder's.
    """

    def __init__(self, encoder: StructuralEncoder, head: LineHead, *, n_lines: int) -> None:
        super().__init__()
        if head.input_dim != encoder.output_dim:
            raise ModelError(
                "head width does not match the encoder",
                encoder_hidden=encoder.output_dim,
                head_input=head.input_dim,
            )
        self.encoder = encoder
        self.head = head
        self.n_lines = n_lines

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Score every line of every unit in the batch.

        Args:
            batch: Must carry ``node_type`` ``[N]``, ``edge_index`` ``[2, E]``, ``line``
                ``[N]``, ``batch`` ``[N]`` and ``n_units``. Under
                ``edge_handling='weighted'`` it must also carry ``edge_type`` ``[E]``.

        Returns:
            ``{"line_logits": [B, L], "function_logits": [B], "line_mask": [B, L]}``.

        Raises:
            ModelError: If a required key is missing.
        """
        for key in ("line", "batch", "n_units"):
            if key not in batch:
                raise ModelError(
                    "batch is missing a required key",
                    missing=key,
                    present=sorted(batch),
                )

        node_states = self.encoder(batch)
        line_states, line_mask = align_nodes_to_lines(
            node_states,
            batch["line"],
            batch["batch"],
            n_lines=self.n_lines,
            n_units=int(batch["n_units"]),
        )
        line_logits, function_logits = self.head(line_states, line_mask)
        return {
            "line_logits": line_logits,
            "function_logits": function_logits,
            "line_mask": line_mask,
        }
