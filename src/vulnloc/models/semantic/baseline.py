"""The LLM-only baseline — Phase 3's deliverable and the first comparator for A1.

Encoder → line alignment → line head. Nothing else: no graph, no fusion. It exists to
produce **the first honest number** (``Phases.md`` Phase 3) and to be one arm of the
experiment the project is actually about — if bidirectional cross-attention cannot beat
this, that is the result and it gets reported.

The composition is deliberately thin. Every part is a registry entry that Phase 5 will
reuse unchanged, so ``llm_only`` is not a special path through the model but the same
path with the structural branch absent. A baseline built differently from the system it
is compared against measures the difference in plumbing as much as the difference in
architecture.

**Windows, not units.** ``clean_eval`` averages 4.17 windows per function and reaches 88
(``Memory.md`` §7 #28), so the encoder sees ``[W, T]`` and the head sees ``[B, L]`` with
``W ≥ B``. ``window_to_unit`` is what ties them back together, and it is required rather
than optional precisely because forgetting it would silently score each window as its own
function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vulnloc.models.alignment import align_tokens_to_lines
from vulnloc.models.heads.line_head import LineHead
from vulnloc.models.semantic.encoder import PretrainedCodeEncoder
from vulnloc.utils.exceptions import ModelError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["SemanticLineModel"]


class SemanticLineModel(torch.nn.Module):
    """Semantic-only per-line vulnerability scorer.

    Args:
        encoder: The token encoder.
        head: The per-line head. Its ``input_dim`` must match the encoder's width.
        n_lines: Line slots, ``L``. Lines beyond this are dropped — see
            :func:`~vulnloc.models.alignment.scatter_mean_lines`.

    Raises:
        ModelError: If the head's width does not match the encoder's.
    """

    def __init__(self, encoder: PretrainedCodeEncoder, head: LineHead, *, n_lines: int) -> None:
        super().__init__()
        if head.input_dim != encoder.hidden_size:
            raise ModelError(
                "head width does not match the encoder",
                encoder_hidden=encoder.hidden_size,
                head_input=head.input_dim,
            )
        self.encoder = encoder
        self.head = head
        self.n_lines = n_lines

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Score every line of every unit in the batch.

        Args:
            batch: Must carry ``input_ids`` ``[W, T]``, ``attention_mask`` ``[W, T]``,
                ``token_to_line`` ``[W, T]``, ``window_to_unit`` ``[W]`` and
                ``n_units``.

        Returns:
            ``{"line_logits": [B, L], "function_logits": [B], "line_mask": [B, L]}``.

        Raises:
            ModelError: If a required key is missing.
        """
        for key in ("token_to_line", "window_to_unit", "n_units"):
            if key not in batch:
                raise ModelError(
                    "batch is missing a required key",
                    missing=key,
                    present=sorted(batch),
                )

        token_states = self.encoder(batch)
        line_states, line_mask = align_tokens_to_lines(
            token_states,
            batch["token_to_line"],
            n_lines=self.n_lines,
            window_to_unit=batch["window_to_unit"],
            n_units=int(batch["n_units"]),
        )
        line_logits, function_logits = self.head(line_states, line_mask)
        return {
            "line_logits": line_logits,
            "function_logits": function_logits,
            "line_mask": line_mask,
        }
