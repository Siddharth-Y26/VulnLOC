"""Both branches, one line axis, one head — the model experiment A1 is run on (Phase 5).

This is :class:`~vulnloc.models.semantic.baseline.SemanticLineModel` and
:class:`~vulnloc.models.structural.baseline.StructuralLineModel` with a fusion step
inserted between the alignment and the head, and with nothing else changed. The encoders,
the alignment functions and the :class:`~vulnloc.models.heads.line_head.LineHead` are the
same objects those baselines use, which is what lets a fused number be compared with the
two arms that closed Phases 3 and 4.

**The batch is flat, and prefixed.** ``training/loop.py`` moves a batch to the device by
walking its top-level values, so a nested ``{"semantic": ..., "structural": ...}`` batch
would arrive half on the GPU with nothing raising. The two branches' keys are therefore
prefixed ``sem_`` and ``str_`` — the naming convention ``CLAUDE.md`` §9 already uses — and
the line-axis keys they share (``line_labels``, ``line_valid``, ``targets``, ``n_units``)
stay unprefixed, because there is exactly one of each and the two branches agreeing on
them is the whole basis of the fusion.

**The line axis is trimmed to the batch.** ``L`` is configured at 512, but attention is
quadratic in it while the corpus mean is 31 lines per function (``Memory.md`` §1.4) — so
attending over the full padded axis spends 250,000 positions per unit per direction to
carry about 1,000 real ones. The active axis is the largest line index anything in the
batch actually reaches, computed from the labels *and* from both branches' line vectors so
a feature can never land past the trim. Logits are padded back to the configured ``L``
before they leave, so the loop, the loss and the metrics see the same ``[B, L]`` contract
the two baselines produce. :func:`~vulnloc.models.fused.FusedLineModel.forward` is
test-pinned to give identical logits with trimming on and off — an optimization that
changes a number is not an optimization.

**A disabled branch is not run.** Under ``fusion=gnn_only`` the semantic encoder is 125M
parameters whose output would be multiplied by zero; it is skipped outright and replaced
by an explicit zero tensor with an all-``False`` mask. The ablation is then a fact about
the graph of the computation rather than an arithmetic accident that a gradient could
still leak through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vulnloc.models.alignment import align_nodes_to_lines, align_tokens_to_lines
from vulnloc.models.fusion.base import FusionModule
from vulnloc.models.heads.line_head import MASKED_LOGIT, LineHead
from vulnloc.models.semantic.encoder import PretrainedCodeEncoder
from vulnloc.models.structural.encoder import StructuralEncoder
from vulnloc.utils.exceptions import ModelError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["FusedLineModel"]

_SEMANTIC_KEYS = ("sem_input_ids", "sem_attention_mask", "sem_token_to_line", "sem_window_to_unit")
_STRUCTURAL_KEYS = ("str_node_type", "str_line", "str_edge_index", "str_batch")


class FusedLineModel(torch.nn.Module):
    """Semantic + structural encoders, fused on the line axis, scored by one head.

    Args:
        semantic: The token encoder.
        structural: The graph encoder.
        fusion: The fusion strategy. Its ``sem_dim``/``str_dim`` must match the encoders
            and its ``output_dim`` must match the head.
        head: The per-line head.
        n_lines: Line slots ``L`` the batch is labelled on. Lines beyond it are dropped by
            :func:`~vulnloc.models.alignment.scatter_mean_lines`, exactly as in both
            baselines.
        trim_line_axis: Run the fusion and the head on only as many line slots as the
            batch reaches, then pad the logits back to ``n_lines``. Numerically identical
            and much cheaper; ``False`` is the unoptimized path the tests compare against.

    Raises:
        ModelError: If the widths of encoders, fusion and head do not line up.
    """

    def __init__(
        self,
        semantic: PretrainedCodeEncoder,
        structural: StructuralEncoder,
        fusion: FusionModule,
        head: LineHead,
        *,
        n_lines: int,
        trim_line_axis: bool = True,
    ) -> None:
        super().__init__()
        if fusion.sem_dim != semantic.hidden_size:
            raise ModelError(
                "fusion does not match the semantic encoder",
                encoder_hidden=semantic.hidden_size,
                fusion_sem_dim=fusion.sem_dim,
            )
        if fusion.str_dim != structural.output_dim:
            raise ModelError(
                "fusion does not match the structural encoder",
                encoder_hidden=structural.output_dim,
                fusion_str_dim=fusion.str_dim,
            )
        if head.input_dim != fusion.output_dim:
            raise ModelError(
                "head width does not match the fusion output",
                fusion_output=fusion.output_dim,
                head_input=head.input_dim,
            )
        if n_lines <= 0:
            raise ModelError("n_lines must be positive", n_lines=n_lines)

        self.semantic = semantic
        self.structural = structural
        self.fusion = fusion
        self.head = head
        self.n_lines = n_lines
        self.trim_line_axis = trim_line_axis

    def forward(
        self, batch: Mapping[str, Any], *, return_attention: bool = False
    ) -> dict[str, Any]:
        """Score every line of every unit in the batch.

        Args:
            batch: A batch from :func:`~vulnloc.training.fused_datasets.collate_fused`.
                Must carry ``n_units``, ``line_valid`` ``[B, L]``, the ``sem_*`` keys when
                the fusion uses the semantic branch, and the ``str_*`` keys when it uses
                the structural one.
            return_attention: Retain the fusion's attention maps. Off during training.

        Returns:
            ``{"line_logits": [B, L], "function_logits": [B], "line_mask": [B, L],
            "sem_line_mask": [B, L], "str_line_mask": [B, L], "n_lines_active": int,
            "attn_sem2str": [B, H, L_active, L_active] or None, "attn_str2sem": ...}``.
            Attention maps are returned on the **active** axis, which is still 1-indexed
            at position ``i - 1``, so an index means the same source line either way.

        Raises:
            ModelError: If a key a live branch needs is missing.
        """
        self._require(batch)
        n_units = int(batch["n_units"])
        active = self._active_lines(batch)

        sem_lines, sem_mask = self._semantic_lines(batch, n_units=n_units, n_lines=active)
        str_lines, str_mask = self._structural_lines(batch, n_units=n_units, n_lines=active)

        fused = self.fusion(
            sem_lines, str_lines, sem_mask, str_mask, return_attention=return_attention
        )
        line_logits, function_logits = self.head(fused.fused, fused.line_mask)

        return {
            "line_logits": self._pad_logits(line_logits),
            "function_logits": function_logits,
            "line_mask": self._pad_mask(fused.line_mask),
            # Kept so an evaluation can report each branch's own coverage ceiling rather
            # than quoting the corpus average at the partition (Memory.md §7 #17, #12).
            "sem_line_mask": self._pad_mask(sem_mask),
            "str_line_mask": self._pad_mask(str_mask),
            "n_lines_active": active,
            "attn_sem2str": fused.attn_sem2str,
            "attn_str2sem": fused.attn_str2sem,
        }

    @torch.no_grad()
    def explain(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Score a batch and return its attention maps — the Phase 7 export API.

        The maps are what ``Architecture.md`` §4.4 promises the explanation layer: for each
        unit, how much each source line attended to each other source line, in each
        direction and each head. They are only produced on request because at the
        configured ``L`` they dominate the memory of a forward pass.

        Args:
            batch: As :meth:`forward`.

        Returns:
            :meth:`forward`'s dictionary, with ``attn_sem2str`` and ``attn_str2sem``
            populated whenever the fusion strategy has attention at all.
        """
        was_training = self.training
        self.eval()
        try:
            return self.forward(batch, return_attention=True)
        finally:
            self.train(was_training)

    def _require(self, batch: Mapping[str, Any]) -> None:
        """Check the batch carries what the live branches need.

        Args:
            batch: The batch.

        Raises:
            ModelError: If a required key is missing.
        """
        needed = ["n_units", "line_valid"]
        if self.fusion.uses_semantic:
            needed += list(_SEMANTIC_KEYS)
        if self.fusion.uses_structural:
            needed += list(_STRUCTURAL_KEYS)
        for key in needed:
            if key not in batch:
                raise ModelError(
                    "batch is missing a required key",
                    missing=key,
                    present=sorted(batch),
                    uses_semantic=self.fusion.uses_semantic,
                    uses_structural=self.fusion.uses_structural,
                )

    def _active_lines(self, batch: Mapping[str, Any]) -> int:
        """How many line slots this batch actually reaches.

        Derived from the label mask *and* from both branches' line vectors. Taking it from
        the labels alone would be one silent-truncation bug away from dropping a feature
        that landed past the last labelled line.

        Args:
            batch: The batch.

        Returns:
            The active axis length, between 1 and ``n_lines``.
        """
        if not self.trim_line_axis:
            return self.n_lines
        valid = batch["line_valid"]
        reach = int(valid.sum(dim=1).max()) if valid.numel() else 0
        if self.fusion.uses_semantic:
            lines = batch["sem_token_to_line"]
            reach = max(reach, int(lines.max()) if lines.numel() else 0)
        if self.fusion.uses_structural:
            lines = batch["str_line"]
            reach = max(reach, int(lines.max()) if lines.numel() else 0)
        return max(1, min(self.n_lines, reach))

    def _semantic_lines(
        self, batch: Mapping[str, Any], *, n_units: int, n_lines: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens and project them onto the line axis.

        Args:
            batch: The batch.
            n_units: ``B``.
            n_lines: Active line slots.

        Returns:
            ``(line_states [B, L_active, D_sem], line_mask [B, L_active])``. All zeros and
            all ``False`` when the fusion does not use this branch.
        """
        if not self.fusion.uses_semantic:
            return self._absent(batch, n_units=n_units, n_lines=n_lines, width=self.fusion.sem_dim)
        if "sem_line_states" in batch:
            return self._cached_semantic_lines(batch, n_lines=n_lines)
        token_states = self.semantic(
            {
                "input_ids": batch["sem_input_ids"],
                "attention_mask": batch["sem_attention_mask"],
            }
        )
        return align_tokens_to_lines(
            token_states,
            batch["sem_token_to_line"],
            n_lines=n_lines,
            window_to_unit=batch["sem_window_to_unit"],
            n_units=n_units,
        )

    def _cached_semantic_lines(
        self, batch: Mapping[str, Any], *, n_lines: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read precomputed frozen-encoder line features instead of encoding (ADR-009 §D9.3).

        The cache holds exactly what :func:`align_tokens_to_lines` returns for a frozen
        encoder in eval mode, so this is the live path with the encoder call removed — and
        everything downstream of it is shared code, identical by construction.

        **One deliberate difference, in training mode only.** The live path applies the
        encoder wrapper's ``dropout`` to *token* states before pooling; a cache cannot, so
        the same dropout module is applied to the pooled *line* states here. Both are
        identity in eval mode, which is where the equivalence test pins them. Every A1 arm
        reads the same cache, so the arms stay identical to each other.

        Args:
            batch: Carries ``sem_line_states`` ``[B, R, D]`` and ``sem_line_mask`` ``[B, R]``.
            n_lines: Active line slots.

        Returns:
            ``(line_states [B, L_active, D_sem], line_mask [B, L_active])``.

        Raises:
            ModelError: If the cache mask is missing, or the cache reaches past the active
                axis — which would mean the trim and the cache disagree about the lines.
        """
        if "sem_line_mask" not in batch:
            raise ModelError("batch has cached semantic states but no sem_line_mask")
        states = batch["sem_line_states"].float()
        mask = batch["sem_line_mask"].bool()
        rows = states.shape[1]
        if bool(mask[:, n_lines:].any()):
            raise ModelError(
                "cached semantic features reach past the active line axis",
                cached_rows=rows,
                n_lines_active=n_lines,
            )
        if rows < n_lines:
            pad = n_lines - rows
            states = torch.nn.functional.pad(states, (0, 0, 0, pad))
            mask = torch.nn.functional.pad(mask, (0, pad), value=False)
        else:
            states, mask = states[:, :n_lines], mask[:, :n_lines]
        dropout = getattr(self.semantic, "dropout", None)
        if isinstance(dropout, torch.nn.Module):
            states = dropout(states)
        return states, mask

    def _structural_lines(
        self, batch: Mapping[str, Any], *, n_units: int, n_lines: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the graph and project it onto the line axis.

        Args:
            batch: The batch.
            n_units: ``B``.
            n_lines: Active line slots.

        Returns:
            ``(line_states [B, L_active, D_str], line_mask [B, L_active])``. All zeros and
            all ``False`` when the fusion does not use this branch.
        """
        if not self.fusion.uses_structural:
            return self._absent(batch, n_units=n_units, n_lines=n_lines, width=self.fusion.str_dim)
        graph_batch = {
            "node_type": batch["str_node_type"],
            "edge_index": batch["str_edge_index"],
            "batch": batch["str_batch"],
        }
        for optional in ("str_edge_type", "str_subtokens"):
            if optional in batch:
                graph_batch[optional.removeprefix("str_")] = batch[optional]
        node_states = self.structural(graph_batch)
        return align_nodes_to_lines(
            node_states,
            batch["str_line"],
            batch["str_batch"],
            n_lines=n_lines,
            n_units=n_units,
        )

    def _absent(
        self, batch: Mapping[str, Any], *, n_units: int, n_lines: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stand in for a branch the fusion does not use.

        Args:
            batch: The batch, read only for device and dtype.
            n_units: ``B``.
            n_lines: Active line slots.
            width: Feature width the fusion expects from this branch.

        Returns:
            ``(zeros [B, L_active, width], all-False mask [B, L_active])``.
        """
        reference = batch["line_valid"]
        device = reference.device
        states = torch.zeros((n_units, n_lines, width), device=device)
        mask = torch.zeros((n_units, n_lines), dtype=torch.bool, device=device)
        return states, mask

    def _pad_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Pad trimmed line logits back to the configured ``L``.

        Args:
            logits: ``[B, L_active]``.

        Returns:
            ``[B, L]``, the tail filled with
            :data:`~vulnloc.models.heads.line_head.MASKED_LOGIT`.
        """
        missing = self.n_lines - logits.shape[1]
        if missing <= 0:
            return logits
        tail = logits.new_full((logits.shape[0], missing), MASKED_LOGIT)
        return torch.cat([logits, tail], dim=1)

    def _pad_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Pad a trimmed line mask back to the configured ``L``.

        Args:
            mask: ``[B, L_active]``.

        Returns:
            ``[B, L]``, the tail ``False``.
        """
        missing = self.n_lines - mask.shape[1]
        if missing <= 0:
            return mask
        tail = torch.zeros((mask.shape[0], missing), dtype=torch.bool, device=mask.device)
        return torch.cat([mask, tail], dim=1)

    def extra_repr(self) -> str:
        """Show the line budget in ``print(model)``."""
        return f"n_lines={self.n_lines}, trim_line_axis={self.trim_line_axis}"
