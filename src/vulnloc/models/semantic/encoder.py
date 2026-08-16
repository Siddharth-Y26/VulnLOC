"""The semantic branch: a pretrained code encoder over token windows.

``Architecture.md`` §4.2 fixes the contract at ``forward(batch) -> [B, T, D_sem]`` — this
module returns *token* states and deliberately stops there. Projection onto lines is
:mod:`vulnloc.models.alignment`'s job, shared with the structural branch, because a branch
that pooled its own way would let the two axes drift apart without anything failing.

CodeBERT and GraphCodeBERT are the same architecture with different pretraining, so they
are one class behind two registry keys rather than two near-identical files. GraphCodeBERT
additionally accepts data-flow edges at input; **this project does not feed them**, and
says so rather than implying otherwise — the structural information enters through the GNN
branch, which is the whole point of the architecture. Used this way GraphCodeBERT is a
differently-pretrained checkpoint, and the A-series ablations should read it as exactly
that.

Memory behaviour is a first-class argument here, not an afterthought
(``Memory.md`` §1.9): ``frozen`` and ``gradient_checkpointing`` are constructor options
because on a 4 GiB card they decide whether the model runs at 22.9 windows/s or spills
into host RAM at 2.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoConfig, AutoModel

from vulnloc.models.registry import register_semantic
from vulnloc.utils.exceptions import ModelError
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["PretrainedCodeEncoder"]

_LOGGER = get_logger(__name__)


@register_semantic("codebert")
@register_semantic("graphcodebert")
class PretrainedCodeEncoder(torch.nn.Module):
    """A HuggingFace encoder over ``[W, T]`` token windows.

    Args:
        model_name: HuggingFace checkpoint id.
        frozen: Run the encoder under ``no_grad`` in eval mode and train only what sits
            above it. Measured at 1,234 MiB against 2,456 MiB trainable
            (``Memory.md`` §1.9) — the fallback, and a legitimate ablation.
        gradient_checkpointing: Recompute activations in the backward pass instead of
            storing them. The default because it is simultaneously the cheapest *and*
            the fastest trainable configuration on this hardware.
        dropout: Dropout applied to the returned token states.
        pretrained: Load pretrained weights. ``False`` builds the same architecture from
            config with random weights — for tests, and for an
            ablation that asks what pretraining is worth.

    Raises:
        ModelError: If the checkpoint cannot be loaded.
    """

    def __init__(
        self,
        model_name: str,
        *,
        frozen: bool = False,
        gradient_checkpointing: bool = True,
        dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            if pretrained:
                self.encoder = AutoModel.from_pretrained(model_name)
            else:
                config = AutoConfig.from_pretrained(model_name)
                self.encoder = AutoModel.from_config(config)  # type: ignore[no-untyped-call]
        except OSError as error:
            # Narrow: a missing checkpoint or an offline cache, not a config mistake.
            raise ModelError(
                "could not load the encoder checkpoint",
                model_name=model_name,
                pretrained=pretrained,
                cause=str(error),
            ) from error

        self.model_name = model_name
        self.frozen = frozen
        self.dropout = torch.nn.Dropout(dropout)

        if gradient_checkpointing and not frozen:
            self.encoder.gradient_checkpointing_enable()
        if frozen:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()

        _LOGGER.info(
            "Semantic encoder ready",
            extra={
                "model_name": model_name,
                "hidden_size": self.hidden_size,
                "frozen": frozen,
                "gradient_checkpointing": gradient_checkpointing and not frozen,
                "trainable_parameters": sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                ),
            },
        )

    @property
    def hidden_size(self) -> int:
        """Width of the token states this encoder returns, ``D_sem``."""
        return int(self.encoder.config.hidden_size)

    def train(self, mode: bool = True) -> PretrainedCodeEncoder:
        """Keep a frozen encoder in eval mode even when the parent is trained.

        Without this, ``model.train()`` would re-enable the encoder's dropout, so a
        "frozen" encoder would return different features every epoch — a slow, confusing
        bug that looks like instability rather than a mistake.

        Args:
            mode: Whether the parent is entering training mode.

        Returns:
            ``self``.
        """
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Encode one batch of token windows.

        Args:
            batch: Must carry ``input_ids`` ``[W, T]`` and ``attention_mask`` ``[W, T]``.

        Returns:
            Token states, ``[W, T, D_sem]``.

        Raises:
            ModelError: If the batch is missing a required key.
        """
        for key in ("input_ids", "attention_mask"):
            if key not in batch:
                raise ModelError(
                    "semantic batch is missing a required key",
                    missing=key,
                    present=sorted(batch),
                )

        if self.frozen:
            with torch.no_grad():
                output = self.encoder(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
            states = output.last_hidden_state.detach()
        else:
            output = self.encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            states = output.last_hidden_state
        return torch.as_tensor(self.dropout(states))
