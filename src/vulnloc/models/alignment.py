"""The shared line axis (``Architecture.md`` §4.3).

The whole architecture turns on one idea: the semantic branch and the structural branch
disagree about what a "position" is — one has tokens, the other has graph nodes — and
they are made comparable by projecting both onto **lines**. Every downstream claim, from
cross-attention to the per-line score to the evidence in an explanation, is stated on
that axis. If this module is wrong, nothing above it can be right, and nothing above it
will complain.

So it is written to make the two failure modes structurally impossible rather than
merely tested against:

**Off-by-one.** Lines are 1-indexed everywhere in this project. Rather than subtracting
one and hoping, the accumulator is allocated with an extra row at index 0 and invalid
positions are *clamped into it*; row 0 is then sliced away. Line ``n`` lives at output
index ``n - 1`` because the slice moved it there, not because arithmetic was done twice.

**Silent contamination.** Special tokens, padding and any node with no line carry
:data:`~vulnloc.representation.tokenization.NO_LINE`. They are masked to zero *before*
the scatter and excluded from the divisor, so they cannot dilute a real line's mean. A
line that received nothing comes back as an explicit ``False`` in the mask rather than as
a plausible-looking zero vector.

Both branches call the same core (:func:`scatter_mean_lines`) so they cannot drift apart:
a change to pooling changes both, or neither.
"""

from __future__ import annotations

import torch

from vulnloc.utils.exceptions import AlignmentError

__all__ = [
    "LineAligner",
    "align_nodes_to_lines",
    "align_tokens_to_lines",
    "scatter_mean_lines",
]


def scatter_mean_lines(
    features: torch.Tensor,
    line_index: torch.Tensor,
    segment_index: torch.Tensor,
    *,
    n_segments: int,
    n_lines: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool flat rows onto a ``[segment, line]`` grid.

    The one primitive both branches use. Rows are tokens for the semantic branch and
    graph nodes for the structural branch; segments are units.

    Args:
        features: ``[M, D]`` row features.
        line_index: ``[M]`` 1-indexed source line per row. Anything below 1 means "no
            line" and is dropped.
        segment_index: ``[M]`` which output segment each row belongs to.
        n_segments: Number of output segments, ``S``.
        n_lines: Number of line slots, ``L``. Lines above this are dropped.

    Returns:
        ``(pooled [S, L, D], mask [S, L])``. ``mask`` is ``True`` where at least one row
        landed, and ``pooled`` is exactly zero everywhere ``mask`` is ``False``.

    Raises:
        AlignmentError: If the inputs disagree in length, or a segment index is out of
            range.
    """
    if features.dim() != 2:
        raise AlignmentError("features must be [M, D]", shape=tuple(features.shape))
    n_rows, n_features = features.shape
    if line_index.shape != (n_rows,) or segment_index.shape != (n_rows,):
        raise AlignmentError(
            "line_index and segment_index must both be [M]",
            rows=n_rows,
            line_index=tuple(line_index.shape),
            segment_index=tuple(segment_index.shape),
        )
    if n_segments <= 0 or n_lines <= 0:
        raise AlignmentError(
            "n_segments and n_lines must be positive", n_segments=n_segments, n_lines=n_lines
        )
    if n_rows and (int(segment_index.min()) < 0 or int(segment_index.max()) >= n_segments):
        raise AlignmentError(
            "segment index out of range",
            n_segments=n_segments,
            observed_min=int(segment_index.min()),
            observed_max=int(segment_index.max()),
        )

    # Row 0 of each segment is a sink for everything that belongs to no line. Clamping
    # into it, then slicing it off, is what keeps the 1-indexed convention honest: no
    # subtraction happens anywhere that could be forgotten or done twice.
    valid = (line_index >= 1) & (line_index <= n_lines)
    slots = torch.where(valid, line_index, torch.zeros_like(line_index)).long()
    flat = segment_index.long() * (n_lines + 1) + slots

    weights = valid.to(features.dtype).unsqueeze(-1)
    sums = features.new_zeros((n_segments * (n_lines + 1), n_features))
    sums.index_add_(0, flat, features * weights)
    counts = features.new_zeros(n_segments * (n_lines + 1))
    counts.index_add_(0, flat, valid.to(features.dtype))

    sums = sums.view(n_segments, n_lines + 1, n_features)[:, 1:, :]
    counts = counts.view(n_segments, n_lines + 1)[:, 1:]
    mask = counts > 0
    pooled = sums / counts.clamp(min=1).unsqueeze(-1)
    return pooled, mask


def align_tokens_to_lines(
    token_states: torch.Tensor,
    token_to_line: torch.Tensor,
    *,
    n_lines: int,
    window_to_unit: torch.Tensor | None = None,
    n_units: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project encoder token states onto the line axis.

    ``[W, T, D] → [U, L, D]``. When a long function was split into overlapping windows,
    ``window_to_unit`` maps each window back to its unit and a line appearing in two
    windows is averaged over the tokens of both — which is the point of striding, and the
    reason the seam does not create a discontinuity in the line scores.

    Args:
        token_states: ``[W, T, D]`` encoder outputs, one row per window.
        token_to_line: ``[W, T]`` 1-indexed line per token,
            :data:`~vulnloc.representation.tokenization.NO_LINE` for specials and padding.
        n_lines: Number of line slots, ``L``.
        window_to_unit: ``[W]`` unit index per window. ``None`` means one unit per window.
        n_units: Number of units, ``U``. Defaults to ``W`` when ``window_to_unit`` is
            ``None``, and to ``window_to_unit.max() + 1`` otherwise.

    Returns:
        ``(line_states [U, L, D], line_mask [U, L])``.

    Raises:
        AlignmentError: If the shapes disagree.
    """
    if token_states.dim() != 3:
        raise AlignmentError("token_states must be [W, T, D]", shape=tuple(token_states.shape))
    n_windows, n_tokens, n_features = token_states.shape
    if token_to_line.shape != (n_windows, n_tokens):
        raise AlignmentError(
            "token_to_line must match token_states in [W, T]",
            token_states=tuple(token_states.shape),
            token_to_line=tuple(token_to_line.shape),
        )

    if window_to_unit is None:
        window_to_unit = torch.arange(n_windows, device=token_states.device)
        segments = n_units if n_units is not None else n_windows
    else:
        if window_to_unit.shape != (n_windows,):
            raise AlignmentError(
                "window_to_unit must be [W]",
                expected=n_windows,
                found=tuple(window_to_unit.shape),
            )
        segments = n_units if n_units is not None else int(window_to_unit.max()) + 1

    rows = window_to_unit.repeat_interleave(n_tokens)
    return scatter_mean_lines(
        token_states.reshape(-1, n_features),
        token_to_line.reshape(-1),
        rows,
        n_segments=segments,
        n_lines=n_lines,
    )


def align_nodes_to_lines(
    node_states: torch.Tensor,
    node_to_line: torch.Tensor,
    batch_index: torch.Tensor,
    *,
    n_lines: int,
    n_units: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project GNN node states onto the line axis.

    ``[N, D] → [B, L, D]``. PyG batches graphs by concatenating nodes, so ``batch_index``
    is the ``Data.batch`` vector.

    Args:
        node_states: ``[N, D]`` node features after message passing.
        node_to_line: ``[N]`` 1-indexed line per node.
        batch_index: ``[N]`` graph index per node.
        n_lines: Number of line slots, ``L``.
        n_units: Number of graphs, ``B``. Defaults to ``batch_index.max() + 1``.

    Returns:
        ``(line_states [B, L, D], line_mask [B, L])``.

    Raises:
        AlignmentError: If the shapes disagree.
    """
    if node_states.dim() != 2:
        raise AlignmentError("node_states must be [N, D]", shape=tuple(node_states.shape))
    segments = (
        n_units
        if n_units is not None
        else (int(batch_index.max()) + 1 if batch_index.numel() else 1)
    )
    return scatter_mean_lines(
        node_states,
        node_to_line,
        batch_index,
        n_segments=segments,
        n_lines=n_lines,
    )


class LineAligner(torch.nn.Module):
    """Stateless module wrapper over the two alignment functions.

    Carries no parameters. It exists so the pooling strategy is a named, swappable
    component — ``Architecture.md`` §4.3 records mean pooling as the v1 default and
    attention pooling as an ablation candidate, and an ablation that has to edit call
    sites is one nobody runs.

    Args:
        n_lines: Number of line slots to project onto.
    """

    def __init__(self, n_lines: int) -> None:
        super().__init__()
        if n_lines <= 0:
            raise AlignmentError("n_lines must be positive", n_lines=n_lines)
        self.n_lines = n_lines

    def tokens(
        self,
        token_states: torch.Tensor,
        token_to_line: torch.Tensor,
        window_to_unit: torch.Tensor | None = None,
        n_units: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project token states. See :func:`align_tokens_to_lines`.

        Returns:
            ``(line_states [U, L, D], line_mask [U, L])``.
        """
        return align_tokens_to_lines(
            token_states,
            token_to_line,
            n_lines=self.n_lines,
            window_to_unit=window_to_unit,
            n_units=n_units,
        )

    def nodes(
        self,
        node_states: torch.Tensor,
        node_to_line: torch.Tensor,
        batch_index: torch.Tensor,
        n_units: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project node states. See :func:`align_nodes_to_lines`.

        Returns:
            ``(line_states [B, L, D], line_mask [B, L])``.
        """
        return align_nodes_to_lines(
            node_states,
            node_to_line,
            batch_index,
            n_lines=self.n_lines,
            n_units=n_units,
        )

    def extra_repr(self) -> str:
        """Show the line budget in ``print(model)``."""
        return f"n_lines={self.n_lines}"
