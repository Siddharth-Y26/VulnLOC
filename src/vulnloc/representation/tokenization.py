"""Source → tokens, with every token still attached to a line (``Architecture.md`` §4.1).

This is the semantic half of the line-alignment bridge, and it is the second place in the
project where a line number can be silently lost. The first was Tree-sitter's byte offsets
(``line_map``); this one is the tokenizer's **character** offsets. They are different
offset spaces and they agree only on ASCII, which is why :class:`LineMap` carries both.

Two properties are load-bearing:

**Every content token knows its line.** ``token_to_line`` is built from the tokenizer's
own ``offset_mapping`` through :meth:`LineMap.char_to_line`, never from counting newlines
in the decoded text. Special and padding tokens carry ``-1`` — a sentinel rather than line
0, so that a bug reading it as an index fails loudly instead of pooling into a real line.

**A function longer than the window is not silently cut.** Big-Vul functions routinely
exceed 512 tokens, and `clean_eval` — the partition carrying every headline line-level
number — averages 138 lines against the corpus's 31 (``Memory.md`` §7 #14). Truncating
there would destroy exactly the labels the project is measured on, so the default is a
**striding window** over the whole function. Plain truncation stays available as
``windowing=False`` because it is what the published baselines do, and the difference
between the two is a number this project owes its reader.

``torch`` is not imported here. Windows are plain tuples of ints; :func:`windows_to_tensors`
does the conversion and imports torch inside itself, following ``to_pyg``. That keeps a
corpus-wide truncation pass — 188,636 functions — out of a 2 GB import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vulnloc.utils.exceptions import AlignmentError
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vulnloc.representation.units import CodeUnit

__all__ = [
    "NO_LINE",
    "LineTokenizer",
    "TokenWindow",
    "TokenizedUnit",
    "windows_to_tensors",
]

_LOGGER = get_logger(__name__)

NO_LINE = -1
"""``token_to_line`` value for a token that belongs to no source line.

Special tokens (``<s>``, ``</s>``) and padding. Deliberately negative: lines are
1-indexed, so a sentinel of 0 would be a *plausible* index and would scatter silently
into whatever tensor row 0 happens to be.
"""


@dataclass(frozen=True)
class TokenWindow:
    """One fixed-length window over a unit's tokens.

    A unit shorter than the window is a single window; a longer one is several,
    overlapping by the tokenizer's stride so no line sits only at a boundary.

    Attributes:
        unit_id: The unit this window came from.
        window_index: 0-based position in the unit's window sequence.
        input_ids: Token ids, including the special tokens, padded to the window length.
        attention_mask: 1 for real tokens, 0 for padding.
        token_to_line: 1-indexed source line per token, :data:`NO_LINE` for special and
            padding tokens. Same length as :attr:`input_ids`.
        n_real_tokens: Tokens before padding, specials included.
    """

    unit_id: str
    window_index: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_to_line: tuple[int, ...]
    n_real_tokens: int

    @property
    def lines_covered(self) -> frozenset[int]:
        """Source lines that at least one token in this window lands on."""
        return frozenset(line for line in self.token_to_line if line != NO_LINE)


@dataclass(frozen=True)
class TokenizedUnit:
    """One function's tokens, windowed, with the line axis intact.

    Attributes:
        unit_id: Stable corpus identifier.
        n_lines: Source lines in the unit, from :class:`CodeUnit`.
        windows: One or more windows covering the unit.
        n_body_tokens: Tokens the source needs *without* special tokens or windowing.
            The honest measure of how long the function is in the encoder's terms.
        max_length: Window length this unit was encoded at.
    """

    unit_id: str
    n_lines: int
    windows: tuple[TokenWindow, ...] = field(repr=False)
    n_body_tokens: int = 0
    max_length: int = 512

    @property
    def n_windows(self) -> int:
        """How many windows the unit needed."""
        return len(self.windows)

    @property
    def lines_covered(self) -> frozenset[int]:
        """Source lines reachable by at least one token, across every window."""
        covered: set[int] = set()
        for window in self.windows:
            covered |= window.lines_covered
        return frozenset(covered)

    @property
    def lines_in_first_window(self) -> frozenset[int]:
        """Lines a plain truncating encoder would have kept.

        The comparison partner for :attr:`lines_covered`: the difference between the two
        is precisely what windowing buys, and what every truncating baseline loses.
        """
        return self.windows[0].lines_covered if self.windows else frozenset()

    @property
    def is_truncated_at_one_window(self) -> bool:
        """Whether a single window would have failed to reach the end of the function."""
        return self.n_windows > 1


class LineTokenizer:
    """Wraps a HuggingFace **fast** tokenizer and keeps the line axis attached.

    A slow tokenizer is rejected rather than worked around: without
    ``offset_mapping`` there is no way to recover a token's character span, and the only
    alternatives are re-deriving lines from decoded text — which round-trips through
    normalisation and is exactly the kind of silent drift ``Rules.md`` §2 bans.

    Args:
        tokenizer: A ``PreTrainedTokenizerFast``.
        max_length: Window length in tokens, special tokens included. CodeBERT and
            GraphCodeBERT are both 512.
        stride: Overlap between consecutive windows, in body tokens. Only used when
            ``windowing`` is true.
        windowing: When false, the unit is truncated to one window — the published
            baseline behaviour, kept so the cost of truncation can be measured rather
            than asserted.

    Raises:
        AlignmentError: If ``tokenizer`` is not a fast tokenizer, or if ``stride`` leaves
            no forward progress.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = 512,
        stride: int = 128,
        windowing: bool = True,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise AlignmentError(
                "a fast tokenizer is required; only it reports offset_mapping",
                tokenizer=type(tokenizer).__name__,
            )
        self._tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.windowing = windowing

        # Specials are added by hand, one pair per window, so the layout has to be known
        # up front. It is *probed* rather than assumed: `<s> body </s>` is RoBERTa's
        # arrangement, not a rule, and the helper methods that used to expose it moved
        # between transformers 4 and 5. A probe is version-proof and would catch a
        # tokenizer that wraps its input differently instead of silently mislabelling
        # two tokens per window.
        self._prefix_ids, self._suffix_ids = _probe_special_tokens(tokenizer)
        self._n_special = len(self._prefix_ids) + len(self._suffix_ids)
        self._body_length = max_length - self._n_special
        if self._body_length <= 0:
            raise AlignmentError(
                "max_length leaves no room for content after special tokens",
                max_length=max_length,
                n_special=self._n_special,
            )
        self._step = self._body_length - stride
        if windowing and self._step <= 0:
            raise AlignmentError(
                "stride is not smaller than the window body; windows would not advance",
                max_length=max_length,
                stride=stride,
                body_length=self._body_length,
            )

    @property
    def pad_token_id(self) -> int:
        """Padding id, falling back to the EOS id when the tokenizer defines none."""
        pad = self._tokenizer.pad_token_id
        return int(pad if pad is not None else self._tokenizer.eos_token_id)

    def encode(self, unit: CodeUnit) -> TokenizedUnit:
        """Tokenize one unit into line-annotated windows.

        Args:
            unit: The unit to encode.

        Returns:
            The unit's windows, with ``token_to_line`` populated.

        Raises:
            AlignmentError: If the tokenizer reports an offset outside the source.
        """
        encoded = self._tokenizer(
            unit.source,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            padding=False,
        )
        body_ids: list[int] = list(encoded["input_ids"])
        offsets: Sequence[tuple[int, int]] = encoded["offset_mapping"]
        body_lines = [self._line_of(start, end, unit) for start, end in offsets]

        starts = self._window_starts(len(body_ids))
        windows = tuple(
            self._build_window(unit.unit_id, index, body_ids, body_lines, start)
            for index, start in enumerate(starts)
        )
        return TokenizedUnit(
            unit_id=unit.unit_id,
            n_lines=unit.n_lines,
            windows=windows,
            n_body_tokens=len(body_ids),
            max_length=self.max_length,
        )

    def _line_of(self, start: int, end: int, unit: CodeUnit) -> int:
        """Return the 1-indexed line a token belongs to.

        The token is assigned to the line its **first character** sits on, matching how
        graph nodes are assigned from ``start_byte``. Keeping the two branches on one
        rule matters more than any refinement, because they are pooled onto a shared
        axis and a disagreement would misalign them against each other.

        Byte-level BPE can emit whitespace-only tokens that straddle a newline
        (``Ċ`` plus the next line's indent). Those carry no content, and assigning them
        to the line they start on is both defensible and consistent.

        Args:
            start: Character offset of the token's first character.
            end: Character offset one past its last.
            unit: The unit being encoded, for its line map.

        Returns:
            The 1-indexed line, or :data:`NO_LINE` for a zero-width token.

        Raises:
            AlignmentError: If the offset falls outside the source.
        """
        if end <= start:
            # Fast tokenizers report (0, 0) for anything synthesised. There should be
            # none here since specials are added by hand, but a zero-width token must
            # never be allowed to claim line 1.
            return NO_LINE
        return unit.line_map.char_to_line(start)

    def _window_starts(self, n_body: int) -> list[int]:
        """Return the body-token offsets each window begins at.

        Args:
            n_body: Total body tokens in the unit.

        Returns:
            Window start offsets. Always at least one, so an empty unit still produces
            a window rather than an empty sequence downstream.
        """
        if n_body <= self._body_length or not self.windowing:
            return [0]
        starts = [0]
        # Stop as soon as a window reaches the end, rather than stepping to the end and
        # trimming afterwards: the trimming version emits a final window that is pure
        # overlap whenever the length lands just past a step boundary.
        while starts[-1] + self._body_length < n_body:
            starts.append(starts[-1] + self._step)
        return starts

    def _build_window(
        self,
        unit_id: str,
        window_index: int,
        body_ids: list[int],
        body_lines: list[int],
        start: int,
    ) -> TokenWindow:
        """Assemble one padded window with its special tokens.

        Args:
            unit_id: The unit's identifier.
            window_index: Position in the window sequence.
            body_ids: All body token ids for the unit.
            body_lines: Line per body token, same length as ``body_ids``.
            start: Body-token offset this window begins at.

        Returns:
            The window.
        """
        chunk_ids = body_ids[start : start + self._body_length]
        chunk_lines = body_lines[start : start + self._body_length]

        ids = [*self._prefix_ids, *chunk_ids, *self._suffix_ids]
        lines = [
            *([NO_LINE] * len(self._prefix_ids)),
            *chunk_lines,
            *([NO_LINE] * len(self._suffix_ids)),
        ]

        n_real = len(ids)
        pad = self.max_length - n_real
        if pad > 0:
            ids = ids + [self.pad_token_id] * pad
            lines = lines + [NO_LINE] * pad
        mask = [1] * n_real + [0] * max(pad, 0)

        return TokenWindow(
            unit_id=unit_id,
            window_index=window_index,
            input_ids=tuple(ids),
            attention_mask=tuple(mask),
            token_to_line=tuple(lines),
            n_real_tokens=n_real,
        )


def _probe_special_tokens(tokenizer: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Discover which ids a tokenizer wraps a single sequence in.

    Encodes one probe both ways and locates the bare body inside the decorated version.
    Works regardless of how the tokenizer arranges its specials, and fails loudly if the
    body is not a contiguous run — which would mean the wrapper cannot be modelled as a
    prefix and a suffix, and that windowing by hand is unsafe for it.

    Args:
        tokenizer: A fast tokenizer.

    Returns:
        ``(prefix_ids, suffix_ids)``.

    Raises:
        AlignmentError: If the bare body does not appear contiguously in the decorated
            encoding.
    """
    probe = "int probe_value = 0;"
    decorated: list[int] = list(tokenizer(probe, add_special_tokens=True)["input_ids"])
    bare: list[int] = list(tokenizer(probe, add_special_tokens=False)["input_ids"])
    for offset in range(len(decorated) - len(bare) + 1):
        if decorated[offset : offset + len(bare)] == bare:
            return tuple(decorated[:offset]), tuple(decorated[offset + len(bare) :])
    raise AlignmentError(
        "cannot locate the body inside the tokenizer's special-token wrapper",
        tokenizer=type(tokenizer).__name__,
        decorated_length=len(decorated),
        bare_length=len(bare),
    )


def windows_to_tensors(windows: Sequence[TokenWindow]) -> dict[str, Any]:
    """Stack windows into the batch a semantic encoder consumes.

    Args:
        windows: Windows of equal length — from one unit or from several.

    Returns:
        ``{"input_ids": [W, T], "attention_mask": [W, T], "token_to_line": [W, T]}``,
        all ``torch.long``.

    Raises:
        AlignmentError: If the windows are not all the same length.
    """
    import torch

    if not windows:
        raise AlignmentError("cannot stack an empty window sequence")
    length = len(windows[0].input_ids)
    for window in windows:
        if len(window.input_ids) != length:
            raise AlignmentError(
                "windows differ in length; they cannot be stacked",
                unit_id=window.unit_id,
                expected=length,
                found=len(window.input_ids),
            )
    return {
        "input_ids": torch.tensor([w.input_ids for w in windows], dtype=torch.long),
        "attention_mask": torch.tensor([w.attention_mask for w in windows], dtype=torch.long),
        "token_to_line": torch.tensor([w.token_to_line for w in windows], dtype=torch.long),
    }
