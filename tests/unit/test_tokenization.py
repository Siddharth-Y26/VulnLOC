"""Guards the semantic half of the line-alignment bridge.

The invariant under test is the one the whole project rests on: **a token's recorded line
is the line its text is actually on.** Everything else here exists to stop that from
breaking quietly — non-ASCII source (where character and byte offsets diverge), special
and padding tokens (which must claim no line at all), and long functions (where a
truncating encoder silently drops the labels `clean_eval` is graded on).

The tokenizer is built locally rather than downloaded. A byte-level BPE with no merges
tokenizes one character at a time, which gives real offset mappings and a real
special-token wrapper while staying deterministic and offline — the logic under test is
ours, not CodeBERT's vocabulary.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from vulnloc.representation.tokenization import (
    NO_LINE,
    LineTokenizer,
    TokenWindow,
    windows_to_tensors,
)
from vulnloc.representation.units import CodeUnit
from vulnloc.utils.exceptions import AlignmentError


@pytest.fixture(scope="module")
def hf_tokenizer():
    """A minimal offline fast tokenizer: one token per character, RoBERTa-style specials."""
    vocab = {"<s>": 0, "</s>": 1, "<pad>": 2, "<unk>": 3}
    for character in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[character] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 1)]
    )
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


@pytest.fixture
def line_tokenizer(hf_tokenizer):
    return LineTokenizer(hf_tokenizer, max_length=64, stride=16)


CLEAN_C = "int add(int a, int b)\n{\n    return a + b;\n}\n"


def unit_of(source: str, unit_id: str = "u1") -> CodeUnit:
    return CodeUnit.from_source(unit_id, source, language="C")


def content_tokens(window: TokenWindow, hf_tokenizer):
    """Yield ``(text, line)`` for every real token that claims a line."""
    for token_id, line in zip(
        window.input_ids[: window.n_real_tokens],
        window.token_to_line[: window.n_real_tokens],
        strict=True,
    ):
        if line == NO_LINE:
            continue
        yield hf_tokenizer.decode([token_id]), line


def assert_lines_match_offsets(window: TokenWindow, source: str, hf_tokenizer) -> None:
    """Check a window's lines against an independent derivation.

    Recomputes each token's line by counting separators before its character offset,
    which shares no code with :class:`LineMap`. Text containment cannot be used here:
    byte-level BPE legitimately emits tokens that are *part* of a multi-byte character,
    and those do not decode to anything findable in the source.
    """
    offsets = hf_tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    body = [line for line in window.token_to_line[: window.n_real_tokens] if line != NO_LINE]
    for (start, end), line in zip(offsets[: len(body)], body, strict=True):
        if end <= start:
            continue
        assert line == source[:start].count("\n") + 1, (start, line)


def test_every_token_lands_on_the_line_its_text_is_on(line_tokenizer, hf_tokenizer):
    unit = unit_of(CLEAN_C)

    encoded = line_tokenizer.encode(unit)

    for text, line in content_tokens(encoded.windows[0], hf_tokenizer):
        if text.strip():
            assert text in unit.lines[line - 1], (text, line)


def test_non_ascii_source_uses_character_offsets_not_byte_offsets(line_tokenizer, hf_tokenizer):
    """The bug this catches shifts every line after a multi-byte character.

    ``é`` is one character and two UTF-8 bytes. Reading the tokenizer's offsets through
    ``byte_to_line`` instead of ``char_to_line`` drifts by one byte per accent, which on
    a real corpus is a rare, silent, off-by-N.
    """
    source = 'const char *note = "café";\nint after = 1;\nreturn after;\n'
    unit = unit_of(source)

    encoded = line_tokenizer.encode(unit)

    assert_lines_match_offsets(encoded.windows[0], source, hf_tokenizer)
    # Line 3 sits past the accent's byte/character divergence; a byte-based reading of
    # the tokenizer's character offsets lands short of it.
    assert 3 in encoded.lines_covered


def test_special_and_padding_tokens_claim_no_line(line_tokenizer):
    unit = unit_of(CLEAN_C)

    window = line_tokenizer.encode(unit).windows[0]

    assert window.token_to_line[0] == NO_LINE
    assert window.token_to_line[window.n_real_tokens - 1] == NO_LINE
    assert all(line == NO_LINE for line in window.token_to_line[window.n_real_tokens :])
    assert all(mask == 0 for mask in window.attention_mask[window.n_real_tokens :])


def test_no_line_is_negative_so_a_misread_cannot_hit_a_real_row():
    # Lines are 1-indexed; a sentinel of 0 would be a plausible index into a [B, L]
    # tensor and would pool silently into the wrong row.
    assert NO_LINE < 0


def test_windowing_reaches_every_line_of_a_long_function(line_tokenizer):
    source = "".join(f"int variable_{index} = {index};\n" for index in range(200))
    unit = unit_of(source)

    encoded = line_tokenizer.encode(unit)

    assert encoded.n_windows > 1
    assert encoded.is_truncated_at_one_window
    assert encoded.lines_covered == frozenset(range(1, unit.n_lines))


def test_truncation_loses_lines_that_windowing_keeps(hf_tokenizer):
    source = "".join(f"int variable_{index} = {index};\n" for index in range(200))
    unit = unit_of(source)

    windowed = LineTokenizer(hf_tokenizer, max_length=64, stride=16).encode(unit)
    truncated = LineTokenizer(hf_tokenizer, max_length=64, windowing=False).encode(unit)

    assert truncated.n_windows == 1
    assert truncated.lines_covered < windowed.lines_covered
    # The measurement Memory.md §7 #14 requires: what truncation costs, not that it costs.
    assert len(windowed.lines_covered - truncated.lines_covered) > 0


def test_first_window_matches_what_a_truncating_encoder_would_keep(hf_tokenizer):
    source = "".join(f"int variable_{index} = {index};\n" for index in range(200))
    unit = unit_of(source)

    windowed = LineTokenizer(hf_tokenizer, max_length=64, stride=16).encode(unit)
    truncated = LineTokenizer(hf_tokenizer, max_length=64, windowing=False).encode(unit)

    assert windowed.lines_in_first_window == truncated.lines_covered


def test_windows_overlap_by_the_stride(hf_tokenizer):
    source = "".join(f"int variable_{index} = {index};\n" for index in range(200))
    tokenizer = LineTokenizer(hf_tokenizer, max_length=64, stride=16)

    encoded = tokenizer.encode(unit_of(source))

    first, second = encoded.windows[0], encoded.windows[1]
    assert first.lines_covered & second.lines_covered, "a line at the seam belongs to both"


def test_a_short_unit_is_one_window(line_tokenizer):
    encoded = line_tokenizer.encode(unit_of(CLEAN_C))

    assert encoded.n_windows == 1
    assert not encoded.is_truncated_at_one_window


def test_empty_source_still_yields_a_window(line_tokenizer):
    encoded = line_tokenizer.encode(unit_of(""))

    assert encoded.n_windows == 1
    assert encoded.lines_covered == frozenset()


def test_every_window_is_exactly_max_length(hf_tokenizer):
    source = "".join(f"int variable_{index} = {index};\n" for index in range(200))
    tokenizer = LineTokenizer(hf_tokenizer, max_length=64, stride=16)

    encoded = tokenizer.encode(unit_of(source))

    for window in encoded.windows:
        assert len(window.input_ids) == 64
        assert len(window.attention_mask) == 64
        assert len(window.token_to_line) == 64


def test_slow_tokenizer_is_refused_rather_than_worked_around():
    class SlowTokenizer:
        is_fast = False

    with pytest.raises(AlignmentError):
        LineTokenizer(SlowTokenizer())


def test_stride_that_would_not_advance_raises(hf_tokenizer):
    # body length is 64 - 2 specials = 62; a stride of 62 makes every window identical.
    with pytest.raises(AlignmentError) as excinfo:
        LineTokenizer(hf_tokenizer, max_length=64, stride=62)

    assert excinfo.value.details["stride"] == 62


def test_max_length_smaller_than_the_special_tokens_raises(hf_tokenizer):
    with pytest.raises(AlignmentError):
        LineTokenizer(hf_tokenizer, max_length=1)


def test_windows_to_tensors_shapes_and_dtypes(line_tokenizer):
    import torch

    encoded = line_tokenizer.encode(unit_of(CLEAN_C))

    batch = windows_to_tensors(encoded.windows)

    assert batch["input_ids"].shape == (1, 64)
    assert batch["token_to_line"].shape == (1, 64)
    assert batch["input_ids"].dtype == torch.long
    assert batch["token_to_line"].dtype == torch.long


def test_windows_to_tensors_rejects_ragged_input(line_tokenizer):
    encoded = line_tokenizer.encode(unit_of(CLEAN_C))
    short = TokenWindow(
        unit_id="u1",
        window_index=1,
        input_ids=(1, 2),
        attention_mask=(1, 1),
        token_to_line=(1, 1),
        n_real_tokens=2,
    )

    with pytest.raises(AlignmentError):
        windows_to_tensors([encoded.windows[0], short])


def test_windows_to_tensors_rejects_an_empty_sequence():
    with pytest.raises(AlignmentError):
        windows_to_tensors([])


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.text(
            alphabet=st.characters(
                min_codepoint=32, max_codepoint=0x2FF, blacklist_characters="\n"
            ),
            min_size=0,
            max_size=24,
        ),
        min_size=1,
        max_size=12,
    )
)
def test_property_every_token_line_is_in_range_and_matches_its_offset(hf_tokenizer, lines):
    """Over randomised text, including non-ASCII and blank lines.

    Two assertions, because they fail differently: an out-of-range line means the map
    broke, while a line disagreeing with the independent newline count means it drifted.
    """
    source = "\n".join(lines)
    unit = unit_of(source)
    tokenizer = LineTokenizer(hf_tokenizer, max_length=512, stride=64)

    encoded = tokenizer.encode(unit)

    for window in encoded.windows:
        for line in window.token_to_line[: window.n_real_tokens]:
            if line != NO_LINE:
                assert 1 <= line <= unit.n_lines
    assert_lines_match_offsets(encoded.windows[0], source, hf_tokenizer)
