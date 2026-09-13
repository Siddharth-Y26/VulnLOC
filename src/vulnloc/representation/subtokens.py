"""Splitting identifiers into subtokens — the other half of ADR-002 §D2.5 (L1).

The structural branch embeds a node's grammar symbol and, per ADR-002 §D2.5, should also
embed the node's own text. Without it two identifiers are the same input, and
``Memory.md`` §1.12 measures what that costs: **76.22% of flagged lines are
indistinguishable from an unflagged one** when node types are all the model has.

``buffer_size``, ``bufferSize`` and ``BUFFER_SIZE`` are the same two words wearing three
conventions, and a vocabulary that treats them as three unrelated symbols learns each from
a third of the evidence. Splitting on case and underscore boundaries is the standard fix
and is what the pre-trained code models this project already uses were trained with.

Deliberately **not** a learned subword model. A BPE vocabulary would be a second trained
artefact with its own fitting procedure, its own digest, and its own way of disagreeing
with the one CodeBERT already carries. Identifier splitting is a pure function of the
string, so two runs of the same corpus produce the same subtokens without an artefact
mediating.

Lowercased, because ``Size`` and ``size`` are one word, and the grammar symbol already
distinguishes a type name from a variable.
"""

from __future__ import annotations

import re

__all__ = ["MAX_SUBTOKENS", "split_identifier"]

MAX_SUBTOKENS = 5
"""Subtokens kept per node.

Measured on the corpus: the overwhelming majority of identifiers split into 1-3 pieces,
so 5 covers the tail without paying for it on every node. A fixed width is what lets the
graph carry ``[N, K]`` rather than a ragged structure, and the cost of the cap is that a
very long identifier loses its last words — the first ones carry more signal.
"""

# Split on: underscores and other separators; a lower-to-upper transition (camelCase); an
# acronym running into a word (HTTPServer -> HTTP, Server); and letter/digit boundaries.
_BOUNDARY = re.compile(
    r"[^0-9A-Za-z]+"  # separators
    r"|(?<=[a-z0-9])(?=[A-Z])"  # camelCase
    r"|(?<=[A-Z])(?=[A-Z][a-z])"  # HTTPServer
    r"|(?<=[A-Za-z])(?=[0-9])"  # buf2
    r"|(?<=[0-9])(?=[A-Za-z])"
)


def split_identifier(text: str) -> list[str]:
    """Split one identifier into lowercased subtokens.

    Args:
        text: The node's source text. Empty for interior nodes, which carry no text of
            their own — see :class:`~vulnloc.representation.graphs.base.GraphNode`.

    Returns:
        Subtokens in source order, lowercased, empty pieces dropped. An operator or
        punctuation token returns ``[]``: ``+`` and ``==`` carry their meaning in the
        grammar symbol, which the type embedding already has, so emitting them again as
        subtokens would double-count the one feature the branch already sees.

    Examples:
        >>> split_identifier("buffer_size")
        ['buffer', 'size']
        >>> split_identifier("HTTPServerPort2")
        ['http', 'server', 'port', '2']
        >>> split_identifier("+")
        []
    """
    if not text:
        return []
    return [piece.lower() for piece in _BOUNDARY.split(text) if piece]
