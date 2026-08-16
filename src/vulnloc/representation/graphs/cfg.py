"""Control-flow graph — second in ADR-002 §D2.2's build order.

A structured-CFG construction over the parse tree. C is not a structured language —
``goto`` and fallthrough exist — so the recursive descent handles the structured forms
and the two unstructured ones are threaded explicitly through a label map and a
break/continue target stack.

Three decisions worth stating, because each is visible in the output:

* **The node set is the AST's, unchanged.** Only ``cfg`` edges are added. ADR-002 §D2.2
  requires the CPG to be a union with "no new extraction", which is only sound if every
  graph type numbers its nodes identically. The consequence is that a CFG-only graph
  contains every expression node too, most of them isolated; that is the price of a
  union that cannot misalign, and isolated nodes still pool onto their line.
* **There is no synthetic entry or exit node.** Adding one would break the shared node
  set. A ``return`` therefore has no outgoing edge — "flows to exit" is represented by
  the absence of a successor, and :func:`terminal_nodes` recovers them.
* **Statement granularity.** An edge connects statements, not expressions: a whole
  ``for`` header is one node. Sub-expression evaluation order is the AST's business.

What this does **not** model, stated plainly rather than discovered later: exceptions,
``setjmp``/``longjmp``, computed ``goto``, ``switch`` on non-constant case ranges (the
GNU extension), and function calls as transfers of control (they are ``call`` edges,
which no builder emits yet).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vulnloc.representation.graphs.ast import transcribe
from vulnloc.representation.graphs.base import CodeGraph, GraphEdge
from vulnloc.representation.registry import register_graph
from vulnloc.utils.exceptions import GraphBuildError

if TYPE_CHECKING:
    from tree_sitter import Node

    from vulnloc.representation.parser import ParseResult

__all__ = ["CfgGraphBuilder", "build_cfg_edges", "build_cfg_graph", "terminal_nodes"]

GRAPH_TYPE = "cfg"

MAX_NESTING_DEPTH = 200
"""Guard on statement nesting.

Set from measurement, not taste: over 20,031 real functions (``clean_eval`` + ``test``)
the deepest statement nesting is **40**, and 99.9% sit below 20. 200 is 5x the observed
worst case while staying inside the interpreter's frame budget — each nesting level
costs roughly three Python frames, so this guard fires at ~600 of the default 1,000,
*before* a :class:`RecursionError` can escape. That ordering is the point: a
RecursionError here would read as a property of the corpus rather than a limit of this
pass. :func:`build_cfg_edges` converts one anyway, so no input can take down a
corpus-wide run."""

#: Nodes the walker descends into looking for statements, but which are not statements.
_TRANSPARENT = frozenset({"declaration_list", "translation_unit"})


class _Context:
    """Mutable state threaded through the descent.

    Attributes:
        edges: Accumulated control-flow edges.
        breaks: Stack of nodes a ``break`` should flow to; empty outside a loop/switch.
        continues: Stack of nodes a ``continue`` should flow to.
        labels: Label name to the node a ``goto`` for it should reach.
        pending_gotos: Gotos seen before their label, resolved at the end.
    """

    def __init__(self) -> None:
        self.edges: list[tuple[int, int]] = []
        self.breaks: list[int | None] = []
        self.continues: list[int | None] = []
        self.labels: dict[str, int] = {}
        self.pending_gotos: list[tuple[int, str]] = []

    def link(self, source: int | None, target: int | None) -> None:
        """Record an edge, ignoring the "flows off the end" case.

        Args:
            source: Origin node id, or None when there is no live predecessor.
            target: Destination node id, or None when control leaves the function.
        """
        if source is not None and target is not None and source != target:
            self.edges.append((source, target))


def _named_children(node: Node) -> list[Node]:
    """Return the named children of a node, skipping punctuation."""
    return [child for child in node.children if child.is_named]


def _statements(node: Node) -> list[Node]:
    """Return the statements directly inside a block.

    Args:
        node: A ``compound_statement`` or similar container.

    Returns:
        Its named children, which for a block are its statements.
    """
    return _named_children(node)


class _Builder:
    """Recursive-descent control-flow construction over one function body."""

    def __init__(self, index: dict[int, int]) -> None:
        self.index = index
        self.ctx = _Context()

    def nid(self, node: Node) -> int:
        """Map a tree-sitter node to its graph node id.

        Args:
            node: Any node from the transcribed tree.

        Returns:
            The corresponding graph node id.

        Raises:
            GraphBuildError: If the node was never transcribed, which would mean the
                node set and the tree have diverged.
        """
        node_id = self.index.get(node.id)
        if node_id is None:
            raise GraphBuildError(
                "control-flow node is absent from the transcribed node set",
                node_type=node.type,
                start_byte=node.start_byte,
            )
        return node_id

    def run(self, root: Node) -> list[tuple[int, int]]:
        """Build the edges for a whole translation unit.

        Args:
            root: The parse tree root.

        Returns:
            The control-flow edges as ``(source, target)`` pairs.
        """
        for function in self._functions(root):
            body = function.child_by_field_name("body")
            if body is None:
                continue
            entry = self.nid(function)
            self.ctx.labels.clear()
            self.ctx.pending_gotos.clear()
            first = self._sequence(_statements(body), None, depth=0)
            # The function definition node is the entry: it flows to the first statement.
            self.ctx.link(entry, first)
            self._resolve_gotos()
        return self.ctx.edges

    def _functions(self, root: Node) -> list[Node]:
        """Find every function body in the tree, including nested ones."""
        found: list[Node] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == "function_definition":
                found.append(node)
            stack.extend(node.children)
        return found

    def _resolve_gotos(self) -> None:
        """Attach gotos to their labels once every label in the function is known.

        A ``goto`` to a label that does not exist in this function is left dangling
        rather than guessed at: Big-Vul functions are fragments, so the target may
        genuinely have been cut away.
        """
        for source, label in self.ctx.pending_gotos:
            target = self.ctx.labels.get(label)
            if target is not None:
                self.ctx.link(source, target)

    # -- statement dispatch ----------------------------------------------------------

    def _sequence(self, statements: list[Node], following: int | None, *, depth: int) -> int | None:
        """Chain a list of statements, returning the entry of the first.

        Args:
            statements: Statements in source order.
            following: Node the last statement flows to, or None.
            depth: Current nesting depth.

        Returns:
            Entry node of the sequence, or ``following`` when it is empty.
        """
        entry: int | None = following
        # Walk backwards: each statement's successor is the entry of the next one.
        for statement in reversed(statements):
            entry = self._statement(statement, entry, depth=depth) or entry
        return entry

    def _statement(self, node: Node, following: int | None, *, depth: int) -> int | None:
        """Build edges for one statement and return its entry node.

        Args:
            node: The statement.
            following: Node control reaches after it, or None.
            depth: Current nesting depth.

        Returns:
            The statement's entry node, or None if it contributes no control flow.

        Raises:
            GraphBuildError: If statement nesting exceeds :data:`MAX_NESTING_DEPTH`.
        """
        if depth > MAX_NESTING_DEPTH:
            raise GraphBuildError(
                "statement nesting exceeds the control-flow builder's limit",
                node_type=node.type,
                depth=depth,
                limit=MAX_NESTING_DEPTH,
            )

        handler = getattr(self, f"_on_{node.type}", None)
        if handler is not None:
            result: int | None = handler(node, following, depth + 1)
            return result
        if node.type in _TRANSPARENT:
            return self._sequence(_statements(node), following, depth=depth + 1)
        if not node.is_named:
            return None
        # Any other named statement — expression_statement, declaration, an expression
        # used as a statement — is a straight-line node.
        entry = self.nid(node)
        self.ctx.link(entry, following)
        return entry

    # -- handlers, one per control-flow form -----------------------------------------

    def _on_compound_statement(self, node: Node, following: int | None, depth: int) -> int | None:
        return self._sequence(_statements(node), following, depth=depth)

    def _on_if_statement(self, node: Node, following: int | None, depth: int) -> int:
        entry = self.nid(node)
        consequence = node.child_by_field_name("consequence")
        alternative = node.child_by_field_name("alternative")

        then_entry = (
            self._statement(consequence, following, depth=depth) if consequence else following
        )
        self.ctx.link(entry, then_entry if then_entry is not None else following)

        if alternative is None:
            # No else: the false branch falls through.
            self.ctx.link(entry, following)
            return entry

        # else_clause wraps the else body, which is itself a statement — an `else if`
        # is just an if_statement in that position.
        body = _named_children(alternative)
        else_entry = self._sequence(body, following, depth=depth)
        self.ctx.link(entry, else_entry if else_entry is not None else following)
        return entry

    def _on_while_statement(self, node: Node, following: int | None, depth: int) -> int:
        entry = self.nid(node)
        body = node.child_by_field_name("body")

        self.ctx.breaks.append(following)
        self.ctx.continues.append(entry)
        # The body flows back to the condition, which is the loop node itself.
        body_entry = self._statement(body, entry, depth=depth) if body else None
        self.ctx.continues.pop()
        self.ctx.breaks.pop()

        self.ctx.link(entry, body_entry if body_entry is not None else entry)
        self.ctx.link(entry, following)
        return entry

    def _on_for_statement(self, node: Node, following: int | None, depth: int) -> int:
        # The whole header — initializer, condition, update — is one node. Splitting it
        # would add nodes the AST does not have, breaking the shared node set.
        entry = self.nid(node)
        body = node.child_by_field_name("body")

        self.ctx.breaks.append(following)
        self.ctx.continues.append(entry)
        body_entry = self._statement(body, entry, depth=depth) if body else None
        self.ctx.continues.pop()
        self.ctx.breaks.pop()

        self.ctx.link(entry, body_entry if body_entry is not None else entry)
        self.ctx.link(entry, following)
        return entry

    def _on_do_statement(self, node: Node, following: int | None, depth: int) -> int | None:
        # The body runs before the test, so the loop's entry is the body's entry.
        condition_node = self.nid(node)
        body = node.child_by_field_name("body")

        self.ctx.breaks.append(following)
        self.ctx.continues.append(condition_node)
        body_entry = self._statement(body, condition_node, depth=depth) if body else None
        self.ctx.continues.pop()
        self.ctx.breaks.pop()

        target = body_entry if body_entry is not None else condition_node
        self.ctx.link(condition_node, target)
        self.ctx.link(condition_node, following)
        return target

    def _on_switch_statement(self, node: Node, following: int | None, depth: int) -> int:
        entry = self.nid(node)
        body = node.child_by_field_name("body")
        if body is None:
            self.ctx.link(entry, following)
            return entry

        cases = [child for child in _statements(body) if child.type == "case_statement"]
        self.ctx.breaks.append(following)
        # Fallthrough: a case with no break flows into the next one, so build backwards.
        next_entry: int | None = following
        entries: list[int | None] = []
        for case in reversed(cases):
            next_entry = self._on_case_statement(case, next_entry, depth)
            entries.append(next_entry)
        self.ctx.breaks.pop()

        for case_entry in entries:
            self.ctx.link(entry, case_entry)
        if not any(case.child_by_field_name("value") is None for case in cases):
            # No `default`, so the switch can also fall past every case.
            self.ctx.link(entry, following)
        return entry

    def _on_case_statement(self, node: Node, following: int | None, depth: int) -> int:
        entry = self.nid(node)
        # The label's value, when present, is the first named child and not a statement.
        value = node.child_by_field_name("value")
        body = [child for child in _named_children(node) if child != value]
        body_entry = self._sequence(body, following, depth=depth)
        self.ctx.link(entry, body_entry if body_entry is not None else following)
        return entry

    def _on_labeled_statement(self, node: Node, following: int | None, depth: int) -> int:
        entry = self.nid(node)
        label = node.child_by_field_name("label")
        if label is not None and label.text is not None:
            self.ctx.labels[label.text.decode("utf-8", errors="replace")] = entry
        body = [child for child in _named_children(node) if child != label]
        body_entry = self._sequence(body, following, depth=depth)
        self.ctx.link(entry, body_entry if body_entry is not None else following)
        return entry

    def _on_goto_statement(self, node: Node, following: int | None, depth: int) -> int:
        del following, depth  # a goto never falls through
        entry = self.nid(node)
        label = node.child_by_field_name("label")
        if label is not None and label.text is not None:
            self.ctx.pending_gotos.append((entry, label.text.decode("utf-8", errors="replace")))
        return entry

    def _on_return_statement(self, node: Node, following: int | None, depth: int) -> int:
        del following, depth  # control leaves the function; absence of an edge says so
        return self.nid(node)

    def _on_break_statement(self, node: Node, following: int | None, depth: int) -> int:
        del following, depth
        entry = self.nid(node)
        if self.ctx.breaks:
            self.ctx.link(entry, self.ctx.breaks[-1])
        return entry

    def _on_continue_statement(self, node: Node, following: int | None, depth: int) -> int:
        del following, depth
        entry = self.nid(node)
        if self.ctx.continues:
            self.ctx.link(entry, self.ctx.continues[-1])
        return entry


def build_cfg_edges(result: ParseResult, index: dict[int, int]) -> tuple[GraphEdge, ...]:
    """Build the control-flow edges for a parsed unit.

    Args:
        result: A parsed unit.
        index: Tree-sitter node identity to graph node id, from
            :func:`~vulnloc.representation.graphs.ast.transcribe`.

    Returns:
        The ``cfg`` edges, deduplicated and in a deterministic order.

    Raises:
        GraphBuildError: If a control-flow node is missing from the node set, or if
            statement nesting exceeds the builder's limit.
    """
    try:
        pairs = _Builder(index).run(result.root)
    except RecursionError as error:
        # The depth guard should fire first. If it ever does not — a future construct
        # costing more frames per level — fail as ourselves rather than as an
        # interpreter limit, so the unit is identifiable in the corpus report.
        raise GraphBuildError(
            "control-flow construction exhausted the interpreter stack",
            unit_id=result.unit_id,
            limit=MAX_NESTING_DEPTH,
        ) from error
    # A join point can be reached twice by the same predecessor (an if whose branches
    # both fall through, say). Parallel edges carry no extra information to a GNN.
    unique = sorted(set(pairs))
    return tuple(
        GraphEdge(source=source, target=target, edge_type="cfg") for source, target in unique
    )


def terminal_nodes(graph: CodeGraph) -> frozenset[int]:
    """Return the statements from which control leaves the function.

    There is no synthetic exit node, so "reaches the exit" is "has no ``cfg``
    successor". Useful for tests and for the eventual interprocedural work.

    Args:
        graph: A graph carrying ``cfg`` edges.

    Returns:
        Node ids with no outgoing control-flow edge, among those that have an incoming
        one or are a ``return``.
    """
    outgoing = {edge.source for edge in graph.edges if edge.edge_type == "cfg"}
    incoming = {edge.target for edge in graph.edges if edge.edge_type == "cfg"}
    reachable = incoming | outgoing
    return frozenset(
        node.node_id
        for node in graph.nodes
        if node.node_id in reachable and node.node_id not in outgoing
    )


def build_cfg_graph(result: ParseResult) -> CodeGraph:
    """Build the control-flow graph for a parsed unit.

    Args:
        result: A parsed unit.

    Returns:
        The CFG, over the AST's node set, carrying only ``cfg`` edges.

    Raises:
        GraphBuildError: If the resulting graph violates a structural invariant.
    """
    parts = transcribe(result)
    graph = CodeGraph(
        unit_id=result.unit_id,
        graph_type=GRAPH_TYPE,
        nodes=parts.nodes,
        edges=build_cfg_edges(result, parts.node_ids),
        n_lines=parts.n_lines,
    )
    graph.validate()
    return graph


@register_graph(GRAPH_TYPE)
class CfgGraphBuilder:
    """Registry entry for the control-flow graph."""

    graph_type = GRAPH_TYPE

    def build(self, result: ParseResult) -> CodeGraph:
        """Build the CFG for one parsed unit.

        Args:
            result: A parsed unit.

        Returns:
            The control-flow graph.
        """
        return build_cfg_graph(result)
