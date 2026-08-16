"""Data-flow graph — reaching definitions over the CFG (ADR-002 §D2.2, third).

Edges run **definition → use**: the identifier node that last wrote a variable, to the
identifier node that reads it. Both endpoints are real AST identifier nodes, so a
data-flow edge is always attributable to a line, which is what makes this usable for
line-level explanation and for the ``labels_v2`` propagation ADR-002 §D2.4 defers.

The analysis is a textbook forward may-reach fixpoint over the control-flow graph:
``IN[n]`` is the union of predecessors' ``OUT``; ``OUT[n] = (IN[n] - kill[n]) | gen[n]``.
Uses in a statement resolve against ``IN``, so ``x = x + 1`` reads the *incoming*
definition, which is what it should do.

**What this deliberately does not model**, so nobody later mistakes its silence for a
guarantee:

* **Aliasing.** Analysis is by *name*. ``p = &x; *p = 1;`` does not define ``x``.
  Alias analysis for C is a research problem in itself and ADR-002 §D2.3 chose in-house
  passes precisely to keep the line mapping ours — not to solve pointer analysis.
* **Scoping.** One flat namespace per function, so two different ``i``s in sibling
  blocks are treated as one variable. This over-approximates: it can add an edge that
  does not exist, never remove one that does.
* **Fields and array elements.** ``s.a`` and ``s.b`` are both "``s``"; ``a[0]`` and
  ``a[1]`` are both "``a``".
* **Stores through a pointer or subscript** — ``*p = v``, ``a[i] = v`` — are recorded as
  a **weak update**: the base name is both used and defined, and the old definition is
  *not* killed. Killing it would claim knowledge the analysis does not have.
* **Calls.** ``f(&x)`` does not define ``x``. There are no interprocedural edges; that
  is what the unused ``call`` edge type is reserved for.

Every one of these makes the graph *miss* edges or *add* conservative ones. None of them
can move a node to the wrong line.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

from vulnloc.representation.graphs.ast import transcribe
from vulnloc.representation.graphs.base import CodeGraph, GraphEdge
from vulnloc.representation.graphs.cfg import build_cfg_edges
from vulnloc.representation.registry import register_graph
from vulnloc.utils.exceptions import GraphBuildError

if TYPE_CHECKING:
    from tree_sitter import Node

    from vulnloc.representation.parser import ParseResult

__all__ = ["DfgGraphBuilder", "build_dfg_edges", "build_dfg_graph"]

GRAPH_TYPE = "dfg"

FIXPOINT_ROUNDS_PER_NODE = 100
FIXPOINT_ROUNDS_FLOOR = 10_000
"""Iteration budget for the fixpoint, **scaled to the function**.

A fixed budget was tried first and was wrong in both directions. The lattice is finite
and the transfer functions monotone, so the analysis always terminates on its own; the
budget exists only to catch a future edit that breaks monotonicity. But a flat 10,000
silently truncated 25 of 400 real functions and halved the corpus's data-flow edges —
one unit lost 1,093 of 13,534. Convergence measured on real code needs about **14
iterations per statement** (worst observed: 14,671 for a 1,041-statement function), so
the budget scales with the graph and keeps ~7x headroom over that.

Exhausting it raises. A partially converged reaching-definitions result is not a
smaller answer, it is a wrong one, and returning it quietly is exactly the silent
failure ``Rules.md`` §4.6 forbids."""

#: Expression forms whose "target" is a location, not a name — a weak update.
_INDIRECT_TARGETS = frozenset(
    {"subscript_expression", "field_expression", "pointer_expression", "parenthesized_expression"}
)

#: Declarator forms that wrap the name being declared. Missing these is not a subtle
#: loss: without `pointer_declarator`, every `int *p` parameter goes undefined and its
#: uses have no reaching definition at all.
_DECLARATOR_WRAPPERS = frozenset(
    {
        "array_declarator",
        "function_declarator",
        "init_declarator",
        "parenthesized_declarator",
        "pointer_declarator",
    }
)

#: Fields to follow when walking down to the name, in priority order.
_BASE_FIELDS = ("argument", "declarator", "value")


def _name(node: Node) -> str | None:
    """Return a node's source text, when it has any."""
    if node.text is None:
        return None
    return node.text.decode("utf-8", errors="replace")


def _base_identifier(node: Node) -> Node | None:
    """Find the identifier a store actually lands on.

    ``a[i].f`` and ``*(p + 1)`` both resolve to their base name. Returns None when the
    target has no single base — a store through a call's result, say.

    Args:
        node: The left-hand side of an assignment.

    Returns:
        The base identifier node, or None.
    """
    current: Node | None = node
    seen = 0
    while current is not None and seen < 64:
        seen += 1
        if current.type == "identifier":
            return current
        if current.type not in _INDIRECT_TARGETS and current.type not in _DECLARATOR_WRAPPERS:
            return None
        nxt: Node | None = None
        for field in _BASE_FIELDS:
            nxt = current.child_by_field_name(field)
            if nxt is not None:
                break
        if nxt is None:
            named = [child for child in current.children if child.is_named]
            nxt = named[0] if named else None
        current = nxt
    return None


class _Facts:
    """Definitions and uses owned by one control-flow node.

    Attributes:
        defs: ``(variable, defining identifier node id)``.
        uses: ``(variable, using identifier node id)``.
        killed: Variables this node definitely overwrites. A weak update is absent
            here on purpose.
    """

    __slots__ = ("defs", "killed", "uses")

    def __init__(self) -> None:
        self.defs: list[tuple[str, int]] = []
        self.uses: list[tuple[str, int]] = []
        self.killed: set[str] = set()


class _Extractor:
    """Collects the definitions and uses each control-flow node owns."""

    def __init__(self, index: dict[int, int], cfg_nodes: set[int]) -> None:
        self.index = index
        self.cfg_nodes = cfg_nodes

    def owned(self, node: Node) -> list[Node]:
        """Return the nodes a statement owns.

        Descent stops at any node that is itself a control-flow node, so an ``if``
        owns its condition but not its body — the body's statements own themselves.

        Args:
            node: The statement.

        Returns:
            The statement's own subtree, root included.
        """
        collected: list[Node] = [node]
        stack: list[Node] = list(node.children)
        while stack:
            current = stack.pop()
            if current.id in self.cfg_nodes:
                continue
            collected.append(current)
            stack.extend(current.children)
        return collected

    def facts(self, node: Node) -> _Facts:
        """Classify a statement's identifiers into definitions and uses.

        Args:
            node: The statement.

        Returns:
            The facts for this node.
        """
        facts = _Facts()
        region = self.owned(node)
        defined_nodes: set[int] = set()

        for current in region:
            if current.type == "assignment_expression":
                self._assignment(current, facts, defined_nodes)
            elif current.type == "init_declarator":
                self._declarator(current, facts, defined_nodes, strong=True)
            elif current.type == "update_expression":
                # i++ both reads and writes.
                target = current.child_by_field_name("argument")
                base = _base_identifier(target) if target is not None else None
                if base is not None:
                    self._define(base, facts, defined_nodes, strong=base.type == "identifier")
            elif current.type == "parameter_declaration":
                declarator = current.child_by_field_name("declarator")
                if declarator is not None:
                    base = _base_identifier(declarator)
                    if base is not None:
                        self._define(base, facts, defined_nodes, strong=True)
            elif current.type == "declaration":
                # `int a[10];` and `char *s;` declare a name without initialising it.
                # That still counts as a definition — an uninitialised one, which is
                # exactly the state a use-of-uninitialised flaw reads.
                type_node = current.child_by_field_name("type")
                for child in current.children:
                    if not child.is_named or child == type_node or child.type == "init_declarator":
                        continue
                    base = _base_identifier(child)
                    if base is not None:
                        self._define(base, facts, defined_nodes, strong=True)

        # Everything else that names something is a use. A node can be both: `x += 1`
        # and `a[i] = v` read what they write.
        for current in region:
            if current.type != "identifier":
                continue
            name = _name(current)
            if name is None:
                continue
            if current.id in defined_nodes and self._is_strong_target(current):
                continue
            facts.uses.append((name, self.index[current.id]))
        return facts

    def _is_strong_target(self, node: Node) -> bool:
        """Whether this identifier is a pure write, reading nothing of its own value.

        Such an identifier is a definition only, never a use. ``x = 1`` and ``int a[10]``
        qualify; ``x += 1``, ``i++`` and ``a[i] = v`` do not, because each reads the name
        it writes.

        Args:
            node: An identifier node.

        Returns:
            True when the identifier is a pure write.
        """
        parent = node.parent
        if parent is None:
            return False
        if parent.type == "assignment_expression":
            operator = parent.child_by_field_name("operator")
            plain = operator is not None and _name(operator) == "="
            return plain and parent.child_by_field_name("left") == node
        # Climb the declarator wrappers: in `int *p` the identifier's parent is a
        # pointer_declarator, not the declaration itself.
        current: Node | None = parent
        depth = 0
        while current is not None and depth < 64:
            depth += 1
            if current.type in {"declaration", "parameter_declaration", "init_declarator"}:
                return True
            if current.type not in _DECLARATOR_WRAPPERS:
                return False
            current = current.parent
        return False

    def _assignment(self, node: Node, facts: _Facts, defined: set[int]) -> None:
        target = node.child_by_field_name("left")
        if target is None:
            return
        base = _base_identifier(target)
        if base is None:
            return
        # Only a plain `name = ...` is a strong update. `*p = v`, `a[i] = v` and every
        # compound operator leave the previous definition reachable.
        operator = node.child_by_field_name("operator")
        plain = operator is not None and _name(operator) == "="
        self._define(base, facts, defined, strong=plain and target.type == "identifier")

    def _declarator(self, node: Node, facts: _Facts, defined: set[int], *, strong: bool) -> None:
        declarator = node.child_by_field_name("declarator")
        if declarator is None:
            return
        base = _base_identifier(declarator)
        if base is not None:
            self._define(base, facts, defined, strong=strong)

    def _define(self, node: Node, facts: _Facts, defined: set[int], *, strong: bool) -> None:
        name = _name(node)
        if name is None:
            return
        facts.defs.append((name, self.index[node.id]))
        defined.add(node.id)
        if strong:
            facts.killed.add(name)


def build_dfg_edges(
    result: ParseResult,
    index: dict[int, int],
    cfg_edges: tuple[GraphEdge, ...],
) -> tuple[GraphEdge, ...]:
    """Compute definition-to-use edges by reaching definitions.

    Args:
        result: A parsed unit.
        index: Tree-sitter node identity to graph node id.
        cfg_edges: The unit's control-flow edges, which the analysis walks.

    Returns:
        The ``dfg`` edges, deduplicated and in a deterministic order.
    """
    reverse = {node_id: ts_id for ts_id, node_id in index.items()}
    cfg_node_ids = {edge.source for edge in cfg_edges} | {edge.target for edge in cfg_edges}
    cfg_ts_ids = {reverse[node_id] for node_id in cfg_node_ids if node_id in reverse}

    ts_by_id: dict[int, Node] = {}
    stack = [result.root]
    while stack:
        current = stack.pop()
        if current.id in cfg_ts_ids:
            ts_by_id[current.id] = current
        stack.extend(current.children)

    extractor = _Extractor(index, cfg_ts_ids)
    facts = {
        index[ts_id]: extractor.facts(node) for ts_id, node in ts_by_id.items() if ts_id in index
    }
    if not facts:
        return ()

    predecessors: dict[int, list[int]] = defaultdict(list)
    successors: dict[int, list[int]] = defaultdict(list)
    for edge in cfg_edges:
        predecessors[edge.target].append(edge.source)
        successors[edge.source].append(edge.target)

    # Forward may-reach fixpoint over sets of (variable, defining node). The queue
    # carries its own membership set: `x in list` is linear, and on a 1,000-statement
    # function that alone turned the pass quadratic.
    out: dict[int, set[tuple[str, int]]] = {node_id: set() for node_id in facts}
    worklist = deque(sorted(facts))
    queued = set(facts)
    budget = FIXPOINT_ROUNDS_PER_NODE * len(facts) + FIXPOINT_ROUNDS_FLOOR
    rounds = 0
    while worklist:
        rounds += 1
        if rounds > budget:
            raise GraphBuildError(
                "reaching-definitions did not converge within its iteration budget",
                unit_id=result.unit_id,
                statements=len(facts),
                budget=budget,
            )
        node_id = worklist.popleft()
        queued.discard(node_id)
        incoming: set[tuple[str, int]] = set()
        for predecessor in predecessors.get(node_id, ()):
            incoming |= out.get(predecessor, set())
        node_facts = facts[node_id]
        survived = {fact for fact in incoming if fact[0] not in node_facts.killed}
        new_out = survived | set(node_facts.defs)
        if new_out != out[node_id]:
            out[node_id] = new_out
            for successor in successors.get(node_id, ()):
                if successor in facts and successor not in queued:
                    worklist.append(successor)
                    queued.add(successor)

    pairs: set[tuple[int, int]] = set()
    for node_id, node_facts in facts.items():
        incoming = set()
        for predecessor in predecessors.get(node_id, ()):
            incoming |= out.get(predecessor, set())
        for variable, use_node in node_facts.uses:
            for def_variable, def_node in incoming:
                if def_variable == variable and def_node != use_node:
                    pairs.add((def_node, use_node))

    return tuple(
        GraphEdge(source=source, target=target, edge_type="dfg") for source, target in sorted(pairs)
    )


def build_dfg_graph(result: ParseResult) -> CodeGraph:
    """Build the data-flow graph for a parsed unit.

    Args:
        result: A parsed unit.

    Returns:
        The DFG, over the AST's node set, carrying only ``dfg`` edges.

    Raises:
        GraphBuildError: If the resulting graph violates a structural invariant.
    """
    parts = transcribe(result)
    cfg_edges = build_cfg_edges(result, parts.node_ids)
    graph = CodeGraph(
        unit_id=result.unit_id,
        graph_type=GRAPH_TYPE,
        nodes=parts.nodes,
        edges=build_dfg_edges(result, parts.node_ids, cfg_edges),
        n_lines=parts.n_lines,
    )
    graph.validate()
    return graph


@register_graph(GRAPH_TYPE)
class DfgGraphBuilder:
    """Registry entry for the data-flow graph."""

    graph_type = GRAPH_TYPE

    def build(self, result: ParseResult) -> CodeGraph:
        """Build the DFG for one parsed unit.

        Args:
            result: A parsed unit.

        Returns:
            The data-flow graph.
        """
        return build_dfg_graph(result)
