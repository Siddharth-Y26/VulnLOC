# ADR-007 — A Relation Is Allowed to Matter, or the Encoder Refuses to Run

- **Status:** **Proposed** — 2026-09-12. The *mechanism* is implemented and the
  *refusal* is default-on, because an edge-blind encoder scoring a CPG is a silent
  failure and shipping that default was not acceptable while the choice is open. What is
  **not** decided is which handling Phase 4 and Phase 5 report against. See D7.4.
- **Date:** 2026-09-12
- **Phase:** 4 — GNN Branch
- **Depends on:** ADR-002 (the CPG is a typed union of AST, CFG and DFG edges over one
  shared node set)
- **Supersedes:** nothing. No number has been produced under any edge handling, so
  nothing here invalidates a result.

---

## Context

ADR-002 fixes the CPG as a **typed union**: all four graph types share one node set, and
the CPG is the union of their edges, distinguished only by `edge_type`. `EDGE_TYPES` is
`("ast", "cfg", "dfg", "call")` and `to_pyg` has always emitted an `edge_type` tensor
alongside `edge_index`.

`StructuralEncoder` ignored it. GCN, GAT and GGNN were each called as
`layer(states, edge_index)`, so a CPG was message-passed as an **untyped** graph: "this
statement is syntactically nested inside that one" and "a value flows from this statement
to that one" were the same edge.

That is not a small modelling simplification. It is the difference between the structural
branch reading a program's dependence structure and reading a bag of adjacency. And it
fails the way this project has already been burned once:

> **§7 #22** — `graph.language` sat in the config unread for a whole phase. A knob that
> nothing consumes looks exactly like a knob that works.

`edge_type` was in exactly that state: computed at L1, carried through the batch,
consumed by nothing. Ablation **A3** selects *which* edge types are built into the graph,
so A3 without edge-type handling measures graph sparsity and calls it relation semantics.

Measured on 60 `val` units built as CPGs: **46,062 edges — `ast` 34,159, `cfg` 3,258,
`dfg` 8,645, `call` 0.** Three relations in wildly different proportions. Averaging them
is a modelling choice, and it was being made by omission.

---

## Decision

### D7.1 — `edge_handling` is a constructor argument with three values

`auto` (default), `ignore`, `weighted`. Config selects it, like everything else here.

### D7.2 — `weighted`: one learned scalar per relation, uniform across all three convolutions

A single parameter vector of length `len(EDGE_TYPES)`, softplus-constrained positive,
indexed by each edge's type and passed to the convolution — as `edge_weight` for GCN and
GGNN, as a one-dimensional `edge_attr` for GAT, which is the same learned number in the
shape each layer accepts.

**Uniformity is the whole point.** Phase 4 owes a comparison *between convolutions* and a
depth sweep at 2/4/6. If GCN got edge weights and GAT got a richer edge-attribute
projection, both comparisons would partly measure the edge mechanism. One mechanism, three
convolutions, as with the shared scaffold everything else in this module already uses.

**Softplus, not a free scalar.** A learned negative weight lets one relation subtract its
neighbours' messages, which is not what "this relation matters less" means and makes the
learned vector unreadable as relation importance. Constraining it positive keeps
`relation_weights()` interpretable, which Phase 7 will want.

### D7.3 — `auto` refuses a multi-relation graph rather than picking for you

`auto` resolves to `ignore` for a graph carrying one relation — an AST, which is the
current builder and every existing number — and **raises** for a graph carrying several.

This is the load-bearing part of the ADR. An AST-only project never notices edge handling;
the moment someone sets `graph.builder=cpg` the question becomes real, and the default
must not answer it silently. `ignore` remains available and is then an explicit statement
that relations should not differ, which is a defensible position and a different thing
from never having considered it.

### D7.4 — Which handling reports the headline numbers is **NOT decided here**

`weighted` is the cheapest mechanism that makes a relation matter. It is **not** the most
expressive one. The documented alternative is a **relational GCN** (Schlichtkrull et al.),
one weight matrix per relation, which PyG provides as `RGCNConv`.

| | `weighted` (implemented) | `relational` (not implemented) |
|---|---|---|
| Parameters per layer | `len(EDGE_TYPES)` scalars | ×`len(EDGE_TYPES)` weight matrices |
| Expressiveness | scales a relation's messages | transforms each relation differently |
| Uniform across GCN/GAT/GGNN | yes | no — it *replaces* the convolution |
| Cost on a 4 GiB card | negligible | unmeasured |

The second column's last two rows are why it is not the default: `relational` would break
the one-scaffold-three-convolutions design that makes Phase 4's A2 comparison clean, and
its memory cost has not been measured against §1.9's budget.

**TODO(decision):** run Phase 4's GNN-only baseline under `ignore` and under `weighted`
on the same seeds and compare, before choosing what Phase 5 fuses. If the gap is within
seed noise, `ignore` is the honest default and this ADR should say so. Do not adopt
`relational` without measuring it against the VRAM budget first.

### D7.5 — The parameter set does not depend on the setting

`edge_type_weight` is allocated under all three settings, so a checkpoint trained under
one loads under another. Making the parameter conditional would turn a config change into
a checkpoint incompatibility, discovered at load time, for no benefit.

---

## Consequences

**Good.** `edge_type` is consumed rather than carried. A CPG cannot be scored as an
untyped union by accident. A3 can measure relation semantics rather than sparsity.
`relation_weights()` gives Phase 7 a readable per-relation importance for free.

**Cost.** One more config knob, and a default that raises on input that used to run — but
only on input that was being silently mishandled.

**Unresolved, and stated rather than buried.** `call` edges are **0** in the measured CPG
sample: the CPG builder does not emit them, so that row of the table will never train and
its weight stays at its initial value. That is a property of the graph builders, not of
this ADR, but anyone reading `relation_weights()` needs to know the fourth number is
untrained rather than uninformative.

**Not claimed.** No result has been produced under any handling. This ADR builds the
choice; it does not make it.

---

## Verification

Real corpus data, 60 `val` units built as CPGs (`ast` 34,159 / `cfg` 3,258 / `dfg` 8,645
/ `call` 0 edges):

- `auto` refused the batch, naming the three relations it found;
- `weighted` ran under **all three** convolutions to `[7947, 64]` node states and onto a
  `[17, 512, 64]` line axis, all finite;
- initial weights are `softplus(0) = 0.6931` for every relation, so nothing is
  preferred before training.

Fifteen tests. The one that matters most asserts that moving a single relation's weight
changes the encoder's output, for each convolution — the test §7 #22 says should have
existed for `graph.language`: *if a config value governs a decision, something must fail
when it is ignored.*
