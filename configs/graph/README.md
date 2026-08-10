# configs/graph/

Empty-but-composable stub for Phase 2 (Code Representation). Primary
graph representation is blocked on ADR-002 (`PRD.md` Q3, `Memory.md` §5).

Expected shape (per `Architecture.md` §5): `ast.yaml`, `cfg.yaml`,
`dfg.yaml`, `cpg.yaml`, each resolving to a `@register_graph` builder in
`src/vulnloc/representation/graphs/`.
