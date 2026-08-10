# configs/fusion/

Empty-but-composable stub for Phase 5 (Cross-Attention Fusion) — the
project's core novelty claim (`ResearchPlan.md` A1).

Expected shape (per `Architecture.md` §5): `cross_attention.yaml`,
`concat.yaml`, `gated.yaml`, `none.yaml`, each resolving to a
`FusionModule` in `src/vulnloc/models/fusion/`. All variants must share
the same interface so `fusion=cross_attention` vs `fusion=concat` is a
flag flip, not a code change.
