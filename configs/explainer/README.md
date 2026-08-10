# configs/explainer/

Empty-but-composable stub for Phase 7 (Explanation Module).

Expected shape (per `Architecture.md` §5): `template.yaml` (7a,
deterministic baseline — build first), `attention.yaml`,
`generative.yaml` (7b, only after 7a is measured), each resolving to an
`Explainer` in `src/vulnloc/explanation/`.
