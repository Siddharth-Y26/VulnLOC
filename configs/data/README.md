# configs/data/

Empty-but-composable stub for Phase 1 (Data Layer). Dataset choice is
blocked on ADR-001 (`PRD.md` Q1, `Memory.md` §5). Do not add dataset
configs here until ADR-001 is accepted — see `Rules.md` §10.

Expected shape (per `Architecture.md` §5): one YAML per registered
dataset (e.g. `dataset_a.yaml`), each resolving to a `@register_dataset`
loader in `src/vulnloc/data/loaders/`.
