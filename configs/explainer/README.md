# configs/explainer/

Phase 7 (Explanation Module). Selected with a leading `+` — `+explainer=template` — and
never added to `config.yaml`'s `defaults` (that would change every run's `config_hash`).

- `template.yaml` — **built, Phase 7a (2026-10-07).** The deterministic template +
  evidence explainer, `EXPLAINER_REGISTRY` key `template`.
- `generative.yaml` — **not built.** 7b is deferred in writing (`docs/Phases.md`): a
  local generator beside CodeBERT does not fit the 4 GiB card. It would have to pass the
  same `explanation/verifier.py` unchanged.

There is no separate `attention.yaml`: the attention maps are evidence the template
explainer reads (`explanation/evidence.py`), not a second explainer.
