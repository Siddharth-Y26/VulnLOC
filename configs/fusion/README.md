# configs/fusion/

The fusion strategies (Phase 5) — the project's core novelty claim and the group
experiment **A1** varies. `default.yaml` carries everything that is *not* the fusion
mechanism; each arm inherits it and changes `name` and `kind` only. Each resolves to a
`FusionModule` in `src/vulnloc/models/fusion/`.

| arm | what it is |
|---|---|
| `cross_attention` | A1's treatment arm — bidirectional MHA over the shared line axis |
| `concat` | A1's control arm — the same module with the attention blocks deleted |
| `sum` | the parameter-free floor |
| `gated` | a learned per-line mixture, with no cross-line reasoning |
| `llm_only` / `gnn_only` | the single-branch ablations (`Phases.md`'s `none`, split per `Architecture.md` §4.4) |

`fusion=cross_attention` against `fusion=concat` being a **flag flip and nothing else** is
a Phase 5 exit criterion. `tests/unit/test_configs.py` asserts the comparator configs
differ only in `name` and `kind`; `tests/unit/test_fusion.py` asserts the arms share the
projections, the masking and the output width. See ADR-008 for why each of those is held
fixed.

## This group is NOT in `config.yaml`'s `defaults`, and that is deliberate

Select it explicitly: `+fusion=cross_attention`, not `fusion=cross_attention`.

Adding a group to `defaults` changes the resolved config — and therefore the
`config_hash` — of **every** run, including the three semantic seeds and the three
structural seeds that closed Phases 3 and 4. `scripts/report_seeds.py` uses that hash to
decide whether two runs are replicates, so it would make the baselines A1 is measured
against incomparable *with each other*. See `Memory.md` §7 #38 and `configs/gnn/README.md`,
which records the same trap one phase earlier.

## `args` is nested on purpose

Everything under `args` is forwarded to the strategy's constructor **verbatim and
unfiltered**, so a key the chosen arm does not accept is a hard failure. That is why
`heads` and `n_layers` appear in `cross_attention.yaml` alone: a `heads` setting that
reached `concat` and quietly did nothing would be `Memory.md` §7 #22 sitting inside the
comparison A1 rests on.
