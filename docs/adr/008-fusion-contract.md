# ADR-008 — What Is Held Fixed So That A1 Has an Answer

- **Status:** **Accepted** by the project owner, 2026-09-15, together with ADR-009 and the
  Phase 3 and Phase 4 gate signatures. D8.7's open question was settled the same day — see
  its resolution note. Proposed 2026-09-14.
- **Date:** 2026-09-14
- **Phase:** 5 — Cross-Attention Fusion
- **Depends on:** ADR-002 (graph representation), ADR-005 (grammar selection),
  ADR-006 (whitespace normalization), ADR-007 (edge handling — **D7.4 is still open and
  D8.7 below is blocked on it**)
- **Supersedes:** nothing.

---

## Context

Experiment **A1** — does bidirectional cross-attention beat concatenation? — is the
question this project exists to answer, and it is the only kind of question that can be
answered wrongly *without anything failing*. Both arms will train. Both will converge.
Both will produce a Top-5 number. Whichever is higher will look like the answer.

So the work of Phase 5 is not mainly the attention module. It is deciding what is held
fixed across the arms, and making the things that must be equal equal **by construction**
rather than by intention. Every decision here is one of those.

The project has already been burned twice by the softer version of this. `Memory.md`
§7 #39 and §7 #40: a config flag that did not exist when an earlier run was made split a
comparison group in two, and two CPU probes at `train.limit=600` were averaged in with
full-corpus runs. Both were live bugs on the same day, both in reports, both invisible.

Phases 3 and 4 closed with three-seed baselines on one shared line axis and one shared
training loop, which is exactly what A1 needs and what did not exist before. This ADR is
about not squandering that.

---

## Decision

### D8.1 — The fused line mask is the **union** of the two branches' coverage

A line is scoreable if *either* branch has evidence for it. This is not a detail: 7.84% of
flagged lines corpus-wide carry no graph node at all (`Memory.md` §7 #17), so the
structural branch cannot score them and a fused model can.

**It is also not the cross-attention result, and must never be reported as one.** `concat`,
`sum` and `gated` have the identical advantage, because the union is a property of the
fused *scaffold*. The arm that isolates it is `gnn_only`, which is why that arm exists.

`fused_line_coverage()` reports the size of the effect per batch —
`flagged_semantic_only`, `flagged_structural_only`, `recovered_by_union` — so it is a
measured quantity rather than an argument.

### D8.2 — The branch projections live in the shared base class, not in the winner

The branches arrive at different widths (768 semantic, 256 structural), so something must
project them into a common space. If that projection lived inside `CrossAttentionFusion`,
`concat` would be "no projection" against "attention **plus** a projection", and the extra
parameters would sit inside the arm A1 hopes will win.

`FusionModule` therefore owns the projections, the re-masking, the shape validation, the
output width and the line axis. A strategy implements `_combine` and nothing else. The
parameter difference between `cross_attention` and `concat` is exactly the multi-head
attention and the two layer norms around it.

**Corollary, enforced in `build_fusion`:** constructor keyword arguments are forwarded
**unfiltered**, so `fusion.args.heads` reaching `concat` raises instead of doing nothing.
`heads` and `n_layers` therefore live in `configs/fusion/cross_attention.yaml` alone. A
knob that exists but is not consumed is §7 #22 sitting inside A1's comparison.

### D8.3 — A line a branch never saw is exactly zero, never a bias vector

`scatter_mean_lines` returns zeros where nothing landed, but `Linear(0)` is the bias — a
learnable constant that the head could read as "the semantic branch is silent here". On a
corpus where branch coverage and label noise are correlated with function size, that is a
feature correlated with the label, arriving through the back door. The projections are
re-masked immediately after they are applied.

### D8.4 — Two learning rates, fixed across every arm

Phase 3 fine-tuned a 124M-parameter pretrained transformer at 2e-5; Phase 4 trained a
~10^5-parameter GNN from scratch at 1e-3. A single rate across the fused model has to be
wrong for one of them: 2e-5 barely moves the GNN, 1e-3 destroys CodeBERT.

`train.encoder_lr` applies to the pretrained encoder, `train.lr` to the GNN, the fusion and
the head. `run_training` gained an **optional** `param_groups` argument; passing `None`
reproduces the previous behaviour exactly, so no Phase 3 or Phase 4 run changes meaning.

This choice affects how good *all* the arms are, not which one wins — but it is held
identical across them, so A1 stays controlled. It is not tuned per arm, and tuning it per
arm would silently turn A1 into a hyperparameter search.

### D8.5 — The two branches are joined on `unit_id`, and the population is reported

`LineLabelledDataset` keeps every line-labelled unit; `GraphLineDataset` drops any whose
graph will not build. Today that difference is empty — every logged Phase 4 run reports
`graph_build_failures=0` at 5,457 train / 488 val units — which is **precisely** why a
positional join would have passed every test and then, on the first partition where it did
not hold, paired one function's tokens with another's control flow, trained, and converged.

`join_branches` intersects on `unit_id`, refuses duplicate ids, and refuses a pair whose
two halves disagree about `line_labels`, `n_lines` or `target`. `collate_fused` re-checks
the same agreement at batch level. `units_dropped_semantic_only` and
`units_dropped_structural_only` are logged as run parameters, because **a fused arm
training on a different population from the baselines it is compared against is a confound,
not a detail**, and it must not be discoverable only at write-up time.

### D8.6 — `Phases.md`'s `none` comparator is built as `llm_only` **and** `gnn_only`

`Phases.md` names four comparators — `concat`, `sum`, `gated`, `none`. `Architecture.md`
§4.4 names five, splitting `none` into `llm_only` and `gnn_only`. The split form is built,
because "no fusion" means something different depending on which branch survives.

These are **not** re-runs of the Phase 3 and Phase 4 baselines, which have their own models
and their own logged seeds. They run a single branch through the fused scaffold, which is
what separates "cross-attention helped" from "the extra projection and layer norm helped".
Their line mask is their own branch's coverage, never the union (see D8.1).

### D8.7 — Which graph the fused arm reads is **NOT decided here**

`gnn.builder` defaults to `ast`, which is what every Phase 4 number was measured under. If
Phase 5 fuses a **CPG** instead, ADR-007 D7.4 becomes live — `ignore` against `weighted` on
the same seeds — and `edge_handling=auto` will refuse the batch outright rather than
silently averaging three relations. That refusal is the correct behaviour and it is a
question for the project owner, not a default to pick here.

Likewise `gnn.use_subtokens` stays `false` by default even though `Memory.md` §1.15
measures it as the single largest improvement to the structural branch. Flipping it is a
deliberate act with a ledger row (`Memory.md` §6.3), not something Phase 5 should do on its
way past.

> **Resolved by the project owner, 2026-09-15.** A1 fuses the **AST** (`gnn.builder=ast`,
> the graph every Phase 4 number was measured on, so ADR-007 D7.4 stays moot) with
> **identifier subtokens on** (`gnn.use_subtokens=true`, the best Phase 4 structural
> configuration: Top-5 0.4302 ± 0.0372 against 0.3645 without, `Memory.md` §1.15). Both are
> set per run in `scripts/run_queue.py`'s A1 constants rather than flipped in
> `configs/gnn/default.yaml`, so every earlier run's resolved config is unchanged. Ledger
> row in `Memory.md` §4.

### D8.8 — The line axis is trimmed per batch, and that must change nothing

Cross-attention is quadratic in the line axis, which is configured at 512 while the corpus
mean is 31 lines per function (`Memory.md` §1.4). The fused model therefore runs the fusion
and the head on the largest line index the batch actually reaches — derived from the label
mask **and** from both branches' line vectors, so a feature can never land past the trim —
and pads the logits back to 512 before they leave, preserving the `[B, L]` contract the
loop, the loss and the metrics share with both baselines.

`fusion.trim_line_axis=false` is the unoptimized path, and a test asserts the two produce
identical logits for every arm. **An optimization that changes a number is not an
optimization**, and the only way to keep that true is to pin it.

---

## Consequences

**Good.**

- A1 differs between its arms in the fusion mechanism and in nothing else, enforced by a
  shared base class and by a config test that asserts the comparator configs differ only in
  `name` and `kind`.
- A negative A1 means what it says. The cross-attention residual is the *pre-attention*
  projected pair, so the module can degrade to roughly `sum` — an architecture that cannot
  fall back to its own baseline makes "it did not help" ambiguous.
- The two branches cannot silently describe different functions.
- Attention maps are a first-class, shaped, retrievable output rather than a debugging
  afterthought, which is what Phase 7 needs of them.

**Costs, stated rather than discovered.**

- The fused arm carries both encoders plus an attention that is quadratic in the active
  line axis. **Now measured** (`Memory.md` §1.17, runs `f96a0744…` / `c1a2fbab…`): it fits,
  at **92.4% of a 4 GiB card** on the worst training unit, with loss and gradients finite
  under AMP. Two things that measurement changed. First, **the batch budgets do not control
  peak memory** — every budget in the grid is dominated by the same oversized unit, because
  a unit over budget gets its own batch by design. Second, 1,922 MiB (46.9%) is static
  optimizer state before a single activation, so the lever for headroom is `llm.frozen` or
  a smaller optimizer, not `max_windows_per_batch`. The two arms cost within 8 MiB of each
  other, which is the fairness property D8.2 is for.
- `llm_only` and `gnn_only` hold the unused encoder in memory. That is the price of running
  the ablation inside the same scaffold, which is the point of the ablation.
- Building both branches per epoch costs both the tokenization pass and the graph pass.
  Phase 4 measured graphs at 3.23 ms/unit on train and 6.50 ms/unit on val; the semantic
  epoch dominates.

**Reversible via config, at no cost today:** `fusion.args.hidden_dim`,
`fusion.args.heads`, `fusion.args.n_layers`, `fusion.trim_line_axis`, `train.encoder_lr`,
`gnn.builder`, `gnn.use_subtokens`. **Irreversible the moment a seed is logged**, because
changing any of them changes `config_hash` and `scripts/report_seeds.py` will then
correctly refuse to aggregate across the change (§7 #38).

---

## Verification

93 tests, all passing; the suite is 791 passed / 2 skipped, with ruff, black, mypy and
import-linter clean.

**2026-09-15:** the overfit test passed on the cached, frozen path for `cross_attention`
(run `117df3a4…`, Top-1 1.000, loss 2.95 → 0.0085 over 150 epochs), `concat` (`aebd4407…`,
Top-1 1.000) and `sum` (`3b32a40d…`, Top-1 1.000) with subtokens off; all six arms were then re-queued with the A1 configuration
(subtokens on). See `Memory.md` §1.20.

**No metric was produced and no run was logged.** What was verified is plumbing, on real
`val` shards through `scripts/train_fused.py`'s own composition path:

- all six arms compose from the real config tree, build against the real node-type
  vocabulary artefact, and return the `[B, L]` batch contract both baselines produce;
- the join drops nothing on `val`, consistent with Phase 4's logged
  `graph_build_failures=0`;
- gradient reaches **both** branches after one backward pass, and reaches exactly one of
  them under `llm_only` and `gnn_only`;
- attention maps come back as `[B, H, L_active, L_active]`, rows normalised over the keys
  the source branch actually covered and zeroed for a unit with no keys at all;
- the trimmed active axis was two orders of magnitude smaller than the configured 512 on
  the batch tried, and produces identical logits to the untrimmed path.

The test that matters most is the one §7 #22 says should exist for every governing config
value: perturbing a branch's features *only where that branch saw nothing* must not change
the output — for every arm.
