# ADR-009 — Freezing the Semantic Encoder for A1, and Caching What It Produces

- **Status:** **Accepted** by the project owner, 2026-09-15 (closes **Q10**). Built and
  verified the same day — see Verification. **Two decisions were amended by measurement
  during implementation, not by preference: D9.5's "bitwise identical" and D9.6's fp16
  storage.** Both amendments are marked in place. Proposed 2026-09-14.
- **Date:** 2026-09-14
- **Phase:** 5 — Cross-Attention Fusion
- **Depends on:** ADR-006 (whitespace normalization — the cache is invalid without it),
  ADR-008 (what is held fixed across A1's arms; **D8.2 and D8.4 are the precedent this
  argument rests on**)
- **Supersedes:** nothing. ADR-008 §D8.4's learning-rate split stands unchanged.

---

## Context

**A1 as currently specified costs 60–80 GPU-hours on a 4 GiB laptop**, and the core
two-arm comparison alone costs 20–30. That is the estimate from measured parts: semantic
runs of 2.25 h / 3.9 h / 4.1 h (§4 ledger), a structural branch that is free by comparison,
and ADR-008's finding that the semantic epoch dominates the fused one. The estimate is
honest and the number is the problem.

It is worth being precise about *where* the time goes, because the obvious cuts are the
wrong ones. Truncation instead of windowing would destroy 38.1% of flagged lines (§7 #23).
Lowering the line cap costs `val` 17.07% of its flagged lines (§7 #26). Fewer seeds walks
straight back into §7 #37, the trap this project has already fallen into twice. None of
those are available.

What *is* available is in the project's own probe. §1.9 measured four encoder modes on the
4 GiB card:

| Mode | Throughput | Peak at bs=16 |
|---|---|---|
| `checkpoint` (AMP + gradient checkpointing) — the current default | 22.9 win/s | 2,456 MiB |
| **`frozen`** | **37.8 win/s** | **1,234 MiB** |

1.65× for freezing. But the throughput number understates it badly, because it measures the
wrong thing: **a frozen encoder is a pure function of its input.** Its output does not
change between epochs, between seeds, or between A1's six arms. Computing it eighteen times
over is not a cost of the experiment; it is a cost of how the experiment is written.

The fused training population is **5,457 train / 488 val units** (ADR-008 §D8.5) — small,
because only line-labelled units are kept. Encoding it once and reading the result from
disk turns every fused run into what a Phase 4 structural run already is: **2–5 minutes**.

---

## Decision

### D9.1 — A1's arms run with `llm.frozen=true`, and that is a property of the experiment

Not a shortcut with a caveat attached. ADR-008 §D8.4 already settled the principle for the
learning-rate split, in words that transfer exactly:

> *This choice affects how good all the arms are, not which one wins — but it is held
> identical across them, so A1 stays controlled.*

A frozen encoder held identical across all six arms leaves A1 controlled in precisely the
same sense. The parameter difference between `cross_attention` and `concat` is still exactly
the multi-head attention and its two layer norms (§D8.2), and that is what A1 measures.

There is a positive argument too, and it should be stated rather than smuggled in as a
consolation. **With the encoder unable to adapt, the fusion mechanism has to do the work
itself.** A fine-tuned CodeBERT can compensate for a weak fusion by reshaping its own line
representations until a trivial combiner suffices — which makes "concat did just as well" an
ambiguous result. Freezing removes that escape route. It is the harder test of the novelty
claim, not the softer one.

**This is a change to what A1 asks, and it must be reported as one.** The claim A1 can
support under this ADR is *"with a fixed semantic representation, bidirectional
cross-attention beats concatenation by X"* — never the unqualified form.

### D9.2 — The cache stores **line-aligned** features, not token states

The cache entry for a unit is the output of `align_tokens_to_lines`, not of the encoder:
`[n_lines_unit, 768]` plus a `[n_lines_unit]` coverage mask.

Caching token states instead would mean storing `[W, T, 768]` — windows × 512 tokens × 768
— which is roughly two orders of magnitude larger and buys nothing, because alignment is
deterministic given the same `token_to_line` map. Caching **after** alignment also means the
cache is indexed by the one axis this whole project is organised around (`CLAUDE.md` §5.2),
which is the axis a mistake would be visible on.

Entries are stored at each unit's **own** line count, never padded to `llm.n_lines`. Padding
every unit to 512 would cost 512 × 768 × 2 B = 786 KB per unit — about 4.7 GB — to store
mostly zeros, against a corpus mean of 31 lines per function (§1.4).

**The size is an estimate until it is measured.** Step 0 of the implementation is to measure
the line-count distribution of the 5,945 joined units and write the real figure into §1.20;
the plan is not costed until that number exists.

> **Measured 2026-09-15.** Line-labelled `train`: 5,457 units, mean 60.3 lines, median 27,
> p99 567. `val`: 488 units, mean 114.2, median 48, p99 998. Stored at each unit's own reach
> (capped at 512): **302,364 + 43,126 rows = 932 MB + 133 MB in fp32** (run `c99764bc…`).

### D9.3 — One insertion point: `FusedLineModel._semantic_lines`

That method already returns `(line_states [B, L_active, D_sem], line_mask [B, L_active])`,
which is exactly what a cache supplies. The change is one branch inside it:

```
if not self.fusion.uses_semantic:   -> _absent(...)          # unchanged
if "sem_line_states" in batch:      -> the cached path       # new
otherwise:                          -> encode and align      # unchanged
```

Everything downstream — the fusion modules, the head, the per-batch trim, the padding back
to `[B, L]` — is untouched, and is therefore identical by construction rather than by
inspection. `gnn_only` never reads the cache because `uses_semantic` is `False`, so the
ablation that isolates the union advantage (§D8.1) is unaffected.

The live path **stays**. It is not replaced, it is not deprecated, and D9.7's confirmation
runs need it.

### D9.4 — The cache refuses itself when it cannot prove it is current

This is the decision the rest of the ADR is in service of. A stale feature cache is a
silent-failure machine of exactly the shape this project keeps getting caught by: it trains,
it converges, it produces a number, and the number is about a corpus that no longer exists.
Nothing raises. It would pass every test written against it.

So the cache carries a digest over **every value that can change its contents**, and the
dataset refuses to load a cache whose digest does not match the live config:

| Source | Why it is in the digest |
|---|---|
| `llm.model_name`, `llm.pretrained` | A different encoder is a different feature space |
| `llm.max_length`, `llm.stride`, `llm.windowing` | Change the windows, change the alignment |
| `llm.n_lines` | The line axis itself |
| `llm.frozen` | A cache built from an unfrozen encoder is one arbitrary draw from training |
| `representation.normalize_whitespace` | **ADR-006.** A cache built from un-normalized text encodes the indentation leak and hands it to every arm |
| Split fingerprint (`0ade7483…`) | A different population |
| Shard digest (`8c1cccdd…`) | Different source text under the same split |

Mismatch is a **hard failure naming the mismatched key**, never a silent rebuild —
`Rules.md` §0.4. A rebuild is minutes; a silently wrong run is a retracted finding.

### D9.5 — Frozen implies eval mode, and dropout must be provably inert

`llm.dropout` is `0.1`. A frozen encoder left in `train()` mode still applies dropout, so its
output is **stochastic** — the cache would be one random draw and the live path would
disagree with it on every batch, by a margin that looks like noise and is not.

`frozen=true` therefore forces the encoder's module into eval mode, and a test asserts that
two consecutive forward passes over the same batch are bitwise identical before anything is
written to disk. This is the same class of trap as §7 #22 — a config value that governs a
decision and is not consumed — except here the value is consumed and its *effect* is the
problem.

> **Amended by measurement, 2026-09-15.** "Bitwise identical" failed on the first real build
> with a maximum difference of 9.5e-7, and the cause was **not** dropout. Measured on the RTX
> 3050 Ti with real CodeBERT: two passes give **bitwise identical token states**; the pooled
> line states differ by up to **1.9e-6**, because CUDA `index_add_` accumulates in no fixed
> order. The live training path carries the same wobble on every batch. For scale, dropout at
> p=0.1 moves the same states by up to 13.8. The check is therefore two-stage: token states
> bitwise, line states within `ALIGNMENT_TOLERANCE = 1e-5`. The tolerance sits seven orders
> of magnitude below what dropout produces, so it cannot hide the failure this section
> guards against.
>
> **A second dropout this section did not name.** `PretrainedCodeEncoder.dropout` sits
> *outside* the frozen HuggingFace module, on the returned token states, and is live in
> `train()` mode. A cache cannot apply it before pooling, so the cached path applies the same
> module to the pooled line states instead. Identity in eval mode, where equivalence is
> pinned; a different noise shape in training, identical across every arm reading the cache.

### D9.6 — fp16 on disk, and an equivalence test per arm

Stored fp16. Under AMP the encoder already computes in fp16, so fp16 storage is closer to
what the live path produces than fp32 would be, and it halves the disk.

The test that makes this safe is ADR-008 §D8.8's, reused deliberately:

> *An optimization that changes a number is not an optimization.*

For **every arm**, the cached path and the live frozen path must produce logits equal within
a pinned tolerance on the same batch — the same shape of test that already pins
`fusion.trim_line_axis`. The tolerance is a constant in the test, not a parameter, and
loosening it is a code review conversation.

> **Amended by measurement, 2026-09-15: stored fp32, not fp16.** The premise above is false.
> Under AMP, RoBERTa's final LayerNorm autocasts to fp32, so the live path's token and line
> states are **fp32**. fp16 storage would add ~3e-5 of rounding the live path never has —
> more than the GPU's own run-to-run noise. The cost is disk and RAM: 1.07 GB instead of
> ~0.54 GB.

### D9.7 — Staged: a frozen screen, then a fine-tuned confirmation

**The frozen result is not automatically the published result.** The mechanism that wins with
a fixed semantic representation could lose with an adapting one, and this ADR does not claim
otherwise.

1. **Screen.** All six arms × 3 seeds, frozen, off the cache. ~5 hours.
2. **Decide.** If A1's margin is comfortable relative to the seed spread, it is reportable
   *as a frozen-encoder result*, in D9.1's qualified form.
3. **Confirm.** Re-run the winner and `concat` **fine-tuned**, 3 seeds each — 6 runs,
   ~20–25 h — and report both. If they disagree, that disagreement is the finding and gets
   written up as one.

The confirmation is triggered by the margin, not skipped by default. **What must not happen
is the frozen number being quoted in the unqualified form** because the confirmation was
inconvenient.

### D9.8 — What this does not cover

- **Phase 7's token-level attribution still needs the live encoder.** The cache holds line
  features, so fusion-level attention maps (`explain()`) survive it intact, but anything
  tracing evidence back to individual tokens must run the encoder. Phase 7 is unaffected on
  the fusion side and must not assume the cache on the semantic side.
- **A5 (CodeBERT vs GraphCodeBERT, frozen vs fine-tuned) is untouched.** It is the ablation
  that measures what D9.1 costs, and it needs its own cache per encoder — which the D9.4
  digest handles by construction, since `model_name` is in it.
- **The structural branch is not cached and should not be.** Phase 4 runs take 2–5 minutes;
  there is nothing to reclaim, and the GNN is trained, not frozen.

---

## Consequences

**Good.**

| | Now | Frozen + cached |
|---|---|---|
| A1 core (2 arms × 3 seeds) | ~20–30 h | **~1.5 h** |
| A1 full (6 arms × 3 seeds) | ~60–80 h | **~5 h** |
| One-time encode pass | — | ~10 min at 37.8 win/s |

- Peak VRAM drops from 2,456 MiB to 1,234 MiB for the semantic half, which on the 4 GiB card
  is the difference between §1.17's 92.4% and real headroom — and §7 #24 is why that matters
  more than it sounds.
- **Iteration stops being a day.** A fused bug currently costs four hours to reproduce. At
  fifteen minutes, the fusion code gets debugged properly rather than carefully.
- The `sum` / `gated` / `llm_only` / `gnn_only` arms stop being a budget question.

**Costs, stated rather than discovered.**

- **A1 answers a narrower question**, and D9.1 requires the narrower phrasing everywhere.
- **New code on the critical path of the load-bearing experiment.** The mitigation is D9.3's
  single insertion point and D9.6's per-arm equivalence test — but the honest statement is
  that this adds a failure mode A1 did not previously have.
- **The screen may need the confirmation anyway** (D9.7), in which case the saving is the
  difference between 60–80 h and ~25 h rather than ~5 h. Still worth it; not free.
- Disk: an estimate until step 0 measures it. Bounded well under the 12.8 GB the checkpoint
  sweep just reclaimed.

**Reversible today at zero cost.** `llm.frozen` is already a config knob and the live path
is untouched by D9.3. **Irreversible the moment a seed is logged**, because `llm.frozen`
changes `config_hash` and `report_seeds.py` will then correctly refuse to aggregate across
it (§7 #38) — which is the intended behaviour and the reason this ADR exists before the
code.

---

## Implementation plan

**Done 2026-09-15**, every step. Where each landed: `training/semantic_cache.py` (digest,
build, load, determinism), `scripts/build_semantic_cache.py`, `FusedLineDataset(semantic_cache=)`
and `collate_fused`, `FusedLineModel._cached_semantic_lines`, `train.semantic_cache` in
`configs/train/fused.yaml`, and `tests/unit/test_semantic_cache.py`. In order, as planned:

0. **Measure the line-count distribution** of the 5,945 joined units and write the real
   cache size into `Memory.md` §1.20. *The plan is not costed until this exists.*
1. `scripts/build_semantic_cache.py` — one pass over `train` and `val`, logged to MLflow as
   a **data artefact, not a result** (the `build_node_vocab.py` / `build_subtoken_vocab.py`
   pattern), writing shards via `torch.save` (`Rules.md` §2 permits it on our own artifacts)
   plus a manifest carrying the D9.4 digest.
2. The D9.5 determinism assertion, run **before** anything is written.
3. `FusedLineDataset` gains the cache path; `collate_fused` stacks `sem_line_states` and
   `sem_line_mask` into the flat `sem_`-prefixed batch (§7 #45 — it must stay flat).
4. D9.3's branch in `FusedLineModel._semantic_lines`.
5. **D9.6's equivalence test, for all six arms.** This is the gate on the whole change.
6. The D9.4 refusal test: mutate each digest input in turn, assert a `ConfigError` naming
   the key.

---

## Verification

All 2026-09-15.

| Check | Result |
|---|---|
| D9.6 equivalence, CPU, every arm, end to end from shards | 6 / 6 pass at `EQUIVALENCE_TOLERANCE = 1e-5` |
| D9.4 refusal, every digest input mutated in turn | 15 / 15 raise `ConfigError` naming the key |
| D9.5 an encoder with live dropout is refused; eval mode passes | pass |
| A fine-tuned run cannot read the cache | `ConfigError` |
| **D9.6 on real hardware** — CodeBERT, RTX 3050 Ti, AMP, 24 real `val` units in 6 budget-sized batches | cached vs live: max line-logit diff **4.9e-4**, function-logit **4.6e-5**, across the five semantic-using arms. **Live vs live, same batches: 4.9e-4 / 3.4e-5.** The cache adds nothing measurable above the GPU's own noise floor. |
| Cache build (run `c99764bc…`) | `train` 5,457 units in 132 s, `val` 488 in 19 s; token difference 0.0, line difference 1.9e-6 |
| Overfit on the cached path | `cross_attention` `117df3a4…`, `concat` `aebd4407…`, `sum` `3b32a40d…`: Top-1 1.000 (subtokens off); all six re-queued on the A1 config |

The per-epoch speed on the overfit config was ~2–3 s for 32 units. The full-population
figure is recorded in `Memory.md` §1.20 once the first A1 seed finishes; the ~5 h estimate
for the whole screen is not re-quoted as measured until then.
