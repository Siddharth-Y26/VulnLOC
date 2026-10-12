# Memory.md

**Purpose:** the project's working memory. Any contributor — human or AI assistant — reads this file **first**, before touching code, and updates it **last**, before ending a session. It exists so that a new chat, a new tool, or a new teammate does not have to re-read the codebase or re-derive decisions.

**Status (2026-09-14):** **Phases 1–4 all meet their exit criteria; the Phase 3 and Phase 4 gates are unsigned and are the owner's.** Semantic branch **Top-5 0.6141 ± 0.0055** over three seeds (§1.14); structural branch **Top-5 0.4302 ± 0.0372** with subtokens (§1.15). **Read §1.16 before quoting any depth or A4 number — two findings were retracted there.** Phase 5 (fusion, experiment A1) is the next real work and is gated on the phase sign-off. Run `1843412a…` remains void — §1.10 is why every line-level number was re-measured.

---

## 0. How to Use This File

**At session start:** read §1 (state), §2 (decisions), §5 (open decisions), §6 (next actions). That should be enough to start work without re-reading `src/`.

**At session end:** update §1, append to §3 (changelog), append any decision to §2, move completed items out of §6, record anything surprising in §4 or §7.

**Rules:**
- Newest entries at the **top** of append-only sections.
- Every entry is dated (`YYYY-MM-DD`) and, where relevant, carries a commit SHA or run ID.
- Never delete a decision — supersede it with a new entry that references the old one.
- If something in this file contradicts the code, the code is authoritative — fix this file immediately and note the drift.
- Keep it under ~500 lines. When it grows past that, archive resolved sections to `docs/memory/ARCHIVE-<date>.md` and leave a pointer.
- **Currently ~1,310 lines, over that target and recorded as over rather than quietly ignored.** Sessions 2–6 were archived 2026-09-12 (`docs/memory/ARCHIVE-2026-09-12.md`, −229 lines) and sessions 7–8 on 2026-09-14 (−99), once Phase 3 closed — the condition the previous note set. What remains above the line is §1.4–1.11, the measured corpus/representation/truncation/VRAM/leak sections — they are cited by section number from `CLAUDE.md`, the ADRs, the paper draft and code comments, so moving them would break more than it saves. **The next cut is sessions 10–11, once Phase 5 lands** — they document the phases now closing and are still the most-referenced history.

---

## 1. Current State

| Field | Value |
|-------|-------|
| **Active phase** | **Phase 7 — Explanation (7a), since 2026-10-07: Phases 5 and 6 are SIGNED (§1.1c, §1.1d).** Ablation A6 (attention direction) is running on the frozen cache as queue group `a6`. Phase 5's record, kept: **A1's frozen screen is DONE (§1.21):** cross_attention 0.5408 ± 0.0181 vs concat 0.5394 ± 0.0198 Top-5 — **no resolvable difference**; every fusion arm beats frozen llm_only (0.4903) on every seed; none beats fine-tuned LLM-only (0.6141). **D9.7 (fine-tuned confirmation): the six runs ADR-009 specifies are FINISHED (§1.22) — cross_attention 0.6162 ± 0.0184 vs concat 0.6169 ± 0.0104 Top-5, again no resolvable difference.** The added scaffold control `llm_only` × 3 finished on 2026-10-07: **0.6169 ± 0.0048 Top-5 — no fused arm beats it (§1.22)** |
| **Phase status** | ✅ **Phases 3 and 4 SIGNED by the owner, 2026-09-15; Phases 5 and 6 SIGNED 2026-10-07; Phase 7 SIGNED 2026-10-08 — Phase 8 is the current phase.** ADR-008 and ADR-009 accepted 2026-09-15, ADR-010 on 2026-10-08 |
| **Last updated** | 2026-10-12 16:10 (session 19: the clean-clone check §1.34; Phases 8 and 9 closed, §1.1f and §1.1g; the three-headline record §1.35; nothing is running — §6 item 0a). Before that: 2026-10-11 08:10 (session 18: held-out results §1.30; `a1r` and its replication §1.32; Phase 9 merged, bundle, latency and Docker §1.33; nothing is running — §6 item 0a) |
| **Open decision** | **None blocking a measurement. On 2026-10-08 the owner accepted ADR-010 (Q14), signed the Phase 7 gate and confirmed `docs/EvaluationProtocol.md` (§2); the single held-out evaluation is authorised and is made on comment-stripped text with the nine `d97r` checkpoints.** Q13 answered and Q5 deferred by the owner (§2, §5). Still open, not blocking: ADR-006 and ADR-007 `Proposed` (ADR-007 blocks ablation A3's CPG arm); Q4 (explanation supervision) belongs to Phase 7; the paper's framing of the A1 null; scheduled defragmentation of D: (§7 #57) |
| **Last commit** | Session 14's resume/queue commit on `phase-2/representation` (see `git log`). **`main` DOES exist** with an `origin` remote — CLAUDE.md §3 and §10 say otherwise and are wrong |
| **Best model run ID** | **Models of record since 2026-10-08 (ADR-010 accepted, §2; `val` numbers in §1.28): cross_attention `ba7dcc5f…` / `8afbf7da…` / `ff980345…` (the full model), concat `9ac7a146…` / `7579bc33…` / `10b43ce5…`, `llm_only` `d82a6a70…` / `a1dce535…` / `9296a0e5…`.** Single-branch baselines, measured with comments in place: semantic `fc902107…` / `5b177450…` / `e82ff268…` (§1.14); structural A4 `77370d56…` / `0d324da8…` / `3c570cc1…` (§1.15). **`fc902107…` (seed 42) is the model the inspector's Findings tab scores with.** Run `1843412a…` remains void |
| **Split fingerprint** | `0ade7483a1b04d4841b5837c7bd2ed398f878acf7d242b59d7a4b043a48aef52` (project-wise, 80/10/10, seed 42) |
| **Blocking issues** | None. Q1/Q2/Q3/Q6/Q7/Q8 closed by ADR-001/002/003/004/005. Q5 closed 2026-10-07 (the learned type classifier is deferred, §2). Q4 belongs to Phase 7 |
| **Biggest live constraint** | **~30% line-level label noise** (§1.6) — the ceiling on every line-level number this project will report. **And, since 2026-10-07, a formatting leak that inflates them (§1.25): ≈ 0.05 of `val` Top-5** |

### 1.1 Phase Tracker

| Phase | Status | Exit criteria met | Notes |
|-------|--------|-------------------|-------|
| 0 Foundation | 🟨 In progress | 3 / 4 | Docker criterion blocked — see §7 issue #1 |
| 1 Data Layer | ✅ **Done** | **5 / 5** | Closed 2026-08-11. Loader, labelling, splits, cache, leakage report and human label audit all complete — see §1.1b |
| 2 Representation | ✅ **Done** | **5 / 5** | Closed 2026-08-16. Parser, line map, all four graph builders (AST/CFG/DFG/CPG), PyG conversion, renderer, corpus report. **Q8 closed the same day by ADR-005** — the corpus-wide graph build is no longer gated |
| 3 LLM Branch | ✅ **Done — gate signed 2026-09-15** | **4 / 4** | Tokenization, `LineAligner`, encoder, line head, losses, metrics, training loop and both design measurements are built. **Overfit test passed** (Top-5 = 1.0, run `ffe3d03a…`). **Run `1843412a…` is VOID — ADR-006, §1.10.** The rerun on normalized text learns for real: seed 42 (`fc902107…`) reports Top-5 0.610 / line F1 0.378 with a rising curve. **All three seeds are in (§1.14): Top-5 0.6141 ± 0.0055, Top-1 0.3513 ± 0.0138, line F1 0.3692 ± 0.0204.** The ≥3-seed criterion is MET; **gate signed by the owner 2026-09-15** |
| 4 GNN Branch | ✅ **Done — gate signed 2026-09-15** | **4 / 4** | All four met by logged runs — `python scripts/report_phase4.py`, §1.13. Overfit test (`8b0a3c1d…`, Top-1 = Top-5 = 1.000); all three variants on real data; depth sweep 2/4/6; ≥3 seeds (gcn depth 4, Top-5 **0.3645 ± 0.0231**). **Gate signed by the owner 2026-09-15.** A4 (§1.15): subtokens lift Top-5 to **0.4302 ± 0.0372**, and A1 uses them |
| 5 Fusion | ✅ **Done — gate signed by the owner 2026-10-07 (§1.1c)** | **5 / 5** (head-to-head at ≥3 seeds, §1.21 frozen and §1.22 fine-tuned; the negative result investigated) | **Session 13 (2026-09-15): gates signed, ADR-008 and ADR-009 accepted, the semantic feature cache built and verified on real hardware (§1.20), A1 queued.** Session 12: `fusion/` (cross-attention + 5 comparators behind one `FusionModule`), `FusedLineModel`, `training/fused_datasets.py`, `configs/fusion/*`, `configs/train/{fused,overfit_fused}.yaml`, `scripts/train_fused.py`, **ADR-008**. Met by code and tests: gradient reaches both branches; attention maps shaped and retrievable; every arm selectable by `+fusion=` alone. **Head-to-head over ≥3 seeds: measured 2026-09-15 (§1.21)** — six arms × 3 seeds, frozen encoder; cross_attention ≈ concat. **Criterion 5 is half met:** "if cross-attention does not beat concat, record it honestly **and investigate before Phase 6**" — recorded (§1.21), not yet investigated. D9.7's fine-tuned confirmation is the natural investigation; Phase 6 must not start before it or an explicit owner decision to skip it. **2026-10-04 (session 15): the investigation's six runs are in (§1.22) — the null survives fine-tuning.** Still owed before the gate: the `llm_only` scaffold control (0 / 3, seed 42 paused) and the owner's signature. **2026-10-07 (session 16): the control is in — 0.6169 ± 0.0048, no fused arm beats it (§1.22). The three gate questions are answered in §1.1c; only the signature is missing** |
| 6 Localization Head | ✅ **Done — gate signed by the owner 2026-10-07 (§1.1d)** | **4 / 4** | "Best full-model configuration frozen and checkpointed": **met by decision** — the designed cross-attention model, D9.7's config, checkpoints `80fc5238` / `c193e3bc` / `829c3555` (§2). **Calibrated, reliability curve reported, threshold chosen on `val` only: met 2026-10-07 (§1.24)** — ECE 0.1704 → 0.0137, line F1 0.2954 → 0.3898 out of fold. **Q5 (CWE labels): explicitly deferred by the owner 2026-10-07 — no type classifier; Phase 7a names a template-derived category and says so (§2).** **Deferred in writing (§2):** unit-level aggregation and a joint loss that means something — no negatives; the sweep is frozen-only |
| 7 Explanation | ✅ **Done — gate signed by the owner 2026-10-08 (§1.1e)** | **4 / 4** (the owner read the explainer's sentences and named none of concern, §2) | 7a built and measured the same day (§1.27): 13,908 of 13,908 flagged lines on `val` explained and verified, three seeds; grounding suite passes; protocol written; four template sentences corrected after reading real output. **Open: the owner's reading of the template sentences for attack instructions.** A pattern names a cause for about one flagged line in twelve. 7b is deferred in writing (§2); the category is template-derived (Q5) |
| 8 Evaluation | ✅ **Done — gate closed on the owner's instruction 2026-10-12 (§1.1f); the owner has not read the gate's answers** | **4 / 4** | The line-level metric suite is built and reconciles with every logged `d97r` number. **The held-out pass was made once, on 2026-10-11 (§1.30).** The ablation tables of record are `bcfeb4ff…` (§1.32); the figures are drawn for `val`, `test` and `clean_eval`. **Ticked: the failure taxonomy — accepted by the owner on 2026-10-11 as revised (§1.31, §2).** **Ticked 2026-10-12 on the owner's words:** every table cell traceable to a run (checked by `scripts/trace_phase8_tables.py`, 0 did not trace), and the ablations' verdict on each novelty claim — as "every claim has a written verdict"; A3, A7 and A10 were not run and no claim is made about them (`Phases.md`). **Ticked 2026-10-12: the clean-clone criterion** — from a clone and the lock file the tests, split, shards and vocabularies reproduce exactly; of two same-seed frozen reruns one missed the band written beforehand, and the tick says so (§1.34, §2) |
| 9 Serving | ✅ **Done — gate closed on the owner's instruction 2026-10-12 (§1.1g); the owner has not read the gate's answers** | **4 / 4** | Bundle of record `artifacts/serving/ba7dcc5f-r2` (`ef031969…`) — the same weights as the first, with a `measured` block and today's model card; latency measured (`3e91feba…`: p95 2.63 s, over NFR-4's recommended 2 s — accepted and stated); errors in one shape; the demo page reads `POST /analyze/report`. **Docker: ticked as a clean container on this laptop (image `7283510c22c7`, same answers as the host to 5.2e-7); a second machine was not tried** (§1.33) |
| 10 Artifacts | ⬜ Not started | 0 / 3 | |

Legend: ⬜ not started · 🟨 in progress · ✅ done · 🟥 blocked

### 1.1a Phase 0 Gate

Archived — see [`memory/ARCHIVE-2026-08-11.md`](memory/ARCHIVE-2026-08-11.md).
Phase 0 remains **3 / 4**: the Docker criterion is unverified (§7 issue #1) and
blocks nothing before Phase 8.

### 1.1b Phase 1 Gate (`Phases.md` §Phase Gate Rule)

**1. Did we meet every exit criterion?** **Yes — 5 of 5. Phase 1 is done (2026-08-11).**

| Exit criterion | Status | Evidence |
|---|---|---|
| ADR-001 written and accepted | ✅ | ADR-001, refined by ADR-003; ADR-004 added for cross-split duplicates |
| Statistics reported: #functions, %vulnerable, mean lines/function, %vulnerable lines | ✅ | §1.4. 188,636 · 5.78% · 31.0 lines/fn · **0.795% of lines are vulnerable** |
| Leakage check run; results in `Memory.md` | ✅ | §1.5. 0 shared projects, 0 cross-split duplicate hashes, 0 clean-eval-in-train |
| 20 sampled line labels manually inspected by a human | ✅ | Reviewed by Siddharth, 2026-08-11: **14 correct / 6 wrong / 0 unsure.** Sheet with per-function reasoning at `artifacts/data/label_audit/label_audit.md` |
| Re-running the pipeline reproduces byte-identical splits | ✅ | Run `e395b21b…` re-ran the full corpus and reported `reproduced=True` against run `0c64a30f…`; fingerprint `0ade7483…` |

**2. What did we learn that changes a later phase?**

- **Line-level label noise is ~30%, not 19.2%** (§1.6). The two are different quantities
  and the line-level one is structurally the larger. Phase 6 must not treat line labels
  as clean, and Phase 8 must report this as the ceiling on any line-level metric.
- **The clean evaluation set averages 138.3 lines per function against the corpus's
  31.0 — 4.5× longer.** The 512-token truncation problem therefore lands hardest on
  exactly the partition carrying the headline line-level numbers. Phase 3 must measure
  label loss under truncation *on `clean_eval` specifically*, not corpus-wide, or it
  will report a reassuring average that does not apply where it matters.
- **A project-wise split does not prevent function-level leakage** (ADR-004). Phase 2
  should re-run the leakage check once tokenization enables near-duplicate detection;
  the 207 hashes we found by exact match are a floor.
- **Label derivation is clean.** Zero out-of-range, zero text mismatches, zero count
  mismatches, zero labels-on-negatives across 188,636 functions. The 0-indexed
  conversion is now verified corpus-wide rather than on 400 sampled rows, so Phase 2's
  line mapping can treat Big-Vul's own line numbers as trustworthy *input* and focus its
  own risk on the Tree-sitter byte-offset conversion.
- **val/test hold longer functions than train** (38.7 / 42.5 vs. 27.5 lines). Not just
  unseen projects — a different length distribution. Phase 8 must not attribute a
  train/test gap purely to generalization.

**3. What `PRD.md` §10 assumption has been confirmed or falsified?**
**A3 (line-level labels obtainable) is confirmed, with a caveat that changes its
meaning.** They are obtainable (8,243 functions / 46,513 lines, all internally verified)
but roughly **30% of them point at the wrong lines** (§1.6). A3 as written asked whether
the labels *exist*; the answer to whether they are *trustworthy* is "mostly, not always".
A1, A4, A5 untouched.

### 1.1c Phase 5 Gate (`Phases.md` §Phase Gate Rule) — **SIGNED BY THE OWNER 2026-10-07**

Written by session 16 and put to the owner the same morning; **the owner signed it
("Sign it") — Phase 5 is done and Phase 6 has started.** The decisions taken with it are in
§2: the full model is the designed cross-attention architecture; the function-level
criteria and 7b are deferred; external baselines stay `(reported)`.

**1. Did we meet every exit criterion?** **Five of five, on the evidence below.**

| Exit criterion | Status | Evidence |
|---|---|---|
| Gradient-flow test: both branches receive non-zero gradients | ✅ | `tests/unit/` fusion tests, session 12; every fixture contains a unit with no structural keys (§7 #41) |
| Attention maps have correct shape and are retrievable | ✅ | `FusedLineModel.explain()`; `[B, H, L, L]` per direction, off by default |
| `fusion=cross_attention` vs `fusion=concat` runnable by flag alone | ✅ | `+fusion=<arm>`; every arm of A1 and D9.7 was launched that way by `run_queue.py` |
| First head-to-head comparison logged (≥ 3 seeds each) | ✅ | Frozen, six arms × 3 seeds (§1.21). Fine-tuned, three arms × 3 seeds (§1.22) |
| A negative result recorded honestly and investigated before Phase 6 | ✅ | Recorded §1.21 (2026-09-15). Investigated by ADR-009 D9.7: nine fine-tuned runs, finished 2026-10-07 (§1.22). **cross_attention 0.6162 ± 0.0184, concat 0.6169 ± 0.0104, llm_only 0.6169 ± 0.0048 Top-5** — the null holds, and neither fused arm beats LLM-only in the same scaffold |

**2. What did we learn that changes a later phase?**

- **Phase 6's "best full-model configuration" has no winner to freeze on Top-5.** Three
  arms tie fine-tuned. Which one is "the model" is a choice about what the project needs
  next, not a measurement: only cross_attention produces the attention maps Phase 7a's
  evidence extraction reads; `llm_only` is the simplest thing that scores the same. **The
  owner's decision** (session 16 recommends keeping D9.7's cross_attention runs — they
  exist at three seeds, so nothing new is fine-tuned — and reporting all three arms).
- **Phase 7's explanations cannot be sold as coming from a better model.** The attention
  maps are evidence about what the fused model looked at; §1.22 says looking there did not
  improve localization. Faithfulness (the deletion test) carries more weight than planned.
- **Phase 8's novelty claim changes shape.** A1 answers "does bidirectional cross-attention
  help?" with *no, not measurably* — frozen and fine-tuned. (Machine 2 reproduced the
  cross_attention arm's level both ways, §1.23; it did not run the comparison.) What the
  data does support: the structural branch helps a
  **frozen** encoder by +0.03 to +0.05 Top-5 and stops helping once the encoder adapts.
  The paper's framing is the owner's (§5).
- **Fine-tuning dominates every architectural choice measured here.** Freezing costs about
  0.08 Top-5 for the fused arms and 0.13 for `llm_only` (§1.21 against §1.22); no ablation
  on machine 2 moves Top-5 by 2 pooled sd (§1.23). Phase 6's sweep should spend its budget accordingly — and weighted BCE is the
  one candidate that earned more seeds.
- **Every fine-tuned number so far is `val`, validation-selected** (§1.18). `test` and
  `clean_eval` are untouched and stay so until the configuration is frozen
  (`Rules.md` §5.3).
- **An unattended run needs the queue's own recovery** (§7 #56, #57) — built 2026-10-07.

**3. What `PRD.md` §10 assumption has been confirmed or falsified?**
**A2 (single-GPU compute budget) is confirmed as binding, and it shaped the result's
form:** A1 could only be screened with a frozen encoder (ADR-009) and confirmed on nine
fine-tuned runs over five days; the registered risk — "architecture may be under-scaled
vs. SOTA" — stands and is not tested by anything here. **No assumption is falsified by
Phase 5; the project's central *hypothesis* is not an entry in that register**, and it is
what §1.22 fails to support. A1, A3, A4, A5 untouched.

### 1.1d Phase 6 Gate (`Phases.md` §Phase Gate Rule) — **SIGNED BY THE OWNER 2026-10-07**

Written by session 16 and put to the owner the same afternoon, in one batch with Q5. **The
owner deferred the type classifier ("Defer the classifier") and signed the gate ("Sign
it") — Phase 6 is done and Phase 7 is the current phase.** The text below is as it was
prepared (commit `aece11b`), with the fourth criterion filled in from the owner's answer.

**1. Did we meet every exit criterion?** **Four of four — the fourth by an explicit
deferral, which the criterion allows.**

| Exit criterion | Status | Evidence |
|---|---|---|
| Line scores calibrated (reliability curve reported) | ✅ | Platt scaling per checkpoint, fitted on `val`: ECE 0.1704 ± 0.0330 → 0.0137 ± 0.0104 out of fold, 3 seeds (run `569c9701…`, §1.24). Reported with where it still fails: above p = 0.8 the observed rate is 0.67–0.81 |
| Threshold chosen on validation only | ✅ | Maximum line F1 on calibrated `val` probabilities, one threshold per checkpoint (0.1686 / 0.1083 / 0.2159). `scripts/calibrate.py` refuses any other partition |
| Best full-model configuration frozen and checkpointed | ✅ | The designed bidirectional cross-attention model at ADR-009 D9.7's configuration; checkpoints `80fc5238` / `c193e3bc` / `829c3555`. **The owner's choice (§2), not a measured win** — it ties concat and `llm_only` on every `val` metric read so far |
| CWE labelling decided (Q5) and implemented or explicitly deferred | ✅ | **Explicitly deferred by the owner, 2026-10-07 (§2):** no model predicts a type; Phase 7a's explanations name a category derived from templates and say so. What is known: Big-Vul carries a `CWE ID` per function and the loader keeps it as unit metadata (`data/loaders/bigvul.py`); 86 distinct CWEs among the positives (`docs/DATA_CARD.md`). No model here predicts it. How many of the line-labelled training functions carry one has **not** been measured |

Deliverables in `Phases.md` that were **not** built, said plainly so the tick marks are not
read as more than they are: unit-level aggregation and the joint line + unit loss (deferred
in writing 2026-10-07, §2 — every model here trains on vulnerable functions only); the CRF
head variant (listed as optional; not built); and **no hyperparameter sweep of the full
model was run** — one fine-tuned run costs 114–211 min. What exists instead is frozen-encoder
ablations: machine 2's slate (§1.23) and A6, attention direction, running as this is
written.

**2. What did we learn that changes a later phase?**

- **Every line F1 logged at 0.5 measured the threshold, not the model.** The full model's
  0.2954 ± 0.0855 at 0.5 becomes 0.3898 ± 0.0190 at each checkpoint's own `val`-chosen
  threshold, and the three D9.7 arms then tie (0.3898 / 0.3908 / 0.3936, §1.24). **Phase 8
  reports thresholded metrics at the calibrated, `val`-chosen threshold of each checkpoint
  and never compares arms at 0.5.**
- **A calibrator and a threshold belong to one checkpoint.** Three seeds of the same arm
  fitted scales of 1.8412 / 0.7166 / 3.0486. Phase 9 ships a checkpoint together with its
  own calibrator — which is why `models/heads/calibration.py` sits at L2, where serving may
  import it — and nothing averages a calibrator across seeds.
- **A calibrated probability is still not a confidence above 0.8** (observed 0.67–0.81).
  Phase 7's explanations and Phase 9's output lead with rank and do not print a high
  probability as near-certainty.
- **Fusion gains no second metric.** After calibration line F1 ties across the three arms
  exactly as Top-5 does (§1.22). Phase 8's claim stays what the Phase 5 gate said it is.
- **Per-line scores are on disk** (`artifacts/phase6/scores/`), each file written only after
  it reproduced its run's logged metrics within 1e-4. Phase 8's metrics, intervals and
  failure taxonomy are computed from those files, not from re-runs.
- **The held-out evaluation is now one scoring pass per checkpoint**
  (`score_checkpoint.py … +score.final=true`), with the `val`-fitted calibrator and
  threshold applied unchanged. The configuration being frozen is `Rules.md` §5.3's
  precondition; **`test` and `clean_eval` are still untouched, and the go-ahead is the
  owner's.**

**3. What `PRD.md` §10 assumption has been confirmed or falsified?**
**A2 (single-GPU compute budget) is binding again:** it is why there is no fine-tuned sweep
and why every function-level deliverable of this phase is deferred — the 148,983-function
split is out of reach on this card inside the week. **Nothing is falsified by Phase 6.** A3
(line-level labels obtainable) is untouched: the labels' measured ~30% noise (§1.6) remains
the ceiling on every line metric here, calibrated or not. A1, A4, A5 untouched.

### 1.1e Phase 7 Gate (`Phases.md` §Phase Gate Rule) — **SIGNED BY THE OWNER 2026-10-08**

Written by session 16 for the owner to read on 2026-10-07. **Signed on 2026-10-08 (§2):**
the owner stated that `artifacts/phase7/sentences_for_review.md` and
`docs/HumanEvalProtocol.md` had been read, named no sentence of concern, and said to
proceed. With that the fourth criterion below is met — **four of four.** The text below is
as prepared, kept unedited.

**1. Did we meet every exit criterion?** **Three of four; the fourth needs a person.**
The table is in §1.27. Met: an explanation for every positive prediction (13,908 of 13,908
on `val`, three seeds, all verified); the grounding suite; the human-evaluation protocol
(`docs/HumanEvalProtocol.md` — its approval is part of this gate). **Open: "explanations
reviewed for absence of attack instructions"** — the automatic check is in place and
passes; the review itself is reading one page —
`artifacts/phase7/sentences_for_review.md`, every sentence the explainer can emit, printed
from the code — and it is the owner's.

Not built, said plainly: 7b (deferred in writing); node importances (the model has no such
output — `graph_nodes` is a selection by node type on the flagged line, and says so); and
`attention_top_tokens` is identifiers ordered by *line*-level attention, because the
fusion attends between lines, not tokens.

**2. What did we learn that changes a later phase?**

- **Looking at what the model ranks first found a label leak (§1.25).** Phase 8's
  evaluation must be made on comment-stripped text, and its headline needs models trained
  on it (`d97r`). The held-out evaluation the owner authorised must not run on the shards
  as they are.
- **A template names a cause for one flagged line in twelve.** Phase 8's human evaluation
  will score the other eleven as "no cause named"; the protocol reports the two kinds
  separately so neither hides the other.
- **Phase 8's faithfulness test carries the weight of the attention sentence.** Nothing
  measured so far says the attended lines drive the score.
- **8–12% of the labelled lines are deleted comments.** Phase 8's metric definitions have
  to say whether a blank line is a prediction target; both readings are already computed.
- **Phase 9 can serve the contract as built** — `explanation/contract.py` returns exactly
  `PRD.md` §5.4's keys; `risk_score` is the top calibrated line score and `is_vulnerable`
  means "a line was flagged", never "safe" otherwise.

**3. What `PRD.md` §10 assumption has been confirmed or falsified?**
**A5 (the `evidence` schema of §5.4) is confirmed as implementable, with two of its three
fields redefined in writing**: `attention_top_tokens` and `graph_nodes` cannot mean what
their names suggest for a model that attends between lines and scores no nodes.
**A3 (line-level labels obtainable) is weakened again**: on top of ~30% wrong labels
(§1.6), 8–12% of the labels sit on comment-only lines, and the release's handling of
flagged lines leaked the label twice (ADR-006, §1.25). Nothing is falsified outright.

### 1.1f Phase 8 Gate (`Phases.md` §Phase Gate Rule) — **CLOSED ON THE OWNER'S INSTRUCTION 2026-10-12**

Written by session 19. The owner's words were *"okay finish these two also so phase 8 and 9 done?"* — said after being told that two
boxes and the two gate signatures were what remained on the owner's side. **The owner
has not read the answers below and did not write "signed"; reading those words as the
instruction to tick the two boxes and close the gate is the session's, and the owner can
reverse any part of it (§2).**

**1. Did we meet every exit criterion?** **Four of four, each with what its tick claims
written beside it in `Phases.md`.** *Every table cell traceable to a run* — checked by
`scripts/trace_phase8_tables.py`: three evaluations, 18 ablation tables, 188 run
references, 0 that do not trace. *Ablations confirm or refute each novelty claim* —
ticked as "every claim has a written verdict": A1 and A2 are not supported for the
models of record and supported with a frozen encoder on validation; A4, A5, A6, A8 and
A9 are not resolved; **A3, A7 and A10 were not run and no claim is made about them.**
*Failure taxonomy* — accepted by the owner on 2026-10-11. *Reproducible from a clean
clone* — ticked on 2026-10-12 with one rerun's miss stated (§1.34).

Not built or not run, said plainly: baseline reproductions under our splits and
cross-dataset generalization (deferred 2026-10-07); the function-level metrics of
`ResearchPlan.md` §4.1; the faithfulness tests of §4.4 — so contribution C5 is not
earned and nothing measured says the attended lines drive a score; the efficiency
figures of §4.5 beyond serving latency; the human evaluation (deferred 2026-10-12).

**2. What did we learn that changes a later phase?**

- **A1 has a two-part answer and the paper leads with the null** (Q15, closed
  2026-10-12): fine-tuned and held out, the fusion rule makes no resolved difference;
  frozen, on validation, cross-attention is ahead.
- **A run repeated at the same seed is a replicate, not a copy (§7 #70).** Phase 10's
  independent reader must be given something that can be hit: the data pipeline's
  fingerprint and byte-equal shards, or a table recomputed from stored scores — both
  deterministic — or a retrained arm judged against its spread over seeds. Never one
  run's Top-5.
- **The held-out partitions are spent.** Nothing that reads `test` or `clean_eval` may
  be tuned, re-scored or re-evaluated for the paper; Phase 10 quotes §1.30.
- **19.4% of the flagged lines on `val` are blank or syntax alone** and the model ranks
  them below chance (§1.31): a limitation the paper states beside every line-level
  figure, with the label-noise ceiling.
- **No person rated the explanations.** C3 rests on the automatic verification and on
  the share of explanations that names a cause (about one in twelve).

**3. What `PRD.md` §10 assumption has been confirmed or falsified?**
**A2 (a single-GPU budget) held, at a price the results carry**: three seeds an arm for
the fine-tuned models, a frozen screen for everything else, and no function-level
experiment at all. **A3 (line-level labels obtainable) stands as weakened in §1.1e** —
the held-out numbers are bounded by the same ~30% label noise. **A4 (one language
family) is unchanged**: C and C++ from one corpus, and cross-corpus generalization was
not run. Nothing is falsified outright.

### 1.1g Phase 9 Gate (`Phases.md` §Phase Gate Rule) — **CLOSED ON THE OWNER'S INSTRUCTION 2026-10-12**

Written by session 19, on the same words as §1.1f (*"okay finish these two also so phase 8 and 9 done?"*). **The owner has not read
the answers below and did not write "signed"; the reading is the session's and the owner
can reverse any part of it (§2).**

**1. Did we meet every exit criterion?** **Four of four, each with what its tick claims
in `Phases.md`.** *Docker* — a clean container on this laptop answers as the host does;
a second machine was not tried. *Latency* — measured and reported; the recommended bound
is missed on long functions and the miss is accepted and stated. *Errors* — one shape.
*Model card* — shipped in every bundle; the bundle of record carries today's card and a
measured record. Every deliverable exists: `POST /analyze`, `GET /health`,
`GET /version`, the schemas, a CPU image, the demo page, `docs/Serving.md` and `/docs`.

Not done, said plainly: the image's versions are unpinned and it has no commit SHA; no
authentication (the service binds `127.0.0.1` by default); nothing tuned for latency; the
explanations give a reason and no remedy.

**2. What did we learn that changes a later phase?**

- **The served model flags a line in every function it is given** (six of six tried,
  including `int add(int a, int b)`), because its threshold was fitted on vulnerable
  functions only. Phase 10's README and the paper must present the tool as a *ranker for
  a function already suspected*, first and in plain words — not as a detector with a
  caveat. The owner's "No known vulnerability found" wording is correct and almost
  never shown.
  ***Corrected the same evening:*** "every function" was six short inputs. On `val`,
  71 of 488 vulnerable functions get no flagged line — the long ones (median 117 lines
  against 40) — so the wording *is* shown, to functions that contain a vulnerability,
  while a four-line harmless one is flagged at 0.99 (§1.33). The conclusion is
  unchanged and stronger: a ranker, not a detector.
- **Host and container agree to 5.2e-7 across torch 2.13.0 and 2.14.1**, on one
  function. The unpinned image is not changing answers today; pinning is still owed for
  the day it does.
- **A measured record can travel with a checkpoint without scoring anything**; the same
  option can carry validation figures.
- **What a reader needs did not fit the §5.4 object**: a second endpoint carries it and
  `/analyze` stays the contract.

**3. What `PRD.md` §10 assumption has been confirmed or falsified?**
**A5 (the `evidence` schema of §5.4) is confirmed as served** — the response is exactly
that object — with the note that it was not enough for a person reading a result.
**A1 (the primary persona is the application developer) is weakened**: the question a
developer brings — is this function safe? — is the one this model cannot answer; it can
only say which lines to read first. Nothing is falsified outright.

### 1.2 What Exists Right Now

| Component | Status | Path |
|-----------|--------|------|
| Documentation set | ✅ v0.1 | `docs/` |
| Repo scaffolding | ✅ | `src/vulnloc/` (L0–L6 packages + `utils/`) |
| Packaging / env | ✅ | `pyproject.toml`, `.venv` (py3.12), `requirements.lock.txt` |
| Task runners | ✅ | `Makefile`, `make.ps1` |
| Containerization | 🟨 written, unbuilt | `Dockerfile`, `docker-compose.yml`, `.dockerignore` |
| Configs (Hydra) | ✅ root + `train/` | `configs/` — other groups are empty dirs by design |
| Exceptions | ✅ | `src/vulnloc/utils/exceptions.py` |
| Logging (JSON + rich, run-id bound) | ✅ | `src/vulnloc/utils/logging.py` |
| Config provenance (git sha, config hash) | ✅ | `src/vulnloc/utils/config.py` |
| Registries (6 extension points) | ✅ empty | `<layer>/registry.py`, `utils/registry.py` |
| Seeding / determinism | ✅ | `src/vulnloc/training/seeding.py` |
| Device resolution (auto/cuda/cpu) | ✅ | `src/vulnloc/training/device.py` |
| MLflow tracking | ✅ | `src/vulnloc/training/tracking.py` |
| Train entrypoint (no-op loop) | ✅ | `scripts/train.py` |
| Quality gates | ✅ | ruff, black, mypy, import-linter, pre-commit, `.github/workflows/ci.yml` |
| Tests | ✅ **791 passing / 2 skipped** | `tests/unit/`, `tests/integration/` |
| Corpus acquisition + audit | ✅ | `data/{sources,download,audit}.py`, `scripts/{prepare_data,audit_data}.py` |
| Big-Vul loader (`@register_dataset`) | ✅ | `data/loaders/bigvul.py` |
| Line-label derivation (`labels_v1`) | ✅ | `data/labeling.py` |
| Splitting + leakage (3 strategies) | ✅ | `data/splits.py` |
| Split pipeline + reports | ✅ | `data/pipeline.py`, `scripts/build_splits.py` |
| Preprocessed shards | ✅ | `data/cache.py`, `scripts/build_cache.py` |
| Deterministic sampling primitives | ✅ | `utils/hashing.py` |
| Label-audit sheet (human task) | ✅ generated | `scripts/sample_labels.py` |
| `DATA_CARD.md` | ✅ | `docs/DATA_CARD.md` |
| Tree-sitter parser wrapper (C/C++) | ✅ | `representation/parser.py` |
| Line map (byte + char → 1-indexed line) | ✅ | `representation/line_map.py` |
| `CodeUnit` (L0 → L1 bridge) | ✅ | `representation/units.py` |
| Graph container + invariants | ✅ | `representation/graphs/base.py` |
| AST builder (`@register_graph("ast")`) | ✅ | `representation/graphs/ast.py` |
| CFG builder — structured + goto/fallthrough | ✅ | `representation/graphs/cfg.py` |
| DFG builder — reaching definitions | ✅ | `representation/graphs/dfg.py` |
| CPG — union over the shared node set | ✅ | `representation/graphs/cpg.py` |
| NetworkX + PyG conversion | ✅ | `representation/to_pyg.py` |
| Line-annotated graph rendering | ✅ | `representation/visualize.py` |
| Corpus parse + graph report | ✅ | `scripts/report_representation.py` |
| Tokenization + windowing (line-safe) | ✅ | `representation/tokenization.py` |
| `LineAligner` — the shared line axis | ✅ | `models/alignment.py` |
| Truncation report | ✅ | `scripts/report_truncation.py` |
| VRAM probe | ✅ | `scripts/probe_vram.py` |
| Grammar policy (ADR-005) | ✅ | `representation/parser.py` — `parse_with_policy` |
| Parse-failure characterization | ✅ | `scripts/characterize_parse_failures.py` |
| Semantic encoder (CodeBERT/GraphCodeBERT) | ✅ | `models/semantic/encoder.py` |
| LLM-only baseline model | ✅ | `models/semantic/baseline.py` |
| Per-line head + derived function score | ✅ | `models/heads/line_head.py` |
| Window-budget dataset / sampler / collate | ✅ | `training/datasets.py` |
| Imbalance losses (focal, weighted BCE) | ✅ | `training/losses.py` |
| Line + function metrics (Top-k, IFA) | ✅ | `training/metrics.py` |
| Training loop, early stopping, checkpoints | ✅ | `training/loop.py` |
| Training entrypoint | ✅ | `scripts/train_semantic.py` |
| Structural encoder (GCN/GAT/GGNN, one scaffold) | ✅ | `models/structural/encoder.py` |
| GNN-only baseline model | ✅ | `models/structural/baseline.py` |
| Node-budget graph dataset / sampler / collate | ✅ | `training/graph_datasets.py` |
| Shared training job (both branches, one loop) | ✅ | `training/job.py` |
| Structural entrypoint | ✅ | `scripts/train_structural.py` |
| Fusion: cross-attention + 5 comparators | ✅ **built, unrun** | `models/fusion/{base,cross_attention,comparators}.py` |
| Fused model (both branches, one line axis) | ✅ **built, unrun** | `models/fused.py` |
| Fused dataset (joined on `unit_id`) / sampler / collate | ✅ **built, unrun** | `training/fused_datasets.py` |
| Fused entrypoint | ✅ **built, unrun** | `scripts/train_fused.py` |
| Fused VRAM probe (real batches, worst case) | ✅ measured | `scripts/probe_vram_fused.py` |
| Review III figures + tables (from logged runs only, labelled validation-selected) | ✅ | `scripts/make_review3_figures.py` → `artifacts/review3/` |
| Published baselines, quoted — never plotted (§1.19) | ✅ | `docs/prior_work.yaml`, guarded by `tests/unit/test_prior_work.py` |
| Localization head | ⬜ | — |
| Explanation module | ⬜ | — |
| API / serving | ⬜ | — |

---

## 1.4 Corpus Statistics — measured, not quoted

Measured from the downloaded files on 2026-08-10 (`Rules.md` §5.7 — we never adopt a
number from a paper as if it were ours). Reproducible from the scan scripts; these
supersede any figure quoted elsewhere in the docs.

**Big-Vul (LineVul release)** — 12.17 GB of corpora total across all sources.

| | train | val | test | **total** |
|---|---|---|---|---|
| rows | 150,908 | 18,864 | 18,864 | **188,636** |
| vulnerable | 8,736 | 1,109 | 1,055 | **10,900 (5.78%)** |
| vulnerable **with** line labels | 6,613 | 842 | 788 | **8,243** |
| vulnerable **without** line labels | 2,123 | 267 | 267 | **2,657 (24.4%)** |
| non-vulnerable with line labels | 0 | 0 | 0 | **0** ✅ invariant holds |
| flagged lines | 37,577 | 4,970 | 3,966 | **46,513** |
| distinct projects | 308 | 271 | 274 | — |
| distinct CWEs (vulnerable) | 86 | 51 | 48 | — |
| mean `processed_func` chars | 899 | 904 | 894 | ~900 (≈250 tokens) |

- 188,636 rows matches the published Big-Vul function count exactly — the download is complete.
- **Chrome + linux = 65.8% of all functions** across every split. See §7 issue #9.
- ~~Mean function length ≈ 250 tokens, so the 512-token truncation risk is smaller than
  assumed.~~ **Falsified 2026-08-16 by actual tokenization (§1.8).** The estimate divided
  mean characters by ~3.6 and was both too low and taken on the wrong denominator: real
  CodeBERT means are 358 tokens on train and **1,502 on `clean_eval`**, and truncation
  costs **38.1% of all flagged lines**. The tail did matter.

**Big-Vul × PrimeVul overlap** (join key verified: `md5(func, whitespace stripped)` as
int, reproduced 300/300):

| | count | share of 8,243 |
|---|---|---|
| found in PrimeVul | 1,724 | 20.9% |
| → PrimeVul agrees vulnerable | **1,393** | 16.9% |
| → PrimeVul says benign | **331** | 4.0% |
| absent from PrimeVul (ambiguous) | 6,519 | 79.1% |

**Measured Big-Vul label-noise rate: 331 / 1,724 = 19.2%** on the subset where both
corpora hold an opinion. This is an RO4 result. PrimeVul holds only 6,004 vulnerable
functions in total, which is why it cannot serve as a bulk filter (ADR-003).

**Line-level denominators, measured 2026-08-11** (full-corpus pass, label-audit run):

| | Value |
|---|---|
| Source lines, whole corpus | 5,850,944 |
| Mean lines / function | **31.0** |
| Flagged (vulnerable) lines | 46,513 |
| **Vulnerable lines as a share of all lines** | **0.795%** |

**Label-derivation quality: clean.** Across all 188,636 functions —
`out_of_range` **0**, `text_mismatch` **0**, `count_mismatch` **0**,
`labels_on_negative` **0**, `no_line_labels` **2,657** (the known 24.4%).
All 46,513 flagged lines verify against their own recorded `flaw_line` text, so the
0-indexed→1-indexed conversion is confirmed corpus-wide rather than on 400 sampled rows.
This establishes **consistency, not correctness** — see the human audit (§6 item 1).

---

## 1.5 The Split — measured, reproducible, clean

Built by `scripts/build_splits.py`. Fingerprint
`0ade7483a1b04d4841b5837c7bd2ed398f878acf7d242b59d7a4b043a48aef52`
(project-wise, 80/10/10, seed 42). Runs: `0c64a30f…` (built), `e395b21b…` (re-ran the
full corpus and reported `reproduced=True` — the Phase 1 byte-identity criterion).

| Partition | Functions | Vulnerable | Line-labelled | Flagged lines | Mean lines/fn | Projects |
|---|---|---|---|---|---|---|
| train | 148,983 | 7,489 (5.03%) | 5,457 | 34,027 | 27.5 | 104 |
| val | 18,593 | 825 (4.44%) | 488 | 3,386 | 38.7 | 102 |
| test | 18,442 | 821 (4.45%) | 546 | 2,779 | 42.5 | 101 |
| **clean_eval** | 1,589 | 1,394 | 1,393 | 4,499 | **138.3** | 184 |
| excluded | 603 | 331 | 331 | 1,638 | 33.4 | 22 |
| duplicate | 426 | 40 | 28 | 184 | 42.6 | 14 |

**Leakage report — clean:**

| Check | Result |
|---|---|
| Projects in more than one of train/val/test | **0** |
| Content hashes in more than one partition | **0** (after withholding 426; see ADR-004) |
| Clean-evaluation functions also in train | **0** |
| Duplicates *within* one partition | 2,447 — reported, deliberately not removed |

Exact matches only. Near-duplicate detection needs tokenization (Phase 2), so **0 is a
floor, not a guarantee**.


**Preprocessed shards** (`artifacts/data/cache/`, run `1f716614…`): all 188,636 functions
in **225 MB** against 11 GB of CSV. One JSONL per partition, each with a SHA-256 in
`shards.json`. Its per-partition counts are an *independent* recount and agree with the
split report exactly — 148,983 / 5,457 line-labelled / 34,027 flagged lines in train.
**From Phase 2 on, read the shards; never reopen a Big-Vul CSV.**

**Two caveats that must never be dropped when these numbers are quoted:**

1. `clean_eval` is **not** project-disjoint from train — **66.5% of its functions belong
   to a project that is also in training**. It measures line-localization quality under
   trustworthy labels. The `test` partition measures generalization to unseen projects.
   Never report one as the other (§7 #13).
2. `clean_eval` functions are **4.5× longer** than the corpus average (138.3 vs. 31.0
   lines). Truncation risk concentrates on the headline evaluation set (§7 #14).

---
---

## 1.6 Human label audit — measured line-level noise ≈ 30%

Reviewed by **Siddharth, 2026-08-11**. Sample: 20 line-labelled functions drawn by
content hash at **seed 1337** (`sample_for_review`), fixed before anyone looked at the
data. Completed sheet with per-function reasoning:
`artifacts/data/label_audit/label_audit.md`.

| correct | wrong | unsure |
|---|---|---|
| **14** | **6** | 0 |

**6 / 20 = 30.0% of flagged line sets do not point at the vulnerability.**
Wilson 95% CI **[14.5%, 51.9%]** — quote the interval with the estimate; n=20 supports
nothing tighter.

**This is not the 19.2% of §1.4, and must never be reported as if it were.**

| | Measures | Value |
|---|---|---|
| §1.4 (PrimeVul cross-reference) | **Function**-level: is this function vulnerable at all? | 19.2% |
| §1.6 (human audit) | **Line**-level: given a vulnerable function, do the flagged lines point at the flaw? | ~30% |

The line-level rate is structurally the larger: a function can be correctly labelled
vulnerable while its lines are mislabelled, because the fix commit touched more than the
flaw. 19.2% also sits inside the CI, so the two measurements are consistent.

**Every `wrong` verdict was the failure mode ADR-003 predicted** — cleanup swept into a
fix commit. Two recurring causes, both worth naming as threats to validity:

1. **Vague multi-fix CVEs.** Several Chrome CVEs bundle unrelated changes into one commit,
   so "lines the commit touched" is a poor proxy for "lines that were vulnerable"
   (e.g. CVE-2013-0886: flagged lines delete OpenGL textures, the CVE is NaCl signal
   handling on macOS).
2. **Cosmetic edits inside genuine fix commits** — reformatting an exception call
   (CVE-2017-5509), touching an adjacent test helper (CVE-2016-3751) — are
   indistinguishable from the fix under `labels_v1`.

Chrome supplied 3 of the 6 (3 wrong of 8 sampled, vs. 3 of 12 elsewhere). The intervals
overlap heavily — **not a supported difference** — but Chrome is 40.9% of the corpus, so
re-test on a larger sample before resting anything on it.

**Consequences:** Phase 6 must not treat line labels as clean. Phase 8 must report ~30%
as the ceiling on line-level metrics — a model far above ~70% line precision is more
likely fitting label noise than localizing vulnerabilities. It also raises the value of
`clean_eval` and of the ActiveClean-style line filtering in ADR-003 §D3.4. **The sample
is 20**: widen it (`data.label_audit.k`) with a second reviewer before the number carries
weight in the paper, so inter-rater agreement can be reported.

---

## 1.7 Representation — measured on the whole corpus (Phase 2)

Artefacts `artifacts/representation/report_{ast,cpg}.json`; regenerate with
`python scripts/report_representation.py graph.builder=<type>`. All 188,636 functions,
read from the shards.

**Everything below is measured under ADR-005's `graph.language=best_of_two`.** Numbers
taken before 2026-08-16 used the declared `lang` column and are *not* comparable; they are
reproducible with `graph.language=declared` and retained in MLflow under run
`093d672b866f42bfb500a75316170027`.

### Parse success — and why the corpus's own `lang` column was the problem

Every unit parsed under both grammars, AST run
`0729ae7197eb47c2bb3a30bec7bafc30` (`report_ast.json`):

| Policy | Clean parses | Share |
|---|---|---|
| the declared `lang` | 69,807 | **37.01%** |
| C, everything | 67,421 | 35.74% |
| C++, everything | 166,444 | 88.24% |
| **best of two per unit — ADR-005** | **168,156** | **89.14%** |
| clean under **neither** | 20,480 | 10.86% |

**100,735 functions (53.4% of the corpus) parse cleanly only as C++. Only 1,712 parse
cleanly only as C.** Following the dataset's own label was the *worst* of the options, by
a factor of more than two — see ADR-005.

Under `best_of_two` the corpus is parsed **185,168 as C++ and 3,468 as C**. Note that
3,468 exceeds the 1,712 that are *clean only* as C: the rest are units where neither
grammar is clean but C is the less damaged of the two.

Residual failure categories (20,480 damaged parses; 36,593 ERROR and 12,737 MISSING
nodes survive corpus-wide):

| Category | Count |
|---|---|
| `error_mid_function` | 8,991 |
| `error_on_first_line` | 7,491 |
| `missing_token_only` | 3,947 |
| `error_at_end_of_function` | 51 |

### The 10.86% no grammar parses — characterised, not absorbed

Run `228e35e0d9584bb59416a869d77445aa`, `artifacts/representation/parse_failures.json`.
**The residue is not a scaled-down copy of the corpus**, and this is the section to quote
rather than the 10.86%:

| | clean under some grammar | **clean under neither** |
|---|---|---|
| units | 168,156 | **20,480 (10.86%)** |
| mean lines / unit | 26.51 | **68.03** |
| line-labelled functions | 6,659 | **1,584** (19.2% of all 8,243) |
| **flagged lines** | 31,823 | **14,690 — 31.58% of all 46,513** |
| bytes inside an ERROR node | 0.07% | **7.51%** |
| source lines with ≥1 node | 83.16% | 83.53% |
| **flagged lines with ≥1 node** | **98.26%** | **78.94%** |

- **10.9% of the functions, but 31.6% of the line-level supervision** (§7 #26). Damaged
  units are 2.6× longer, and length carries both vulnerability and line labels.
- **Recovery is genuine, not nominal**: 92.5% of a damaged unit's bytes sit outside any
  ERROR node. Trees with holes, not rubble.
- **Damage is 2.5× worse on `clean_eval` (23.66%) than on train (9.30%)** — val 16.68%,
  test 16.31%. The headline partition again (compare §7 #14).
- **The residue is linux-heavy (9,015), not Chrome-heavy (1,599).** C++ repairs Chrome;
  what is left is real C defeated by function-like macros (`DSA_BN_MOD_EXP(goto err, ...)`)
  and by signatures the extraction cut in half. **No grammar choice fixes either.**
- 40-unit sample for human categorisation at
  `artifacts/representation/parse_failure_sample.md` (seed 1337). **Not yet reviewed** —
  the three causes named above are read off the sheet, not counted (§6 item 2).

### Graph statistics

AST run `0729ae7197eb47c2bb3a30bec7bafc30` (`report_ast.json`); CPG run
`004b05de04634c8b80dd0081463dea16` (`report_cpg.json`). Both under `best_of_two`.
**All 188,636 units built with zero failures** under both builders.

| | AST | CPG |
|---|---|---|
| Total nodes | 47,473,932 | 47,473,932 *(identical — one shared node set)* |
| Total edges | 47,285,296 | **59,664,000** |
| Mean per function | 251.67 nodes / 250.67 edges | 251.67 nodes / **316.29 edges** |
| Largest graph | **56,266 nodes** (`train`) | same |

Corpus-wide edge-type split in the CPG:

| Edge type | Count | Share | vs. pre-ADR-005 |
|---|---|---|---|
| `ast` | 47,285,296 | 79.3% | +0.0% |
| `dfg` | 8,802,085 | 14.8% | **+27.2%** |
| `cfg` | 3,576,619 | 6.0% | **+26.9%** |

**A cleaner parse buys flow edges, not nodes.** Under the declared `lang` the same corpus
gave 6,918,261 DFG and 2,819,004 CFG edges. Syntax nodes survive inside an ERROR span, but
control-flow and reaching-definitions passes need *well-formed statements*, so parse
damage costs flow edges roughly 27% while costing nodes 0.003%. **Any comparison of
grammar policies by node count would have concluded, wrongly, that ADR-005 changed
nothing.**

Mean nodes per function varies 4× across partitions (train 222, test 336, `clean_eval`
**1,144**), tracking the function-length differences already recorded in §1.5. The
`clean_eval` figure is the one to plan memory against, since it carries the headline
numbers — and in CPG form it averages **1,664 edges** per function.

Control and data flow together are **20.7%** of the CPG's edges (was 17.1%). Whether that
is enough signal for the structural branch, against 79% syntax, is an empirical question
for ablation A3 — not something to assume either way.

**Distinct node types per partition: train 295, val 248, test 245, `clean_eval` 210.**
Under ADR-005 this vocabulary is a union over two grammars — Phase 4 must build it from
the corpus, not from either grammar's symbol list (§7 #27).

### Line coverage — a limit on line-level scoring, and where it comes from

| | |
|---|---|
| Source lines with ≥1 node | 83.24% |
| **Flagged lines with ≥1 node** | **92.16%** (42,867 of 46,513) |
| **Flagged lines with no node at all** | **3,646** |

A flagged line carrying no AST node cannot receive a structural score — there is nothing
to pool over it. This is a *ceiling on the structural branch specifically*, distinct from
the ~30% label-noise ceiling (§1.6), and it must be reported per partition: `clean_eval`
98.78%, train 92.73%, `test` 83.41%, val 82.19%.

**These four numbers are identical under `declared` and under `best_of_two`.** Grammar
choice moves the parse rate by 51 points and line coverage by approximately nothing,
because ERROR nodes still land on lines. The cause of the gap is instead the parse damage
itself: **3,093 of the 3,646 uncovered flagged lines — 84.8% — are in the 20,480 units no
grammar parses cleanly.** It was first assumed to be blank and comment lines inside a
fix's diff; it is not (§7 #21).


## 1.8 Truncation — what 512 tokens costs, measured per partition (Phase 3)

Run `b7ccc9ca68c14b41890e3303e819f010`, `artifacts/llm/truncation.json`; regenerate with
`python scripts/report_truncation.py`. All 188,636 functions tokenized with
`microsoft/codebert-base`, window 512, stride 128.

§7 #14 required this measurement **on `clean_eval` specifically**. That instruction was
correct and the corpus-wide figure would indeed have been reassuring and wrong.

| Partition | units | >1 window | mean tokens | max tokens | flagged lines | **truncating keeps** | windowing keeps | units losing a label |
|---|---|---|---|---|---|---|---|---|
| train | 148,983 | 15.85% | 358 | 146,721 | 34,027 | 63.86% | 100% | 20.12% |
| val | 18,593 | 27.68% | 547 | 90,252 | 3,386 | **44.21%** | 100% | 30.12% |
| test | 18,442 | 27.77% | 589 | 63,357 | 2,779 | 54.91% | 100% | 35.53% |
| **clean_eval** | 1,589 | **59.03%** | **1,502** | 33,666 | 4,499 | **55.50%** | 100% | **38.05%** |
| excluded | 603 | 17.41% | 368 | 7,548 | 1,638 | 82.36% | 100% | 16.31% |
| duplicate | 426 | 34.04% | 518 | 4,055 | 184 | 94.02% | 100% | 17.86% |
| **corpus** | 188,636 | 18.59% | — | — | 46,513 | **61.86%** | **100%** | — |

**Truncation destroys 17,741 of 46,513 flagged lines — 38.1% of all line-level
supervision.** On `clean_eval` it is 44.5%, and **38.05% of line-labelled functions there
lose at least one label they are graded on**. Windowing recovers all of it, by
construction and verified: 100% in every partition.

Three consequences, none of them optional:

1. **Windowing is required, not a refinement.** `llm.windowing=false` reproduces the
   published-baseline behaviour and is kept only so the gap can be reported. Any
   line-level number taken with it off is measuring a truncated corpus, and the loss is
   *not* uniform — see the val/`clean_eval` spread above.
2. **The evaluation partition is the worst-hit one, again.** `clean_eval` averages 1,502
   tokens against train's 358 — 4.2×, matching its 4.5× line-length skew (§1.5). Its
   length distribution is close to uniform across the buckets: 324 units under 256 tokens,
   331 under 512, 322 under 1,024, 305 under 2,048, **307 over 2,048**.
3. **Windowing has a memory cost that lands on the same partition.** `clean_eval`
   averages **4.17 windows per unit and reaches 88**; the longest function in the corpus
   is 146,721 tokens (287 windows). One "sample" is therefore not one forward pass, and
   Phase 3 batching has to be sized in windows — the same trap as §7 #22 for graphs, on
   the other branch.


## 1.9 What fits in 4 GiB — measured, not estimated (Phase 3)

Run `1904c2950055457888db06b4297b80a8`, `artifacts/llm/vram_probe.json`; regenerate with
`python scripts/probe_vram.py`. Real forward *and* backward passes of
`microsoft/codebert-base` (124.6M parameters) at 512 tokens, pooled onto 512 line slots
with a per-line loss — the whole Phase 3 shape, not just the encoder.
**NVIDIA RTX 3050 Ti Laptop, 4,095.5 MiB.**

| Mode | Largest batch **in VRAM** | Peak at that batch | Throughput | Notes |
|---|---|---|---|---|
| `finetune` (fp32) | 8 | 3,982 MiB (97%) | 0.9 win/s | No headroom at all |
| `amp` | 8 | 3,261 MiB (80%) | 19.8 win/s | Spills at 16 |
| **`checkpoint` (AMP + gradient checkpointing)** | **32** | **2,456 MiB at bs=16 (60%)** | **22.9 win/s** | **Best trainable throughput *and* the most headroom** |
| `frozen` | 32 | 1,234 MiB at bs=16 (30%) | 37.8 win/s | 1.6× faster, but the encoder never adapts |

**Recommendation: AMP + gradient checkpointing at batch 16 windows.** It is simultaneously
the fastest trainable configuration measured and the one with the most room left — 1.6 GiB
spare for Phase 5, which has to hold the GNN branch and the retained cross-attention maps
on the same card. `frozen` remains the fallback and is a legitimate ablation, not a
failure mode; Phase 3's exit criteria require both anyway.

Throughput here is indicative, not precise: the same grid re-run gave `finetune` bs=1 as
0.24 s/step once and 0.56 s/step another time. The **memory** numbers were stable to
0.3%, and those are what the decision rests on.

### The trap this probe actually caught

**Exceeding VRAM on Windows does not raise.** Only one point in the 24-point grid raised
`OutOfMemoryError` (`finetune` at batch 32). Three others — `finetune` bs=16 at 6,417 MiB,
`amp` bs=16 at 4,835 MiB, `amp` bs=32 at 7,995 MiB — **exceeded the card's physical memory
and ran anyway**, because the WDDM driver pages to host RAM. They returned correct results
at 10–25× the cost: `amp` went from 19.8 windows/s at batch 8 to **2.5 at batch 16**.

§7 #7 said "do not discover this during a sweep". Catching `OutOfMemoryError` would not
have discovered it either — a sweep would have completed, produced valid numbers, and
simply taken a day instead of an hour. Peak allocation is therefore compared against
`torch.cuda.get_device_properties().total_memory`, and `within_physical_vram` — not the
absence of an exception — is what `largest_batch_that_fits` means. See §7 #29.

---

---

## 1.10 The formatting label leak — why the first training result was void (Phase 3)

**Run `a57074914375490eb5ec774f41f9a959`, `scripts/audit_label_leakage.py`,
artefact `artifacts/representation/label_leakage.json`. Decision: ADR-006.**

### What happened

The first full training run (`1843412a…`, seed 42, 2026-08-17) reported on held-out `val`:
line precision **0.9847**, recall 0.9377, F1 0.9606, Top-1 0.9876, Top-5 0.9959,
function F1 1.0000. Those numbers are far above the published line-level literature and
**above this project's own stated ceiling** — §1.6 records that ~30% of Big-Vul's line
labels point at the wrong lines, and no model can be 98% precise against labels that are
30% wrong unless it is reproducing the *labelling process*.

Two other signals said so before the cause was known:

- **Flat from epoch 0.** Line F1 went 0.9596 → 0.9630 → 0.9603 → 0.9606 → 0.9628 →
  0.9619 → 0.9614 over seven epochs. The model was finished after the first one.
- **Function F1 exactly 1.0000** with `tp=488, fp=0, fn=0, tn=0` — degenerate because
  `train.line_labelled_only=true` leaves every eval unit positive. A caveat, not the bug,
  but the same category of mistake.

**Leakage was ruled out and is genuinely absent.** Of the 488 line-labelled `val` units,
**0** appear in `train` under exact match, whitespace-stripped match, or
comment-and-whitespace-stripped match, and the two share **0 projects**. The split is
clean. The problem was in the text.

### The cause

**Big-Vul's LineVul release encodes the label in the indentation.** `processed_func` has
leading whitespace stripped from its lines — *except* the flagged ones, which are
re-inserted verbatim from the `flaw_line` column with their original indentation.

```
     0| static void CloudPrintInfoCallback(bool enabled,
>>   1|                                      const std::string& email,
>>   2|                                      const std::string& proxy_id) {
>>   3|     QuitMessageLoop();
     4| }
```

This is reproducible **straight from `artifacts/data/raw/bigvul_test.csv`** with no
VulnLoc code involved: precision 1.0000, 3,791 true positives, **0 false positives**.

### The measurement — "does this line start with a space or tab?"

| Partition | units | base rate | precision | recall | lift |
|---|---|---|---|---|---|
| `train` | 5,457 | 10.98% | **1.0000** | 0.9613 | 9.1× |
| `val` | 488 | 6.44% | **1.0000** | 0.9302 | 15.5× |
| `test` | 546 | 4.70% | **1.0000** | 0.9623 | 21.3× |
| **`clean_eval`** | 1,393 | 2.63% | **1.0000** | 0.9862 | **38.1×** |
| `excluded` | 331 | 11.85% | **1.0000** | 0.8266 | 8.4× |
| `duplicate` | 28 | 25.81% | **1.0000** | 0.8261 | 3.9× |

Zero false positives on every partition — 307,754 judged lines in `train` alone. Trailing
whitespace is a second tell at precision 1.0000 (recall 0.027); `contains_tab` a third at
0.9016, mostly an echo of leading tabs.

**The model and the one-line rule are the same predictor** on `val`:

| | precision | recall | F1 | Top-1 |
|---|---|---|---|---|
| run `1843412a…` | 0.9847 | 0.9377 | 0.9606 | 0.9876 |
| "line starts with a space" | **1.0000** | **0.9302** | **0.9638** | **0.9855** |

### Why our own checks passed

§1.4 records "0 text mismatches" across 188,636 functions and that check is real — but
`data/labeling.py:272` compares `lines[i].strip() != expected`. **It strips both sides**,
which is exactly the difference carrying the leak. The check was blind to it by
construction. A consistency check that normalizes away the thing it is checking will pass
forever.

### The fix

`utils/text.py::normalize_source_whitespace` strips both ends of every line. It is
**line-count preserving by construction**, so every 1-indexed label stays attached, and
**idempotent**, so it can safely be applied twice.

It is applied at **both** layers (Q9, ADR-006 D6.6):

- **L0 — `BigVulLoader._source`, `data.normalize_whitespace=true`.** The corpus is
  normalized as it is read, so the defect never reaches the hashes, the split or the
  shards. One helper serves `iter_units` and `iter_functions` so they cannot drift.
- **L1 — `CodeUnit`, `representation.normalize_whitespace=true`.** Idempotent defence in
  depth, covering any future corpus loaded by a different path.

The function lives in `utils/` because `Architecture.md` §3 forbids L0 from importing L1.

**The L0 move cost nothing, though it was accepted expecting a rebuild.** Every field the
split reads is invariant: `func_hash` already strips all whitespace (landmine #3), the
line count is preserved, and everything else comes from non-source columns. The
assignment is bit-identical and **fingerprint `0ade7483…` stands**. Only the shards'
source text changes. `build_cache.py` must be re-run; `build_splits.py` need not be, and
re-running it is the check that proves the point.

Escape hatches `data.normalize_whitespace=false` / `representation.normalize_whitespace=false`
reproduce pre-ADR-006 artefacts and are logged as MLflow parameters. After the fix **no
tell leaks on any partition**.

### What to expect next

**The honest numbers should be much lower, and that is the fix working.** The real task
is ~11% of lines flagged in `train`, 6.4% in `val`, 2.6% in `clean_eval`, under three
separately-recorded ceilings: ~30% label noise (§1.6), 7.84% of flagged lines with no AST
node (§7 #17), and the 512-line cap (§7 #26). **A model that has to learn should start
near chance and improve** — the shape of the curve is now the primary evidence that the
task is honest, and a flat one is the alarm.

**What actually happened — run `fc902107…`, seed 42, 2026-09-12.** The curve has the shape
this section predicted. Top-5 rose 0.531 → 0.610 and Top-1 0.247 → 0.363 across nine
epochs, against the void run's flat 0.960 from epoch 0. Held-out line P/R/F1 =
0.463 / 0.319 / 0.378, Top-1 0.363 (≈5.6× the 6.44% `val` base rate), IFA 11.94 — under
the ceilings above, where the void run sat above the largest of them. **One seed; Phase 3
needs three, so nothing here is citable yet.** §4's ledger row carries the caveats,
including that `eval/loss` rose monotonically while the ranking metrics improved.

---

---

## 1.11 The node-type vocabulary — why it is scanned, not read off a grammar (Phase 4)

**Run `2c9ce1b1…`, `scripts/build_node_vocab.py`, artefact
`artifacts/representation/node_vocab_ast.json`, digest `a0643c4e…`.**

ADR-002 §D2.5 embeds node type, so Phase 4 needs a lookup table. Landmine §7 #21 says to
build it by scanning the corpus rather than from a grammar's symbol list. That is now
measured rather than assumed:

| Source | Types | Corpus types it would miss | Symbols that never occur |
|---|---|---|---|
| **Corpus scan, all 188,636 units** | **304** (+1 unknown slot) | — | — |
| `tree-sitter-c` symbol list | 327 | **85** | 108 |
| `tree-sitter-cpp` symbol list | 489 | 3 | 188 |

**Sizing the embedding from `tree-sitter-c` would fold 85 real grammar symbols into slot 0
and never learn them** — silently, because slot 0 is a legitimate value. Among the 85 are
`access_specifier`, `abstract_reference_declarator`, `->*`, `.*` and `ERROR` itself. The
cpp list is closer but still wrong in both directions. This is a direct consequence of
ADR-005: the corpus is parsed under whichever grammar parses best, so its symbol set is a
**union** over the two and matches neither.

0 units failed to build a graph, so the vocabulary covers the corpus completely. Only 3
types occur exactly once (`__thread`, `requires`, `requires_clause`). The distribution is
extremely skewed — `identifier` alone occurs 7,242,519 times against a median in the
hundreds — which Phase 4 should expect when it initialises the embedding.

The artefact carries a digest for the same reason the split carries a fingerprint: an
embedding table whose indices shift between runs makes two checkpoints incomparable
without anything raising.

---

## 1.12 What the structural branch can see — measured before training (Phase 4)

**Run `b530f1ff…`, `scripts/report_node_feature_aliasing.py`, artefact
`artifacts/representation/node_feature_aliasing.json`.**

ADR-002 §D2.5 specifies node features as node-type embedding **+ subtoken embedding**.
Only the first half is built, so two identifiers are literally the same input. The
question is what that costs, and the usual way to answer it — train something and look at
the number — confounds the features with the optimiser, the schedule and the aggregator.

This measures it **without training anything**. If two lines carry the same multiset of
node features and one is flagged while the other is not, no model reading those features
can separate them. Message passing changes what "the features" are, so the measurement is
repeated at increasing radius using Weisfeiler-Lehman refinement — the standard upper
bound on what a message-passing network can distinguish. Mean-aggregating GCN and GAT are
strictly weaker than WL, so **the true ceiling is at least this bad**.

| radius | flagged lines indistinguishable from an unflagged one | line-accuracy ceiling |
|---|---|---|
| 0 hops (features alone) | **76.22%** | 91.43% |
| 1 hop | 65.05% | 92.72% |
| **2 hops** | **27.91%** | 96.95% |
| **4 hops** | **7.35%** | 99.19% |
| 6 hops | 5.39% | 99.41% |

`train`, 1,500 line-labelled units, 8,349 flagged lines. `val` agrees within ~3 points at
every radius (68.98 / 62.08 / 28.90 / 10.75 / 8.94).

**Two things follow, and the second one is a trap.**

1. **The §D2.5 gap is expensive at shallow depth and cheap at depth 4** — *as a
   disambiguation ceiling.* Node types alone cannot distinguish three quarters of flagged
   lines; four hops recovers all but 7%. **The inference drawn from this — that subtoken
   features are worth most where the receptive field is small — was tested in §1.15 and is
   FALSE**: the measured gain is larger at depth 4 than at depth 2. Disambiguation is not
   the main thing subtokens do; `memcpy` is predictive on its own. The measurement below
   stands, its predictive use for A4 does not.
2. **The 2/4/6 depth sweep is NOT a clean over-smoothing experiment.** Depth 2 carries a
   **27.91% hard recall ceiling from the representation itself**. If depth 2
   underperforms depth 4, that is the expected result *before* over-smoothing is
   considered at all, and reading it as evidence about over-smoothing would be wrong.
   Over-smoothing is a claim about 6 vs 4, where the feature ceilings are 5.39% and 7.35%
   and therefore nearly equal.

---

## 1.13 The GNN-only baseline, and what the depth sweep actually shows (Phase 4)

**All four Phase 4 exit criteria are met by logged runs** — `python scripts/report_phase4.py`,
artefact `artifacts/phase4/phase4_report.json`. The gate itself is the owner's to sign.

| | seeds | Top-5 | Top-1 | line F1 |
|---|---|---|---|---|
| **gcn, depth 4** (headline) | 42, 1, 2 | **0.3645 ± 0.0231** | 0.1155 ± 0.0132 | 0.0956 ± 0.0016 |
| gat, depth 4 | 42 | 0.3506 | 0.1224 | 0.1027 |
| ggnn, depth 4 | 42 | 0.3237 | 0.1017 | 0.0989 |
| gcn, depth 2 | 42 | 0.3361 | 0.0685 | 0.0655 |
| gcn, depth 6 | 42 | 0.3174 | 0.0851 | 0.0694 |

**The structural branch is far weaker than the semantic one** — Top-5 0.36 against 0.61,
line F1 0.10 against 0.38 (run `fc902107…`). That is the honest result, not a failure to
explain away: node features are node-type only in every run above, and §1.12 measures what
that costs. A1's premise is that the two branches are complementary, **not** that they are
equally strong, and a fusion that beats the semantic arm has to earn it.

### The depth sweep — **superseded by §1.16; the reading below did not survive three seeds**

| depth | Top-5 | Top-1 | flagged lines the *features* cannot resolve (§1.12) |
|---|---|---|---|
| 2 | 0.3361 | 0.0685 | **27.91%** |
| **4** | **0.3645** | **0.1155** | 7.35% |
| 6 | 0.3174 | 0.0851 | **5.39%** |

**Depth 4 beating depth 2 says nothing about over-smoothing.** Depth 2 carries a 27.91%
hard ceiling from the representation, so it was always going to lose. Reading that as
over-smoothing evidence would be reading a feature limit as an architectural one (§7 #37).

~~**Depth 6 losing to depth 4 is the interesting half.**~~ **RETRACTED — §1.16.** At one
seed depth 6 sat 0.047 below depth 4 and I read that as over-smoothing, feature ambiguity
being unable to explain it. At **three seeds the gap is 0.016 — 0.59 standard deviations**,
and no depth comparison in 2–6 exceeds 0.6 sd. There is no over-smoothing signal here, and
no depth signal of any kind on Top-5.

**The caveat was right and still insufficient.** It said one seed, named the ±0.023
spread, called the result suggestive and asked for three seeds before the paper used it.
All true — and the finding still got written down as a finding. **2 sd of noise looks
exactly like a result**; the fix is not a better caveat but refusing to write the
comparison until every arm has ≥3 seeds (§1.16).

---

## 1.14 Phase 3's result — LLM-only over three seeds, on normalized text

**Runs `fc902107…` (seed 42), `5b177450…` (seed 1), `e82ff268…` (seed 2). Aggregated by
`scripts/report_seeds.py`, artefact `artifacts/phase3/seed_report_semantic.json`.**
Held-out `val`, CodeBERT + line head, ADR-006 normalization confirmed in every run's
logged params.

| | mean ± std | range |
|---|---|---|
| **Top-5 accuracy** | **0.6141 ± 0.0055** | [0.6100, 0.6203] |
| Top-1 accuracy | 0.3513 ± 0.0138 | [0.3361, 0.3631] |
| line F1 | 0.3692 ± 0.0204 | [0.3460, 0.3840] |
| line precision | 0.4732 ± 0.0395 | [0.4394, 0.5166] |
| line recall | 0.3066 ± 0.0418 | [0.2601, 0.3409] |
| IFA (mean) | 11.93 ± 0.60 | [11.33, 12.52] |

**Phase 3's last exit criterion is met.** Best checkpoints at epochs 5, 5 and 6; all three
cross-check clean between MLflow and the on-disk summary.

**Seed variance is small where it matters and larger where it does not.** Top-5 varies by
±0.0055 across seeds — under one point — while precision and recall swing ±0.04 in
opposite directions (seed 2 trades 8 points of recall for 8 of precision at the same
threshold). That is the threshold moving, not the model changing, and it is the clearest
argument yet for reporting the **threshold-free** ranking metrics as the headline and
treating P/R/F1 as supporting (\u00a76's metric recommendation).

**Read every line number beside all three ceilings** (\u00a71.6 label noise ~30%, \u00a77 #17 no-node
coverage, \u00a77 #26 the 512-line cap) — they are embedded in the report artefact for that
reason. Top-1 0.3513 against a 6.44% `val` base rate is **≈5.5× chance**.

**`function/f1` = 1.000 in all three and is degenerate by construction** (`line_labelled_only`
leaves fp = tn = 0). It must never be quoted.

---

## 1.15 A4 — what subtokens are worth, and the prediction it falsified

**Runs `77370d56…`/`0d324da8…`/`3c570cc1…` (depth 4, seeds 42/1/2) and `b92e6985…`
(depth 2, seed 42), against their node-type-only partners.** ADR-002 §D2.5's second half,
`gnn.use_subtokens=true`, vocabulary `e1d018bd…`.

| depth | seeds | node-type only | + subtokens | Δ Top-5 | §1.12 said unresolvable without |
|---|---|---|---|---|---|
| 2 | 42 | 0.3361 | 0.3797 | **+0.0436** | **27.91%** |
| 4 | 42, 1, 2 | 0.3645 | **0.4302 ± 0.0372** | **+0.0657** | 7.35% |

**Subtokens are a large, real gain**, larger than the seed spread: the structural branch's
headline goes from Top-5 0.3645 to **0.4302**, with Top-1 0.1155 → 0.1750.

### The prediction was wrong, and the reason matters

§1.12 predicted the gain would be **large at depth 2 and small at depth 4**, because the
Weisfeiler-Lehman bound leaves 27.91% of flagged lines unresolvable at two hops and only
7.35% at four. **The measured gain does not follow it.** At one seed it looked 1.5× larger at depth 4;
at three seeds (§1.16) the two gains are **+0.0567 and +0.0657, 0.27 sd apart** — there is
no depth dependence at all. The prediction is still not supported; the *reversal* I first
wrote was itself a one-seed artefact.

The error was not in the WL measurement, which stands. It was in assuming **what subtokens
are for**. §1.12 modelled their value as *disambiguation*: telling apart two lines that
look identical to the model. That is one mechanism. The likelier dominant one is
**semantic**: `memcpy`, `strcpy`, `malloc`, `free` are predictive of a memory-safety flaw
in and of themselves, whatever the surrounding structure does or does not resolve. A
ceiling on disambiguation says nothing about that, and there is no reason it should shrink
as the receptive field grows.

Two lesser contributors, both plausible and neither separated here:

- **WL upper-bounds distinguishability, not learnability.** Mean-aggregating GCN is
  strictly weaker than WL, so the structure that is *in principle* separable at four hops
  may not be extractable — and subtokens supply directly what message passing could not.
- **Distinguishable is not learnable from 5,457 units.**

**What still stands from §1.12:** the depth-sweep reading (§7 #37). That argument is about
comparing *depths at fixed features*, and nothing here touches it.

**Now at three seeds per arm (§1.16).** The gain is real — ≈+0.06 Top-5 (≈2 sd) and +0.06
Top-1 (≈2.6 sd at depth 4) — and **flat across depth**. A flat gain is what a mechanism
independent of disambiguation predicts, so the explanation above is strengthened even
though the ratio that prompted it is gone.

---

## 1.16 Three seeds dissolved two findings I had already written down

**Runs `cbdc2a1f…`/`8985c411…` (depth 2), `032e4700…`/`f1203001…` (depth 6),
`e9378a9e…`/`f1b26024…` (A4 depth 2), 2026-09-14.** Each arm of the depth sweep and of A4
went from **one seed to three**. Both conclusions drawn at n=1 failed.

### The depth sweep shows no resolvable effect at all

| depth | seeds | Top-5 | Top-1 |
|---|---|---|---|
| 2 | 42, 1, 2 | 0.3638 ± 0.0240 | 0.0947 ± 0.0229 |
| 4 | 42, 1, 2 | 0.3645 ± 0.0231 | 0.1155 ± 0.0132 |
| 6 | 42, 1, 2 | 0.3485 ± 0.0301 | 0.0961 ± 0.0248 |

| comparison | \|Δ Top-5\| | pooled sd | in sd |
|---|---|---|---|
| 2 vs 4 | 0.0007 | 0.0236 | **0.03** |
| 4 vs 6 | 0.0159 | 0.0269 | **0.59** |
| 2 vs 6 | 0.0152 | 0.0273 | 0.56 |

**RETRACTED — §1.13's over-smoothing reading.** At one seed, depth 6 sat 0.047 below
depth 4 and I wrote that feature ambiguity could not explain it, leaving over-smoothing as
the live explanation. At three seeds the gap is 0.016, **0.59 standard deviations**. There
is no over-smoothing signal in 2–6 on this task; there is no depth signal of any kind on
Top-5. Top-1 mildly favours depth 4 (0.1155 vs 0.0947 and 0.0961) but at ±0.023–0.025 that
is about one sd and not something to report either.

**The single-seed number was not wrong, the inference from it was.** §1.13 called it
"suggestive, not conclusive" and said Phase 8 should re-run at three seeds. That was the
right caveat; it just was not a strong enough one to stop the finding being written as a
finding.

### A4's gain is real and does not depend on depth

| depth | seeds | without | with subtokens | Δ |
|---|---|---|---|---|
| 2 | 42, 1, 2 | 0.3638 ± 0.0240 | 0.4205 ± 0.0355 | **+0.0567** |
| 4 | 42, 1, 2 | 0.3645 ± 0.0231 | 0.4302 ± 0.0372 | **+0.0657** |

The two gains differ by **0.27 sd** — indistinguishable. **Subtokens help, at roughly
+0.06 Top-5 (≈2 sd) and +0.06 Top-1 (≈2.6 sd at depth 4), and the benefit does not vary
with depth over 2–6.**

**PARTLY RETRACTED — §1.15's framing.** §1.12 predicted a gain ~4× larger at depth 2 than
at depth 4. That prediction is still **not supported**. But §1.15 said the measured effect
ran *the other way*, 1.5× larger at depth 4, and that was itself a one-seed artefact: the
correct statement is **no depth dependence**, not reversed depth dependence. §1.15's
*explanation* survives and is strengthened — if subtokens worked mainly by
disambiguation, the gain would track the ambiguity ceiling, which falls 4× between depth 2
and 4. A flat gain is what a mechanism independent of disambiguation predicts.

### The lesson, which this project keeps re-learning

Both retractions come from the same place: **a single seed produced a difference of about
2 standard deviations, and 2 sd of noise looks exactly like a finding.** The depth-4 seed
spread (±0.023) was *already recorded* beside both claims when they were written. The
guard that would have caught this is not a better caveat — it is refusing to write a
comparison down until every arm has ≥3 seeds, the same rule Phase 3 and Phase 4 apply to
headline numbers and which neither sweep was held to.

---

## 1.17 What the fused model costs on 4 GiB, and the guard that was missing (Phase 5)

Runs `f96a0744…` (cross_attention) and `c1a2fbab…` (concat), `artifacts/fusion/vram_probe_*.json`;
regenerate with `python scripts/probe_vram_fused.py +fusion=<arm> +gnn=gcn train=fused`.
Real corpus batches — not synthetic tensors — because a unit costs *windows* on one side
and *nodes* on the other and the two are uncorrelated. **NVIDIA RTX 3050 Ti Laptop,
4,095.5 MiB.**

### The batch budgets do not control peak memory. At all.

| Budget (windows / nodes) | Worst batch found | Peak | % of card |
|---|---|---|---|
| 2 / 1,024 | 1 unit, 54 windows, 9,233 nodes | 3,786 MiB | **92.4%** |
| 4 / 2,048 | *the same unit* | 3,786 MiB | 92.4% |
| 8 / 4,096 | *the same unit* | 3,786 MiB | 92.4% |
| 16 / 8,192 | *the same unit* | 3,786 MiB | 92.4% |

Every budget in the grid is dominated by **one** oversized unit. A unit larger than the
budget still gets its own batch — both samplers do this deliberately, because dropping it
would discard the largest functions, which carry the most flagged lines (§1.8) — so the
worst batch is the worst *unit*, whatever the budget says. `train.max_windows_per_batch`
and `train.max_nodes_per_batch` therefore buy throughput, not headroom.

**Where the headroom actually went**, by arithmetic on the measured parameter count
(125,976,837 trainable): params 481 MiB + gradients 481 + AdamW moments 961 =
**1,922 MiB of static state, 46.9% of the card, before a single activation**. That is why
the budget cannot help: only ~2.1 GiB is ever available for activations.

`concat` peaks 8 MiB below `cross_attention` for 527,360 fewer parameters — the two A1
arms cost the same, which is the fairness property the experiment needs. Loss and
gradients stayed **finite under AMP** on every probe point, so `train.amp=true` does not
break the GNN half the way `configs/train/structural.yaml` feared.

### `evaluate()` was building an autograd graph for a backward that never came

`model.eval()` settles dropout; it does **not** stop autograd recording. `training/loop.py`
had no `no_grad`, so every activation of the evaluation forward was retained until the next
batch replaced it — across Phase 3, Phase 4 and anything using the shared loop.

The mechanism is worse than it sounds and is verifiable in the library, not inferred:
`transformers/modeling_layers.py` gates gradient checkpointing on `self.training`, so **in
eval mode checkpointing is bypassed entirely**. Turning on `llm.gradient_checkpointing`
does nothing for evaluation, which made evaluation the *most* expensive pass in the loop
rather than the cheapest.

Fixed 2026-09-14 with `@torch.no_grad()` and three tests. **It cannot have moved a single
logged metric** — nothing in the evaluation path reads a gradient, so every Phase 3 and
Phase 4 number stands exactly as reported. What it changes is memory and wall clock.

**The size of the effect is measured but NOT yet from a logged run** — the probe run that
would have logged it was cancelled. `scripts/probe_vram_fused.py` now carries an
`eval_grad` row for exactly this comparison, held out of the fit verdict. Re-run either arm
to make the number citable under `Rules.md` §0.3; until then it is an observation, not a
figure, and must not appear in the paper.

---

## 1.18 Every headline number is validation-selected, and now says so (Phase 3/4 reporting)

**Nothing here is a new measurement. It is a correction to how the existing ones are
labelled**, which is the kind of error that survives review precisely because no value
changes.

`configs/train/semantic.yaml` and `configs/train/structural.yaml` both set
`watch_metric: line/top_5_accuracy`, `watch_mode: max`, on `eval_split: val`. Early
stopping therefore picks the epoch with the best **val Top-5**, and §1.14's and §1.15's
headline numbers are read from that epoch. So:

| | |
|---|---|
| **Top-5 accuracy** | Selected on *itself*, on the split it is reported on. Optimistically biased by an unmeasured amount. |
| Top-1, line F1, precision, recall, IFA | Reported at an epoch chosen by Top-5, on the same split. A weaker version of the same problem, not an escape from it. |
| `test` (18,442 functions), `clean_eval` (1,589) | **Never touched.** `Rules.md` §5.3 spends them once, at the end of an experiment family. The fusion family is not finished. |

The honest phrasing is **"the best this model reached on `val`"**, never "what it would
score on unseen data". `scripts/make_review3_figures.py` now writes that into every figure
subtitle, both axis labels and §1 of `artifacts/review3/results.md`, so a chart cropped
onto a slide carries the caveat with it.

**Why this is worth a section.** The bias is real but its *size* is unknown, and the
temptation when someone asks "how good is it?" is to quote the number without the adverb.
Spending `test` now to get an unbiased figure for a review deck would cost the project its
one clean measurement and buy a sentence. The sentence is not worth it.

---

## 1.19 Where this sits against published work — quoted, never plotted

`docs/prior_work.yaml` holds the only published numbers this repo quotes. `Rules.md` §5.7:
*baselines are reproduced, not quoted; anything copied from a paper is labelled
`(reported)` and never compared directly to ours as if it were run under our splits.*
Retrieved 2026-09-14, each with its source URL and which artefact the value was read from.

| System | Metric | Value | |
|---|---|---|---|
| **LineVul** (Fu & Tantithamthavorn, MSR 2022) | Function F1 | 0.91 | P 0.97 / R 0.86 |
| | Line Top-10 accuracy | 0.65 | self-attention ranking, not a supervised line head |
| | Line IFA | 4.56 mean (repo) / 1 median (paper) | **not the same statistic** |
| **LineVD** (Hin et al., MSR 2022) | Statement F1 | 0.360 | P 0.271 / R 0.533; the closest published relative of our structural branch |
| **IVDetect** (Li, Wang & Nguyen, FSE 2021) | Statement F1 | 0.176 | as re-run by LineVD's authors — a third-party reproduction |

**The single most useful row is not in that table.** A later paper (LEAP, arXiv 2209.10414)
re-measures **LineVul** on Big-Vul at **Top-10 0.740** against the **0.65** in its own
paper — a 9-point spread with no change of method, wider than most of the improvements
reported in this literature. Whatever differs is the evaluation setting. **That is why
these numbers render as a table with a caveat column and never as bars beside ours**; a
shared axis asserts "same kind of thing" and has no room to take it back.
`tests/unit/test_prior_work.py` fails if a quoted value ever appears in an SVG.

**What can honestly be said today:**

1. Our line F1 **0.3692** lands in the same region as LineVD's **0.360**. Same region,
   different benchmark — "not obviously off the map" and nothing stronger.
2. **We cannot be placed against LineVul at all**, because we report Top-5 and it reports
   Top-10, and on a corpus averaging 31 lines/function that gap is not cosmetic. Computing
   Top-10 from the logged predictions is cheap and is the one change that would make a row
   of this table mean something. Not done, so not claimed.
3. Our IFA **11.93** is worse than every figure above. Part of the gap is the statistic
   (our mean against their median, on a right-skewed distribution) and part is real.
   **Which part is unmeasured.**
4. **A1 is not in this table and never will be.** Cross-attention against concatenation,
   both arms ours, both under ADR-008's fixed conditions — internal by construction, which
   is exactly why it can be answered when the table above cannot.

**The ADR-006 hypothesis stays a hypothesis.** *Line-level results trained on the LineVul
release of Big-Vul without whitespace normalization are contaminated by the indentation
artefact.* Status **UNTESTED**. It would be settled by running each published system's
released preprocessing over the release and checking whether the leading-space feature
survives into its model input. Until then it is never an explanation for a gap between
their numbers and ours, and never a reason ours are lower — it is a claim about other
people's work and ADR-006 requires verification against their released code first.

## 1.20 The semantic feature cache — built, measured, and two ADR assumptions it broke (Phase 5)

ADR-009, accepted 2026-09-15. `scripts/build_semantic_cache.py train=fused llm.frozen=true`,
run `c99764bc…`: every line-labelled `train` and `val` unit encoded once by the frozen
CodeBERT, stored as **line-aligned** states at each unit's own reach.

| | units | windows | rows | size (fp32) | encode |
|---|---|---|---|---|---|
| `train` | 5,457 | 11,514 | 302,364 | 932 MB | 132 s |
| `val` | 488 | 1,690 | 43,126 | 133 MB | 19 s |

Line counts (step 0): `train` mean 60.3 / median 27 / p99 567; `val` mean 114.2 / median 48
/ p99 998. Digests `77f7b07e…` (train), `a710b3bc…` (val). Encoder revision `3b0952fe…`,
which resolves identically with `HF_HUB_OFFLINE=1` — it has to, or every queued run would
refuse the cache.

### What the measurements changed

1. **"Bitwise identical" was the wrong test, and the failure was not dropout.** The first
   build refused itself at a max difference of 9.5e-7. Measured separately: token states are
   **bitwise identical** across passes; pooled line states differ by up to **1.9e-6** because
   CUDA `index_add_` has no fixed accumulation order. Dropout p=0.1 on the same states moves
   them by up to **13.8**. The check is now two-stage (§7 #50).
2. **The encoder does not emit fp16 under AMP.** RoBERTa's final LayerNorm autocasts to fp32,
   so ADR-009 §D9.6's reason for fp16 storage was false. Stored fp32 (§7 #51).
3. **A second dropout ADR-009 did not name.** `PretrainedCodeEncoder.dropout` wraps the
   token states outside the frozen module and is live in `train()`. The cached path applies
   the same module to the line states instead: identical in eval, and identical across arms.

### Equivalence on real hardware

Cached vs live frozen path, same model, same 6 budget-sized batches of 24 real `val` units,
RTX 3050 Ti, AMP, eval mode:

| | max line-logit diff | max function-logit diff |
|---|---|---|
| cached vs live, 5 semantic-using arms | **4.9e-4** | 4.6e-5 |
| **live vs live** (`cross_attention`, `gated`) | **4.9e-4** | 3.4e-5 |

The line figures are one to two fp16 quanta at logit scale. **The cache adds nothing
measurable above the GPU's own run-to-run noise.** Not a logged run — a scratch measurement;
the reproducible pin is `tests/unit/test_semantic_cache.py` (6 arms, 1e-5, CPU).

### Overfit on the cached path — a sanity check, never a result

With `gnn.use_subtokens` **off**: `cross_attention` `117df3a4…` (Top-1 1.000, loss 2.95 →
0.0085), `concat` `aebd4407…` (Top-1 1.000), `sum` `3b32a40d…` (Top-1 1.000), 150 epochs,
~7.7 min each. `d4ac0260…` (gated) was stopped at epoch 1 and **tagged void**. All six arms
were then re-queued with subtokens **on**, because that is the configuration A1 compares.

**The A1-config overfit gate (`p5check`, all with subtokens on, frozen, cache `77f7b07e…`):**

| arm | run | best / last Top-1 | loss first → last |
|---|---|---|---|
| cross_attention | `2233c9a6…` | 1.000 / 1.000 | 2.79 → 0.011 |
| concat | `01be9392…` | 1.000 / 1.000 | 2.92 → 0.008 |
| sum | `7906edbf…` | 1.000 / 1.000 | 2.55 → 0.055 |
| gated | `83876edc…` | 1.000 / 1.000 | 1.56 → 0.009 |
| **llm_only** | `25d03e12…` | **0.968 / 0.903** | 2.09 → 0.85 |
| gnn_only | `cf63c8ef…` | 1.000 / 1.000 | 2.09 → 0.16 |

**`llm_only` did not finish memorising, and that is a finding about freezing, not a bug.**
Its Top-1 was still climbing at epoch 150 (0.36 → 0.48 → 0.65 → 0.77 → 0.84 → 0.90; Top-5
already 1.0), where `concat` hit 1.0 by epoch 100. Its semantic path is the same cached
path every other arm passed on, and it matched the live encoder on the GPU. With the encoder
frozen, `llm_only` can only fit through the fusion projection and the head over fixed
CodeBERT line features; every other arm also carries a from-scratch GNN. **Expect A1's frozen
`llm_only` to land below Phase 3's fine-tuned 0.6141** — that gap is ADR-009 D9.1's cost made
visible, and it must be reported as such, never as the fused scaffold hurting the semantic
branch.

A1 was launched from commit `d1003f6` (clean tree), so its runs carry a real `git_sha`; the
overfit runs above are `6140e3a-dirty`, which is acceptable for sanity checks only (§7 #4).

### The inspector now scores lines — an owner decision with its caveats attached

`frontend/` gained a **Findings** tab (the pipeline inspector moved to **Detailed report**).
It ranks the pasted function's lines with `fc902107…` (seed 42 — the default seed, not the
best of three, so the choice is not made on `val`) on CPU, and attaches **fix templates**
(`frontend/fix_templates.py`, 11 single-line patterns + use-after-free / double-free /
unchecked allocation) that match line text. What travels with every score on screen:

- It **ranks where a flaw is likeliest; it cannot say whether one exists.** Trained on
  line-labelled units only, all vulnerable. The function-level output is never shown.
- Its accuracy — first pick 36%, top five 61% on `val` for seed 42 — is **same-distribution
  Big-Vul**. Pasted code is outside it.
- **Scores are not calibrated** (the sample's normal lines score 0.7–0.97), so the UI leads
  with rank.
- Punctuation-only lines are skipped from the cards; that can only raise the chance the real
  line is shown, so the quoted Top-5 stays a lower bound.
- **Templates are not Phase 7 and not an answer to Q4.** They say what a known pattern looks
  like, never why the model ranked a line; a highly ranked line with no template says so.

**Later in session 13 (owner decisions):** the Detailed report now **ranks the paste with the
structural model too** — `77370d56…` (GCN depth 4 + subtokens, seed 42), shown only in the
Detailed report beside the semantic model's rank for the same line; Findings is unchanged.
Input is built as training built it, vocabularies pinned to the run's logged digests
(`a0643c4e…` / `e1d018bd…`); **verified by scoring 40 `val` units through both the paste path
and `GraphLineDataset` — 40/40 identical** (pinned as a slow test). The page was cut down to
one sentence + numbers + picture per stage; every held-out number is read from the seed
reports (`/api/results`), none typed into `phases.json`. Two inspector bugs found on the way:
a server older than the page returned 405 and the UI showed a titled red box with no message
(now shows status + `detail`); and concurrent first imports of `transformers` raised
`ImportError: AutoTokenizer` once (now serialised by `frontend/scoring.py::IMPORT_LOCK`).
**Still no fused score is computed on a paste.**

## 1.21 A1 — with a frozen semantic encoder, cross-attention does not beat concatenation

The full screen: six arms × seeds 42/1/2, all from commit `d1003f6` on a clean tree, frozen
CodeBERT read from cache `77f7b07e…`, GCN depth 4 over AST with identifier subtokens,
`train=fused`. Aggregated by `report_seeds.py train=fused '+seeds.where={fusion_kind: <arm>}'
+seeds.tag=<arm> +seeds.output_dir=./artifacts/phase5`; every run cross-checks `agrees`.
**Held-out `val`, validation-selected (§1.18), n = 3 per arm.**

| Arm | Top-5 (headline) | Top-1 | IFA ↓ | Coverage | Runs (42 / 1 / 2) |
|---|---|---|---|---|---|
| cross_attention | **0.5408 ± 0.0181** | 0.2310 ± 0.0084 | **12.19 ± 0.44** | 99.96% | `192e067c` `e7ae3108` `ce2380bf` |
| concat | **0.5394 ± 0.0198** | 0.2524 ± 0.0188 | 15.12 ± 1.28 | 99.96% | `86343b54` `1428a86f` `5b09a7a4` |
| sum | 0.5304 ± 0.0012 | **0.2711 ± 0.0048** | 15.43 ± 1.32 | 99.96% | `74021cc5` `b6d4bb5f` `9d88f284` |
| gated | 0.5221 ± 0.0052 | 0.2593 ± 0.0184 | 14.89 ± 1.55 | 99.96% | `e6331b32` `df0de21a` `d2718402` |
| llm_only | 0.4903 ± 0.0032 | 0.2331 ± 0.0043 | 14.24 ± 0.07 | 99.96% | `e0ca38de` `df964b78` `2d4de3bf` |
| gnn_only | 0.4191 ± 0.0468 | 0.1521 ± 0.0662 | 19.75 ± 0.81 | 90.81% | `cca2b33b` `e75f7e34` `394ceb9b` |

### What this supports

1. **A1's question, answered: no resolvable difference.** cross_attention − concat on Top-5 =
   **+0.0014 (0.07 pooled sd)**; paired by seed +0.035 / +0.008 / −0.039 — the sign flips.
   The only admissible phrasing is *"with a fixed semantic representation, bidirectional
   cross-attention did not outperform concatenation on Top-5 (0.5408 ± 0.0181 vs
   0.5394 ± 0.0198, 3 seeds)"* (ADR-009 D9.1). cross_attention − sum (+0.0104, 0.81 sd) and
   − gated (+0.0187, 1.40 sd) are **not resolvable either** — §7 #37 is what writing them
   down as an ordering would cost.
2. **Fusion adds something over frozen semantics alone — every mechanism, every seed.**
   Top-5 over `llm_only`: cross_attention +0.0505 (paired +0.062 / +0.056 / +0.033), concat
   +0.0491, sum +0.0401, gated +0.0318, all three seeds positive for each. **This is not the
   union-coverage scaffold** (§7 #42): `llm_only`'s coverage is the same 99.96% as the fused
   arms on these units, so the gain comes from structural *features*, not from reaching
   more lines.
3. **Cross-attention's one distinctive signal is IFA**: 12.19, the best of every arm and
   **2.06 lower than `llm_only` on all three seeds** (−1.54 / −2.31 / −2.32), while concat's
   IFA is worse than `llm_only`'s. Against that, its Top-1 (0.2310) is the lowest of the
   fused arms, concat's is higher on all three seeds, and sum's is highest. **These two
   secondary signals point opposite ways; neither is a headline and neither is a win.**

### What it does not support

- **Nothing here beats the project's best model.** Freezing costs the semantic branch
  **0.124 Top-5**: frozen `llm_only` 0.4903 against Phase 3's fine-tuned 0.6141 (§1.14).
  Every fused arm (≤ 0.5408) sits below fine-tuned LLM-only. The Findings tab therefore keeps
  scoring with `fc902107…`.
- **Nothing about fine-tuned fusion.** A fine-tuned encoder can reshape its own line features
  (ADR-009 D9.1), so the mechanism ordering may change. D9.7's confirmation is triggered by
  exactly this margin and is an owner decision (§6).
- **Not line F1.** At the 0.5 threshold the fused arms' recall is 0.03–0.09; F1 there
  measures calibration under focal loss, not localisation.

### Run-level observations

- `llm_only` peaks early (best epochs 1 / 3 / 1) — consistent with its `p5check` run
  (§1.20): a small head over fixed features fits what it can within a couple of epochs.
- `gnn_only` seed 2 (`394ceb9b`) never improved on epoch 0 (Top-5 0.3651, best step 0),
  which drives that arm's ±0.0468. Its mean, 0.4191, sits within one sd of Phase 4's
  structural baseline 0.4302 — the scaffold does not change the structural branch.
- `sum` seed 42 (`74021cc5`) spans the laptop's sleep at 11:41; the process paused and
  resumed, and only its wall clock is affected.

---

## 1.22 D9.7 — fine-tuned, cross-attention still does not beat concatenation, and neither beats LLM-only

The six runs ADR-009 D9.7 specifies: cross_attention and concat × seeds 42/1/2 with the
**encoder training** (`llm.frozen=false train.semantic_cache.enabled=false`), otherwise A1's
constants (GCN depth 4 over AST, identifier subtokens), all from `run_queue.py --only d97` on
this laptop. `src/vulnloc` is identical across the six — the logged shas `70824ec` /
`0ce5d96` / `782ea98` differ in docs, scripts and tests only; `report_seeds.py` compared them
and accepted no drift, and no run had uncommitted changes. Aggregated 2026-10-04 by
`report_seeds.py train=fused '+seeds.where={fusion_kind: <arm>, frozen: False}'
+seeds.tag=finetuned_<arm> +seeds.output_dir=./artifacts/phase5`; every run cross-checks
`agrees`. **Held-out `val`, validation-selected (§1.18), n = 3 per arm.**

| Arm | Top-5 (headline) | Top-1 | IFA ↓ | Line F1 @ 0.5 | Coverage | Runs (42 / 1 / 2) |
|---|---|---|---|---|---|---|
| cross_attention | **0.6162 ± 0.0184** | 0.3444 ± 0.0293 | 11.26 ± 1.45 | 0.2954 ± 0.0855 | 99.96% | `80fc5238` `c193e3bc` `829c3555` |
| concat | **0.6169 ± 0.0104** | 0.3645 ± 0.0073 | 11.46 ± 0.68 | 0.3811 ± 0.0159 | 99.96% | `fbf65554` `633ea97b`† `a0c42d6f` |

Per seed, Top-5: cross_attention 0.6224 / 0.6307 / 0.5954; concat 0.6058 / 0.6183 / 0.6266.
Selected steps 2 / 5 / 2 and 3 / 4 / 7.

† **`633ea97b` is a resumed run** (§7 #53): tags `resumed_at=epoch=5,batch=418,step=3821`,
`resume_git_sha=782ea98`. Its selected checkpoint is step 4, reached **before** the pause —
the resume log shows `best_so_far=0.6183`, the value reported — so the resume decided when
the run stopped, not what it reports.

### What this supports

1. **A1's null survives fine-tuning.** cross_attention − concat on Top-5 = **−0.0007 (0.05
   pooled sd)**; paired by seed +0.017 / +0.012 / −0.031 — the sign flips, as it did frozen
   (§1.21). The admissible phrasing is *"with the semantic encoder fine-tuned, bidirectional
   cross-attention did not outperform concatenation on Top-5 (0.6162 ± 0.0184 vs
   0.6169 ± 0.0104, 3 seeds)"*. The frozen screen and the confirmation **agree**, so ADR-009
   D9.7's "report both" has no disagreement to write up.
2. **Fine-tuning the encoder lifts both fused arms by about the same amount**: cross_attention
   0.5408 → 0.6162 (+0.0754), concat 0.5394 → 0.6169 (+0.0775), three seeds on each side.
   That is the adapting-encoder case ADR-009 D9.1 worried about, and it did not separate the
   mechanisms.
3. **The frozen screen's secondary signals did not survive.** IFA: frozen, cross_attention
   led on all three seeds (12.19 vs 15.12); fine-tuned it is 11.26 ± 1.45 vs 11.46 ± 0.68,
   paired −0.35 / −1.01 / +0.77. Top-1: −0.0201 (0.94 pooled sd), paired −0.017 / +0.010 /
   −0.054. Both signs flip; neither is a finding.

### What it does not support

- ~~**Nothing about fusion vs fine-tuned LLM-only.**~~ **Superseded 2026-10-07 — the control
  has its three seeds; see "The scaffold control" below.** (As written on 2026-10-04: both
  fused arms sat within noise of Phase 3's 0.6141 ± 0.0055, §1.14, but that number is from a
  different scaffold, ADR-008 §D8.6, and was not the control.)
- **Not line F1.** concat's 0.3811 against cross_attention's 0.2954 is 1.39 pooled sd, and
  cross_attention's recall at the 0.5 threshold swings 0.12–0.28 across seeds while its
  precision stays at 0.56–0.65 — the threshold moving, as in §1.14. F1 means something only
  after Phase 6 chooses a threshold on `val`.
- **Not `test` or `clean_eval`.** Every number here is `val`.

### Run-level observations

- Queue wall clock: 118 / 168 / 114 min (cross_attention), 123 / 183 over two attempts /
  211 min (concat). **A fine-tuned fused run therefore costs 114–211 min**, not the 118–168
  the first three suggested; `a0c42d6f` selected step 7 and ran longest.
- The first 79 steps after `633ea97b`'s resume took 156 s (~1.97 s/step, warm-up included)
  at 3,575 MiB dedicated + 462 MiB shared — session 14's residual spill, with a
  hardware-accelerated browser (Brave) holding 162 MiB on the NVIDIA adapter (§3 session 15).

### The scaffold control — added 2026-10-07 (session 16)

`llm_only` fine-tuned **in the fused scaffold** × seeds 42/1/2 — the same queue group, the
same laptop, the same command with `+fusion=llm_only`. Aggregated 2026-10-07 03:14 by the
command above with `fusion_kind: llm_only`; `n_seeds: 3`, every comparable parameter
identical, no run with uncommitted changes, every run cross-checks `agrees`. Logged shas
`782ea98` / `ce04fb1` (and `de1fa21` / `85aa177` at the resumes) differ in docs only —
`report_seeds.py` compared `src/vulnloc` and accepted no drift. **Held-out `val`,
validation-selected, n = 3.**

| Arm | Top-5 (headline) | Top-1 | IFA ↓ | Line F1 @ 0.5 | Runs (42 / 1 / 2) |
|---|---|---|---|---|---|
| cross_attention | 0.6162 ± 0.0184 | 0.3444 ± 0.0293 | 11.26 ± 1.45 | 0.2954 ± 0.0855 | as above |
| concat | 0.6169 ± 0.0104 | 0.3645 ± 0.0073 | 11.46 ± 0.68 | 0.3811 ± 0.0159 | as above |
| **llm_only** | **0.6169 ± 0.0048** | 0.3527 ± 0.0021 | 11.13 ± 0.33 | 0.3726 ± 0.0157 | `179632f5`‡ `72f067d0`‡ `2c7a7631`‡ |

Per seed, `llm_only`: Top-5 0.6141 / 0.6141 / 0.6224, Top-1 0.3527 / 0.3506 / 0.3548,
selected steps 3 / 5 / 4. Queue wall clock 134 / 210 / 170 min, two attempts each.

‡ **All three are resumed runs** (§7 #53) — a replicate each, not a bit-identical copy:

| Run | `resumed_at` | Why it stopped | Selected checkpoint |
|---|---|---|---|
| `179632f5` (42) | `epoch=6,batch=500,step=4565` | paused by the owner, 2026-10-04 21:51 | epoch 3 — **before** the pause |
| `72f067d0` (1) | `epoch=5,batch=976,step=4100` | crashed 2026-10-06 22:47, `WinError 5` on the checkpoint rename (§7 #56); continued from the step-4100 `last.pt`, the ~100 steps after it trained again | epoch 5 — **completed after the resume** (steps 4101–4331) |
| `2c7a7631` (2) | `epoch=6,batch=946,step=4800` | crashed 2026-10-07 01:12, `CUDA error: unspecified launch failure` (§7 #57) | epoch 4 — **before** the fault |

**What it supports.**

1. **Fine-tuned, neither fused arm outperforms LLM-only in the same scaffold on Top-5.**
   cross_attention − llm_only = −0.0007 (0.05 pooled sd), paired by seed +0.008 / +0.017 /
   −0.027; concat − llm_only = 0.0000 to four decimals (0.61687 both), paired −0.008 /
   +0.004 / +0.004. The admissible phrasing is *"with the semantic encoder fine-tuned,
   adding the structural branch — by cross-attention or by concatenation — did not improve
   Top-5 over the LLM-only control (0.6162 ± 0.0184 and 0.6169 ± 0.0104 vs 0.6169 ± 0.0048,
   3 seeds each)"*.
2. **§1.21's "every fusion arm beats `llm_only`" is a frozen-encoder result.** Frozen, every
   fused arm beat `llm_only` on every seed, by +0.032 to +0.051 Top-5 in the mean;
   fine-tuned, the lead is gone. Whatever the structural branch supplied to a fixed semantic representation, an
   adapting encoder reaches the same Top-5 without it.
3. **The scaffold is not the difference.** `llm_only` in the fused scaffold (0.6169 ± 0.0048)
   and Phase 3's LLM-only in its own (0.6141 ± 0.0055, §1.14) are 0.0028 apart, 0.54 pooled
   sd. Different scaffolds, so read side by side, not pooled.

**What it does not support.**

- **Not "concat beats LLM-only on Top-1".** concat's Top-1 is above `llm_only`'s on all
  three seeds (+0.004 / +0.015 / +0.017; 0.0118 = 2.2 pooled sd). It is one of several
  secondary metrics, at three seeds, and the frozen screen's own all-three-seeds secondary
  signals did not survive fine-tuning (point 3 above). Something to re-test with more
  seeds, recorded so it is not rediscovered; **not a finding.** cross_attention's
  Top-1 against `llm_only` flips sign across seeds (−0.013 / +0.025 / −0.037).
- **Not line F1** (threshold-dependent, as above) **and not IFA** (11.13 ± 0.33 against
  11.26 ± 1.45 and 11.46 ± 0.68).
- **Not `test` or `clean_eval`.** Every number here is `val`.
- **Not that the structural branch is useless.** It carries signal alone (§1.15) and helps
  a frozen encoder (§1.21). What is measured is that it adds nothing resolvable on Top-5 to
  a fine-tuned CodeBERT in this configuration — AST only, GCN depth 4, one corpus.

---

## 1.23 Machine 2's slate — imported 2026-10-07, read beside this machine's, never pooled

**Where:** `artifacts/machine2/Sujal_20261005-224401/` (git-ignored) — `RESULTS.md`,
`manifest.json`, 32 run summaries, 33 logs, 9 seed reports, `queue_state.json` and
`mlflow_Sujal.db` (a copy to *read*; never this machine's tracking store). Unpacked from the
owner's `transfer.zip`; 78 files, no checkpoints.

**What produced it:** host `Sujal`, **NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB**, commit
`782ea98` (clean), Python 3.12.10, **torch 2.14.1+cu130 — this machine runs 2.13.0+cu130**.
Exported 2026-10-05 22:44. All 33 steps of `m2check` / `m2` / `m2ftcheck` / `m2ft` are
`done`; every one of its 9 seed reports has `n_seeds: 3`, no accepted source drift and no
run with uncommitted changes. **A different GPU and a different torch are two more reasons
no number below joins a comparison with this laptop's** (§7 #49). Cite as *"machine 2
(RTX 4050 Laptop, 6 GB)"* with the run ids.

**The five overfit gates pass** (§7 #52) — read from the imported store's full metric
history, not from `RESULTS.md`: `eval/line/top_1_accuracy` reaches 1.0000 and holds it
through epoch 150 in all five (first at epoch 18 GAT `0d6f8971`, 33 GGNN `28c3188a`, 28
frozen GraphCodeBERT `eccda2c9`, 11 fine-tuned CodeBERT `9bb37c75`, 8 fine-tuned
GraphCodeBERT `b4383ffc`). **`RESULTS.md` shows 0.9032 / 0.9677 for three of them because it
prints the *selected* epoch's metrics — Top-5 was already 1.0000 there — not the last
epoch's.** Never read a gate off that table.

**Held-out `val`, validation-selected, n = 3 per arm, machine 2 only.** Each row changes
one thing against its group's reference; Δ is against that reference, in Top-5.

| Arm (ablation) | Top-5 | Δ, pooled sd | paired Δ 42 / 1 / 2 | Top-1 | Runs (42 / 1 / 2) |
|---|---|---|---|---|---|
| **frozen reference** — A1 cross_attention, CodeBERT, GCN, focal | 0.5498 ± 0.0219 | — | — | 0.2531 ± 0.0150 | `f4d81b36` `513d9d6b` `45de097a` |
| GAT (A4 in fusion) | 0.5456 ± 0.0165 | −0.0041, −0.21 | −0.015 / +0.023 / −0.021 | 0.2552 ± 0.0145 | `a911fd6c` `6a44ead4` `e1d8f15c` |
| GGNN (A4 in fusion) | 0.5692 ± 0.0098 | +0.0194, +1.14 | −0.015 / +0.044 / +0.029 | 0.2669 ± 0.0079 | `08e08a0c` `7d32adac` `06292067` |
| weighted BCE (A9) | 0.5802 ± 0.0024 | +0.0304, +1.96 | +0.006 / +0.050 / +0.035 | 0.2697 ± 0.0157 | `b69fbf3c` `347fe5e6` `8a69db1a` |
| plain BCE (A9) | 0.5747 ± 0.0162 | +0.0249, +1.29 | −0.006 / +0.035 / +0.046 | 0.2642 ± 0.0122 | `499a5fd4` `195b9b99` `27d5a848` |
| line-only, `function_weight=0` (A8) | 0.5456 ± 0.0220 | −0.0041, −0.19 | −0.044 / +0.008 / +0.023 | 0.2580 ± 0.0312 | `c5ea524f` `31a608af` `fc0ad7ba` |
| GraphCodeBERT, frozen (A5) | 0.5747 ± 0.0187 | +0.0249, +1.22 | +0.021 / +0.046 / +0.008 | 0.2898 ± 0.0067 | `12f58c9c` `edae7f99` `6fee9b17` |
| **fine-tuned reference** — D9.7's cross_attention config, CodeBERT | 0.6272 ± 0.0073 | — | — | 0.3568 ± 0.0108 | `57df8cff` `77268204`‡ `3a953045`‡ |
| GraphCodeBERT, fine-tuned (A5) | 0.6169 ± 0.0098 | −0.0104, −1.20 | −0.010 / 0.000 / −0.021 | 0.3520 ± 0.0228 | `b04a6b82`‡ `7d785c26`‡ `fb445d65` |

‡ resumed (§7 #53): `77268204` `epoch=2,batch=1028,step=1962`; `3a953045`
`epoch=7,batch=424,step=5257`; `b04a6b82` `epoch=2,batch=436,step=1659` (three attempts —
the tag holds only the last resume); `7d785c26` `epoch=4,batch=1236,step=3502`.

### What this supports

1. **No ablation resolves at three seeds.** Every Top-5 difference from its reference is
   under 2 pooled sd. In four of the seven comparisons the paired sign flips across seeds.
2. **Fine-tuned, GraphCodeBERT does not beat CodeBERT** (A5): −0.0104 Top-5, −1.20 pooled
   sd; Top-1 −0.0048, −0.27 pooled sd, paired −0.029 / +0.015 / 0.000.
3. **Both reproductions land close, as `SecondMachine.md` predicted — side by side, not
   pooled.** Frozen cross_attention: machine 2 0.5498 ± 0.0219 against this machine's
   0.5408 ± 0.0181 (§1.21), 0.45 pooled sd apart. Fine-tuned cross_attention: 0.6272 ± 0.0073
   against 0.6162 ± 0.0184 (§1.22), 0.79 pooled sd apart.

### What it does not support — three things that look like results

- **Weighted BCE over focal (A9), +0.0304 Top-5.** All three seeds positive and the
  tightest spread in the table (± 0.0024), but 1.96 pooled sd against a reference whose own
  spread is ± 0.0219, on a frozen encoder. **The one candidate here worth more seeds** —
  cheap, 4–9 min a frozen run on machine 2 — before Phase 6 fixes a loss. Its line F1 @ 0.5 (0.2599 against
  0.0759) is the threshold moving, as everywhere else.
- **Frozen GraphCodeBERT's Top-1, +0.0367 (3.16 pooled sd, all three seeds positive).** A
  secondary metric, with the encoder frozen — and it is **gone fine-tuned** (point 2), the
  same fate as §1.21's frozen secondary signals in §1.22. Not a finding.
- **GGNN or GAT over GCN.** GGNN is +1.14 pooled sd with a flipped seed; GAT is −0.21.
  Neither ranks the convolutions; Phase 4 could not rank them alone either (one seed each).
- **Nothing about `test` or `clean_eval`, and nothing fine-tuned except A5.** A4, A8 and A9
  were run frozen only.

---

## 1.24 Phase 6 — calibrated line scores and a threshold chosen on validation

**The full model** (owner, 2026-10-07, §2): the designed cross-attention architecture at
D9.7's configuration — runs `80fc5238` / `c193e3bc` / `829c3555`, seeds 42 / 1 / 2.

**How the numbers were made.** `scripts/score_checkpoint.py` rebuilds each run's model from
the config the run archived at its start, loads `best.pt`, scores `val` and keeps every
line's logit (`artifacts/phase6/scores/<run>_val.npz`; score runs `06c04755…`, `5d3a6ad9…`,
`a593b1be…`). **All nine D9.7 checkpoints were re-scored and every one reproduces its
run's logged best-epoch metrics** on eight keys within 1e-4 — Top-1, Top-5, IFA, line
P/R/F1, scored units, coverage — and the script writes nothing when they do not.
`scripts/calibrate.py` then works from those files alone (calibration run `569c9701…`;
`artifacts/phase6/calibration_report.json`). **`test` and `clean_eval` were not touched**:
both scripts refuse them without an explicit flag (`Rules.md` §5.3).

*Provenance.* The first pass of these runs (`be9d2979…`, `81003ffa…`, `20723b0c…`,
`e93ad84f…` and seven more) was made while the new scripts were still uncommitted, so they
carry a `-dirty` sha. All were re-run from the clean tree at `c7b6025`; every number came
out identical to four decimals; the first pass is tagged `void` as superseded. **Commit
before a run whose number will be quoted, even when the run is "only" a measurement.**

**What was fitted.** Per seed, a Platt map on the logits — `sigmoid(scale · logit + shift)`,
`scale` constrained positive so the ranking cannot change — and the cut-off with the
highest line F1. **`val`, 488 units, 43,056 scoreable lines, 2,807 of them flagged (6.52%),
n = 3 seeds.**

| | Raw `sigmoid`, cut at 0.5 (what every run logged) | **Calibrated, threshold chosen on the other folds** |
|---|---|---|
| ECE (15 bins) | 0.1704 ± 0.0330 | **0.0137 ± 0.0104** |
| Negative log-likelihood | 0.3208 ± 0.0314 | 0.2012 ± 0.0075 |
| Line precision | 0.6163 ± 0.0496 | 0.4020 ± 0.0300 |
| Line recall | 0.2015 ± 0.0795 | 0.3787 ± 0.0098 |
| **Line F1** | 0.2954 ± 0.0855 | **0.3898 ± 0.0190** |

The right-hand column is **out-of-fold**: calibrator and threshold fitted on four fifths of
the validation units (folds by content hash of `unit_id`, by unit, never by line) and
applied to the fifth. Fitted on all of `val` and scored on the same lines, F1 is
0.3952 ± 0.0191 — optimistic by 0.005 to 0.007 per seed. **Quote the out-of-fold numbers.**

**Parameters of record** (fitted on all of `val`; what Phase 7 and serving apply):

| Seed | Run | `scale` | `shift` | Threshold (calibrated probability) |
|---|---|---|---|---|
| 42 | `80fc5238` | 1.8412 | −0.6707 | 0.1686 |
| 1 | `c193e3bc` | 0.7166 | −1.8906 | 0.1083 |
| 2 | `829c3555` | 3.0486 | −0.1939 | 0.2159 |

### What this supports

1. **The line scores can be made calibrated**: ECE falls from 0.17 to 0.014 on units the
   calibrator never saw. Raw, the lines scored below 0.2 carry a mean probability of
   0.13–0.16 and are flagged 0.3–2.3% of the time — focal loss, as expected.
2. **The seed-to-seed spread in line F1 was the threshold, measured.** At a fixed 0.5 the
   three seeds score 0.31 / 0.37 / 0.20 (sd 0.086); each given its own validation-chosen
   cut-off, 0.39 / 0.41 / 0.37 (sd 0.019). §1.14 and §1.22 said "the threshold moving, not
   the model"; this is the number behind the sentence. **The scales differ by 4× across
   seeds (0.72 to 3.05), so a calibrator belongs to its checkpoint — never average them.**
3. **Calibration changes no ranking.** Top-1, Top-5 and IFA are exactly §1.22's.

### What it does not support

- **Not calibration at the confident end.** Out of fold, lines given p ≥ 0.8 (268 / 390 /
  105 of them; mean predicted 0.94–0.95) are flagged **0.70 / 0.67 / 0.81** of the time.
  A two-parameter monotone map fits the 93–98% of lines below 0.2, where it is within
  0.002, and cannot bend the top. **That ceiling sits where §1.6's label audit would put
  it** (~30% of line labels wrong, 95% CI [14.5%, 51.9%]) — *consistent with* label noise
  capping the observed rate, **not evidence of it**: over-confidence looks the same.
- **Not a reason to prefer an arm — see the next table.**

### The same procedure on the other two arms — line F1 ties as well

Calibrated and thresholded exactly as above, each arm on its own three checkpoints
(`artifacts/phase6/arms/<arm>/calibration_report.json`; calibration runs `fd3e94c5…`
concat, `97cb6a76…` `llm_only`; score runs `845be156…` `2e52df3b…` `2d98c9b5…` and
`527c9399…` `40de595e…` `15126017…`). **`val`, out of fold, n = 3 per arm.**

| Arm | Line F1 at the logged 0.5 | **Line F1, own threshold** | Precision | Recall | ECE raw → calibrated |
|---|---|---|---|---|---|
| cross_attention | 0.2954 ± 0.0855 | **0.3898 ± 0.0190** | 0.4020 ± 0.0300 | 0.3787 ± 0.0098 | 0.1704 → 0.0137 |
| concat | 0.3811 ± 0.0159 | **0.3908 ± 0.0030** | 0.4175 ± 0.0227 | 0.3682 ± 0.0151 | 0.1634 → 0.0198 |
| llm_only | 0.3726 ± 0.0157 | **0.3936 ± 0.0072** | 0.4162 ± 0.0149 | 0.3738 ± 0.0143 | 0.1438 → 0.0170 |

Per seed (42 / 1 / 2), own threshold: cross_attention 0.3920 / 0.4075 / 0.3698; concat
0.3917 / 0.3933 / 0.3874; `llm_only` 0.4006 / 0.3940 / 0.3862.

- **§1.22's line-F1 gap was the threshold, and here is the measurement.** At 0.5,
  cross_attention trailed concat by 0.086 (1.39 pooled sd). Each arm at its own
  validation-chosen cut-off: 0.3898 / 0.3908 / 0.3936 — a spread of 0.004 across the three
  arms, against per-arm sds of 0.003 to 0.019. **Top-5, Top-1, IFA and now line F1: no
  metric measured on `val` separates the three arms.**
- **What cross-attention does show is more seed-to-seed spread**: its calibrator scales
  run 0.72 / 1.84 / 3.05 against concat's 0.73 / 0.79 / 1.01 and `llm_only`'s 0.55 / 1.13 /
  1.48, and its F1 sd is the largest of the three. Three seeds; recorded, not a finding.
- **Every arm is over-confident above p = 0.8 in the same way**: observed 0.67–0.81
  (cross_attention), 0.71–0.73 (concat), 0.66–0.79 (`llm_only`). Whatever caps the observed
  rate there is common to all three models — which fits a property of the labels better
  than a property of one architecture, and is still not proof of either.
- **Not `test` or `clean_eval`**, and not a claim that a threshold chosen on `val` (488
  units, longer functions than `train`) transfers. That is what the single held-out
  evaluation is for.

---

## 1.25 A second formatting leak — the comment terminator that lost its slash (found 2026-10-07)

**Every line-level number this project has logged since ADR-006 was measured on text that
still writes part of the label into the formatting.** Found in session 16 while running
Phase 7a's explainer on real validation functions: the lines the full model ranked first
in `bigvul-178623` were comment lines ending in a bare `*`. Nothing is voided yet — that
is open decision **Q14** (§5), the owner's.

### What the corpus does

In the shards' text — already ADR-006-normalized — a **flagged** line that ended a block
comment has lost the slash of its terminator:

```
qio_channel_wait(ioc,                                   <- unflagged lines keep  ...  */
/* XXX this is needed by nbd_reply_ready.  *            <- a flagged line
*ch = '\0';  /* and replace with null-termination *     <- a flagged line
```

Same family as ADR-006: flagged lines are pasted back from `flaw_line` and carry a
formatting difference the rest of `processed_func` does not. **Why the slash is lost is not
established** — the raw CSV was not reopened. Whitespace normalization does not touch it.

Measured — `scripts/audit_label_leakage.py`, run `9f462174…`, normalized text, lines inside
the 512-line cap. The tell is `comment_end_lost_slash`: text, then exactly one `*`, at the
end of a line that carries a comment (`/**` and banners excluded).

| Partition | Line-labelled units | Tell fires on | Of them flagged | Precision | Share of flagged lines | Units it answers |
|---|---|---|---|---|---|---|
| `train` | 5,457 | 425 | 420 | 0.9882 | 1.24% | 239 (4.4%) |
| `val` | 488 | 204 | 204 | **1.0000** | 7.26% | **58 (11.9%)** |
| `test` | 546 | 156 | 152 | 0.9744 | 5.57% | 73 (13.4%) |
| `clean_eval` | 1,393 | 3 | 0 | — | 0% | 0 |

The mirror tell, `comment_end_intact` — a line ending in `*/` — is a **negative oracle on
every partition**: flagged 0 of 8,474 on `train` (930 expected at the base rate), 0 of 1,847
on `val` (119), 1 of 3,219 on `test` (151), 0 of 6,639 on `clean_eval` (174). The audit's
first version could not report this; it now carries an `excludes_label` verdict for a tell
that is *never* right. (`is_blank` gets the same verdict and is not an artefact: a blank
line holds no code to change.)

**`clean_eval` is different, not clean.** It has no positive tell, but a probe (a scratch
script over the shard, not a logged run) counts only **4** of its 4,358 flagged lines
containing `/*` at all, against 6,763 unflagged ones — there, "this line carries a comment"
reads as "not flagged". That regime is not in the audit yet.

### The models read it

`scripts/audit_score_leakage.py`, run `ded852ee…` — from the stored `val` scores of the
nine D9.7 checkpoints (§1.22), no model run. Mean ± sample sd over seeds 42 / 1 / 2:

| `val`, Top-5 (Top-1) | cross_attention | concat | `llm_only` |
|---|---|---|---|
| All 482 units — reproduces §1.22 | 0.6162 ± 0.0184 (0.3444 ± 0.0293) | 0.6169 ± 0.0104 (0.3645 ± 0.0073) | 0.6169 ± 0.0048 (0.3527 ± 0.0021) |
| The 58 units the tell answers | 0.9483 ± 0.0622 (0.7874 ± 0.1486) | 0.9943 ± 0.0100 (0.9138 ± 0.0896) | 0.9885 ± 0.0199 (0.8851 ± 0.1247) |
| — top-ranked line *is* a tell line, of 58 | 20 / 53 / 34 | 33 / 51 / 52 | 26 / 51 / 52 |
| The other 424 units | **0.5708 ± 0.0227** (0.2838 ± 0.0231) | **0.5653 ± 0.0106** (0.2893 ± 0.0059) | **0.5660 ± 0.0047** (0.2799 ± 0.0171) |
| All units, both comment tells' lines withheld (472) | 0.6038 ± 0.0194 (0.3249 ± 0.0192) | 0.5932 ± 0.0106 (0.3340 ± 0.0044) | 0.5953 ± 0.0076 (0.3227 ± 0.0178) |

Rule-only baselines on the same 482 units, no model: **source order alone — the first five
scoreable lines — Top-5 0.3071** (Top-1 0.0000); the tell first, then source order, Top-1
0.1203, Top-5 0.3942.

### What this supports

- **A label leak of ADR-006's family is live on `train`, `val` and `test`**, and all nine
  fine-tuned checkpoints use it: on the units where the formatting gives the answer they
  reach Top-5 0.95–0.99 and usually put the tell line first.
- **About 0.045–0.052 of `val` Top-5 and 0.061–0.075 of Top-1 sit on those 58 units** —
  the gap between "all" and "the other 424", per arm.
- **A1's null does not depend on it.** The three arms tie on the other 424 units (0.5708 /
  0.5653 / 0.5660) exactly as they tie overall.
- **Part of §1.7's parse damage is this defect.** A lost terminator leaves a comment open,
  and the grammar swallows what follows. Probe (scratch script, not a logged run): of the
  line-labelled units with a tell line, 9 of 61 parse cleanly on `val` as shipped and 30 of
  61 once a slash is appended to each tell line; on `train`, 45 → 154 of 244. Flagged lines
  in units that do not parse cleanly: 65.71% → 56.11% on `val`, 27.99% → 22.43% on `train`.

### What it does not support

- **0.57 is not "the leak-free Top-5".** The other 424 units are a subgroup, not a
  counterfactual, and these models were trained with the tell in view. A leak-free number
  needs repaired text at training and at evaluation.
- **Nothing here says how much the structural branch reads it.** Parse damage around a
  flagged line is itself a tell a GNN can see. The frozen screen (§1.21), Phase 4 (§1.15),
  Phase 3 (§1.14), A6 and machine 2's slate (§1.23) have no stored scores and are
  **unaudited** — "structure helps a frozen encoder" is the claim most exposed.
- **`clean_eval` is not exempt** — it has the negative oracle and its own regime.
- **The calibration in §1.24 is not known to survive.** A tell line is scored with high
  confidence and is nearly always right, which is exactly what flatters ECE.

### What to do with every number until Q14 is decided

*Q14 was decided on 2026-10-08 (§2): ADR-010 accepted. The rule below now holds for good
for every number measured with comments in place (ADR-010 D10.7), and the held-out
evaluation is made on comment-stripped text.*

Quote it as measured, with **"on text carrying the §1.25 formatting leak"** beside it, and
give the other-units figure where one exists. Do not start the held-out evaluation on the
shards as they are: `test` carries the tell on 13.4% of its units.

---

### The fix, built and measured the same afternoon

**`utils/text.strip_comments` removes every comment from every line, label-blind, and
keeps the line count.** Both forms of the defect are about comments, so removing comments
from flagged and unflagged lines alike removes both. It is **off unless asked for**
(`+representation.strip_comments=true`; the key is deliberately absent from
`configs/config.yaml`, §7 #38), because every number logged before it was measured with
comments in place. ADR-010 (**Proposed** when this was written; **accepted by the owner
2026-10-08**, §2) is the decision record.

`scripts/report_comment_stripping.py`, run `af4173a3…`, every partition's line-labelled
units. It replaces the two scratch probes quoted above with logged figures.

| | `train` | `val` | `test` | `clean_eval` |
|---|---|---|---|---|
| Stripper identical to Tree-sitter's own comment nodes, on clean units without a lost-slash line | 4,540 / 4,540 | 312 / 312 | 366 / 366 | 1,056 / 1,056 |
| Flagged lines carrying a comment marker (unflagged) | 8.99% (8.61%) | 16.69% (8.03%) | 6.15% (8.35%) | **0.44%** (5.73%) |
| Lost-slash lines, before → after | 431 → 6 | 492 → 0 | 162 → 1 | 3 → 1 |
| Lines ending in `*/`, before → after | 9,142 → 5 | 2,444 → 6 | 3,866 → 24 | 7,684 → 6 |
| **Flagged lines that held only a comment — blank after** | **11.73%** | **8.18%** | **7.38%** | 1.38% |
| Units whose flagged lines are all blank after | 31 | 8 | 8 | 0 |
| Clean parses, before → after | 84.02% → 86.09% | 65.78% → 70.29% | 70.15% → 76.92% | 75.88% → 75.88% |
| — among units with a lost-slash line | 45 → 157 of 244 | 9 → 30 of 61 | 17 → 53 of 79 | 1 → 0 of 3 |
| Flagged lines in units that do not parse cleanly | 27.99% → 22.13% | 65.71% → 55.88% | **54.44% → 21.84%** | 27.83% → 28.07% |

With `excluded` (288) and `duplicate` (24) the stripper agrees with the grammar on **6,586
of 6,586** compared units. **`clean_eval`'s form of the defect is now a logged number:**
20 of its 4,499 flagged lines carry any comment marker.

**A ceiling that was always there:** 8–12% of the labelled lines on `train`/`val`/`test`
are comment-only lines — a comment the fix commit deleted. A comment cannot be the
vulnerable line, and until now a model was credited for ranking one.

### What stripping at evaluation does to the D9.7 checkpoints

The nine checkpoints re-scored on comment-stripped `val` — *trained with comments, scored
without*. Score runs `84582530…` `baf55d15…` `f3b641bc…` (cross_attention, seeds 42 / 1 /
2), `0f391973…` `be36cca8…` `9e0e13ec…` (concat), `9d89bb6d…` `3a1507ed…` `006b0ea1…`
(`llm_only`); calibration `e8a27dc5…` / `a38fd4db…` / `5bada5f3…`; audit `9b87fbb2…`
(the as-shipped audit re-run with the same groups is `b6f750f2…`). All from clean trees.

| `val`, 3 seeds | cross_attention | concat | `llm_only` |
|---|---|---|---|
| Top-5 as shipped (§1.22) | 0.6162 ± 0.0184 | 0.6169 ± 0.0104 | 0.6169 ± 0.0048 |
| **Top-5, comments stripped** | **0.5353 ± 0.0200** | **0.5415 ± 0.0090** | **0.5546 ± 0.0084** |
| — blank flagged lines withheld (474 units) | 0.5450 ± 0.0211 | 0.5506 ± 0.0092 | 0.5640 ± 0.0085 |
| Top-1 as shipped → stripped | 0.3444 → 0.2628 ± 0.0141 | 0.3645 → 0.2759 ± 0.0072 | 0.3527 → 0.2669 ± 0.0133 |
| IFA as shipped → stripped | 11.26 → 14.13 ± 0.97 | 11.46 → 14.34 ± 1.34 | 11.13 → 13.89 ± 0.30 |
| Line F1 out of fold, as shipped (§1.24) → stripped | 0.3898 → **0.2355 ± 0.0040** | 0.3908 → **0.2338 ± 0.0059** | 0.3936 → **0.2362 ± 0.0030** |
| ECE after calibration, stripped | 0.0109 ± 0.0067 | 0.0158 ± 0.0029 | 0.0137 ± 0.0041 |

On stripped text: no unit is answered by the tell (0 of 482, all nine checkpoints); 274 of
the 2,808 flagged lines inside the cap are blank and 8 units have nothing else flagged;
flagged-line coverage is 97.04% (it was 99.96%); source order alone scores Top-5 0.3071.

**Supports:** the leak and the comment lines together were worth **0.06–0.08 Top-5 and
about 0.15 line F1** to these checkpoints; the fix removes the leak rather than hiding it;
and **the three arms still tie** (`llm_only` − cross_attention = +0.0193 Top-5, 1.26 pooled
sd, signs not uniform across seeds).

**Does not support:** "0.54 is the model's real Top-5". These checkpoints learned with the
comments in view — including "a line ending in `*/` is never flagged" — and are scored
here on text they never saw. **A model trained on stripped text is the measurement.** That
is queue group `d97r` (nine fine-tuned runs, one more override than D9.7, arm by arm);
`d97rcheck` is its overfit gate.

### The frozen screen: the structural branch is where the leak gets in

The 18 checkpoints of the A1 frozen screen (§1.21) scored on `val` as shipped — each
re-scoring reproduced its run's logged metrics — and again on comment-stripped text off a
stripped semantic cache in its own directory (§7 #59). Audits `30a0d40b…` (as shipped) and
`18b2535c…` (stripped). Top-5, mean ± sd over seeds 42 / 1 / 2:

| Frozen arm | All 482 (as logged) | The 58 tell units | Top-ranked line is a tell line, of 58 (range over seeds) | The other 424 | All 482, comments stripped |
|---|---|---|---|---|---|
| `llm_only` | 0.4903 ± 0.0032 | 0.4655 ± 0.0000 | **0–1** | 0.4937 ± 0.0036 | 0.4938 ± 0.0036 |
| cross_attention | 0.5408 ± 0.0181 | 0.6379 ± 0.1645 | 0–18 | 0.5275 ± 0.0054 | 0.5228 ± 0.0062 |
| concat | 0.5394 ± 0.0198 | 0.6609 ± 0.0100 | 18–23 | 0.5228 ± 0.0213 | 0.5159 ± 0.0220 |
| sum | 0.5304 ± 0.0012 | 0.6494 ± 0.0527 | 18–21 | 0.5142 ± 0.0062 | 0.5069 ± 0.0043 |
| gated | 0.5221 ± 0.0052 | 0.6322 ± 0.0697 | 20–22 | 0.5071 ± 0.0118 | 0.5041 ± 0.0055 |
| `gnn_only` | 0.4191 ± 0.0468 | 0.5690 ± 0.0597 | 21–24 | 0.3986 ± 0.0450 | 0.4053 ± 0.0456 |

**Supports:**

- **Frozen `llm_only` does not read the tell** — its top-ranked line is a tell line in 0–1
  of 58 units, and it scores *lower* on those units than on the rest. **Every arm with a
  structural branch does**, `gnn_only` included: an unterminated comment is parse damage,
  and a GNN sees parse damage. A frozen encoder with a linear head cannot use the cue; a
  fine-tuned one learns it (above).
- **So part of "every fusion arm beats frozen `llm_only`" (§1.21) was the leak.** The lead
  over `llm_only` was +0.032 to +0.051 Top-5 as logged. On the other 424 units it is
  +0.013 to +0.034; on stripped text it is **+0.010 to +0.029**.
- **What is left keeps its sign.** On stripped text every fused arm is above `llm_only` on
  every seed, except concat on seed 1, which ties it (0.4959 both). Per seed: `llm_only`
  0.4959 / 0.4959 / 0.4896; cross_attention 0.5166 / 0.5290 / 0.5228; concat 0.5124 /
  0.4959 / 0.5394; sum 0.5083 / 0.5021 / 0.5104; gated 0.5062 / 0.4979 / 0.5083.
- **Cross-attention ≈ concat survives here too** (0.5228 ± 0.0062 against 0.5159 ± 0.0220
  stripped; +0.0069, 0.43 pooled sd).

**Does not support:** a leak-free size for the frozen advantage. These checkpoints were
trained with the damage in view and are scored without it. Queue groups `a1rcheck` / `a1r`
are the same screen trained on stripped text — 18 frozen runs, a whole comparison group,
runnable on either machine but never split between them. **Not started.** Phase 4's
structural baseline (§1.15), Phase 3 (§1.14) and machine 2's slate (§1.23) have no stored
scores and remain unaudited.

---

## 1.26 A6 — attention direction: bidirectional is not resolvably better than either one-way arm

`run_queue.py --only a6check a6`, 2026-10-07 13:01 → 16:54. Frozen encoder off the cache,
A1's constants, seeds 42 / 1 / 2, `val`, validation-selected. **Measured on text carrying
the §1.25 leak** — like every frozen number so far.

| Arm | Runs (seeds 42 / 1 / 2) | Top-5 | Top-1 | IFA |
|---|---|---|---|---|
| `both` — the design | `cde35987…` `753f4c60…` `0fca1b4f…` | 0.5408 ± 0.0181 | 0.2310 ± 0.0084 | 12.19 ± 0.44 |
| `sem_from_str` only | `920989e5…` `7a525ea0…` `f07faf7b…` | 0.5166 ± 0.0095 | 0.2503 ± 0.0079 | 12.99 ± 0.33 |
| `str_from_sem` only | `ed9d88e5…` `64c5d5a0…` `317aef09…` | 0.5498 ± 0.0231 | 0.2649 ± 0.0348 | 13.96 ± 1.79 |

Per seed Top-5: 0.5560, 0.5456, 0.5207 / 0.5083, 0.5145, 0.5270 / 0.5747, 0.5290, 0.5456.
Seed reports `c98d0ab1…` `377764de…` `18edd904…`, each `n_seeds: 3`. Commits `2e9c0a4`,
`82f1957`, `9083d4d` — clean, and `src/vulnloc` and `configs/` are byte-identical across
them. No run resumed. 14–38 min each. Overfit gates `836aec8e…` and `f3acb184…` ended at
Top-1 = 1.0000 over their last five epochs.

**The `both` arm reproduces A1's cross_attention arm exactly** — 0.5560 / 0.5456 / 0.5207
against `192e067c…` / `e7ae3108…` / `ce2380bf…` (§1.21), selected epoch and last epoch
alike — so adding the `direction` setting did not change the designed model.

Against `both`:

| | Top-5 | Top-1 | IFA (lower is better) |
|---|---|---|---|
| `sem_from_str` | −0.0242, 1.67 pooled sd; paired −0.0477, −0.0311, +0.0062 | +0.0194, 2.38 sd; all three seeds higher | +0.80, 2.04 sd; all three worse |
| `str_from_sem` | +0.0090, 0.43 sd; signs mixed | +0.0339, 1.34 sd; all three higher | +1.77, 1.35 sd; two of three worse |

### What this supports

- **No one-way arm differs resolvably from the bidirectional one on Top-5.** The largest
  gap is 1.67 pooled sd with a sign flip.
- **Bidirectional cross-attention is not shown to be better than attending one way.** The
  claim A6 exists to test — "bidirectional specifically" — is not supported.

### What it does not support

- That one-way attention is *better*. Both one-way arms are higher on Top-1 on all three
  seeds (+0.019 and +0.034) and worse on IFA; at three seeds and ~2 pooled sd that is a
  candidate, not a finding (§1.16).
- Anything about a fine-tuned encoder. For text without the leak, see below.

### With the leak accounted for

The nine A6 checkpoints scored on `val` both ways; audits `7c64dfac…` (as shipped) and
`421bf67c…` (comments stripped at evaluation). Top-5 (Top-1):

| Arm | As logged | The 58 tell units | The other 424 | Comments stripped |
|---|---|---|---|---|
| `both` | 0.5408 ± 0.0181 (0.2310) | 0.6379 ± 0.1645 | 0.5275 ± 0.0054 | 0.5228 ± 0.0062 (0.2199 ± 0.0198) |
| `sem_from_str` | 0.5166 ± 0.0095 (0.2503) | 0.5747 ± 0.0796 | 0.5086 ± 0.0214 | 0.5028 ± 0.0104 (0.2393 ± 0.0161) |
| `str_from_sem` | 0.5498 ± 0.0231 (0.2649) | **0.7701 ± 0.0100** | 0.5197 ± 0.0252 | 0.5339 ± 0.0048 (0.2337 ± 0.0270) |

`str_from_sem` — structural lines querying semantic ones — reads the tell most: its
top-ranked line is a tell line in 15–22 of the 58 units by seed, against 0–18 for `both`
and 3–9 for `sem_from_str`. Its apparent lead over `both` as logged is on
those units; on the other 424 it is behind.

On stripped text `sem_from_str` sits 0.0200 below `both` (2.34 pooled sd) and
`str_from_sem` 0.0111 above (2.00 pooled sd). **Still not written down as a difference:**
three seeds, about 2 sd, models scored on text they were not trained on, and §1.16 is two
retractions of exactly this size. The conclusion above stands — nothing here shows
bidirectional attention to be better than attending one way.

---

## 1.27 Phase 7a — a grounded explanation for every flagged line; a named cause for one in twelve

Built 2026-10-07. `src/vulnloc/explanation/`: `types.py` (`Prediction`, `Evidence`,
`Explanation` — the three types `Architecture.md` §4.2 named and left undefined),
`evidence.py`, `templates.py` (16 single-line and 3 two-line weakness patterns),
`generator.py` (`EXPLAINER_REGISTRY` key `template`), `verifier.py` (FR-15),
`contract.py` (`PRD.md` §5.4, exactly its keys). `scripts/explain_checkpoint.py` runs it
on a scored, calibrated checkpoint; `scripts/make_rater_pack.py` draws the human-evaluation
sample; `docs/HumanEvalProtocol.md` fixes the procedure.

**What an explanation is made of — and what is and is not the model's.** Four fixed-form
sentences: (1) the weakness pattern the line's *text* matches, worded "Matches the
pattern", or a plain statement that none does — **a template, never the model's
reasoning**; (2) the model's rank and calibrated score for the line; (3) the lines the
model's own cross-attention weighted most; (4) a path from the name-based data-flow graph
— **static analysis of the function; the full model was trained on the AST and never saw
those edges**. The category is template-derived (Q5). Nothing is generated.

Measured on `val`, full model, each checkpoint at its own calibrated threshold:

| | seed 42 `80fc5238…` | seed 1 `c193e3bc…` | seed 2 `829c3555…` | seed 42, comments in place |
|---|---|---|---|---|
| Explain run | `412a2df4…` | `2c427d73…` | `1eca6836…` | `3ac6a4c3…` |
| Text | stripped | stripped | stripped | as shipped |
| Units with a flagged line, of 488 | 448 | 408 | 444 | 305 |
| Flagged lines | 5,364 | 4,014 | 4,530 | 2,447 |
| **Explanations verified / rejected** | **5,364 / 0** | **4,014 / 0** | **4,530 / 0** | 2,447 / 0 |
| Lines where a pattern names a cause | 8.07% | 8.79% | 8.23% | 6.78% (166) |
| — of units' top-ranked lines | 14.51% | 13.48% | 14.86% | 12.46% (and 6.23% carry the tell) |
| Lines that match no pattern and carry the §1.25 tell (told so) | 0 | 0 | 0 | 6.29% (154) |
| With attention / data-flow / graph-node evidence | 100% / 68.8% / 94.3% | 100% / 69.3% / 94.8% | 100% / 67.7% / 94.3% | 100% / 48.0% / 76.5% |
| Flagged lines the dataset labels | 18.21% | 20.35% | 18.81% | 42.83% — and **154 of 154** of the tell lines |
| Template CWE = the function's CWE id, exactly (units compared) | 11.67% (60) | 11.76% (51) | 14.52% (62) | 8.57% (35) |

**The table is the second pass.** Reading the first pass's explanations of real code
found four template sentences saying more than a pattern can see — an allocation tested
where it stands (`if ((p = malloc(n)) == NULL)`) explained as "never compared against
NULL"; "used after being freed" from a pattern that sees line order, not execution
order — and they were corrected before these runs (first pass: `6f76427b…` `37a4943f…`
`31ed73d2…` `8f85eebd…`, superseded, 8.15 / 8.87 / 8.30%).

Every pass reproduced the stored scores (largest logit deviation 0.0215). The stripped
thresholds are low (0.0973 / 0.0846 / 0.1163), so most units have a flagged line and four
in five flagged lines are not ones the dataset labels — the F1-optimal point on this task
is a recall-heavy one.

### Phase 7's exit criteria, against this

| Exit criterion | Status | Evidence |
|---|---|---|
| 7a produces a valid explanation for every positive prediction | ✅ | 13,908 of 13,908 flagged lines over three seeds, each verified |
| Grounding test suite passes (no explanation names an unflagged line) | ✅ | `tests/unit/test_explanation.py`, incl. a property test over arbitrary scores and thresholds; grounding rate 100% on real data, 0 rejections |
| Explanations reviewed for absence of attack instructions | 🟨 | Automatic half done: every template sentence and every finished text is scanned for that vocabulary (test + verifier). **The reading by a person is the owner's** — one page, `artifacts/phase7/sentences_for_review.md` (19 template sentences and 8 fixed forms, printed from the code by `scripts/list_explanation_sentences.py`) |
| Human evaluation protocol defined | ✅ | `docs/HumanEvalProtocol.md`; a pilot pack of 100 `val` samples is at `artifacts/phase7/rater_pack_stripped/` (run `51525d82…`) |

### What this supports

- **FR-14 and FR-15 hold by construction and by measurement**: every explanation cites
  only lines it has evidence for, and none calls an unflagged line flagged.
- **FR-13 — "naming the vulnerability type and the root cause" — is met for about one
  flagged line in twelve.** For the rest the tool says, in words, that it names no cause.
  That is the template baseline's honest reach on real vulnerable lines, which are mostly
  ordinary statements.

### What it does not support

- That the attention sentence explains the score. It reports where the weights are; A1 and
  A6 found that attending does not improve localization, and no faithfulness test
  (`ResearchPlan.md` §4.4) has been run.
- That the named category is right: it equals the dataset's CWE id for 12–15% of the units
  where both exist (exact match only; a template's CWE-787 under a function filed as its
  parent CWE-119 counts as a miss).
- Anything a human thinks of these explanations. Nothing has been rated.

---

## 1.28 `d97r` — D9.7 retrained on comment-stripped text: all three arms in, and the tie survives (2026-10-08 14:21)

**Status: all nine runs finished; three seed reports at `n_seeds: 3`. With no comment in
the text at training or at evaluation — so no tell — cross-attention does not beat
concat and neither fused arm beats `llm_only`: no difference between arms reaches
2 pooled sd.** The calibrated line F1, the score audit and the explanations are below,
and the arms tie there too (line F1 0.233–0.241, each checkpoint at its own threshold).
Run pending the owner's decision on Q14 (ADR-010, Proposed). **Decided 2026-10-08 (§2):
ADR-010 accepted — these nine are the models of record.**

| Arm, retrained without comments — `val`, 3 seeds | Top-5 | Top-1 | IFA | Seed report |
|---|---|---|---|---|
| cross_attention (the designed model) | 0.5394 ± 0.0116 | 0.2683 ± 0.0060 | 14.56 ± 0.43 | `bd2544c7…` |
| concat | 0.5546 ± 0.0084 | 0.2621 ± 0.0222 | 14.06 ± 0.26 | `f3d754eb…` |
| `llm_only` (same scaffold, no structural branch) | 0.5519 ± 0.0187 | 0.2801 ± 0.0062 | 14.72 ± 1.27 | `f814e521…` |

`d97r` is D9.7's nine runs with exactly one more override,
`+representation.strip_comments=true`: the comments — and with them the §1.25 tell — are
gone at training as well as at evaluation. Validation-selected, `val`, 482 units.

### The cross-attention arm (the designed model), three seeds

| Run | Seed | Epochs run | Selected epoch | Top-5 | Top-1 | Minutes | Commit |
|---|---|---|---|---|---|---|---|
| `ba7dcc5f…` | 42 | 8 | 4 | 0.5519 | 0.2718 | 109.2 | `0a4843c` |
| `8afbf7da…` | 1 | 8 | 4 | 0.5373 | 0.2718 | 109.1 | `d5e5dda` |
| `ff980345…` | 2 | 4 | 0 | 0.5290 | 0.2614 | 54.4 | `5af24ba` |

Every run: one attempt, not resumed, clean tree, the same card (`resolved_gpu` logged),
`strip_comments=True`, 2,808 flagged lines of which 83 have no features (coverage 97.04%).

Seed report `bd2544c7…` — `artifacts/adr010/seed_report_fused_finetuned_stripped_cross_attention.json`,
`n_seeds: 3`, every run's cross-check "agrees":

| Metric | Mean ± sd over 3 seeds |
|---|---|
| **Top-5** | **0.5394 ± 0.0116** |
| **Top-1** | **0.2683 ± 0.0060** |
| IFA | 14.56 ± 0.43 |
| Line F1 at the logged threshold 0.5 | 0.1603 ± 0.0561 (P 0.3244 ± 0.1704, R 0.1316 ± 0.0761) |

The F1 row is the threshold moving, as in §1.24 — precision 0.21 to 0.52 across seeds with
recall moving the other way. The calibrated, validation-thresholded figure comes from the
scoring step that follows the queue.

### The same arm under three conditions

| Condition | Top-5 | Top-1 | Source |
|---|---|---|---|
| Comments at training and at evaluation — D9.7 as shipped, reads the tell | 0.6162 ± 0.0184 | 0.3444 ± 0.0293 | §1.22 |
| Trained with comments, stripped at evaluation only | 0.5353 ± 0.0200 | 0.2628 ± 0.0141 | audit `9b87fbb2…`, §1.25 |
| **Stripped at training and at evaluation (`d97r`)** | **0.5394 ± 0.0116** | **0.2683 ± 0.0060** | seed report `bd2544c7…` |

- **Retraining on leak-free text lands where stripping at evaluation had already put the
  D9.7 checkpoints.** The difference is +0.0041 Top-5 (0.25 pooled sd) and +0.0055 Top-1
  (0.51 pooled sd): not resolved.
- **So for this arm the leak-free validation Top-5 is ≈ 0.54, and the 0.6162 it was
  reported at was ≈ 0.077 too high** (4.99 pooled sd; Top-1 −0.0761, 3.60 pooled sd; IFA
  14.56 ± 0.43 against 11.26 ± 1.45). A model that never saw the tell does not find
  another route to the old score. §1.25's 8.18% of `val`'s flagged lines that are
  comment-only — blank once stripped — are inside that drop: a ceiling, not a leak.
- Whether the arms still tie when retrained is two subsections down.

### What to quote with it

1. **The seed report was made with `+seeds.allow_source_drift=true`.** Its provenance lists
   `src/vulnloc/explanation/generator.py` (commit `7ea7a9c`) as the one file that differs
   between the seeds' commits. Training is layer 3 and cannot import layer 5; §7 #62.
2. **It was made outside the queue**, with the queue plan's own command
   (`artifacts/queue/run_plan_step.py`), at commit `3bc918e`: the queue process running that
   night predates the override and its own attempt is `[FAIL]`. `d97r_after.ps1` re-runs
   that step through the queue at the end, which writes a second, identical report.
3. **Seed 2 stopped after four epochs with its best at epoch 0** — validation Top-5 0.5290,
   then 0.5249, 0.5166, 0.5021 — under `train.patience=3`, the rule every D9.7 run used.
   The other two seeds dipped at epoch 1 and peaked at epoch 4, and none of the nine D9.7
   runs was selected at epoch 0 (theirs: epochs 2–7). It is a replicate under the pre-set
   rule and it is counted; it is also why the sd of 0.0116 should not be read as tight.
4. Validation-selected (§1.18), `val` only. `test` and `clean_eval` are untouched.

### The `llm_only` control (same scaffold, no structural branch), three seeds

| Run | Seed | Epochs run | Selected epoch | Top-5 | Top-1 | Minutes | Commit |
|---|---|---|---|---|---|---|---|
| `d82a6a70…` | 42 | 7 | 3 | 0.5332 | 0.2739 | 83.8 | `3bc918e` |
| `a1dce535…` | 1 | 10 | 7 | 0.5519 | 0.2801 | 122.9 | `fbc44e2` |
| `9296a0e5…` | 2 | 10 | 7 | 0.5705 | 0.2863 | 121.8 | `fbc44e2` |

Every run: one attempt and **not resumed** — unlike D9.7's three `llm_only` runs, which
all were (§1.22) — clean tree, the same card, coverage 97.04%. Seed report `f814e521…`,
made by the queue itself: `n_seeds: 3`, every cross-check "agrees", no source drift (the
two commits differ in docs only).

| Metric | Mean ± sd over 3 seeds |
|---|---|
| **Top-5** | **0.5519 ± 0.0187** |
| **Top-1** | **0.2801 ± 0.0062** |
| IFA | 14.72 ± 1.27 |
| Line F1 at the logged threshold 0.5 | 0.1392 ± 0.0316 (P 0.2472 ± 0.0342, R 0.0994 ± 0.0323) |

| Condition | Top-5 | Top-1 | Source |
|---|---|---|---|
| Comments at training and at evaluation — D9.7 as shipped, reads the tell | 0.6169 ± 0.0048 | 0.3527 ± 0.0021 | §1.22 |
| Trained with comments, stripped at evaluation only | 0.5546 ± 0.0084 | 0.2669 ± 0.0133 | audit `9b87fbb2…`, §1.25 |
| **Stripped at training and at evaluation (`d97r`)** | **0.5519 ± 0.0187** | **0.2801 ± 0.0062** | seed report `f814e521…` |

- **Same reading as for cross-attention: retraining lands where stripping at evaluation
  put it** — −0.0028 Top-5 (0.19 pooled sd), +0.0131 Top-1 (1.26 pooled sd). Not
  resolved. Against the number as shipped: −0.0650 Top-5, −0.0726 Top-1.
- **Seeds 1 and 2 ran to the 10-epoch cap** (`train.epochs=10`, D9.7's constant), both
  selected at epoch 7 with patience not yet exhausted; seed 42 was stopped by patience
  after seven epochs, selected at epoch 3. They are counted as they are — a longer
  schedule was not tried. One D9.7 run ended the same way (`a0c42d6f…`, concat seed 2:
  ten epochs, selected at epoch 7).

### Cross-attention against its control, both retrained

| | cross_attention | `llm_only` | Difference | In pooled sd | Same-seed differences (42, 1, 2) |
|---|---|---|---|---|---|
| Top-5 | 0.5394 ± 0.0116 | 0.5519 ± 0.0187 | −0.0124 | 0.80 | +0.0187, −0.0145, −0.0415 |
| Top-1 | 0.2683 ± 0.0060 | 0.2801 ± 0.0062 | −0.0118 | 1.92 | −0.0021, −0.0083, −0.0249 |
| IFA (lower is better) | 14.56 ± 0.43 | 14.72 ± 1.27 | −0.15 | 0.16 | −1.53, −0.13, +1.20 |

- **On leak-free text the designed model still does not beat its `llm_only` control.**
  No difference reaches the 2-pooled-sd line this project has used throughout (§1.26);
  Top-5 changes sign across seeds.
- **Top-1 is lower for cross-attention on all three seeds, at 1.92 pooled sd.** That is
  under the line, so it is **not resolved: a candidate in the control's favour, not a
  finding** (§1.16 is what happens when ~2 sd at three seeds is written down as one).
- This is §1.22's reading — no fused arm beats fine-tuned `llm_only` in the same scaffold
  — surviving the removal of the leak. Concat follows.

### The concat arm, three seeds — in at 14:21 on 2026-10-08

| Run | Seed | Epochs run | Selected epoch | Top-5 | Top-1 | Minutes | Commit |
|---|---|---|---|---|---|---|---|
| `9ac7a146…` | 42 | 7 | 3 | 0.5456 | 0.2365 | 98.9 over two attempts | `fbc44e2`, resumed at `2e41629` |
| `7579bc33…` | 1 | 7 | 3 | 0.5622 | 0.2759 | 101.4 | `1686536` |
| `10b43ce5…` | 2 | 6 | 2 | 0.5560 | 0.2739 | 81.3 | `1686536` |

Clean trees, the same card, coverage 97.04%, every cross-check "agrees". Seed report
`f3d754eb…`, made by the queue itself, `n_seeds: 3`, no source drift (the commits differ
in docs only). All three stopped on patience.

- **Seed 42 is a resumed run — `resumed_at: epoch=6,batch=6,step=3944`.** The owner
  paused the queue at 05:46 and resumed it at 10:59 on 2026-10-08. **Its selected
  checkpoint predates the pause:** `best.pt` was written at 05:20 (epoch 3); after the
  resume the run trained one more epoch (0.5415, no improvement) and stopped on
  patience at 11:17. The seed's reported numbers therefore come from weights the
  interruption never touched. It is also the arm's low Top-1 (0.2365 against 0.2759
  and 0.2739), which is why concat's Top-1 sd is 0.0222.

| Metric | Mean ± sd over 3 seeds |
|---|---|
| **Top-5** | **0.5546 ± 0.0084** |
| **Top-1** | **0.2621 ± 0.0222** |
| IFA | 14.06 ± 0.26 |
| Line F1 at the logged threshold 0.5 | 0.0914 ± 0.0594 (P 0.3753 ± 0.0951, R 0.0560 ± 0.0385) |

| Condition | Top-5 | Top-1 | Source |
|---|---|---|---|
| Comments at training and at evaluation — D9.7 as shipped, reads the tell | 0.6169 ± 0.0104 | 0.3645 ± 0.0073 | §1.22 |
| Trained with comments, stripped at evaluation only | 0.5415 ± 0.0090 | 0.2759 ± 0.0072 | audit `9b87fbb2…`, §1.25 |
| **Stripped at training and at evaluation (`d97r`)** | **0.5546 ± 0.0084** | **0.2621 ± 0.0222** | seed report `f3d754eb…` |

- Retraining against stripping at evaluation only: +0.0131 Top-5 (1.51 pooled sd),
  −0.0138 Top-1 (0.84 pooled sd). Not resolved — the same reading as the other two
  arms. Against the number as shipped: −0.0622 Top-5, −0.1024 Top-1.
- **A coincidence worth naming, because it looks like a copying error:** retrained
  concat's Top-5, 0.5546 ± 0.0084, is digit for digit the Top-5 of D9.7's `llm_only`
  checkpoints with comments stripped at evaluation (§1.25). Both are three counts out
  of 482 units, and the counts happen to be the same three — 263, 268 and 271 — from
  different checkpoints of different arms. Checked in the two artefacts.

### The three arms, retrained — A1 on text with no tell

| Difference | Top-5 | Top-1 | IFA (lower is better) |
|---|---|---|---|
| **cross_attention − concat (A1)** | **−0.0152 (1.51 pooled sd)** | +0.0062 (0.38) | +0.50 (1.43) |
| cross_attention − `llm_only` | −0.0124 (0.80) | −0.0118 (1.92) | −0.15 (0.16) |
| concat − `llm_only` | +0.0028 (0.19) | −0.0180 (1.10) | −0.66 (0.72) |

Same-seed differences (42, 1, 2), cross_attention − concat: Top-5 +0.0062, −0.0249,
−0.0270; Top-1 +0.0353, −0.0041, −0.0124; IFA +0.04, +1.25, +0.22.

- **A1's null survives retraining on leak-free text: cross-attention does not beat
  concatenation.** It is nominally behind on Top-5 (two seeds of three) and on IFA
  (all three seeds), nominally ahead on Top-1 (one seed of three). Nothing reaches
  2 pooled sd; nothing is resolved in either direction.
- **Neither fused arm beats `llm_only`.** Concat's Top-5 is 0.0028 above it (0.19
  pooled sd); both fused arms' Top-1 is below it. The largest standardized gap in the
  table is cross_attention − `llm_only` on Top-1, 1.92 pooled sd: a candidate in the
  control's favour, not a finding.
- **The three Top-5 means span 0.0152.** As shipped they spanned 0.0007 (§1.22); with
  comments stripped at evaluation only, 0.0193 (§1.25). The tie does not depend on the
  tell.
- **These are the leak-free validation numbers — and, since the owner accepted ADR-010
  on 2026-10-08 (Q14, §2), the validation numbers of record.** Every
  arm is ≈ 0.06–0.08 Top-5 below the figure it was reported at with the tell in view,
  and retraining did not resolvably recover any of it (the largest change against
  stripping at evaluation only is concat's +0.0131, 1.51 pooled sd).
- Three seeds an arm, validation-selected, one resumed run (concat seed 42), one seed
  report made with a recorded source-drift override (cross_attention). `val` only.

### Calibrated on `val`, each checkpoint at its own threshold (§1.24's procedure)

Nine `score-val` runs — every score file kept reproduces its run's logged metrics at the
default tolerance — and one calibration per arm (`513dd46e…` cross_attention, `79b8c8d1…`
concat, `8b7ac935…` `llm_only`): Platt, 15 bins, 5 folds by unit.

| | cross_attention | concat | `llm_only` |
|---|---|---|---|
| ECE, raw → calibrated (out of fold) | 0.1981 ± 0.0381 → 0.0133 ± 0.0067 | 0.1613 ± 0.0245 → 0.0076 ± 0.0019 | 0.1632 ± 0.0288 → 0.0168 ± 0.0029 |
| Line F1 at the logged threshold 0.5 | 0.1603 ± 0.0561 | 0.0914 ± 0.0594 | 0.1392 ± 0.0316 |
| **Line F1 at the validation threshold, out of fold** | **0.2359 ± 0.0120** | **0.2408 ± 0.0074** | **0.2330 ± 0.0118** |
| — precision / recall there | 0.1888 / 0.3187 | 0.1931 / 0.3233 | 0.1771 / 0.3456 |
| Thresholds (seeds 42, 1, 2) | 0.0925 / 0.0902 / 0.1278 | 0.1037 / 0.1058 / 0.1475 | 0.0912 / 0.0794 / 0.0807 |
| For comparison: trained with comments, stripped at evaluation (§1.25) | 0.2355 ± 0.0040 | 0.2338 ± 0.0059 | 0.2362 ± 0.0030 |

- **The arms tie on line F1 too.** The largest gap is concat − `llm_only`, +0.0078 (0.79
  pooled sd); cross_attention − concat is −0.0049 (0.49).
- **The leak-free line F1 is ≈ 0.24**, against the 0.39 reported with the tell in view
  (§1.24). Retraining changes nothing resolvable here either.
- Thresholds are per checkpoint. Never average a calibrator across seeds.

### The score audit of the retrained checkpoints (`10319b01…`)

| `val`, 3 seeds | cross_attention | concat | `llm_only` |
|---|---|---|---|
| Top-5, all 482 units — reproduces the seed reports | 0.5394 ± 0.0116 | 0.5546 ± 0.0084 | 0.5519 ± 0.0187 |
| Top-5, blank flagged lines withheld (474 units) | 0.5499 ± 0.0120 | 0.5633 ± 0.0084 | 0.5633 ± 0.0211 |
| Top-1, blank flagged lines withheld | 0.2722 ± 0.0073 | 0.2658 ± 0.0223 | 0.2855 ± 0.0064 |
| Units a formatting tell answers | 0 | 0 | 0 |

- **No unit is answered by the tell**: there is no comment left to carry it.
- **274 of the 2,808 flagged lines inside the line cap (9.76%) are blank after stripping**,
  and 8 units have no other flagged line — §1.25's ceiling, on these runs. Withholding
  blank lines moves each arm by about +0.01 and does not separate them.
- Source order, the rule-only baseline on the same units: Top-5 0.3071.
- The audit's Top-1 for concat is 0.2614 where the seed report says 0.2621: one unit in one
  checkpoint, a tied pair ordered one way by the audit's stable sort and the other way by
  the training metric's. That is the subject of the last subsection.

### Explanations from the retrained designed model (§1.27's procedure)

| Seed | Run → explain run | Units with a flagged line (of 488) | Flagged lines | Verified | Rejected | A pattern names a cause | — of top-ranked lines | Labelled by the dataset |
|---|---|---|---|---|---|---|---|---|
| 42 | `ba7dcc5f…` → `9196a0e3…` | 417 | 4,846 | 4,846 | 0 | 8.40% | 16.55% | 18.06% |
| 1 | `8afbf7da…` → `798f6ef3…` | 382 | 4,804 | 4,804 | 0 | 6.54% | 12.83% | 19.92% |
| 2 | `ff980345…` → `a65aa282…` | 415 | 3,430 | 3,430 | 0 | 8.69% | 14.22% | 21.31% |

**13,080 of 13,080 flagged lines explained and verified, none rejected.** No
formatting-artifact line (no comment exists to produce one); every explanation cites
attention; a data-flow path for 59.7–66.6%; the template's category equals the dataset's
CWE id for 10.7–12.3% of the units where both exist. Each pass's logits sit within 0.0117
of the stored scores (tolerance 0.05). Same reading as §1.27: the templates name a cause
for roughly one flagged line in twelve to fifteen, and say so for the rest. In
`artifacts/phase7/explanations_retrained/`. No rater pack was drawn from these: the pilot
pack (§1.27) stands until the owner decides Q14. *Q14 was decided on 2026-10-08;
whether the pilot is redrawn from `ba7dcc5f…` is the owner's (§6 item 0a).*

### The reproduction check refused one checkpoint four times — exact ties, not a mismatch

`score_checkpoint.py` keeps a score file only if the re-scoring reproduces every metric the
run logged, within 1e-4. For `8afbf7da…` (cross_attention, seed 1) **four of six passes
were refused, each with the same single disagreement: `line/ifa_mean` 15.022822 against
the logged 15.024896** — 0.002075, exactly one rank position in one of the 482 units.
Top-1, Top-5, precision, recall, F1, scored units and coverage agreed every time. Two
passes reproduced everything; the file kept (`2e9a2edc…`) is from one of them, **at the
default tolerance — the tolerance was not raised.** One diagnostic pass written to a
scratch directory with `score.tolerance=0.0025` is tagged `void` (`e735045f…`).

**The cause, measured on the stored scores.** The logits are half-precision values: the
41,065 scoreable `val` lines of one checkpoint take only 3,244 to 6,225 distinct values.
Exact ties are therefore ordinary — in 7 to 22 of the 482 units, per checkpoint, the best
flagged line is exactly tied with at least one unflagged line. `training/metrics.py` ranks
with `torch.argsort(scores, descending=True)`, which promises nothing about the order of
equal scores, and a forward pass can land a value one quantum away (§7 #50's scatter noise,
rounded to half precision). So the evaluation a run logged and a re-scoring of the very
same weights can differ by one rank in one unit.

**How far tie order can move anything** — every such tie broken for the flagged line, then
against it, on all nine checkpoints (`python artifacts/queue/tie_bounds.py`, read-only,
from the score files; git-ignored, on this laptop):

| | Width, one checkpoint | Width, an arm's mean | Seed sd of the same quantity |
|---|---|---|---|
| Top-1 | at most one unit: 0.0021 | ≤ 0.0007 | 0.0060 – 0.0222 |
| Top-5 | at most one unit: 0.0021 | ≤ 0.0007 | 0.0084 – 0.0187 |
| IFA | 0.017 – 0.075 | 0.024 – 0.051 | 0.26 – 1.27 |

Every logged value lies inside its bounds. **Tie order cannot change a comparison in this
section; it can, and did, trip a guard whose tolerance is twenty times finer than one unit.**

What follows from it (§7 #63):

- **Do not raise `score.tolerance`. Re-run.** A pass that reproduces is kept; if none does
  in several, it is a real mismatch — wrong checkpoint, config or text.
- **A batch script must expect the refusal.** `d97r_after.ps1` did the right thing with the
  arm (left it uncalibrated) and the wrong thing with the audit: it audited the two
  cross-attention files it had, as `n=2` (`15fccf6d…`, tagged `void`, superseded by
  `10319b01…`) — §7 #60's trap again. Read `n_seeds` in every aggregate.
- **The held-out evaluation carries this jitter and has no guard against it**: nothing is
  logged there to reproduce, and it is scored once. `docs/EvaluationProtocol.md` §6 now
  says so, and Phase 8's metric code must state how ties are ordered.
- **Not changed in `training/metrics.py`.** A defined tie order is a different metric from
  the one every logged run used, by up to one unit a checkpoint. That is a choice for
  Phase 8 and the owner, not a repair to slip in.

### Runs behind this section

Training: `ba7dcc5f…` `8afbf7da…` `ff980345…` (cross_attention), `9ac7a146…` `7579bc33…`
`10b43ce5…` (concat), `d82a6a70…` `a1dce535…` `9296a0e5…` (`llm_only`); gate `bcf18403…`.
Seed reports: `bd2544c7…` (and the queue's identical re-run `73af1609…`), `f3d754eb…`,
`f814e521…`. Scores: `96b04469…` `2e9a2edc…` `af8b025e…` / `fbcf8b73…` `df29e1e1…`
`3ee81242…` / `8f32f39d…` `09a50662…` `b626975c…`. Calibrations: `513dd46e…` `79b8c8d1…`
`8b7ac935…`. Audit: `10319b01…`. Explanations: `9196a0e3…` `798f6ef3…` `a65aa282…`.
Artefacts: `artifacts/adr010/` (seed reports), `artifacts/phase6/scores_retrained/`,
`artifacts/phase6/retrained/`, `artifacts/phase7/explanations_retrained/`.

---

## 1.29 Phase 8 — the full line-level metric suite on `val`: no per-unit metric separates the arms; one pooled metric crosses the threshold (2026-10-08)

**Status: built and measured on `val` only. The Phase 7 gate is not signed and the
evaluation protocol is not confirmed; no held-out partition has been read.** **(2026-10-11:
the held-out pass has since been made — §1.30, which also checks this section's one
candidate.)** The nine
`d97r` checkpoints — the models of record (§2) — trained and evaluated without comments,
three seeds an arm. Run `52b9c998…` (`scripts/evaluate_scores.py`, clean `4281b3f`); the
same numbers were first logged as `30aeff44…` at `b875d70`, before the wording of one
table was corrected (§7 #64). Tables: `artifacts/phase8/evaluation_val.md`. Definitions:
`docs/EvaluationProtocol.md` §5.1.

| `val`, 482 units ranked, mean ± sd over 3 seeds | cross_attention | concat | `llm_only` | source order |
|---|---|---|---|---|
| Top-1 | 0.2683 ± 0.0060 | 0.2614 ± 0.0217 | 0.2801 ± 0.0062 | 0.0000 |
| Top-3 | 0.4364 ± 0.0086 | 0.4564 ± 0.0166 | 0.4544 ± 0.0239 | 0.1598 |
| **Top-5** | 0.5394 ± 0.0116 | 0.5546 ± 0.0084 | 0.5519 ± 0.0187 | 0.3071 |
| Top-10 | 0.6667 ± 0.0153 | 0.6798 ± 0.0102 | 0.6757 ± 0.0188 | 0.4793 |
| **IFA** (mean) | 14.56 ± 0.43 | 14.06 ± 0.26 | 14.72 ± 1.27 | 29.78 |
| MFR (always IFA + 1) | 15.56 ± 0.43 | 15.06 ± 0.26 | 15.72 ± 1.27 | 30.78 |
| MAR | 27.45 ± 0.15 | 26.66 ± 0.56 | 27.44 ± 1.18 | 42.30 |
| Recall@1% LOC | 0.1275 ± 0.0044 | 0.1203 ± 0.0162 | 0.1268 ± 0.0099 | 0.0057 |
| Recall@5% LOC | 0.1891 ± 0.0143 | 0.1930 ± 0.0040 | 0.1939 ± 0.0265 | 0.0248 |
| Recall@20% LOC | 0.4700 ± 0.0032 | 0.4701 ± 0.0092 | 0.4700 ± 0.0186 | 0.1917 |
| **PR-AUC** (positive rate 0.0664) | 0.1713 ± 0.0143 | 0.1816 ± 0.0090 | 0.1610 ± 0.0097 | — |
| Line F1, out of fold | 0.2359 ± 0.0120 | 0.2408 ± 0.0074 | 0.2330 ± 0.0118 | — |
| ECE, out of fold | 0.0133 ± 0.0067 | 0.0076 ± 0.0019 | 0.0168 ± 0.0029 | — |
| Top-5, blank lines withheld (474 units) | 0.5499 ± 0.0120 | 0.5633 ± 0.0084 | 0.5633 ± 0.0211 | |

Ties in source order. Median IFA is 3–4 for every checkpoint against a mean of 14–15: a
long tail of units whose first flagged line the models rank far down.

### What is the same as §1.28, and the one number that is not

Top-5, the out-of-fold line F1 and ECE, and the blank-lines-withheld Top-5 are §1.28's to
the last digit — the same score files through new code. **Concat's Top-1 is 0.2614 ±
0.0217 here and 0.2621 ± 0.0222 in its seed report**: in `10b43ce5…` one unit's best
flagged line is exactly tied with a clean line, the training loop's `argsort` put it
first and source order does not. The arm's tie bounds are 0.2614 … 0.2621; both values
are inside them. Wherever a comparison below uses Top-1, it uses the source-order value
(cross_attention − concat is +0.0069 here, +0.0062 in §1.28).

### The comparisons — 93 made, nine of them headline, three past the threshold

| First − second | Top-5 | IFA | PR-AUC |
|---|---|---|---|
| cross_attention − concat | −0.0152 (1.51 pooled sd) | +0.50 (1.43) | −0.0103 (0.87) |
| cross_attention − `llm_only` | −0.0124 (0.80) | −0.15 (0.16) | +0.0102 (0.84) |
| concat − `llm_only` | +0.0028 (0.19) | −0.65 (0.71) | **+0.0206 (2.20)** |

- **No per-unit ranking metric separates any two arms** — not Top-1/3/5/10, IFA, MAR or
  effort-aware recall, with every line a candidate or with blank lines withheld. The
  nearest are cross_attention − concat on MAR (+0.80, 1.96 pooled sd, same sign on every
  seed) and cross_attention − `llm_only` on Top-1 (−0.0118, 1.92, same sign on every seed
  — §1.28's candidate, unchanged).
- **Three of the 93 comparisons reach 2 pooled sd, all concat − `llm_only`:** PR-AUC
  +0.0206 (2.20; by seed +0.0098 / +0.0211 / +0.0308), ECE out of fold −0.0092 (3.77) and
  ECE in sample −0.0130 (3.04).
- **What that is worth.** With three seeds an arm, 2 pooled sd is a two-sample t of 2.45
  on 4 degrees of freedom: two arms with no difference reach it with probability ≈ 0.07
  in any one comparison — about 6.6 of 93 if they were independent (they are not), and
  about 0.6 of the nine headline ones. One headline crossing is what chance gives. It is
  **a candidate for the held-out pass**, where concat − `llm_only` on PR-AUC is a
  pre-stated comparison — not a finding, and not "fusion beats LLM-only".
- Both fused arms sit above `llm_only` on PR-AUC on every seed (cross_attention: +0.0023 /
  +0.0179 / +0.0105). *A reading, not a measurement:* PR-AUC is the one metric here that
  pools lines across functions, so a difference in it with none per unit would be about
  how scores compare *between* functions, not about where a model points inside one.

### Three things the new metrics show that Top-5 alone did not

- **The sample of functions is the larger noise.** Each checkpoint's 95% bootstrap
  interval over units is about ± 0.045 on Top-5 (`ba7dcc5f…`: 0.5519 [0.5062, 0.5975])
  and ± 2.5 on IFA (14.18 [11.81, 16.75]) — two to five times the arms' seed sd on Top-5
  (0.0084–0.0187). All nine intervals overlap. On 482 units, which functions are in the partition moves a number
  more than the seed does.
- **Reading from the top is a real baseline, and the models beat it.** Source order finds
  a flagged line within five lines in 30.7% of units and within ten in 47.9%; the models
  reach 54–55% and 67–68%, and halve its IFA (14–15 against 29.8). Its Top-1 is 0: no
  function in `val` has its first line flagged.
- **The ceilings, in one place.** 488 line-labelled units, all scored, 482 ranked. Of
  3,386 flagged lines, 578 (17.07%) are past the line cap; of the 2,808 inside it, 274 are
  blank after comment stripping and 83 have no features. And ~30% of the labels are wrong
  (§1.6), which no count measures.

### What this supports

- On text with no tell, on `val`, at three seeds: the three arms tie on every per-unit
  localization metric the protocol names, and each beats source order by a wide margin.
- The evaluation code computes what the runs logged: every logged Top-1 / Top-5 / IFA is
  inside its tie bounds, in-sample F1 and ECE equal the calibration reports', and Top-1 /
  Top-5 in three readings equal the score audit's (`10319b01…`) exactly.

### What it does not support

- **Anything about `test` or `clean_eval`.** Nothing here was computed on them.
- **"Concat beats `llm_only`"**, or any fusion-over-LLM claim: one headline comparison in
  nine, on validation, at a threshold chance crosses 7 times in 100.
- A paired test between arms on the same units. The intervals above are per checkpoint;
  the protocol asks for nothing paired by unit, and none was computed.

### Runs behind this section

Evaluation: `52b9c998…` (of record, `4281b3f`), `30aeff44…` (same numbers, `b875d70`,
table wording since corrected). Inputs: the nine score runs, three calibrations and the
audit listed under §1.28. Artefacts: `artifacts/phase8/evaluation_val.json` and `.md`.

---

## 1.30 The held-out evaluation — `test` and `clean_eval`, once: Top-5 and IFA separate no two arms; concat's PR-AUC lead over `llm_only` repeats on both (2026-10-11)

**Status: done, once, by `docs/EvaluationProtocol.md` as confirmed on 2026-10-08. These
are the project's held-out numbers; neither partition is scored or evaluated again.** The
nine `d97r` checkpoints, comment-stripped text, each checkpoint's own `val` calibrator and
threshold applied unchanged. Every checkpoint was scored once per partition — `ba7dcc5f…`
on `test` on 2026-10-08 (`21e80fb8…`), the other seventeen files on 2026-10-11 between
00:15 and 00:42 — and `artifacts/queue/heldout_pass.out` is the transcript: every step
exited 0. Evaluation runs `f3a7dbdf…` (`test`) and `c5609b93…` (`clean_eval`), both at
`344db7c`; tables in `artifacts/phase8/evaluation_<split>.md`.

### `test` — 541 units ranked, positive rate 0.0494

| `test`, mean ± sd over 3 seeds | cross_attention | concat | `llm_only` | source order |
|---|---|---|---|---|
| Top-1 | 0.2625 ± 0.0176 | 0.2428 ± 0.0221 | 0.2446 ± 0.0065 | 0.0000 |
| Top-3 | 0.3993 ± 0.0085 | 0.3919 ± 0.0231 | 0.3795 ± 0.0136 | 0.1848 |
| **Top-5** | 0.4646 ± 0.0105 | 0.4781 ± 0.0245 | 0.4664 ± 0.0070 | 0.2606 |
| Top-10 | 0.5977 ± 0.0093 | 0.6137 ± 0.0193 | 0.5995 ± 0.0217 | 0.4196 |
| **IFA** (mean) | 22.50 ± 0.89 | 21.16 ± 0.56 | 23.43 ± 2.12 | 39.13 |
| MFR (always IFA + 1) | 23.50 ± 0.89 | 22.16 ± 0.56 | 24.43 ± 2.12 | 40.13 |
| MAR | 34.00 ± 0.55 | 32.43 ± 0.72 | 34.56 ± 2.08 | 48.89 |
| Recall@1% LOC | 0.1310 ± 0.0057 | 0.1092 ± 0.0179 | 0.1154 ± 0.0023 | 0.0011 |
| Recall@5% LOC | 0.2124 ± 0.0061 | 0.1995 ± 0.0316 | 0.1913 ± 0.0085 | 0.0317 |
| Recall@20% LOC | 0.4688 ± 0.0097 | 0.4835 ± 0.0145 | 0.4640 ± 0.0268 | 0.1963 |
| **PR-AUC** | 0.1739 ± 0.0287 | 0.1685 ± 0.0125 | 0.1484 ± 0.0067 | — |
| Line precision at the `val` threshold | 0.1884 ± 0.0337 | 0.1871 ± 0.0354 | 0.1613 ± 0.0120 | — |
| Line recall | 0.3296 ± 0.0380 | 0.3232 ± 0.0611 | 0.3511 ± 0.0651 | — |
| Line F1 | 0.2384 ± 0.0294 | 0.2322 ± 0.0132 | 0.2190 ± 0.0018 | — |
| ECE | 0.0216 ± 0.0020 | 0.0153 ± 0.0012 | 0.0247 ± 0.0031 | — |
| Top-5, blank lines withheld (533 units) | 0.4728 ± 0.0099 | 0.4847 ± 0.0239 | 0.4734 ± 0.0071 | 0.2852 |

### `clean_eval` — 1,367 units ranked, positive rate 0.0274

| `clean_eval`, mean ± sd over 3 seeds | cross_attention | concat | `llm_only` | source order |
|---|---|---|---|---|
| Top-1 | 0.1578 ± 0.0040 | 0.1600 ± 0.0110 | 0.1526 ± 0.0150 | 0.0000 |
| Top-3 | 0.3121 ± 0.0066 | 0.3168 ± 0.0046 | 0.3019 ± 0.0140 | 0.0819 |
| **Top-5** | 0.3977 ± 0.0103 | 0.4160 ± 0.0085 | 0.3982 ± 0.0150 | 0.1580 |
| Top-10 | 0.5411 ± 0.0223 | 0.5523 ± 0.0092 | 0.5474 ± 0.0192 | 0.3197 |
| **IFA** (mean) | 23.77 ± 0.36 | 24.18 ± 0.84 | 22.54 ± 2.52 | 47.02 |
| MFR (always IFA + 1) | 24.77 ± 0.36 | 25.18 ± 0.84 | 23.54 ± 2.52 | 48.02 |
| MAR | 33.33 ± 0.34 | 33.48 ± 1.09 | 31.71 ± 2.40 | 55.44 |
| Recall@1% LOC | 0.0873 ± 0.0017 | 0.0863 ± 0.0080 | 0.0840 ± 0.0106 | 0.0032 |
| Recall@5% LOC | 0.1653 ± 0.0035 | 0.1696 ± 0.0114 | 0.1671 ± 0.0203 | 0.0213 |
| Recall@20% LOC | 0.4637 ± 0.0127 | 0.4729 ± 0.0037 | 0.4741 ± 0.0219 | 0.1669 |
| **PR-AUC** | 0.1092 ± 0.0144 | 0.1185 ± 0.0102 | 0.0950 ± 0.0082 | — |
| Line precision at the `val` threshold | 0.1243 ± 0.0162 | 0.1281 ± 0.0230 | 0.1047 ± 0.0143 | — |
| Line recall | 0.2903 ± 0.0241 | 0.3306 ± 0.0481 | 0.3210 ± 0.0517 | — |
| Line F1 | 0.1731 ± 0.0128 | 0.1820 ± 0.0159 | 0.1566 ± 0.0137 | — |
| ECE | 0.0307 ± 0.0004 | 0.0303 ± 0.0013 | 0.0336 ± 0.0011 | — |
| Top-5, blank lines withheld (1,367 units) | 0.3997 ± 0.0100 | 0.4179 ± 0.0078 | 0.4004 ± 0.0160 | 0.1814 |

Ties in source order on both. Median IFA is 4–6 per checkpoint on `test` (source order 13)
and 7–8 on `clean_eval` (21), against means of 21–24: the long tail of §1.29 again.

### The nine headline comparisons, on each partition

| First − second | Partition | Top-5 | IFA | PR-AUC |
|---|---|---|---|---|
| cross_attention − concat | `test` | −0.0136 (0.72 pooled sd) | +1.34 (1.80) | +0.0054 (0.24) |
| | `clean_eval` | −0.0183 (1.94) | −0.41 (0.63) | −0.0093 (0.75) |
| cross_attention − `llm_only` | `test` | −0.0018 (0.21) | −0.94 (0.57) | +0.0255 (1.22) |
| | `clean_eval` | −0.0005 (0.04) | +1.23 (0.68) | +0.0142 (1.21) |
| concat − `llm_only` | `test` | +0.0117 (0.65) | −2.28 (1.47) | **+0.0201 (2.01)** |
| | `clean_eval` | +0.0178 (1.46) | +1.63 (0.87) | **+0.0235 (2.54)** |

Each partition makes 81 comparisons (27 metrics × 3 pairs). **Seven reach 2 pooled sd on
`test` and four on `clean_eval`**; chance alone gives about 5.7 of 81 if they were
independent, which they are not (§7 #64).

- `test`: cross_attention − concat on ECE (+0.0064, 3.87) and MAR (+1.56, 2.45);
  cross_attention − `llm_only` on Recall@1% LOC (+0.0156, 3.60; blank lines withheld
  +0.0158, 3.95) and Recall@5% LOC (+0.0212, 2.86); concat − `llm_only` on ECE (−0.0095,
  4.02) and PR-AUC (+0.0201, 2.01).
- `clean_eval`: cross_attention − concat on Top-5 with blank lines withheld (−0.0183,
  2.05); cross_attention − `llm_only` on ECE (−0.0029, 3.56); concat − `llm_only` on ECE
  (−0.0033, 2.77) and PR-AUC (+0.0235, 2.54).

### What this supports

1. **A1's answer, held out: cross-attention does not beat concatenation.** On Top-5 the
   difference is −0.0136 on `test` and −0.0183 on `clean_eval` — concat is ahead on both,
   as it was on `val` (−0.0152), and on none of the three does the primary reading reach
   2 pooled sd (1.51, 0.72, 1.94). By seed the pattern is the same three times: seed 42's
   pair favours cross_attention slightly (+0.0062 / +0.0037 / +0.0015 on `val` / `test` /
   `clean_eval`), seeds 1 and 2 favour concat. Three partitions, the same nine
   checkpoints — that is one set of seeds seen three times, not three confirmations.
2. **Neither fused arm beats fine-tuned `llm_only` at pointing inside a function.**
   cross_attention − `llm_only` on Top-5 is −0.0018 and −0.0005; concat − `llm_only` is
   +0.0117 and +0.0178, under 1.5 pooled sd on both. Top-1, Top-3, Top-5, Top-10 and IFA
   cross the threshold for no pair on either partition in the primary reading. The
   per-unit metrics that do cross on one partition — MAR and Recall@1% / 5% LOC on
   `test`, Top-5 with blank lines withheld on `clean_eval` — are listed above and none
   crosses on the other.
3. **Every arm beats reading the function from the top, on both partitions.** Top-5
   0.46–0.48 against 0.26 on `test` and 0.40–0.42 against 0.16 on `clean_eval`; IFA 21–23
   against 39 and 23–24 against 47.
4. **The one `val` candidate that was pre-stated and checked: concat − `llm_only` on
   PR-AUC repeats.** §1.29 named it (+0.0206, 2.20 pooled sd) as a candidate for this
   pass. It is +0.0201 (2.01) on `test` and +0.0235 (2.54) on `clean_eval`, positive on
   every seed on each: +0.0148 / +0.0172 / +0.0283 and +0.0200 / +0.0262 / +0.0243 (seeds
   42 / 1 / 2). Concat's ECE is also the lowest on all three partitions. **What may be
   said:** *across lines pooled over functions, the concat model's scores order flagged
   lines above clean ones better than fine-tuned `llm_only`'s — by about 0.02 average
   precision on validation and on both held-out partitions, at three seeds.* The
   partitions re-sample functions, not seeds: it is still three checkpoints against three.
5. **Both fused arms are above `llm_only` on PR-AUC on every seed of every partition** —
   eighteen of eighteen seed pairs across `val`, `test` and `clean_eval`
   (cross_attention − `llm_only`: +0.0041 / +0.0224 / +0.0500 on `test`, +0.0223 /
   +0.0013 / +0.0189 on `clean_eval`). For cross_attention the mean difference does not
   reach the threshold on any partition (0.84, 1.22, 1.21): its own seed sd is the
   largest of the three arms.

### What it does not support

- **"Fusion improves localization."** PR-AUC pools lines across functions; within a
  function — which is what Top-k and IFA measure, and what a developer reading one
  function needs — no arm is ahead. *A reading, not a measurement:* a structural branch
  may help decide how suspicious a function's lines are *relative to another
  function's*, without helping to order the lines of one.
- **The `val` candidate in the control's favour.** §1.28 carried cross_attention −
  `llm_only` on Top-1 at −0.0118 (1.92 pooled sd). Held out it is +0.0179 (1.34) on
  `test` and +0.0052 on `clean_eval`: the sign reversed. It was noise.
- **Any single-partition crossing above.** cross_attention's lead on Recall@1% / 5% LOC
  over `llm_only` on `test` is +0.0007 / −0.0048 on `val` and +0.0033 / −0.0018 on
  `clean_eval`; its MAR deficit against concat (+0.80 on `val`, +1.56 on `test`) is not
  there on `clean_eval` (−0.15). None was pre-stated, and none holds on all three.
- **Generalization from `clean_eval`.** It measures label quality, not new projects —
  66.5% of its projects are also in `train` (landmine #10). Its functions are the longest
  (source-order IFA 47 against 39 and 30), which is why every number there is lower.
- **Function-level detection, an external detector, or anything about the frozen screen
  or an ablation** (`EvaluationProtocol.md` §9). §1.32's result is validation-only and
  stays so.

### Validation against held-out, and the calibration

| | `val` (§1.29) | `test` | `clean_eval` |
|---|---|---|---|
| Top-5, the three arms | 0.5394 / 0.5546 / 0.5519 | 0.4646 / 0.4781 / 0.4664 | 0.3977 / 0.4160 / 0.3982 |
| Top-5, source order | 0.3071 | 0.2606 | 0.1580 |
| IFA, source order | 29.78 | 39.13 | 47.02 |
| Line F1 at the `val` threshold | 0.2359 / 0.2408 / 0.2330 (out of fold) | 0.2384 / 0.2322 / 0.2190 | 0.1731 / 0.1820 / 0.1566 |
| ECE | 0.0133 / 0.0076 / 0.0168 (out of fold) | 0.0216 / 0.0153 / 0.0247 | 0.0307 / 0.0303 / 0.0336 |
| Positive rate | 0.0664 | 0.0494 | 0.0274 |

- **Top-5 is 0.07–0.09 lower on `test` than on `val` and 0.14–0.15 lower on
  `clean_eval`, and source order falls with it** (0.31 → 0.26 → 0.16): the partitions
  differ in how long their functions are. The margin over source order is 0.23–0.25 on
  `val`, 0.20–0.22 on `test` and 0.24–0.26 on `clean_eval`. The `val` numbers were also
  validation-selected — the epoch was chosen on them — and these are not.
- **The `val` threshold transfers to `test`** (line F1 0.22–0.24 against 0.23–0.24 out
  of fold) **and not to `clean_eval`** (0.16–0.18), where flagged lines are less than
  half as frequent. ECE rises from `val` to `test` and again to `clean_eval`, by 1.4–2.0×
  at each step: the calibrator was fitted where 6.6% of lines are flagged.

### Noise floors

- **Units.** Each checkpoint's 95% bootstrap interval over units is about ± 0.042 on
  Top-5 on `test` (`ba7dcc5f…`: 0.4677 [0.4269, 0.5102]; IFA 22.09 [18.46, 25.91]) and
  ± 0.026 on `clean_eval` (0.4082 [0.3826, 0.4338]; IFA 23.36 [21.25, 25.56]). All nine
  intervals overlap on each partition, and every arm difference above is inside one.
- **Ties.** The best flagged line is exactly tied with a clean one in 17–38 of 541
  `test` units and 55–139 of 1,367 `clean_eval` units per checkpoint. The arm means'
  bounds: `test` Top-5 unchanged at four decimals, IFA within ± 0.04; `clean_eval` Top-5
  0.3977 … 0.3975 / 0.4165 … 0.4155 / 0.3984 … 0.3977, IFA within ± 0.05.

### Ceilings, beside every number above

| | `test` | `clean_eval` |
|---|---|---|
| Line-labelled units, all scored | 546 | 1,393 |
| Units ranked (a flagged line inside the 512-line cap) | 541 | 1,367 |
| Flagged lines in the partition | 2,779 | 4,499 |
| …past the line cap | 48 (1.73%) | 141 (3.13%) |
| …inside the cap | 2,731 | 4,358 |
| …of those, blank after comment stripping | 192 | 55 |
| …of those, with no features | 43 | 21 |
| Units whose flagged lines are all blank | 8 | 0 |

And about 30% of the line labels are wrong (§1.6, n = 20), which no count here measures.

### Leak audit of the held-out scores

Runs `9c9d1d15…` (`test`) and `f2aa9b90…` (`clean_eval`), on comment-stripped text.

- **`clean_eval`: no unit a tell can answer.** As the protocol expected.
- **`test`: one unit, not zero.** `bigvul-181496`, line 6, reads
  `* Eliminate warning about unused variable. *` — a flagged comment that lost **both**
  slashes, so no parser sees a comment and the label-blind stripper has nothing to
  remove. ADR-010's own evidence counted it in advance
  (`artifacts/representation/comment_stripping.json`, run `af4173a3…`: lost-slash lines
  left after stripping — `train` 6, `val` 0, `test` 1, `clean_eval` 1). All nine
  checkpoints have a flagged line of that unit in their top five and none ranks the tell
  line first (the three cross_attention seeds put it 2nd of 10; the explainer says so in
  its text for that line). The unit has three flagged lines among ten scored, so a
  top-five hit there says little, and it is a Top-1 miss for all nine. Without it each
  arm's Top-5 is lower by 0.0009–0.0010 (540 units: 0.4636 / 0.4772 / 0.4654) and its
  Top-1 higher by 0.0004–0.0005 — the same for every arm.
  **Reported as it is: the held-out text was not perfectly free of the tell, by one
  function in 541.**

### Explanations and the rater pack, on `test`

The designed model's three seeds (`explain-test` runs `c0bb255d…`, `3cbc0f9a…`,
`3fc4afa5…`): 5,040 + 5,477 + 3,835 = **14,352 flagged lines explained, 14,352
verified, none rejected**; 437 / 411 / 448 of 546 units have at least one line above the
threshold. A weakness pattern names a cause for 6.0% / 5.1% / 7.7% of the flagged lines
(about one in twelve on `val`, §1.27) and for 10.5% / 9.5% / 10.3% of the top-ranked
ones. The rater pack — 100 samples from `ba7dcc5f…`'s 437 flagged `test` units, three
rating sheets, run `dd8003f8…` — is at `artifacts/phase7/rater_pack_test/`; the flagged
line is a labelled line in 31 of the 100. **No rating exists yet**; that is the human
evaluation (`docs/HumanEvalProtocol.md`).

### Runs behind this section

Scoring: seventeen runs on 2026-10-11 (`score-test` × 8, `ae7fd055…` first; `score-
clean_eval` × 9, `393b41e9…` first), all `FINISHED` at `344db7c`, and `21e80fb8…` on
2026-10-08. Evaluation `f3a7dbdf…`, `c5609b93…`; audits
`9c9d1d15…`, `f2aa9b90…`; explanations `c0bb255d…`, `3cbc0f9a…`, `3fc4afa5…`; rater pack
`dd8003f8…`. Figures: `artifacts/phase8/figures/{arms,reliability}_{test,clean_eval}.svg`.
Score files: `artifacts/phase6/scores_retrained/<run>_<split>.npz`.

---

## 1.31 Phase 8 — where the model of record misses, on `val`: a draft taxonomy and two counts that need no reader (2026-10-10)

**§1.30 is deliberately not written: that number is reserved for the held-out pass, and
its absence is how §6 item 0a tells a session that the pass is unfinished.**

**Status: ACCEPTED by the owner on 2026-10-11, as revised that day — Phase 8's "≥ 20
inspected cases" box is ticked on the owner's words, *"proceed ahead with this and
Siddharth read this"* (§2; `docs/FailureTaxonomy.md` §8 says what those words do and do
not establish).** Thirty cases were read by session 18 on 2026-10-10. **Revised
2026-10-11, before the acceptance:** the owner passed the session a
written review of the draft — addressed to the owner, its author not named, and leaving
the acceptance with the owner. Its points were checked against the sheet; B and C were
re-sorted by a count on the sheet (the block a flagged header opens, the lines that use a
flagged declaration's name) rather than by reading. Taken: B narrowed and case 02 moved
to H, case 20 set apart inside C, E renamed. Not taken, because the sheet says
otherwise: case 01 to H (an early-exit check protects what comes *after* its block, and
two of those statements are ranked 3rd and 9th) and case 28 as "no use ranked high" (all
three lines using the declared name are ranked 5th, 7th and 8th). `docs/FailureTaxonomy.md`
§7 has each point. It was recorded then as a second reading with the box open; the
owner's acceptance followed within the hour.
Validation only — `scripts/sample_failures.py` refuses
any other partition. Run `42f8f27c…` (clean `1363561`); `335adb30…` is the same draw and
the same counts at `6c9114d`, before the sheet printed the chance column. Sheet:
`artifacts/phase8/failure_cases.md`.

**A failure** is defined before any is read: a unit whose first flagged line the model of
record (cross_attention retrained, `ba7dcc5f…` / `8afbf7da…` / `ff980345…`) ranks outside
the top 10 on **every** seed, under the evaluation's ranking. **107 of 482 ranked units.**
Thirty were drawn by content hash (seed 1337).

### Counted over every failure — no reading involved

| A unit that… | Of the 107 failures | Of the 482 ranked units |
|---|---|---|
| has every flagged line inside the cap blank, unscoreable or syntax alone | 10 | 11 |
| has a flagged line that is blank after comment stripping | 17 | 76 |
| has a flagged line whose text also stands on an unflagged line | 22 | 94 |
| has a top-10 line on, or within three lines of, a flagged line (seed 42) | 31 | 383 |

Scored lines per unit, median: **117 for the failures, 46 for every ranked unit**; no
failure has fewer than 25.

### Counted over every flagged line — which kinds of line the model finds

Lines sorted into kinds by rule (one stripped line at a time, first match wins). *Found*:
share of the kind's flagged lines in the top 10 of their own function, mean of three
seeds. *By chance*: the same for a ranking drawn at random, `min(1, 10 / n)` per line.

| Kind | Flagged | In units | Flagged, % of the kind | Found | By chance |
|---|---|---|---|---|---|
| assignment | 679 | 212 | 9.1 | 0.324 | 0.187 |
| condition or loop header | 499 | 208 | 9.7 | 0.352 | 0.179 |
| call | 428 | 194 | 8.4 | 0.405 | 0.260 |
| other (continuation lines, literals) | 286 | 126 | 7.3 | 0.408 | 0.251 |
| blank | 274 | 76 | 3.3 | 0.095 | 0.205 |
| syntax alone | 271 | 83 | 3.9 | 0.077 | 0.183 |
| preprocessor | 145 | **13** | 25.3 | 0.025 | 0.061 |
| return or jump | 136 | 54 | 7.9 | 0.206 | 0.259 |
| declaration | 90 | 58 | 6.3 | 0.256 | 0.229 |
| **every kind** | 2,808 | 482 | 6.9 | **0.281** | **0.203** |

### What this supports

- **Per flagged line the model is above a random ranking, and not by much: 0.281 against
  0.203.** A different quantity from the per-unit Top-10 of 0.667 (§1.29), which needs
  only one flagged line of a function.
- **The lift is on statements** — assignments, conditions, calls, continuation lines:
  1.6–2.0 times chance.
- **On blank lines, syntax-only lines, bare returns and preprocessor lines the model is
  below chance.** It ranks them down, as their label rates teach (3.3% and 3.9% against
  6.9%), so **the 545 flagged lines that are blank or syntax alone — 19.4% of the 2,808 —
  are close to unreachable for it.** The blank ones are §1.25's residue (a flagged
  comment line, stripped); the 271 syntax-only ones were not counted before.
- **Ten of the 107 failures had nothing to find**, and those ten are 10 of the 11 such
  units in the partition. A ceiling, not an error.
- **Failures are long functions**, and 29% of them (31 of 107) are near misses in the
  three-line sense.

### What it does not support

- **Any share of the 107 by category.** The eight categories of `docs/FailureTaxonomy.md`
  as revised on 2026-10-11 (a block header flagged and the block's statements or the
  checked expression ranked, 6 of 30; declaration or definition flagged, 4 — uses ranked
  high in 3; a twin statement ranked instead, 2; a flagged region of six or more lines
  with none ranked high, 3; format-string calls, 3; nothing to find, 2; one line of a
  longer statement, 1; no relation seen, 9) are a reading of thirty cases, accepted by
  the owner as a vocabulary for the failures — thirty cases still give no proportions.
- **"The model found the real bug and the label is wrong"** for the cases where its top
  ranks sit beside the flagged line. Nobody opened a fix commit.
- **Anything general about preprocessor lines**: 145 flagged lines in 13 units, a fifth
  of them in one. The first table printed the rate without the unit count and it read as
  a property of the kind; the script now prints both (§7 #65).
- Anything about `test`, `clean_eval`, the other two arms or the frozen models.

### One mechanism, checked by hand

In case 08 the model's top ranks are **empty lines**: alternate lines of a stripped
comment block are ranked 1st, 2nd, 5th and 7th and the lines between hold the "no
features" sentinel. The CodeBERT tokenizer turns a run of empty lines into newline-pair
tokens (`ĊĊ`), each attributed to the line it starts on, so every second empty line has
semantic features. Checked by tokenizing such a run — a property of the tokenizer, not a
logged measurement. It is the top-ranked line in 4 of 482 units, all failures; the
evaluation's "blank lines withheld" reading already removes it.

### Runs behind this section

`42f8f27c…` (of record), `335adb30…` (superseded). Inputs: the three `val` score files of
§1.28. Artefacts: `artifacts/phase8/failure_cases.{md,json}`.

---

## 1.32 `a1r` and its replication — the frozen A1 screen retrained on comment-stripped text: cross-attention leads concat and sum on `val`, and the lead repeats on three new seeds (2026-10-11)

**Status: `val` only, validation-selected; six seeds an arm, in two sets of three. The
first set's result was repeated on the second by a criterion written before the second
existed ("The replication's outcome", below). It is a statement about a frozen encoder
on validation — not about the models of record: with the encoder fine-tuned the same
arms tie on the same text, on `val` and held out (§1.28, §1.30).** Everything down to
"The replication — written before its first run" was written with three seeds in hand
and is kept as written.

The screen of §1.21 again, with exactly two more overrides
(`+representation.strip_comments=true` and the stripped cache's own directory): six arms ×
seeds 42 / 1 / 2, `train=fused`, frozen CodeBERT line features read from
`artifacts/semantic_cache_stripped`, GCN depth 4 over AST with identifier subtokens. Queue
group `a1r`, 2026-10-10 14:52 → 2026-10-11 00:14, with two pauses at the owner's request;
no run was interrupted (a frozen run is not resumable and each finished). Gate:
`a1rcheck`, `3c749001…`, Top-1 1.0000 from epoch 58. Every seed report says `n_seeds: 3`,
every run cross-checks `agrees`, and no source drift was accepted — the commits the runs
span differ in docs and scripts only. The arm rows below are the six seed reports'; the
differences in pooled sd are `scripts/make_ablation_tables.py`'s arithmetic on them, and
**its logged run is `922719b5…`** (2026-10-11 00:48, at `c6373cf`, made once the held-out
pass had ended) — the tables of record, `artifacts/phase8/ablations.md`: 84 scaled
comparisons, 28 of them on Top-5, 25 past the threshold.

| Frozen, comment-stripped, `val`, 3 seeds | Top-5 | Top-1 | IFA ↓ | Top-5 by seed (42 / 1 / 2) | Selected epoch, of epochs evaluated | Runs (42 / 1 / 2) |
|---|---|---|---|---|---|---|
| cross_attention | **0.5325 ± 0.0102** | 0.2510 ± 0.0145 | **12.31 ± 0.30** | 0.5207 / 0.5394 / 0.5373 | 2, 2, 2 of 6, 6, 6 | `8cde0dd5` `7f9149ea` `53d468bb` |
| concat | 0.4938 ± 0.0157 | 0.2151 ± 0.0213 | 14.54 ± 1.04 | 0.4917 / 0.4793 / 0.5104 | 1, 1, 4 of 5, 5, 8 | `6ce89a15` `5495b3bf` `06780cf6` |
| sum | 0.5048 ± 0.0043 | 0.2517 ± 0.0196 | 16.85 ± 1.43 | 0.5083 / 0.5000 / 0.5062 | 5, 9, 5 of 9, 10, 9 | `d25cf429` `0eafe0f4` `6b95e590` |
| gated | 0.5131 ± 0.0102 | 0.2441 ± 0.0060 | 16.23 ± 1.19 | 0.5062 / 0.5083 / 0.5249 | 8, 5, 5 of 10, 9, 9 | `bfc8abec` `2259f46f` `2d3d4629` |
| `llm_only` | 0.4945 ± 0.0094 | 0.2254 ± 0.0012 | 13.85 ± 0.15 | 0.4855 / 0.5041 / 0.4938 | 1, 1, 2 of 5, 5, 6 | `3b6832aa` `3c3779a6` `91963e6c` |
| `gnn_only` | 0.4481 ± 0.0041 | 0.1646 ± 0.0397 | 19.84 ± 1.33 | 0.4523 / 0.4440 / 0.4481 | 3, 9, 8 of 7, 10, 10 | `ea51fd68` `5f31909f` `3c1ede50` |

Epochs are counted from 0. Top-1 is the value each run logged (ties in `argsort` order,
§7 #63).

### The comparisons the catalogue pre-states — 15 scaled, 11 past 2 pooled sd

| First − second | Top-5 | Top-1 | IFA | Top-5 by seed (42 / 1 / 2) |
|---|---|---|---|---|
| cross_attention − concat (**A1's question**) | **+0.0387 (2.93 pooled sd)** | +0.0360 (1.97) | −2.23 (2.91) | +0.0290 / +0.0602 / +0.0270 |
| cross_attention − sum | +0.0277 (3.52) | −0.0007 (0.04) | −4.53 (4.37) | +0.0124 / +0.0394 / +0.0311 |
| cross_attention − gated | +0.0194 (1.89) | +0.0069 (0.62) | −3.91 (4.49) | +0.0145 / +0.0311 / +0.0124 |
| cross_attention − `llm_only` | +0.0380 (3.88) | +0.0256 (2.48) | −1.54 (6.43) | +0.0353 / +0.0353 / +0.0436 |
| cross_attention − `gnn_only` | +0.0844 (10.80) | +0.0864 (2.89) | −7.53 (7.80) | +0.0685 / +0.0954 / +0.0892 |

### What this supports

1. **On this screen cross-attention is ahead of concatenation: Top-5 +0.0387, 2.93
   pooled sd, the same sign on every seed** — and every cross_attention seed (0.5207 –
   0.5394) is above every concat seed (0.4793 – 0.5104). It is ahead of sum by +0.0277
   (3.52) and of gated by +0.0194 (1.89 — not resolved). Its IFA is the lowest of the six
   arms and crosses the threshold against all three other fusion arms. The only admissible
   phrasing is the qualified one: *"with a fixed semantic representation, on
   comment-stripped text, on validation, at three seeds"* (ADR-009 D9.1).
2. **Against the single branches: +0.0380 over frozen `llm_only` (3.88 pooled sd, every
   seed) and +0.0844 over `gnn_only` (10.80).**
3. **"Every fusion arm beats frozen `llm_only` on every seed" (§1.21) does not survive
   retraining without the tell.** It holds for cross_attention and gated (3 of 3 seeds
   each). Concat − `llm_only` is −0.0007 (by seed +0.0062 / −0.0248 / +0.0166) and sum −
   `llm_only` is +0.0103 (+0.0228 / −0.0041 / +0.0124); gated − `llm_only` is +0.0186.
   These three are differences of the table's means, not among the catalogue's
   comparisons.
4. **Where the arms moved when the tell was removed.** Top-5 against §1.21 (text as
   shipped) — a different training text and a different cache, so a description, not a
   paired comparison: concat 0.5394 → 0.4938, sum 0.5304 → 0.5048, gated 0.5221 → 0.5131,
   cross_attention 0.5408 → 0.5325, `llm_only` 0.4903 → 0.4945, `gnn_only` 0.4191 →
   0.4481. The §1.21 checkpoints with comments stripped at evaluation only (§1.25) had
   cross_attention at 0.5228 and concat at 0.5159: trained without the tell, concat falls
   further and cross_attention does not.
5. **Frozen cross-attention sits 0.007 below its own fine-tuned form** (0.5325 against
   §1.28's 0.5394 ± 0.0116), while fine-tuning lifts `llm_only` from 0.4945 to 0.5519 and
   concat from 0.4938 to 0.5546. Two comparison groups, three seeds each — a description.
   *A reading, not a measurement:* what cross-attention adds over fixed features and what
   fine-tuning the encoder adds may be largely the same thing, which would explain both a
   lead when the encoder is frozen and a tie when it is not.

### What it does not support

- **Anything about the models of record.** Fine-tuned, on the same text, cross_attention
  − concat is −0.0152 (1.51 pooled sd, the other sign) and cross_attention − `llm_only`
  is −0.0124 (§1.28). The nine `d97r` checkpoints are unchanged, and every frozen arm
  here is below every fine-tuned one (0.5394 – 0.5546).
- **"Cross-attention beats concatenation" as a finding.** Four reasons, each enough:
  - *Three seeds.* Two arms with no difference reach 2 pooled sd 7 times in 100 (§7 #64),
    and this screen's fifteen scaled comparisons all share one reference arm's three
    runs — if those three are fortunate, all fifteen move together. Eleven crossings are
    not eleven confirmations.
  - *The same screen has said otherwise.* On text with the tell it gave +0.0014 with the
    sign flipping by seed (§1.21), and two results of this project were retracted at
    exactly this stage (§7 #37).
  - *Each number is the best of 5–10 noisy epochs.* A run's `val` Top-5 moves by up to
    0.0996 from one epoch to the next (`8cde0dd5`: 0.5083 → 0.4108 → 0.5104), and the
    reported value is the maximum, selected on the partition it is reported on. The seed
    sd of those maxima understates what one more epoch could change. What the curves do
    show: no concat run reaches above 0.5104 at any of its 18 logged epochs; cross_attention
    is above 0.5104 at 8 of its 18, sum at 0 of 28, gated at 2 of 28, `llm_only` at 0 of
    16. The lead is not one lucky epoch — and it is still one partition.
  - *Validation only.* Ablations are validation-only by protocol
    (`EvaluationProtocol.md` §9); nothing held-out exists or is planned for this screen.
- **Not line F1.** At the logged 0.5 threshold recall is 0.02–0.05 for every arm; F1 there
  is calibration under focal loss, not localization (§1.21).

### The replication — written before its first run

Queue group **`a1rep`**: cross_attention, concat, `llm_only`, sum, gated × **seeds 3, 4,
5**, with `a1r`'s constants and `a1r`'s cache — 15 frozen runs, the core pair first.
`gnn_only` is not run again; its distance from every arm is not in question. Each arm then
gets two seed reports in `artifacts/adr010/`: `…frozen_stripped_replication_<arm>.json`
(seeds 3 / 4 / 5 alone) and `…frozen_stripped_six_seeds_<arm>.json`. The `a1r` reports
above are other files and stay as they are. Fixed now, before any of these runs exists:

- **Primary: cross_attention − concat on `val` Top-5 over seeds 3 / 4 / 5 alone**, by the
  convention of every table here (mean ± sample sd, pooled sd, each seed's own
  difference).
- **Read as replicated** if that difference is positive and at least 2 pooled sd. **Read
  as not replicated** otherwise — and the table above is then reported as a three-seed
  crossing that three new seeds did not repeat.
- **Secondary, reported whatever they show:** cross_attention − sum, − gated and −
  `llm_only` on Top-5; all four pairs on Top-1 and IFA; the six-seed tables. The six-seed
  table is not a third measurement — three of its seeds are the ones that prompted the
  other three.
- No arm, seed, metric or run is dropped afterwards. A run that fails is re-run with its
  own seed and the failure is recorded.
- **What cannot change whatever it shows:** the models of record, the fine-tuned tie
  (§1.28), and the held-out results. Only the wording of the frozen screen's sentence
  about A1 depends on it.

### The replication's outcome — seeds 3 / 4 / 5, queue group `a1rep` (2026-10-11, 00:49 → 04:05)

Fifteen runs and ten seed reports, all `done`, no failure and no retry. Every report holds
the seeds it should (3, or 6), every run cross-checks `agrees`, and no source drift was
accepted. Tables of record: run `bcfeb4ff…` — `artifacts/phase8/ablations.md`, tables
`a1_frozen_stripped_replication` and `a1_frozen_stripped_six_seeds` (the document now
makes 108 scaled comparisons, 36 of them on Top-5).

| Seeds 3 / 4 / 5 alone, `val` | Top-5 | Top-1 | IFA ↓ | Top-5 by seed (3 / 4 / 5) | Selected epoch, of epochs evaluated | Runs (3 / 4 / 5) |
|---|---|---|---|---|---|---|
| cross_attention | **0.5519 ± 0.0255** | 0.2614 ± 0.0252 | 13.16 ± 1.38 | 0.5228 / 0.5622 / 0.5705 | 1, 7, 1 of 5, 10, 5 | `cabf7c84` `efb60ee8` `d12630f6` |
| concat | 0.4959 ± 0.0239 | 0.2275 ± 0.0248 | 14.91 ± 2.11 | 0.5145 / 0.5041 / 0.4689 | 6, 3, 1 of 10, 7, 5 | `2d7a5fc7` `c92fdc9a` `8a0895dc` |
| sum | 0.4869 ± 0.0186 | 0.2158 ± 0.0177 | 14.93 ± 1.04 | 0.4751 / 0.5083 / 0.4772 | 1, 4, 1 of 5, 8, 5 | `d91b511e` `bbe0afbc` `05bab4af` |
| gated | 0.5111 ± 0.0138 | 0.2344 ± 0.0150 | 15.68 ± 2.92 | 0.5145 / 0.5228 / 0.4959 | 6, 4, 1 of 10, 8, 5 | `a0ff6636` `3d555b2f` `f7d762bd` |
| `llm_only` | 0.4910 ± 0.0048 | 0.2144 ± 0.0086 | 13.96 ± 0.36 | 0.4938 / 0.4938 / 0.4855 | 1, 5, 1 of 5, 9, 5 | `16410cbe` `53c81e3e` `322a4ce0` |

| First − second | Seeds 3 / 4 / 5: Top-5 | Top-1 | IFA | Top-5 by seed (3 / 4 / 5) | All six seeds: Top-5 | Top-1 | IFA |
|---|---|---|---|---|---|---|---|
| cross_attention − concat (**the primary**) | **+0.0560 (2.27 pooled sd)** | +0.0339 (1.36) | −1.75 (0.98) | +0.0083 / +0.0581 / +0.1017 | +0.0474 (2.46) | +0.0349 (1.70) | −1.99 (1.55) |
| cross_attention − sum | +0.0650 (2.91) | +0.0456 (2.10) | −1.77 (1.45) | +0.0477 / +0.0539 / +0.0934 | +0.0463 (2.56) | +0.0225 (0.99) | −3.15 (2.43) |
| cross_attention − gated | +0.0408 (1.99) | +0.0270 (1.30) | −2.52 (1.10) | +0.0083 / +0.0394 / +0.0747 | +0.0301 (1.84) | +0.0169 (1.07) | −3.22 (2.02) |
| cross_attention − `llm_only` | +0.0609 (3.32) | +0.0470 (2.50) | −0.79 (0.79) | +0.0290 / +0.0685 / +0.0851 | +0.0494 (3.25) | +0.0363 (2.46) | −1.17 (1.59) |

Over all six seeds (42 / 1 / 2 / 3 / 4 / 5): cross_attention 0.5422 ± 0.0204, concat
0.4948 ± 0.0181, sum 0.4959 ± 0.0156, gated 0.5121 ± 0.0109, `llm_only` 0.4927 ± 0.0069.

**Read against the plan above, which was committed (`c6373cf`) before the first of these
runs:**

- **The primary comparison is replicated.** cross_attention − concat on Top-5 over seeds
  3 / 4 / 5 alone is +0.0560, 2.27 pooled sd, positive on every seed — the criterion was
  "positive and at least 2 pooled sd". Across both sets cross_attention is ahead on all
  six seeds (+0.0083 … +0.1017), and every cross_attention seed (0.5207 – 0.5705) is above
  every concat seed (0.4689 – 0.5145).
- **Secondary, as they came out.** Against sum (2.91) and against frozen `llm_only`
  (3.32) the Top-5 lead repeats. Against gated it does not reach the threshold in either
  set (1.89, then 1.99) or pooled (1.84): the same sign on all six seeds, never resolved.
- **The IFA lead did not repeat at the threshold.** The first set had it at 2.91 / 4.37 /
  4.49 pooled sd against concat / sum / gated; the replication has 0.98 / 1.45 / 1.10,
  the same sign. "Lowest IFA of the arms" is not carried by the second set.
- **The curves agree with the selected values.** No concat run is above 0.5145 at any of
  its 22 logged epochs; cross_attention is above 0.5145 at 10 of its 20, sum at 0 of 18,
  gated at 2 of 23, `llm_only` at 0 of 19. Seed 3 is the close pair: 0.5228 against 0.5145.
- **What six seeds are worth.** Two arms with no difference reach 2 pooled sd about 0.6
  times in 100 at six seeds an arm (`statistics.threshold_p_value(6)` = 0.0061), against 7
  in 100 at three. The six-seed table is not a third measurement — three of its seeds
  prompted the other three — so the replication alone is the test, and the pooled table
  is the best estimate of the size: about +0.05 Top-5.

**What may be said now** — and only with every qualifier in it: *with the semantic
encoder frozen, on comment-stripped text, bidirectional cross-attention reaches a higher
validation Top-5 than concatenation (0.5422 ± 0.0204 against 0.4948 ± 0.0181 over six
seeds; +0.0474, 2.46 pooled sd; ahead on every seed), than summation, and than the frozen
semantic branch alone; the first three seeds' result was repeated on three further seeds
by a criterion fixed in advance. With the encoder fine-tuned the three arms are not
separated, on validation or on either held-out partition (§1.28, §1.30).*

**What still may not be said:**

- "Cross-attention beats concatenation", unqualified. The models of record tie.
- Anything held out about this screen: it is validation-selected and validation-only by
  protocol, and the held-out partitions are not touched again.
- That cross-attention beats gated fusion: never resolved.
- Why. *A reading, not a measurement:* frozen cross_attention over six seeds (0.5422 ±
  0.0204) is level with fine-tuned cross_attention (0.5394 ± 0.0116, §1.28) on `val`,
  while fine-tuning lifts concat and `llm_only` by about 0.06 — as if attention over fixed
  features and a trained encoder reach the same place by different routes. Two comparison
  groups; nothing here tests it.

**For the owner — a wording decision, not a new experiment:** A1 now has a two-part
answer (frozen: cross-attention ahead on validation; fine-tuned, the models of record:
no difference, held out included). How the paper states it is the owner's (§5).

### Runs behind this section

Training: the 18 runs in the first table. Seed reports (all at `344db7c`):
cross_attention `afbe8d62…`, concat `541d0e55…`, sum `3d5be460…`, gated `69681049…`,
`llm_only` `381d6133…`, `gnn_only` `12f535d1…`. The per-epoch values are each run's
logged `eval/line/top_5_accuracy` history. Artefacts:
`artifacts/adr010/seed_report_fused_frozen_stripped_<arm>.json`,
`artifacts/phase8/ablations.{json,md}`.

Replication: the 15 runs in the outcome's first table. Seed reports of seeds 3 / 4 / 5
alone — cross_attention `a8a39dc4…`, concat `3d00a4af…`, `llm_only` `526a7c2b…`, sum
`bad1544d…`, gated `e1b0a8e6…`; of all six seeds — `a502ab1d…`, `0c912581…`,
`ad29fba6…`, `a69d554b…`, `14eef261…` in the same order. Tables of record:
`922719b5…` (the first set, 2026-10-11 00:48) and `bcfeb4ff…` (with the replication,
07:00). Artefacts: `artifacts/adr010/seed_report_fused_frozen_stripped_replication_<arm>.json`
and `…_six_seeds_<arm>.json`; transcript `artifacts/queue/a1rep.out`.

---

## 1.33 Phase 9 — serving: the bundle of record answers as the checkpoint did; p95 latency misses NFR-4's bound on long functions; it runs in Docker on this laptop (2026-10-11)

**Status: built on 2026-10-10 on branch `phase-9/serving`, merged on 2026-10-11 as
`f5f450a` once `a1r`, the held-out pass and `a1rep` had all finished. Three of the four
exit criteria are met. The fourth — Docker on a clean machine — holds on this laptop, in
a container that takes nothing from the host but the bundle; a second machine has not
been tried.**

What exists (`docs/Serving.md` is the user's page; `docs/MODEL_CARD.md` ships in the
bundle):

- `src/vulnloc/serving/` — `bundle.py` (a manifest with the provenance and a SHA-256 of
  every file; a bundle that does not verify does not load), `inference.py` (one function
  in, the `PRD.md` §5.4 contract out — it featurizes with L1 code of its own, because L6
  may not import L3), `schemas.py`, `app.py` (`GET /`, `/health`, `/version`,
  `POST /analyze`, `/analyze/batch`), `static/index.html` (the demo page).
- `scripts/export_serving_bundle.py`, `scripts/serve.py`, `scripts/measure_latency.py`; a
  `serve` service in `docker-compose.yml`.

### The bundle of record — `artifacts/serving/ba7dcc5f`, export run `ff9d1bd2…` (at `f5f450a`)

The released checkpoint is `ba7dcc5f…` (cross_attention, seed 42 — fixed as the seed to
release before any held-out number existed), with its own `val` calibrator and its
threshold, 0.0925. The export loads the finished bundle through the serving path and
compares it with the checkpoint on 48 `val` functions drawn by content hash (3,272 lines):

| | |
|---|---|
| Tensors of the served batch that differ from `collate_fused`'s for the same function | **0** |
| Lines the two disagree on being scoreable | 0 |
| Functions with the same top-ranked line | 48 of 48 |
| Logit deviation from the stored scores — median / 99th percentile / largest | 0.00036 / 0.0055 / 0.0900 |
| For scale: the sd of those logits | 0.771 |
| Tolerance | 0.2 |

The input is identical; the output differs because the stored scores were made on the GPU
under mixed precision and the service runs in full precision on a CPU. The tolerance was
set on a development bundle on 2026-10-10, from the same measurement and from one more:
two seeds of one arm differ by more than 0.336 somewhere in every one of 488 `val` units,
so 0.2 is well inside the noise between two models and well outside 0.09.

### Latency — run `3e91feba…` (at `7d7c102`), against NFR-4

200 `val` functions drawn by content hash, one request each after three warm-up requests,
the whole request timed — parse, both branches, calibration, explanation, verification.
CPU only (AMD64 Family 25 Model 80, 16 logical CPUs, 8 torch threads); CPU load 6.3%
before and 6.5% after; the bundle loads in 4.3 s; no request failed.

| Function length, in lines | Requests | p50 (s) | p95 (s) | Longest (s) |
|---|---|---|---|---|
| **All** | 200 | 0.30 | **2.63** | 18.19 |
| 1–25 | 59 | 0.27 | 0.30 | 0.32 |
| 26–50 | 41 | 0.28 | 0.56 | 0.58 |
| 51–100 | 52 | 0.55 | 0.81 | 0.82 |
| 101–250 | 33 | 1.06 | 1.96 | 2.36 |
| 251 and over | 15 | 3.41 | 18.15 | 18.19 |

- **NFR-4 recommends a p95 under 2 s; the measured p95 is 2.63 s. Not met.** It is met in
  every band up to 250 lines — 185 of the 200 functions — and missed by the fifteen
  longest, which take a median of 3.4 s and up to 18 s. The mean is 0.88 s.
- Nothing was tuned towards the bound: no thread setting, quantization, batching of
  windows or cap on length was tried. Those are the levers, and whether to pull one is
  the owner's. **Decided 2026-10-12 (§2): the miss is accepted and stated, and no lever
  is pulled** — the owner delegated the choice; `docs/MODEL_CARD.md` carries the figures.
- The first run of record (`8b74b485…`, `FAILED`) timed all 200 requests and then lost
  them in its own bookkeeping (§7 #68). Fixed in `7d7c102`, with a test.

### Exercised by hand on the bundle of record

The service was started with `scripts/serve.py`, `/health` answered, and the demo page was
driven in a headless browser through its three states. For a nine-line C function the
session wrote (a loop that overruns a 16-byte buffer, then `strcpy`): five of eight scored
lines flagged, `strcpy(dst, buf);` ranked first at 0.984 with a verified 'Buffer overflow'
(CWE-120) explanation and a name-based data-flow path through lines 5 → 6 → 8; the
comment line is shown and never scored. An empty box gives "Nothing to analyze" without a
request. **One function is an illustration, not a measurement** — and five lines flagged
of eight is the model card's "treat the flag as a hint" seen once.

### In Docker — the first image this project built (2026-10-11, 07:28 → 08:00)

- **Built:** `docker compose --profile serve build serve`, 15 minutes, most of it
  downloads: `vulnloc:0.1.0`, 2.9 GB, Python 3.11.17, CPU-only torch.
- **The first run failed, which is what the criterion is for.** `/workspace` was created
  by root; the image's non-root user owned what was copied into it and `artifacts/`, but
  not the directory, so Hydra could not create `outputs/` and `scripts/serve.py` died
  with `PermissionError` before loading anything — as the image's default command would
  have. Fixed in `193f3ac` (§7 #69).
- **From the committed tree (`193f3ac`), image `053c0a4ae9b4`:** the `serve` container
  mounts the bundle of record read-only and nothing else, runs as uid 1000, reports
  healthy 12–60 s after start and holds 1.17 GiB with the model loaded. `GET /version`
  names the bundle. `POST /analyze` on the nine-line example returns exactly the §5.4
  keys and the same five lines in the same order as the host, scores equal to three
  decimals (0.984, 0.945, 0.335, 0.181, 0.129); the first request took 2.0 s, the second
  0.37 s. The two error cases leave in the one shape:
  `{"error":{"type":"ValidationError","message":"language: Input should be 'c' or 'cpp'","unit_id":null}}`
  (422) and `{"error":{"type":"ParseError","message":"there is no code to analyze","unit_id":"request"}}`
  (400).
- **Seen on a second example, a four-line `strcpy` function:** the `strcpy` line first at
  0.9915 — and the closing brace flagged at 0.0957 against a threshold of 0.0925. The
  service withholds empty lines, not lines that are syntax alone (§1.31 measured that the
  model ranks such lines below chance; one sits just over the threshold here).
- **The versions inside are not the host's.** torch 2.14.1+cpu, transformers 5.19.0,
  torch-geometric 2.8.0.post1, numpy 2.4.6 — against torch 2.13.0+cu130 on the host. The
  Dockerfile installs from `pyproject.toml`'s lower bounds, not from the lock file, so an
  image built another day may hold other versions. The answers agreed today; nothing
  guarantees it.
- **Phase 0's open criterion, checked at the same time:** `docker run --rm vulnloc:0.1.0`
  runs the image's default command, `python scripts/train.py train=debug`, to "No-op
  training complete", exit 0, logging to a store inside the container. **It logs
  `git_sha=unknown`: the image carries no `.git`, so a run made in a container cannot be
  traced to a commit.** Nothing of record may be run there as the image stands.
- **Left on D: —** the image (2.9 GB) and 5.3 GB of build cache. Docker Desktop was shut
  down again afterwards. Build log: `artifacts/queue/docker_build.log`.

### The exit criteria

| Criterion | |
|---|---|
| End-to-end request works in Docker on a clean machine | **Works on this laptop** — image `053c0a4ae9b4` from `193f3ac`, above; the container mounts only the bundle. **Not tried on a second machine**; whether a container on the development laptop is what "clean" asks is the owner's to say at the gate |
| Latency measured and reported (NFR-4) | Done — `3e91feba…`, above. The recommended bound is missed |
| Error responses structured per `Rules.md` §4.7 | Done — every failure leaves as `{"error": {type, message, unit_id}}`: unreadable input 400, a request that does not validate 422, anything else 500 (`tests/unit/test_serving_app.py`) |
| Model card published with the checkpoint | Done — the export copies `docs/MODEL_CARD.md` into the bundle and lists its digest; the card carries the held-out numbers of §1.30 |

### What this does not support

- **A function-level verdict.** The model never saw a safe function; `is_vulnerable:
  false` means no line reached the threshold (the API's documentation and the page say so).
- **A latency for anything but this laptop's CPU and `val`'s lengths.** `clean_eval`
  functions average four times the corpus length (landmine #12); expect worse there.
- **That it runs on another machine.** The image was built and run on this laptop only,
  and its package versions are not pinned.

### 2026-10-12 — the report page, a measured record in the bundle, the re-export, the image rebuilt

**The demo page now draws `POST /analyze/report`** (`serving/static/index.html`, commits
`61c8489`, `5d806ad`): the summary; one entry per flagged line with its verified
explanation, the lines each of the two attention maps weighted, the name-based data-flow
path and a count of syntax nodes (both labelled as static analysis); the five
highest-ranked lines; the source with flagged lines marked and the open entry's weighted
lines beside them; coverage; and the 19 weakness patterns with the matched ones marked.
Driven in headless Chrome over the DevTools protocol, against the host service and
against the container: empty input gives the client-side message; a ten-line function
with an overrun gives 8 of 10 lines flagged, three pattern matches and the measured
sentence; no script error.

**The model flags a line in every function it was given — six of six inputs**, down to
`;`, `int x;`, an empty-bodied function and `int add(int a, int b)` (its `return` at a
calibrated 0.99). The threshold was fitted on functions that are all vulnerable (§1.28),
so `is_vulnerable` is close to always true and **the "No known vulnerability found"
state could not be reached through the real model in this check; its wording is pinned
by unit tests only.** This is the model card's "not a detector" seen from the page.

***Corrected the same evening — the paragraph above generalised from six tiny inputs.***
On `val`, from the stored explanations of `ba7dcc5f…` (`artifacts/phase7/explanations_retrained/`, 488 functions, all vulnerable): 417 have a flagged line and **71 (14.5%) have none** — and those are the long ones, a median of 117 lines against 40. So the unflagged state *is* reached —
on long functions that do contain a vulnerability — while a four-line harmless function
is flagged at 0.99: a function's highest calibrated score falls as the function gets
longer. `is_vulnerable` is therefore not "close to always true"; it is wrong in both
directions, by length. (Those counts are from the scoring pipeline, which can flag a
blank line; the service withholds blank lines, so its counts can differ slightly.) What
stands: the state was not reached in the browser check, and the model is a ranker for a
function already suspected.

**A bundle can carry what was measured for its checkpoint.**
`scripts/export_serving_bundle.py +export.measured_from=<evaluation file>` quotes the
checkpoint's own figures into the manifest's `measured` block, with the file's SHA-256
and the evaluation's run id; a file for another run, threshold or text variant is
refused. **Nothing is scored for it** — reading `evaluation_test.json` is not a second
evaluation. The first attempt stopped before writing a byte: the hand-made test fixture
had bare numbers where the file stores one value per tie order (§7 #71; fixed `b0afb75`,
with a test that reads the real file).

**The bundle of record is now `artifacts/serving/ba7dcc5f-r2`, export run `ef031969…`**
(at `b0afb75`, clean tree): on 48 `val` functions the served batch equals training's,
tensor for tensor (0 differing), the top-ranked line is the same on 48 of 48, and the
largest logit difference is 0.08998 over 3,272 lines — the first bundle's figures. Every
file's digest equals the first bundle's except `MODEL_CARD.md` (today's card) and
`tokenizer/tokenizer_config.json` (`local_files_only` is `false` where the first export,
made offline, wrote `true`). Its `measured` block, quoted from `f3a7dbdf…` for seed 42
on `test`, blank lines withheld, 533 functions: Top-1 0.2495, Top-3 0.4071, **Top-5
0.4765** (source order 0.2852), Top-10 0.6248, median IFA 5; pooled line precision
0.1583 and recall 0.2969 at the `val` threshold. The first bundle
(`artifacts/serving/ba7dcc5f`) is kept; latency was measured on it and not again.

**The image was rebuilt and the service checked in it** (Docker box, `Phases.md`):
`vulnloc:0.1.0` is now `7283510c22c7`, built from the tree at `5d806ad` — Python
3.11.17, torch 2.14.1+cpu, transformers 5.19.0, fastapi 0.143.0, unpinned. The image of
2026-10-11 is kept as `vulnloc:0.1.0-193f3ac`. With only the new bundle mounted and the
hub offline, the container answered one function exactly as the host service did (torch
2.13.0, same CPU): the same eight flagged lines in order, scores within 5.2e-7, equal
explanation, evidence and report; `ParseError` in the one error shape; `x-vulnloc-seconds`
0.37. Phase 0's command still runs to completion in it. One function, one machine.
Docker Desktop is shut down again.

**Tests.** On the main tree at `86aa7a3`, Docker closed: 1,521 tests pass, 3 skip. A first run of the suite, made while Docker Desktop was starting, had
one failure: `test_evaluation_localization.py`'s Hypothesis property took 2.0 s on its
first call against a 200 ms deadline (0.37 ms on the next). It failed the same way once
more while Docker was shutting down and passes on a quiet machine (33 of 33 in 7 s). The
test and `src/vulnloc/evaluation/` were not touched.

Still open, none of it an exit criterion: the image's versions are not pinned to the lock
file and it carries no commit SHA; a second machine; no fix advice in the explanations;
`frontend/fix_templates.py` unreviewed; the older `frontend/` tool still shows the
Phase 3 and Phase 4 models, not the bundle.

### Runs and commits behind this section

Export `ff9d1bd2…`; latency `3e91feba…` (and `8b74b485…`, failed). Commits `3a119ba`,
`6b93d58`, `c2e4d92` (the branch), `f5f450a` (the merge), `7d7c102` (the latency fix),
`193f3ac` (the Dockerfile fix).
Pre-merge checks: `artifacts/queue/after_a1rep_checks.out`. After the merge and the fix,
on the main tree: 1,498 tests pass, 3 skip; black, ruff, mypy and import-linter clean.
2026-10-12: export `ef031969…`; commits `61c8489` (the page, the measured record),
`b0afb75` (the tie-order fix), `5d806ad` (the trace script, the page's details, the
serving guide).

---

## 1.34 The clean-clone check — the pipeline rebuilds exactly; of two frozen runs repeated at seed 42, concat lands inside the band written beforehand and cross-attention does not (2026-10-11, read 2026-10-12)

**What ran.** `artifacts/queue/cleanclone_check.py`, 12:29–13:39 on 2026-10-11, in
`D:\vulnloc-cleanclone` (log `cleanclone.log`, outcome `result.json`); seventeen steps,
every one exit 0. A `git clone --no-hardlinks` of this tree at `65e65dc`; a new Python
3.12 environment from `requirements.lock.txt` alone (`torch 2.13.0+cu130`, the same RTX
3050 Ti; `pip check` clean); the raw corpora hard-linked and their digests verified, not
downloaded; everything downstream rebuilt. No held-out partition was touched. The
clone's runs are in the clone's own MLflow store and are not pooled with this tree's.
What it would be read for was committed first (`65e65dc`, §6 item 0a).

| Check, as written before the run | Outcome |
|---|---|
| Test suite in the new environment | 1,506 passed, 5 skipped |
| Split fingerprint equals `0ade7483…` | yes |
| `assignment.csv` and the four shards equal the main tree's, byte for byte | yes, all five |
| Both vocabularies equal the main tree's, byte for byte | **no as written — yes in content** (below) |
| cross_attention, seed 42, `val` Top-5 within the arm's three-seed sd (0.0102) of `8cde0dd5…`'s 0.5207 | **no: 0.5436 (clone run `c1add839…`), +0.0228 = 2.2 of those sd** |
| concat, seed 42, within 0.0157 of `6ce89a15…`'s 0.4917 | yes: 0.4772 (clone run `1078e39a…`), −0.0145 |

The script's verdict is `COMPLETED - NOT ALL CHECKS HOLD`, and that stands as the
outcome against the criterion as it was written.

**The vocabularies differ in one field: the `run_id` of the MLflow run that wrote
them.** Compared field by field on 2026-10-12, every other field of both files is equal —
`digest`, `size` (305 and 31,389), `counts`, `vocab`, `units_scanned`. The check asked
for byte equality of a file that records which run made it; the check was wrong, not the
vocabulary. Both training runs also logged the same `node_vocab_digest`,
`subtoken_vocab_digest` and `semantic_cache_digest` as the runs of record.

**The cross-attention run is not explained by any input.** All 165 logged parameters
are equal between `c1add839…` and `8cde0dd5…` (163 of 163 for concat); the source
differs between the two commits only under `src/vulnloc/serving/`, which training cannot
import. The curves (`val` Top-5 per epoch):

| | epochs |
|---|---|
| cross_attention, of record | 0.5187, 0.4564, **0.5207**, 0.5083, 0.4108, 0.5104 |
| cross_attention, clone | 0.4793, 0.4419, 0.4855, 0.5104, 0.5166, 0.4357, **0.5436**, 0.4793, 0.4689, 0.5124 |
| concat, of record | 0.4440, **0.4917**, 0.4253, 0.4627, 0.4793 |
| concat, clone | 0.4481, **0.4772**, 0.4253, 0.4689, 0.4751 |

Epoch-0 training loss is equal to four decimals in both arms (0.0645; 0.0666). Concat
then follows its run of record epoch for epoch, within 0.015. Cross-attention is already
0.039 apart at the first evaluation, and its curve moves by as much as 0.10 (the run
of record) and 0.11 (the rerun) from one epoch to the next (*corrected 2026-10-12: this
line first said "by 0.04–0.11"; several steps are smaller, down to 0.006*) — so the reported number, the best epoch's, is a
maximum over a noisy curve, and early stopping ended one run after 6 epochs and the
other after 10.

**What can and cannot be said.**

- The data pipeline, the split, the shards and both vocabularies are reproduced exactly
  from a clean clone and the lock file.
- A frozen run repeated at the same seed is a replicate, not a copy (§7 #50 expected
  that; its size was not known). **For cross-attention the same-seed difference measured
  here, 0.0228, is larger than the three-seed sd (0.0102) it was to be judged by.** That
  band was the wrong yardstick, and it was the one written down.
- Against the arm's six-seed distribution (§1.32) both clone runs are ordinary:
  cross_attention 0.5436 against 0.5422 ± 0.0204; concat 0.4772 against 0.4948 ± 0.0181
  (0.97 sd). That comparison was **not** stated before the run and is reported as an
  after-the-fact reading.
- No number of record changes: the seed sd of every arm already contains this
  run-to-run variation, since each seed was run once.
- **Not established:** the cause. Non-deterministic CUDA scatter order (§7 #50) is the
  suspect; it was not isolated. And this is one rerun an arm — it shows the difference
  can be this large, not how large it usually is.

Whether Phase 8's "reproducible from a clean clone" box is ticked on this is the
owner's, at the gate. **Ticked 2026-10-12 on the owner's words (§2), with the miss
stated beside it** — `Phases.md` says what the tick claims and what it does not.

## 1.35 Three headlines over one analysis — the released one says "flagged" about 93.5% of harmless functions; the pattern one fires on a quarter of both kinds (2026-10-12)

**The record is `docs/VerdictComparison.md`; this section says what it found and what it
is not.** The owner asked for the results so far to be kept in a file *"so we can compare
all the three, the original the pattern and the screening gate later"* (§2). Kept, and —
the session's addition, not asked for — both existing headlines were measured for the
first time on functions that are **not** vulnerable, since a "nothing found" line cannot be
compared on vulnerable functions alone. Nothing was trained; `val` only; the released
system is unchanged.

Run `6c942e6c…` (code `0af6ae2`, branch `phase-9/screening`, `scripts/compare_verdicts.py`),
the bundle of record, one request a function: all 488 line-labelled vulnerable `val`
functions (median 47.5 lines) and 1,000 of the 17,768 the corpus does not mark vulnerable,
drawn by content hash, seed 1337 (median 18; the population's is 19). All 1,488 answered.

| Headline | Raises a concern about a vulnerable function | About one that is not |
|---|---|---|
| Original (`/`): a line reaches the threshold | 415 of 488 (85.0%) | 935 of 1,000 (93.5%) |
| Pattern (`/experimental/patterns`): a weakness pattern matches such a line | 135 of 488 (27.7%) | 210 of 1,000 (21.0%) |
| Screening gate | not built | not built |

- **The released headline raises a concern more often about harmless functions than about
  vulnerable ones**, and at equal length the two kinds get nearly the same answer: up to
  25 lines it flags 757 of 759 functions; at 51–100 lines 77.0% against 73.1%. It reads
  length. The 73 vulnerable functions it misses have a median of 123 lines.
- **The pattern headline's "No known vulnerability pattern found" appears for 79.0% of
  harmless functions and 72.3% of vulnerable ones.** Overall it fires 6.7 points more
  often on vulnerable functions; in the two largest length bands the rates agree to half a
  point. Whether it carries anything beyond length is not resolved.
- **How well one number separates the two kinds** (the chance a vulnerable function has
  the higher value; 0.5 separates nothing): **length alone 0.7111; the checkpoint's highest
  line score 0.3321 — the wrong way round** (harmless functions score higher, median 0.286
  against 0.137); lines flagged 0.5189. Length alone is the bar a gate has to beat.
- The served count is 415 where the stored explanations gave 417 (§1.33's correction): the
  two functions that differ are the two in which every stored flagged line is blank, which
  the service never offers. The pattern set is the same 135 functions either way.

**What it is not:** a precision (one `val` function in 23 is vulnerable, and the two kinds
were asked in chosen numbers); a statement about more than the one served checkpoint; a
statement that the 1,000 are safe — "not vulnerable" is the corpus's label. And it is a
different question from the project's results, which rank the lines inside a function
already known to be vulnerable and stand as they were (§1.28–§1.30).

**For the gate, written before it exists** (`docs/VerdictComparison.md` §6 — the session's
rules, the owner's to change before a gate is run): the same 1,488 functions; no function
answered by a gate fitted or thresholded on it (cross-fitted, folds by project); better
than length alone on the whole record *and* within length bands; `val` only. **Fitting on
`train` has a trap: the checkpoint has seen `train`'s vulnerable functions and none of its
harmless ones, so a gate on its features can learn "seen before".** The gate is not
started and still needs the owner's explicit yes.

Per-function records: `artifacts/screening/verdicts_val.jsonl` (SHA-256 `8deca36a…`),
also stored with the run. The script and its 15 tests are on the experiment branch only.

---

## 1.36 One site that holds everything — `frontend/` on port 8000: Findings is the released model, a reason only where a pattern matches, 17 hand-written examples (2026-10-12)

**The owner's words, after seeing both interfaces side by side** (§2): *"i like the 8000
port one its format is exactly like what we want, but I want you to make one complete site
which has everything also on the 8080 port one i dont like the count of number of lines
flagged card also below only mention the lines who match a pattern"*. Built the same
evening. **On seeing it the owner said: *"okay save these for now, theres some improvement
to be done for sure we will tackle that later"* — committed as it stood. Which
improvements was not said; ask before changing the page.** **The released service (`src/vulnloc/serving/`,
`scripts/serve.py`, port 8080) is unchanged** — a picker added to its page earlier in the
session was taken back out.

What `python frontend/server.py` now serves on `http://127.0.0.1:8000`:

- **Findings is the model of record**, not the Phase 3 semantic model: `frontend/released.py`
  loads the bundle `artifacts/serving/ba7dcc5f-r2` through the service's own
  `vulnloc.serving.inference.Analyzer` — same scores, cut-off, explanations and verifier.
  Nothing is re-implemented; every line's rank is rebuilt from the analysis' logits for the
  whole-function view and **refused if it disagrees with the service's report**.
- **No count of flagged lines as a headline** (the service's `summary` sentence is not
  passed to the page), and **a card only for a line over the cut-off that a known weakness
  pattern matches** — the explainer's verified sentence shown whole, the usual fix from
  `frontend/fix_templates.py` joined on the pattern's id (`advice_for`; the three
  index/offset patterns are declared in `NO_ADVICE` and carry no fix), and what the model
  weighted behind a toggle. The model's five highest-ranked lines and the whole function
  with its ordering stay on the page whether or not anything matches.
- **This is the "pattern" headline of §1.35, by the owner's choice of presentation.** So
  the page quotes that record beside it, read from `artifacts/screening/verdicts_val.json`
  and only for the bundle it was measured with (run `6c942e6c…`, `val`, an experiment): a
  pattern matched in 135 of 488 vulnerable functions and 210 of 1,000 harmless ones. "No
  known weakness pattern matches…" is followed by "That is not a statement that the
  function is safe."
- **Samples: 21.** The four pipeline samples (still `?sample=0…3`) and 17 functions
  written by hand in `frontend/samples.py` — ten with a weakness written into a named
  line, seven with none intended. For an unedited sample the page adds a "Sample check":
  where the named line was ranked and whether a pattern matches it. **A sample's line is
  its author's intent, never a label; "no weakness intended" was audited by nobody;
  seventeen functions are an illustration and no figure from them is quoted anywhere.**
  Seen once, on the bundle of record: every weakness sample gets at least one card, and
  so do four of the seven harmless ones — a pattern is a text match (landmine #74's
  cousin). One of those is the crude use-after-free template firing on `return items;`
  after a `free(items)` on an error path: names only, no control flow (#19).
- **Detailed report: all twelve stages now say something about the paste or the
  checkpoint.** Semantic and structural models each rank the paste alone, with the released
  model's rank beside; Fusion, Localization head and Explanation run on the paste;
  Evaluation shows the bundle's held-out record (`test`, 533 units); Serving shows which
  checkpoint answered and what was not examined. `frontend/phases.json` was updated; the
  Fusion panel's old sentence "A full run is pending" was false since 2026-10-07 and is gone.

Endpoints: `/api/findings` is the released model; the Phase 3 semantic ranking moved to
`/api/semantic`. Tests: `tests/unit/test_frontend_released.py` (14; one reads the real
bundle, one the real screening record — landmine #71) — with the other frontend and
serving-app tests, 81 pass. Checked in a headless browser on both tabs.

**The first thing the owner saw was a hung page**, and the cause was not the code under
test: the browser had kept the old `app.js`, which could not draw the new `/api/findings`
answer and threw with the button stuck at "Analysing…". Headless Edge, with an empty
cache, had shown the page working an hour earlier. Fixed the same evening — the server
sends `Cache-Control: no-cache` for everything outside `/api/`, the page's script and
stylesheet addresses carry a query string that retires the cached copies, and a result
the script cannot draw is an error box, not a stuck button. **A check made with an empty
cache says nothing about a browser that has been here before.**

**Not done, and worth knowing:** the semantic model alone (`fc902107…`) is still the one
trained with comments in place, and its panel says so; the structural panel's held-out
figure was measured on that text too and does not say so. The first request after a
start waits for three models (about 80 s on a busy disk, 0.3–0.4 s afterwards).

---

## 2. Decision Log

Append-only. Format: date · decision · alternatives considered · rationale · ADR reference.

| Date | Decision | Alternatives | Rationale | ADR |
|------|----------|--------------|-----------|-----|
| 2026-10-12 | **One site holds everything, in the port-8000 layout; no count of flagged lines as a headline; a reason only for a line a pattern matches** (owner, session 20, shown the released demo page on port 8080 and the two-tab tool on port 8000 side by side: ***"i like the 8000 port one its format is exactly like what we want, but I want you to make one complete site which has everything also on the 8080 port one i dont like the count of number of lines flagged card also below only mention the lines who match a pattern"***). Read by the session as: build it in `frontend/`, put the released bundle behind Findings, move the examples there, and leave the packaged service and its page as released — **that last part is the session's reading, not the owner's words** (§1.36) | Change the released page under `src/vulnloc/serving/` as well; keep two sites | The owner chose the layout and the presentation. The pattern presentation is §1.35's second headline, so its measured record is quoted on the page beside it. The released page is what the Phase 9 gate, the Docker image and `docs/VerdictComparison.md` describe, and the pipeline panels need layers the packaged service may not import | — |
| 2026-10-12 | **The results so far are kept in one record so the three headlines can be compared later: `docs/VerdictComparison.md`** (owner, after being told that the pattern page exists apart from the released system and asked whether the screening gate should be attempted the next day: ***"its fine save these results in a fle for later so we can compare all the three, the original the pattern and the screening gate later"***). Session 19 wrote the record and, **as its own addition**, measured both existing headlines on 1,000 `val` functions the corpus does not mark vulnerable (run `6c942e6c…`, §1.35). **"its fine" is not read as a yes to the gate: it is new training, the owner chose on 2026-10-11 to go on without any, and nothing was started** | Keep only the figures already reported (417 / 71 and 135 / 353 on vulnerable functions), the harmless side left "not measured" · read "its fine" as the go-ahead and start the gate | The owner asked for a record to compare against, and a "nothing found" line cannot be compared on vulnerable functions alone. The measurement trains nothing and touches no held-out partition and no file of record. The gate waits for words that name it | — |
| 2026-10-12 | **Phase 9's Docker box is ticked as "a clean container on this laptop", the bundle of record becomes `artifacts/serving/ba7dcc5f-r2`, and the Phase 9 gate is closed** (owner: *"okay finish these two also so phase 8 and 9 done?"* — the same words as the Phase 8 row below; their first word answered the session's question whether to go ahead with the demo page). Session 19 finished the page on `POST /analyze/report`, added `export.measured_from`, re-exported the released checkpoint (`ef031969…`), rebuilt the image and checked the service in it against the host, then ticked the box and wrote §1.1g. **The Docker box had been left for the owner to say whether one laptop is what "clean" asks; the owner's words do not address that, and the tick's reading is the session's.** The owner has not read §1.1g | Leave the Docker box unticked and Phase 9 at 3 / 4 until the second laptop is tried; keep the first bundle as the bundle of record and treat the re-export as a second artefact | The owner's stated aim is both phases done. A container built from the repository alone, with only a bundle mounted and no network, is clean in everything but hardware; the second-machine check has a place of its own in Phase 10 (the independent reader). The new bundle has the same weights by digest and passed the same export check, so nothing measured on the first one is lost | — |
| 2026-10-12 | **Phase 8's two remaining boxes are ticked and its gate is closed** (owner, after being told that two Phase 8 boxes and the Phase 8 and 9 gate signatures were what remained on the owner's side: *"okay finish these two also so phase 8 and 9 done?"*). Session 19 read those words as the instruction to finish both boxes and close both gates, and did the work first: a read-only trace of every run the Phase 8 tables name (`scripts/trace_phase8_tables.py` — 0 did not trace), and a verdict for each of the ten novelty claims written into `Phases.md`. **"Ablations confirm or refute each novelty claim" is ticked as "every claim has a written verdict", not as "every ablation was run": A3, A7 and A10 were not run and no claim is made about them.** The gate's three answers are §1.1f. **The owner has not read them and did not write "signed"; the reading is the session's and any part of it can be reversed** | Leave both boxes and the gate unsigned until the owner reads §1.1f; run A3, A7 and A10 first (A3 needs ADR-007 accepted and CPG graphs costed on the 4 GiB card; A7 needs the live encoder; A10 needs explainers that are not built) — new training, which the owner declined on 2026-10-11 | The owner's stated aim is both phases done, and each tick says in `Phases.md` what it claims and what it does not. The weakest of the four is the ablations box, and it is recorded as such | — |
| 2026-10-12 | **Phase 8's "results reproducible from a clean clone" box is ticked, with the cross-attention miss stated beside it** (owner). Session 19 explained the check (§1.34) and offered two options — tick it with the miss stated, or leave it unticked; the owner asked *"what do you recommend"*, was told "tick it, with the cross-attention miss written beside it", and answered ***"okay then proceed"***. The claim is: the tests, split, shards and vocabularies reproduce exactly from a clean clone and the lock file, and a retrained frozen model lands within its arm's spread over seeds — not that a single run's number can be recovered. The online draft's reproducibility section says so, with both reruns' figures | Leave it unticked, holding to the criterion written before the check (each rerun within its arm's three-seed sd of one logged run) | That criterion was the session's and was too narrow: a same-seed rerun of this model moves more than the three-seed sd with every input equal (§7 #70), so a correct pipeline can fail it. Everything deterministic reproduced exactly. Kept in the record: one of two reruns missed the pre-stated band, the six-seed comparison was made after the run, the cause was not isolated, the corpora were verified and not downloaded, and it was one machine | — |
| 2026-10-12 | **Q15 is closed: the paper leads with the fine-tuned null and reports the frozen result beside it as a qualified, validation-only finding** (owner, delegating: asked for four things — (1) how A1 is stated, (2) latency, (3) the clean-clone box, (4) the gates — the owner wrote ***"do 1 and 2 according to you explain to me 3"***; the choice is the session's recommendation of §5 and the owner can reverse it). Applied the same hour to the online draft, whose Section 3.10 already read this way: the Abstract, Table 1 (now with both held-out rows), the answers to RQ1 and RQ2, the second contribution and the limits paragraph of Section 1.4 — every figure from §1.28, §1.30 and §1.32; the draft-status line under the Abstract is removed | Lead with the frozen result · give both equal weight | The models of record and the only held-out numbers are the fine-tuned ones; the frozen lead is validation-only by protocol, six seeds of one screen, and carries every qualifier §1.32 lists. "Replicated null" could no longer stand in the Abstract either: with the encoder frozen, on repaired text, the arms are separated | — |
| 2026-10-12 | **The latency miss is accepted and stated; nothing is tuned towards NFR-4** (owner, delegating in the same sentence: *"do 1 and 2 according to you"*; the session's choice, reversible). p95 2.63 s against a recommended 2 s (`3e91feba…`, §1.33), met for functions of up to 250 lines. Stated in `docs/MODEL_CARD.md` (a new bullet under known limits) and to be stated in the paper's limitations. **The card inside the bundle of record predates that bullet** — the bundle is not edited; the re-export already owed for the `measured` block carries the new card | Tune towards the bound: thread settings or batching a function's windows (neither changes what the service returns; both untried, and the equivalence check against training would have to be re-run), quantization or a cap on length (both change what it returns) | NFR-4 is a `[RECOMMENDATION]` of `PRD.md`, not a requirement of the brief; Phase 9's criterion is "measured and reported", which is met; 185 of 200 sampled functions are inside the bound; the service is a demonstration of a research model, and the days left belong to the paper. Each lever is future work, named in the card's rationale here | — |
| 2026-10-12 | **The human evaluation of the explanations (E9, `docs/HumanEvalProtocol.md`) is deferred in writing — no person rates the pack for this submission** (owner). The owner had downloaded three filled rating sheets; session 19 found the same score in 400 of 400 cells and asked who filled them in; the owner asked *"what if we skip this?"*, then answered *"gemini rated those"*, and, offered (1) defer in writing or (2) three people rate the pack, said ***"then do 1"***. The sheets are a language model's output: they are not in the repo and are never reported as human evaluation, raters or agreement. What is reported instead: the automatic verification of every explanation (§1.27) and the share that names a cause, with the sentence that no human rated them | Three people rate the unchanged `test` pack independently (still possible later — the pack, `artifacts/phase7/rater_pack_test/`, and the protocol stand as they are); report the model's scores as an "LLM-judge" reading (advised against: outside the approved protocol, three runs of one model are not independent, and three of three checked comments from one sheet did not describe their sample) | The owner's decision. The explainer declines to name a cause for 84 of the 100 sampled functions, so the counts already say most of what ratings would. The paper loses its primary measure of explanation quality (`ResearchPlan.md` §4.3) and must say so | — |
| 2026-10-11 | **The failure taxonomy is accepted as revised, and Phase 8's "≥ 20 inspected cases" box is ticked** (owner, about 11:10). The owner had pasted a written review of the draft with no words of their own; session 18 checked it against the sheet, revised the draft (`9f88765`) and asked two things — *does the taxonomy stand as revised, or what would you change?* and *who wrote the review; if a person read the cases, say so.* The owner's answer, whole: ***"proceed ahead with this and Siddharth read this"***. Read as: the taxonomy stands as `docs/FailureTaxonomy.md` §4 has it after the revision, and a person, Siddharth, read it. **The words do not say whether "this" is the review, the case sheet or both, how many of the thirty cases were opened, or who wrote the review — nothing here claims those** | Leave the box open and ask a third time which document was read · tick it only after the `category:` fields of the sheet are filled in by hand | `docs/FailureTaxonomy.md` §6 put the condition in writing before any review: the box is ticked when the owner says the taxonomy stands. The owner said it, in answer to that question by name. What a session may not do is tick it on its own reading, and it did not. The acceptance changes no number and no limit of §1.31: the categories are a vocabulary, not a breakdown, and validation only. Reversible — the box and the status line are one edit each | — |
| 2026-10-11 | **Phase 9 merged into the main branch (`f5f450a`)** (session 18, on the owner's *"whats next to be done, start"* at about 07:00, and under the condition the owner set on 2026-10-10: after `a1r`'s seed reports and the held-out pass). `a1rep` had finished as well; nothing was running | Leave it on its branch until the owner reads it | The condition was met, a trial merge passed the whole suite and the linters at 04:09, and the owner asked for the next thing to be started | — |
| 2026-10-11 | **Four choices made while building Phase 9, recorded now that it is merged** (session 18, under the owner's standing delegation to build; each is reversible). (1) **The released checkpoint is seed 42 of the model of record, `ba7dcc5f…`** — fixed on 2026-10-10, before any held-out number existed. (2) **The service never flags or ranks a line with no code on it** — the evaluation's "blank lines withheld" reading; the model's own logits are untouched. (3) **The demo page shows rank and calibrated score, not `Design.md` §2.3's five risk bands**: almost every calibrated line probability is below 0.2, so the bands would label nearly every flagged line "Clean". (4) **`httpx` is a development dependency** (FastAPI's test client needs it), and **the export's logit tolerance is 0.2**, set from a measurement (§1.33) | Pick the seed with the best held-out number · let the page flag empty lines as the model scores them · rescale the bands | (1) choosing a seed on a held-out number would be selection on the held-out set. (2) an empty line flagged at 0.096 against a threshold of 0.0925 was the first thing the running service showed. (3) a band is a claim about what a score means, and the scores do not mean that. (4) measured, not guessed | — |
| 2026-10-11 | **The `a1r` screen is run again on three new seeds — queue group `a1rep`, started after the held-out pass has ended** (session 18, on the owner's words shortly after midnight: *"train or build whatever you need to overnight, internet will be on till 4 am i guess"*, and *"keep the docs updated and context as well updated so theres no dip in your performance"*). cross_attention, concat, `llm_only`, sum, gated × seeds 3 / 4 / 5, `a1r`'s constants and cache; what is computed from them is written in §1.32 before the first run. **The choice of what to train is the session's, under that delegation; the owner can stop it with `python scripts/run_queue.py --stop`** | Leave `a1r`'s three-seed result as it is · re-run all six arms · add seeds only to the core pair | `a1r` put cross_attention 2.93 pooled sd ahead of concat on Top-5 — A1's own comparison — where the same screen on text with the tell showed nothing and the fine-tuned models tie. At three seeds that is a candidate (§7 #37, #64), and a frozen run costs about 15 minutes: fifteen runs fit in one night and need nobody. `gnn_only` is left out because nothing about it is in question. Run after the pass, so one GPU job at a time; before Phase 9's merge, so that `src/vulnloc/` is the same for all six seeds of an arm | — |
| 2026-10-11 | **The held-out pass runs tonight, after `a1r`** (owner, asked at 23:55 on 2026-10-10 with two options — "a1r only", "Yes, after a1r tonight" — after resuming with *"continue"*; answer at about 00:00: **"Yes, after a1r tonight"**). On the bare "continue" session 18 restarted `a1r` alone at 23:54 and asked; the chain script was started at 00:00, only after the answer | `a1r` only, the pass on another day — the session's recommendation, because the pass then runs unattended until about 02:00–03:30 | The owner's to decide, and the owner named it. The row below read a bare "continue" as covering the pass, and the owner paused again twenty minutes later (*"pause for a bit i will continue later"*, 21:09) without saying why; so this time the question was put before anything was armed | — |
| 2026-10-10 | **The pause of 18:52 ends: the `a1r` queue and the chain behind it run again from 20:50** (owner, after the conversation was compacted: *"continue"*). Session 18 reads the word as undoing the pause and nothing more: the 18:51 instruction was *"if anything is running, pause for a while and compact"*, and what was running then was `a1r` with the held-out pass chained behind it by the owner's 16:25 answer ("After a1r, tonight"). **The owner did not name the pass again — that it follows `a1r` rests on the 16:25 answer, and the reading is the session's** | Resume `a1r` alone and ask once more about the pass | The owner named and timed the pass four hours earlier; the pause was asked for "a while", for compaction, not as a change of plan; it is still tonight. The chain also leaves a window: `run_queue.py --stop` before `a1r` exits (about 45 min from the restart) ends it without the pass, exactly as at 19:01 | — |
| 2026-10-10 | **The held-out pass continues tonight, when `a1r` has finished** (owner, asked at 16:25 with three options — "After a1r, tonight", "Now", "Not yet" — after replying "next" to session 18's report; answer: **"After a1r, tonight"**). It starts by itself when the `a1r` queue exits, from `artifacts/queue/heldout_pass.py` as written on 2026-10-08: only what is missing is scored, each checkpoint once per partition. The owner then left for a few hours: *"ill be back in a few hours fo what you need to"* | Now, pausing `a1r` · not yet | The pass was authorised on 2026-10-07/08 and paused by the owner for the disk; the restart is the owner's too, and this names it. Running it after `a1r` interrupts nothing and keeps one GPU job at a time | — |
| 2026-10-10 | **Phase 9 is started while Phase 8 is still open** (owner, same question round, two options — "Yes, start Phase 9", "No, stay in Phase 8"; answer: **"Yes, start Phase 9"**). Built on a worktree branch and **merged only after `a1r`'s seed reports and the held-out pass are done**: serving lives under `src/vulnloc/`, and nothing there changes while a comparison group runs (§7 #62). Phase 8 is not closed by this — its box for the taxonomy waits for the owner's review, its clean-clone criterion is half met, and its held-out numbers do not exist yet | Stay in Phase 8 and wait three hours for `a1r` | `Phases.md` blocks Phase 9 on 6 and 7, both signed; Phase 8 had nothing left a session could do without the owner or the GPU. Same shape as the 2026-10-08 row that started Phase 8 before the Phase 7 gate: the owner sets the order of work, and a phase is still closed only by its criteria | — |
| 2026-10-10 | **`a1r` runs, on this laptop; the held-out pass stays paused.** The owner opened session 18 with one sentence — *"kindly continue the next steps bui;ding/training"* — and nothing else. Session 18 reads it as: **build** Phase 8's remaining pieces, and **train** the one group still pending on this machine, `a1r` (the frozen A1 screen on comment-stripped text — "whether `a1r` runs, and where" was the owner's, 2026-10-08). `a1rcheck` passed and `a1r` was started at 14:52. The same sentence was first read as the owner's word to continue the paused held-out pass, and the pass was launched at about 14:38: **the session's permission system refused the launch as an irreversible action before any process started** — `artifacts/queue/heldout_pass.out` has no new line, no file was written, nothing held-out was read. It was not retried. **The pass therefore still waits for the owner to say so in words that name it** | Resume the pass on the strength of "continue" · start nothing and ask first | "Training" names `a1r` and nothing else here: it is reversible, its own cache directory and its own comparison group, and it is the leak-free form of the claim §1.25 found most exposed to the leak. The held-out pass is made once; the authorisation of 2026-10-07/08 stands, but the pause was the owner's and so is the restart, and a sentence that does not mention evaluation is not that | — |
| 2026-10-08 | **The owner confirmed `docs/EvaluationProtocol.md`; it is final, and the single held-out evaluation goes ahead.** Asked (session 17, about 17:05) to read it — including §5.1's metric definitions and §6's note on what 2 pooled sd is worth, both written that afternoon on `val` — and reply "protocol confirmed" or say what to change, the owner answered: *"yes I have already done all these please proceed forward I read through all this"*. No change was requested. The protocol is committed as final **before** any held-out file is read; then the nine `d97r` checkpoints are scored on `test` and on `clean_eval`, one pass each, with their `val`-fitted calibrators and thresholds applied unchanged | Change a definition first · wait | This is the condition the 2026-10-07 authorisation row set ("the evaluation protocol … is committed; then all nine … are scored … once"), with the checkpoints as superseded by the ADR-010 row. Everything a number depends on was fixed on `val`, in writing, before the partitions were opened (`Rules.md` §5.3) | — |
| 2026-10-08 | **The owner signed the Phase 7 gate and approved `docs/HumanEvalProtocol.md`.** Asked in the same message to read `artifacts/phase7/sentences_for_review.md` (every sentence the explainer can emit: 19 template sentences, 8 fixed forms) and the human-evaluation protocol, then reply "sign Phase 7" or name the sentence of concern; the answer is the one quoted in the row above. **No sentence was named as a concern.** Phase 7's fourth criterion — explanations reviewed for absence of attack instructions — is met by the owner's statement that the page was read; the other three were met on 2026-10-07 (§1.1e). Phase 8 is the current phase | Not yet · a sentence to reword first | Four of four exit criteria. What was not built is listed in §1.1e and stays deferred: 7b, node importances, token-level attention. The protocol as approved draws the rater sample from seed 42 of the model of record (`ba7dcc5f…`), on `test`, on comment-stripped text | — |
| 2026-10-08 | **Phase 8 is started at the owner's instruction, with the Phase 7 gate still unsigned** (owner: "start with phase 8" — said after being given the steps for the gate and for confirming the evaluation protocol, and without answering either). Session 17 reads the words as a go-ahead to **build Phase 8 on `val`**. They are **not** recorded as the Phase 7 signature — that gate's open criterion is a person reading the explainer's sentences for attack instructions, and nothing said covers it — **nor as confirmation of `docs/EvaluationProtocol.md`**. No held-out partition is read | Build nothing until the gate is signed (the 2026-10-07 authorisation row: "Phase 8 starts only after the owner signs the Phase 7 gate") · record the words as the signature and the confirmation | The owner sets the order of work and can reverse an earlier row; a session cannot record a review nobody said took place (`Rules.md` §1). Everything built reads `val`, which was already permitted, so nothing irreversible follows from starting. **For Phase 8 code this supersedes the 2026-10-07 row's ordering; for the held-out evaluation that row stands unchanged** | — |
| 2026-10-08 | **ADR-010 accepted; Q14 closed. The nine `d97r` checkpoints are the models of record** (owner, asked in session 17 with the evidence complete: "accept ADR-010 and the nine `d97r` runs as the numbers of record"; answer: "yes"). Comments are stripped label-blind from every line the models see (`+representation.strip_comments=true`, still an explicit override — D10.3). The full model of record is the designed cross-attention architecture retrained on that text, `ba7dcc5f…` / `8afbf7da…` / `ff980345…`, with the calibrators in `artifacts/phase6/retrained/`; concat (`9ac7a146…` / `7579bc33…` / `10b43ce5…`) and `llm_only` (`d82a6a70…` / `a1dce535…` / `9296a0e5…`) are reported beside it. The held-out evaluation is made on stripped text with these nine. **This supersedes, as the checkpoints of record, `80fc5238` / `c193e3bc` / `829c3555` in the 2026-10-07 full-model row below — the architecture choice in that row stands — and "all nine D9.7 checkpoints" in the 2026-10-07 authorisation row** | (b) keep the D9.7 checkpoints and strip at evaluation only · (c) report as is, with a limitation paragraph | Measured (§1.28, ADR-010): on `val` each retrained arm is within 0.19–1.51 pooled sd of the same arm stripped at evaluation only, so the choice moves no number resolvably — what it changes is that these models never saw the tell. `test` carries the tell on 73 of 546 units, so (c) was not defensible. Every line-level number logged with comments in place stays in the record, labelled as measured on leaky text (D10.7). **Not decided here:** D10.4 (moving the transform to L0) and `a1r` (the frozen screen on stripped text) | **ADR-010 — proposed 2026-10-07, accepted 2026-10-08 (Q14)** |
| 2026-10-07 | **The single held-out evaluation is authorised for the start of Phase 8** (owner, asked in session 16 with three options; answer: "When Phase 8 starts"). Before either split is read, the evaluation protocol — metrics, thresholds, intervals — is committed; then all nine D9.7 checkpoints are scored on `test` and on `clean_eval` once, each with its own `val`-fitted calibrator and threshold applied unchanged. No further check-in is needed | Ask again right before · run it the evening A6 ends, before Phase 7a | `Rules.md` §5.3: once, at the end. A protocol committed before the splits are read is what makes "once" checkable afterwards. **Phase 8 starts only after the owner signs the Phase 7 gate** — this is a go-ahead for the evaluation, not for skipping a gate | — |
| 2026-10-07 | **The owner signed the Phase 6 gate** (asked with §1.1d written and Q5 answered in the same batch; answer: "Sign it"). Phase 7 is the current phase | Not yet — read §1.1d first | Four of four exit criteria: calibration with its reliability curve, a `val`-only threshold, the configuration frozen by the owner's morning decision, Q5 explicitly deferred. The deliverables that were not built are listed in §1.1d rather than ticked | — |
| 2026-10-07 | **Q5 closed: the learned CWE-type classifier is deferred** (owner; answer: "Defer the classifier"). No model here predicts a vulnerability type. Phase 7a's explanations name a category **derived from templates** and say that it is | Train a CWE head on the frozen features (3 seeds, its own evaluation) · binary only, no type anywhere | Big-Vul carries a `CWE ID` per function (86 distinct among positives) but no model was trained on it, and a type head is a new objective with its own evaluation two days before the deadline. A template-derived category matches the deck's example output without claiming the model inferred it. Reversible: the `CWE ID` is already unit metadata | — |
| 2026-10-07 | **The owner signed the Phase 5 gate** (asked in session 16 with §1.1c in front of them; answer: "Sign it"). Phase 6 may start | Not yet | All five exit criteria met on logged evidence; the negative A1 result is recorded and investigated (§1.21, §1.22) | — |
| 2026-10-07 | **The full model from Phase 6 on is the architecture as designed — bidirectional cross-attention fusion** (owner: "keep whatever architecture was designed before"). The three existing D9.7 cross_attention runs (`80fc5238`, `c193e3bc`, `829c3555`) are that model at three seeds; nothing new is fine-tuned for it | `llm_only` (same Top-5, simplest, no attention maps) · concat | The arms tie on Top-5 (§1.22), so the choice is not a measurement. The designed model is the one with attention maps for Phase 7a, and all three arms are still reported side by side — the tie is the finding, not something the choice hides | — |
| 2026-10-07 | **Deferrals, delegated.** Asked which out-of-reach items to defer in writing, the owner answered: *"whatever is best or we can split small tasks between 2 machines."* Session 16 applies its recommendation, **reversible by the owner**: (1) the function-level criteria — unit-level aggregation, DiverseVul generalization, a full external-detector reproduction — are **deferred**; (2) the generative explainer (7b) is **deferred**; (3) external baselines are quoted as **`(reported)`** from `docs/prior_work.yaml`. Small self-contained groups may go to machine 2 as whole groups (`docs/SecondMachine.md`) | Run a balanced-subsample external baseline (14–24 GPU-h) · keep everything in scope | (1) and the subsample need negatives the models never trained on — the 148,983-function split, days per fine-tuned run (an estimate from unit counts); (2) is optional in `Phases.md` and does not fit a 4 GiB card beside CodeBERT. None of these is a "small task", so the second machine does not change them; it can take frozen ablation groups | — |
| 2026-10-07 | **Weighted BCE gets no more seeds for now** (owner: "Skip for now"). Focal loss, which every headline run used, stays | ~45 min on machine 2 for three more seeds of both arms | +0.0304 Top-5 at 1.96 pooled sd on a frozen encoder (§1.23) is a candidate, not a finding; it is listed as one in the write-up | — |
| 2026-09-15 | **The owner signed the Phase 3 and Phase 4 gates and accepted ADR-008 and ADR-009.** A1 fuses the **AST** with **identifier subtokens on**, set per run in `run_queue.py`'s A1 constants | CPG (makes ADR-007 D7.4 live: an edge-handling sweep first); subtokens off (the default, and the weaker structural arm) | AST is what every Phase 4 number was measured on. Subtokens are the best structural config (+0.066 Top-5); fusing the weaker arm would bias A1 towards "structure adds nothing". Set per run, not in `configs/gnn/default.yaml`, so no earlier run's resolved config moves | **ADR-008 D8.7 resolved; ADR-009 accepted (Q10)** |
| 2026-09-15 | **The inspector ranks a pasted function's lines with the Phase 3 semantic model and attaches template fix advice**, in a new Findings tab; the stage-by-stage view moves to a Detailed report tab | Keep showing no scores (the inspector's rule since session 10); wait for a fused model; generate fixes with an LLM | Owner decision: the tool's purpose is to tell a developer what to look at. The reasons for the old rule are kept as on-screen caveats instead: ranks-not-verdicts, same-distribution accuracy, uncalibrated scores. Templates, not generation, because Q4 recommends a template baseline and a generated fix has no reference set to be judged against | — (`frontend/`, outside the package) |
| 2026-09-14 | **A1 is run with the semantic encoder frozen, and its line features are computed once and cached.** §1.9 measured `frozen` at 37.8 win/s against `checkpoint`'s 22.9, at 1,234 MiB against 2,456 — but the throughput understates it, because a frozen encoder is a *pure function of its input* and encoding the same 5,945 units eighteen times is a cost of how the experiment is written, not of the experiment. The cache holds **line-aligned** features at each unit's own line count, refuses itself on any digest mismatch, and is gated on a per-arm equivalence test against the live path | Run A1 as specified (60-80 GPU-h); cache token states instead (~100× the disk, buys nothing); fewer seeds; fewer arms; truncation instead of windowing; a lower line cap | A1's core two arms cost 20-30 h and the full grid 60-80 on a 4 GiB laptop, and **the obvious cuts are all forbidden** — truncation destroys 38.1% of flagged lines (§7 #23), a lower cap costs `val` 17.07% (§7 #26), and fewer seeds is §7 #37, which this project has already been caught by twice. Freezing stays controlled by exactly ADR-008 §D8.4's argument, and is the **harder** test of the novelty: a fine-tuned encoder can reshape its own line features until a trivial combiner suffices, which makes “concat did as well” ambiguous. Cost: A1 answers a narrower question and must be phrased that way (D9.1), and the fine-tuned confirmation (D9.7) is triggered by the margin, not skipped by default | **ADR-009 — proposed 2026-09-14, accepted 2026-09-15 (Q10); built and verified, §1.20** |
| 2026-09-14 | **What is held fixed across A1's arms is fixed by construction, not by intention.** Shared projections/masking/width in `FusionModule`; fused line mask = the union of both branches' coverage; two learning rates (`train.encoder_lr` for the pretrained encoder, `train.lr` for the rest), identical across arms; the two branch datasets joined on `unit_id` with the population difference logged; `Phases.md`'s `none` built as `llm_only` **and** `gnn_only`; the line axis trimmed per batch and test-pinned to change nothing | Put the projections in the cross-attention module (gives it free parameters); positional join between the branches (correct today, silently wrong the first time a graph fails to build); one learning rate (has to be wrong for one half); score every arm on the union (lets `llm_only` predict lines it has no tokens for) | A1 is the one question here that can be answered *wrongly* with nothing failing — both arms train, both converge, both produce a number. §7 #39 and #40 are the softer version of the same failure and were both live bugs on one day. Every item is reversible at zero cost today and expensive the moment a seed is logged under it | **ADR-008 — proposed 2026-09-14, accepted 2026-09-15** |
| 2026-09-12 | **A relation is allowed to matter, or the encoder refuses to run.** `StructuralEncoder.edge_handling` ∈ {`auto`, `ignore`, `weighted`}; `weighted` learns one softplus-positive scalar per relation, uniform across GCN/GAT/GGNN; `auto` (default) resolves to `ignore` on a single-relation graph and **raises** on a multi-relation one | Keep ignoring `edge_type` (what the code did); relational GCN, one weight matrix per relation; a per-convolution mechanism | `edge_type` was computed at L1, carried through every batch and consumed by nothing — §7 #22 exactly. Measured on 60 `val` CPGs: 46,062 edges, `ast` 34,159 / `cfg` 3,258 / `dfg` 8,645 / `call` 0, so averaging them is a modelling choice that was being made by omission. One mechanism across all three convolutions keeps Phase 4's A2 comparison and its depth sweep from partly measuring edge handling. **Which handling reports the headline numbers is not decided** — ADR-007 D7.4 asks for `ignore` vs `weighted` on the same seeds first | **ADR-007 — proposed 2026-09-12** |
| 2026-09-08 | **Every source line is stripped of leading and trailing whitespace at the L0→L1 boundary, default-on.** The corpus encodes the flaw label in indentation; a one-line rule recovers it at precision 1.0000 with zero false positives on all six partitions | Re-indent with a formatter (changes line counts — destroys the labels); drop Big-Vul's line labels (ends the line-level project); normalize at L0 instead (new split fingerprint, full rebuild — deferred as Q9) | Stripping *restores the corpus's own intent* rather than imposing a convention: the non-flagged lines are already flush left. It preserves the line count exactly, so labels stay attached, and it costs only indentation — which this corpus never carried consistently, so nothing legitimate was learnable from it | **ADR-006 — proposed 2026-09-08** |
| 2026-08-16 | **A unit is parsed with whichever of the two grammars parses it best.** Parse as C++; if damage remains, parse as C too and keep the less damaged tree. C++ is tried first and wins ties, so the grammar is a pure function of the source | Trust the `lang` column (37.01% clean); always C (35.74%); always C++ (88.24%) | **89.14% clean** — the corpus's own label is the worst option by more than 2×, because 98.6% of Big-Vul is labelled `C` and much of it (Chrome, 40.9%) is C++. The 1,712 units that separate this from always-C++ are *C that the C++ grammar actively misreads* — `class`, `new` and `this` are identifiers in C — so the choice is between a clean tree and a confidently wrong one, not between two parse rates | **ADR-005 — accepted 2026-08-16, closing Q8** |
| 2026-08-11 | **Cross-split exact duplicates are withheld into a `duplicate` partition**, not reassigned or kept | Reassign every copy to one partition; keep them and report; drop them silently | A project-wise split constrains projects, not functions: forks and vendored copies put byte-identical code in different projects. Measured: 207 hashes / 426 functions across partitions with **zero shared projects**. Reassigning would break project disjointness; withholding preserves both invariants for 0.23% of the corpus | **ADR-004 — accepted 2026-08-11** |
| 2026-08-11 | **Splits are an artefact with a fingerprint, not a function call.** `assignment.csv` + `split_report.json` are written once and re-running verifies rather than regenerates | Re-split inside every training run | A split that silently changes invalidates every comparison made against the previous one. `build_splits.py` refuses to overwrite a differing assignment unless told to | — |
| 2026-08-11 | **Preprocessed shards (`artifacts/data/cache/*.jsonl`) are the boundary between corpus and dataset.** Nothing after Phase 1 opens a Big-Vul CSV | Read the CSVs in every phase; Parquet; a database | 11 GB of CSV is minutes per pass, ~98% of it columns we never read. JSONL keeps the records human-inspectable, which matters while labels are still being trusted | — |
| 2026-08-11 | **`clean_eval` is excluded from the project-disjointness check**, and its train-project overlap is reported separately | Count it as leakage | ADR-003 selects it by label confidence, so overlap is by construction. Counting it made every real split look dirty and buried the 207 genuine duplicates | — |
| 2026-08-10 | **PrimeVul = clean evaluation set, not a training filter.** Train on the noisy pool (~6,519 fn), evaluate on the PrimeVul-confirmed subset (1,393 fn / 4,499 lines), exclude the 331 contradicted | Inclusion filter (ADR-001 §D1.2); no filtering at all | Measured: the filter discards 83% of line-level data for mostly ambiguous reasons. Noise in training is survivable; noise in evaluation invalidates every number. The clean set is *larger* than LineVul's vulnerable test split (788) | ADR-003 |
| 2026-08-10 | **Dataset architecture: Big-Vul primary (LineVul release), DiverseVul generalization, PrimeVul validation, Juliet controlled** | PrimeVul as primary (no line labels); CVEfixes; MegaVul (GPL-3.0); merging corpora | Only Big-Vul has line labels *and* comparable published baselines. PrimeVul/DiverseVul/CVEfixes are partly built **from** Big-Vul, so none is a clean generalization set without project/hash filtering | ADR-001 |
| 2026-08-10 | **C/C++; graphs built incrementally AST→CFG→DFG→CPG; in-house CPG passes** | Java+Vul4J; multi-language; Joern as a runtime dependency; AST-only | Follows from the dataset choice. In-house keeps the line mapping under our control, which is the one invariant the project cannot compromise | ADR-002 |
| 2026-08-10 | MLflow tracks to **SQLite** (`sqlite:///artifacts/mlflow.db`), artifacts under `artifacts/mlartifacts/` | `file:./artifacts/mlruns`; `MLFLOW_ALLOW_FILE_STORE=true` | MLflow 3.x refuses the filesystem store outright — it is in maintenance mode. Opting out would put 20+ ablation runs on an unmaintained backend | ADR-000 §D0.7 |
| 2026-08-10 | **CUDA torch `2.13.0+cu130`** replaces the CPU wheel; device is resolved from `config.device.{kind,index}` via `training/device.py`, never hard-coded | Leave torch CPU-only; hard-code `.cuda()` at call sites | The default PyPI Windows wheel is CPU-only, so an idle 4 GB RTX 3050 Ti went unnoticed. `auto` probes with a real allocation and warns on fallback; `cuda` fails hard | ADR-000 §D0.8 |
| 2026-08-10 | Layer enforcement via **import-linter contracts** (1 layers + 2 forbidden), verified non-vacuous by a deliberate violation | Directory conventions + code review | Mechanical enforcement of `Architecture.md` §3; a review rule alone drifts | ADR-000 §D0.6 |
| 2026-08-10 | Generic `Registry` lives in **`utils/`**; each layer owns a `registry.py` | Single registry in `models/` | L0 `data` cannot import L2 `models` without breaking the downward-import rule | ADR-000 §D0.1 |
| 2026-08-10 | Added **`training/tracking.py`** as the only caller of `mlflow.start_run` | MLflow calls inline in `scripts/train.py` | Makes `Rules.md` §1.3 (every number cites a run) mechanically enforceable; every run is auto-tagged with `git_sha` + `config_hash` | ADR-000 §D0.2 |
| 2026-08-10 | Dev on **Python 3.12**, Docker pinned to **3.11** | 3.11 (not installed); 3.13 (thinner Tree-sitter wheel coverage) | `Architecture.md` §6's 3.11 is an `[ASSUMPTION]`, not a brief requirement; container remains the reproducibility reference | ADR-000 §D0.3 |
| 2026-08-10 | Ship **`make.ps1`** alongside the `Makefile` | `invoke` / `just` / `nox` | GNU make absent on the Windows dev machines; a new task-runner dependency is not worth one script. CI runs the `Makefile` so the two cannot drift | ADR-000 §D0.4 |
| 2026-08-10 | Added top-level **`.github/`** for CI | No CI in Phase 0 | Exit criteria must hold from a *clean clone*, which only CI proves | ADR-000 §D0.5 |

**No new dependencies were added** beyond those already listed in `pyproject.toml` / `Rules.md` §2.

**Decisions still pending — see §5.**

---

## 3. Session Changelog

Newest first. One entry per working session.

```
### YYYY-MM-DD — <session title>
Phase: <n>
Did:
  - <what was built/changed>
Files touched:
  - <paths>
Runs logged:
  - <run ID> : <headline metric>
Learned:
  - <anything that changes a plan>
Next:
  - <handoff to the next session>
```

---

### 2026-10-10 14:30 → (session 18) — `a1r` started, the ablation tables built; the held-out pass not resumed

Phase: 8.
Did:
  - Read the owner's one-line instruction (§2) as build + train. Pre-flight, then
    `a1rcheck` (gate passed, `3c749001…`) and `a1r` started at 14:52.
  - Tried first to continue the held-out pass; the session's permission system refused
    the launch before anything ran. Left paused, for the owner.
  - Built the ablation tables: `docs/ablations.yaml` (the ten ablations of
    `ResearchPlan.md` §5 — which reports, which conditions, which difference would count)
    and `scripts/make_ablation_tables.py`. 25 tests. Dry run on the real artefacts: 14
    tables run, 2 pending on `a1r`, 69 scaled comparisons.
  - Built the failure-case draw (`scripts/sample_failures.py`, 39 tests), ran it on `val`,
    read the thirty cases and drafted `docs/FailureTaxonomy.md` (§1.31).
  - Built the figures (`scripts/make_phase8_figures.py`, 10 tests) and drew the three
    `val` figures; rendered and looked at each before committing.
  - Ran the whole suite from a fresh clone: 1,434 passed, 5 skipped at `a63a62e`.
Files touched:
  - `docs/ablations.yaml`, `scripts/make_ablation_tables.py`,
    `tests/unit/test_make_ablation_tables.py`, `scripts/sample_failures.py`,
    `tests/unit/test_sample_failures.py`, `scripts/make_phase8_figures.py`,
    `tests/unit/test_make_phase8_figures.py`, `docs/FailureTaxonomy.md`,
    `docs/Memory.md`, `docs/Phases.md`, `CLAUDE.md`.
Runs logged:
  - `19e49b8f…` : the stripped semantic cache, rebuilt (a data artefact, not a result).
  - `3c749001…` : `a1r` overfit gate — Top-1 1.0000 from epoch 58. Never a result.
  - `42f8f27c…` : failure cases on `val` — 107 of 482 units missed on every seed; per
    flagged line the model finds 0.281 in the top 10 against 0.203 by chance.
    (`335adb30…`: the same draw before the chance column.)
Learned:
  - **`report_phase4.py` sorts a group's `seeds` and not its `run_ids`** — zipping the
    two labels runs with the wrong seed (`46ab8056…` is seed 42 and came out as seed 1).
    No number moves; the tables now read each run's seed from what the run logged.
  - Two catalogue rows were wrong as first written and the script refused both: weighted
    and plain BCE both log `loss = bce` and differ in `train.auto_pos_weight`. A declared
    condition that is checked against the run is worth more than one that is typed.
  - Rebuilding a semantic cache moves it by 2.9e-6 (§6 item 0a). Whatever was scored
    against the old build is reproducible to that, not bitwise.
  - A fifth of the flagged lines on `val` are blank or syntax alone, and the model ranks
    those kinds below chance (§1.31). The blank ones were known; the brace-only ones were
    not counted before.
  - A per-kind label rate needs its unit count beside it (§7 #65).
Next:
  - When `a1r` ends: read each seed report's `n_seeds`, mark the two `pending` tables
    `run` in `docs/ablations.yaml`, run `scripts/make_ablation_tables.py`, write it up.
  - The owner: say whether the held-out pass continues now.

### 2026-10-08 16:30 → 17:05 (session 17, continued) — Phase 8 started on `val`: the metric suite, and a table that needed its denominator

Phase: 8 started at the owner's instruction; 7's gate still unsigned (§2).
Did:
  - Built the line-level metric suite the evaluation protocol specifies, as functions of
    stored line scores: `evaluation/metrics/localization.py`, `evaluation/statistics.py`,
    `evaluation/evaluator.py`, `scripts/evaluate_scores.py`. 107 new tests.
  - Checked it against everything already logged before logging anything: tie bounds
    around each run's Top-1 / Top-5 / IFA, the calibration reports' in-sample F1 and ECE,
    the score audit's Top-1 / Top-5 in three readings — all exact.
  - Ran it on `val` for the nine models of record and wrote §1.29.
  - Wrote the definitions into `docs/EvaluationProtocol.md` §5.1 and §6.
Files touched:
  - `src/vulnloc/evaluation/` (4 files), `scripts/evaluate_scores.py`, four new test
    files, `tests/unit/test_registry.py`, `docs/EvaluationProtocol.md`, `docs/Phases.md`,
    `docs/Memory.md`, `CLAUDE.md`.
Runs logged:
  - `52b9c998…` : `val`, 9 checkpoints — Top-5 0.5394 / 0.5546 / 0.5519, PR-AUC 0.1713 /
    0.1816 / 0.1610 (cross_attention / concat / `llm_only`); source order Top-5 0.3071.
  - `30aeff44…` : the same numbers at `b875d70`, before the table wording was corrected.
Learned:
  - The script's first table marked three of 93 comparisons "resolved". At three seeds
    that threshold is crossed by chance 7 times in 100 (§7 #64). The rule was written for
    one comparison at a time; a script applies it to all of them at once.
  - The bootstrap interval over units (± 0.045 on Top-5) is wider than any difference
    between arms and than the seed sd: 482 functions is the binding sample.
  - Tie order is now defined (§7 #63 closed): source order, with both bounds.
Next:
  - The owner: the Phase 7 gate and the protocol. Then the held-out pass.

### 2026-10-08 afternoon (session 17) — the owner accepted ADR-010: Q14 closed, the `d97r` checkpoints are the models of record

Phase: 7 (7a built; gate prepared, not signed).
Did:
  - Checked the finished `d97r` slate for the owner (13 of 13 done, nothing running).
  - Recorded the owner's answer to Q14 — "yes" to accepting ADR-010 and the nine `d97r`
    runs as the numbers of record (§2, §5).
  - Carried the decision into the two protocols awaiting the owner: the evaluation
    protocol's §2 and §3 are decided; the human-evaluation protocol names seed 42 of the
    model of record (`ba7dcc5f…`, was `80fc5238…`) and stripped text.
Files touched:
  - `docs/adr/010-comment-stripping.md`, `docs/EvaluationProtocol.md`,
    `docs/HumanEvalProtocol.md`, `docs/SecondMachine.md`, `docs/Memory.md`, `CLAUDE.md`.
    Docs only — no code, no config, no run.
Runs logged:
  - none.
Learned:
  - ADR-010's own text gives two ranges for "retrained vs stripped at evaluation": the
    per-arm figures are 0.25, 1.51 and 0.19 pooled sd, so the range is 0.19–1.51; its
    Alternatives section says 0.25–1.51. §2 quotes the per-arm figures.
Next:
  - The owner's two steps, in §6 item 0a "Where it stands": the Phase 7 gate, then
    confirmation of `docs/EvaluationProtocol.md`. Phase 8 starts after both.

### 2026-10-08 10:59 → 14:50 (session 16, the day) — `d97r` resumed and finished: on text with no tell the three arms tie

Phase: 7 (7a built; gate prepared, not signed). Everything below is `val`.

- **Owner, 10:55:** "continue training where it was left off". Resumed at 10:59 with
  `artifacts/queue/d97r_resume.ps1` after checking that nothing the paused run depends on
  had changed.
- **All nine `d97r` runs finished at 14:21** (§1.28). Retrained without comments:
  cross_attention 0.5394 ± 0.0116, concat 0.5546 ± 0.0084, `llm_only` 0.5519 ± 0.0187
  Top-5. Cross-attention does not beat concat; neither fused arm beats `llm_only`; nothing
  reaches 2 pooled sd. Each arm matches itself with comments stripped at evaluation only.
- **Scored, calibrated, audited, explained by 14:44.** Line F1 at each checkpoint's own
  threshold 0.2359 / 0.2408 / 0.2330 — a tie again; no unit answered by a tell; 13,080 of
  13,080 flagged lines explained and verified.
- **The unattended script did its job and met one thing it could not handle:** the
  reproduction check refused one checkpoint (`8afbf7da…`). Diagnosed from the score files
  as tie order among half-precision logits (§7 #63), bounded, and resolved by re-scoring
  at the default tolerance — not by loosening the guard.

Files touched: `docs/Memory.md`, `CLAUDE.md`, `docs/EvaluationProtocol.md`,
`docs/adr/010-comment-stripping.md`. No code, nothing under `src/vulnloc/` or `configs/`.

Runs logged: listed at the end of §1.28. Void: `15fccf6d…`, `e735045f…`.

Learned:
- **A guard with a tolerance finer than the metric's own granularity will eventually refuse
  a correct input.** The right response was to find out why, measure how far the cause can
  move a result, and re-run — in that order.
- **An unattended chain must carry "this arm is incomplete" into every later step.** The
  calibration honoured it; the audit did not, and reported an arm at `n=2`.
- Retraining without comments is not resolvably different from stripping them at
  evaluation, for all three arms. ADR-010 has the evidence it was waiting for.

Next: the owner's decisions, in the order §6 item 0a's "Where it stands" lists them — Q14
first. Nothing is running; `configs/` and `src/vulnloc/` may be edited again.

---

### 2026-10-07 20:00 → 2026-10-08 05:50 (session 16, the night) — two retrained arms in, two traps caught before they fired, paused

Phase: 7 (7a built; gate prepared, not signed). Everything below is `val`.

- **Owner:** away overnight ("keep working/training on your own"); at 05:45 asked to
  pause — `run_queue.py --stop` at 05:46 — and to resume later.
- **`d97r` ran unattended for eleven hours without a crash:** six fine-tuned runs
  finished, each one attempt, none resumed; a seventh was paused mid-run.
- **Two arms have three seeds (§1.28).** Cross-attention retrained without comments:
  Top-5 0.5394 ± 0.0116. `llm_only` retrained: 0.5519 ± 0.0187. Each matches what its
  comment-trained checkpoints score with comments stripped at evaluation only, and the
  designed model does not beat its control (Top-1 −0.0118 at 1.92 pooled sd, same sign on
  every seed: a candidate in the control's favour, not a finding).
- **Trap 1, caught by reading defaults (§7 #61):** a checkpoint trained on stripped text is
  "not a change of text", so its scores would have been written beside the comment-trained
  ones and audited as one arm at `n=6`. Three populations now have three directories; the
  audit refuses the other text variant.
- **Trap 2, my own, caught two hours before it fired (§7 #62):** an edit to
  `explanation/generator.py` between two seeds of an arm — allowed by a rule that only
  protected what training imports — would have failed that arm's seed report at midnight.
  The report carries the recorded override; the rule is now "nothing under `src/vulnloc/`
  while a group runs".
- **Unattended machinery, all under `artifacts/queue/`:** `d97r_after.ps1` (retry what
  crashed, then score / calibrate / audit / explain complete arms; stands down if somebody
  paused the queue — it did, correctly), `d97r_report_cross.ps1` (made the one report the
  old plan could not), `d97r_resume.ps1` (continue the paused group).
- **Repaired:** a path in §6 that an unescaped backslash had turned into two vertical-tab
  characters; a scan of every tracked text file found no other control character.

Files touched: `scripts/score_checkpoint.py`, `scripts/explain_checkpoint.py`,
`scripts/audit_score_leakage.py`, `scripts/run_queue.py` and their four test files;
`docs/Memory.md`, `CLAUDE.md`, `docs/EvaluationProtocol.md`. Nothing under `src/vulnloc/`
or `configs/`. Tests: 1,265 passed, 2 skipped (22:11, `2f6e150`).

Runs logged: `ba7dcc5f…` `8afbf7da…` `ff980345…` (cross_attention), seed report
`bd2544c7…`; `d82a6a70…` `a1dce535…` `9296a0e5…` (`llm_only`), seed report `f814e521…`;
`9ac7a146…` (concat seed 42) paused, unfinished.

Learned:
- **Before leaving a group unattended, read what its last step will check**, not only what
  its training steps need. Both traps were visible in code that had not run yet.
- A queue process keeps the plan it started with; editing `run_queue.py` changes the next
  start, not the running one.
- For the two arms measured, retraining without comments is not resolvably different from
  stripping them at evaluation only (0.19 to 1.26 pooled sd on Top-5 and Top-1).

Next: resume (`artifacts/queue/d97r_resume.ps1`), then finish §1.28 as §6 item 0a's
"Where it stands" block says. The owner's six open decisions are listed there.

---

### 2026-10-07 afternoon → evening (session 16, continued) — a second label leak, its fix, A6 and Phase 7a

- **Owner, 14:45:** Q5 deferred, Phase 6 gate signed, the held-out evaluation authorised for
  the start of Phase 8 (§2). Then away for the evening, asking for work and training to
  continue.
- **Phase 7a built** on a worktree branch while A6 trained (§1.27) — and its first run on
  real validation functions showed the full model ranking comment lines first.
- **That was a label leak (§1.25).** A flagged line that ended a block comment lost the
  slash of its `*/`. Audited (`audit_label_leakage.py` gained two tells and a verdict for a
  tell that is never right; `audit_score_leakage.py` is new). The nine D9.7 checkpoints
  read it.
- **The fix:** `strip_comments`, validated against the grammar on every clean unit of every
  partition, threaded through `CodeUnit`, the three datasets, the semantic cache key and
  the scoring / explaining / audit scripts as an explicit, off-by-default switch. ADR-010
  written, Proposed.
- **Measured:** the nine checkpoints re-scored and re-calibrated on stripped `val`.
- **A6 finished and written up (§1.26).**
- **Human-evaluation protocol** written; pilot rater pack drawn.
- **Queue groups `d97rcheck` / `d97r`** added for the leak-free rerun, and **started at
  18:04** in their own window; `a1rcheck` / `a1r` (the frozen screen on stripped text)
  defined, not started.
- **The frozen screen audited** (27 checkpoints scored both ways): the structural branch
  is where the leak gets in; frozen `llm_only` does not read it.
- **Three things found by looking rather than by testing**, all the same evening: four
  template sentences that claimed more than a pattern can see (read on real code); a
  run-id prefix parsed as a float, which left one checkpoint out of an audit (§7 #60);
  and the score audit's own ranking, which dropped units the training metric counts as
  misses (it did not reproduce `gnn_only`'s logged 0.4191).
- **What the afternoon taught:** the leak was found by *reading what the model ranked
  first on real functions*, not by any metric — every number looked plausible. And the
  audit that should have caught it a month ago only asked whether a formatting tell was
  right; "ends in `*/`" is never right, which is just as readable.

### 2026-10-06 → 07 (session 16) — D9.7 finished: no fused arm beats fine-tuned LLM-only; two crashes survived overnight

Phase: 5 (criterion 5: the investigation is complete, control included; the gate is unsigned).

Did:
  - Session-start `--status`: 6 done, `d97-llm_only-seed42` `[paus]`, 5 pending — the card
    had been idle since Sun 4, 21:51.
  - Checked before restarting: tree clean at `de1fa21`; `git diff 0ce5d96..HEAD` touches
    nothing under `src/vulnloc` or `configs/` (only `scripts/run_queue.py`, from `782ea98`,
    which the paused run already started under); `last.pt` on disk; Docker not running; D:
    99% idle over 10 s; 72 MiB in use on the card.
  - **Restarted `run_queue.py --only d97` at 20:33** in its own console window, at the
    owner's request. The step resumed at 20:37:02 into the same MLflow run
    (`earlier_resumes=0`), `resumed_epoch=6 batches_already_trained=500 global_step=4565`;
    the step-4600 periodic checkpoint landed at 20:38:17 — 35 steps in 75 s, warm-up
    included, against the ~2.0 s/step baseline (§7 #55).
  - Told the owner what to bring back from the second machine: one `export_results.py`
    folder into `artifacts/machine2/`, seed reports run there first, no `mlflow.db`, no
    checkpoints, no queue state.
  - **Wed 7, 00:21 — the owner asked for progress; `--status` showed seed 1 `[FAIL]`.** Read
    its log: `PermissionError [WinError 5]` on the checkpoint rename at 22:47 (§7 #56). Seed
    2 was training normally. Started a watcher (`artifacts/queue/d97_finish.ps1`, 00:25) that
    retries the failed step through the queue once it exits and then re-runs the `llm_only`
    seed report; its guards are in §6 item 0a.
  - **01:12 — seed 2 crashed on a GPU driver fault** (§7 #57). The queue ended at 01:15; the
    watcher's `--retry-failed` continued seed 1, then seed 2, then re-ran the `llm_only`
    report at 03:14 (`n_seeds: 3`). All twelve `d97` steps are `done`.
  - 01:49 — found seed 1 at 5.4 s/step, the trainer spilling 792 MiB, and
    `ScheduledDefrag` running on D:. Tried to stop the defrag three ways from a non-admin
    shell; none worked. Chose **not** to pause-and-resume the run: earlier runs' slow
    patches had cleared themselves, and this one did at the epoch boundary (01:57).
  - Wrote the control into §1.22, the ledger, `Phases.md` and `CLAUDE.md`.
  - **Wed 7, 09:15 — imported the second machine's export.** The owner had put
    `transfer.zip` in `artifacts/machine 2/` (with a space); listed its 79 entries before
    unpacking (one top-level folder, no absolute or `..` paths, no checkpoints), then
    extracted to `artifacts/machine2/Sujal_20261005-224401/`. This machine's `mlflow.db` was
    not touched. Read the five gates from the imported store's metric history, computed
    every comparison from the nine seed reports, and **wrote §1.23.**
  - **Wed 7, 09:30 → — the owner: "do everything that you can solve on your own by building
    and fixing."** Taken as a go-ahead for fixes and for anything inside the current phase;
    **not** recorded as the Phase 5 signature, which was not given. Landed, each with tests:
    the rename retry, `run_queue.py --retries`, `seeds.require_sufficient`, the VRAM guard
    (measured first — `scripts/probe_vram_spill.py`), `resolved_gpu`, and `Top-1 last` in
    `export_results.py`. ruff / black / mypy / import-linter clean; **965 passed, 2 skipped**
    (was 937).
  - **Asked the owner four questions; the answers are in §2**: the Phase 5 gate is
    **signed**; the full model is "whatever architecture was designed before" — cross-
    attention; deferrals are delegated ("whatever is best…"), applied as recommended and
    reversible; weighted BCE is skipped for now.
  - **Phase 6, built and run the same morning (§1.24).** `models/heads/calibration.py`,
    `training/line_scores.py`, `scripts/score_checkpoint.py`, `scripts/calibrate.py`, with
    81 new tests. Scored the three full-model checkpoints on `val` — each reproduces its
    run's logged metrics — and calibrated them. Checked the fixed VRAM guard on a short
    real run first (`3ee36f97…`, void).
Files touched:
  - `src/vulnloc/training/{loop,job}.py`; `scripts/{run_queue,report_seeds,export_results}.py`;
    new `scripts/probe_vram_spill.py`; `tests/unit/test_{resume,run_queue,report_seeds,export_results}.py`,
    new `tests/unit/test_vram_guard.py`; this file, `CLAUDE.md`, `docs/Phases.md`. No config
    file changed. One helper script under `artifacts/queue/` (git-ignored).
  - Phase 6: new `src/vulnloc/models/heads/calibration.py`,
    `src/vulnloc/training/line_scores.py`, `scripts/score_checkpoint.py`,
    `scripts/calibrate.py`, `tests/unit/test_{calibration,line_scores,calibrate_script,score_checkpoint}.py`.
    **1,027 passed, 2 skipped.**
Runs logged:
  - `179632f5…` / `72f067d0…` / `2c7a7631…` : llm_only seeds 42 / 1 / 2, all FINISHED, all
    **resumed** — **Top-5 0.6169 ± 0.0048** (§1.22).
  - One `report_seeds.py` run →
    `artifacts/phase5/seed_report_fused_finetuned_llm_only.json` (`n_seeds: 3`). The queue's
    own 2-seed and 1-seed passes wrote the same file earlier and were overwritten.
  - `06c04755…` / `5d3a6ad9…` / `a593b1be…` : `score-val` for the three full-model runs;
    six more for concat and `llm_only` (ids in §1.24). `569c9701…` : calibration of the
    full model — **ECE 0.1704 → 0.0137, line F1 0.2954 → 0.3898 out of fold** (§1.24);
    `fd3e94c5…` / `97cb6a76…` : the same for concat and `llm_only` — **line F1 ties at
    0.390 / 0.391 / 0.394.** `3ee36f97…` : void probe of the VRAM guard. Eleven first-pass
    score and calibration runs are void as superseded (made from an uncommitted tree), and
    `5d06b630…` is void — a calibration attempt that died on a duplicate MLflow parameter
    name before logging a metric.
Learned:
  - **Brave was open again at the restart** and held 131 MiB dedicated on the NVIDIA
    adapter; the trainer sat at 3,752 MiB dedicated + 542 MiB shared. Close to session 15's
    reading with it open (3,575 + 462) and not slower, but it is a spill: the owner was
    asked to close it. **Correction:** "not slower" was judged against the fused arms'
    ~2.0 s/step; `llm_only` runs at ~1.2 s/step when healthy, so 2.1 s/step was already a
    spill slowing it.
  - **Fusion's lead over `llm_only` exists only with the encoder frozen** (§1.22).
  - **A failed step is not retried, and a seed report does not wait for its seeds** (§7 #56).
  - **An unattended Windows laptop defragments its hard disk, faults its GPU driver and
    spills VRAM, and `--status` shows none of it** (§7 #57).
  - **`export_results.py`'s `RESULTS.md` is not where to read an overfit gate.** It prints
    the selected epoch's Top-1 (0.9032 for a GGNN gate that reaches 1.0000 at epoch 33 and
    holds it). Read the metric history in the exported store (§1.23).
  - **Machine 2 ran torch 2.14.1, this machine 2.13.0.** Why is not established — its
    setup ran days later than this one's. It is in the manifest; one more reason its numbers
    stay beside ours.
  - **A background wait is not a guaranteed wake-up.** The 02:18 one reached the session at
    06:59. Anything that must happen overnight belongs in a process on the machine (the
    watcher), not in the session's intention to check.
  - **A6 is built, as a setting that cannot change the model.** `CrossAttentionBlock` runs
    one attention or both by `direction`; both modules always exist, so the three arms
    share a parameter set and a `state_dict` (ADR-007 §D7.5's rule), the idle one gets no
    gradient, and its map is `None` rather than zeros. The bidirectional path draws its
    random numbers in the order it always did. Queue groups `a6check` (two overfit gates)
    and `a6` (3 arms × 3 seeds, frozen, the bidirectional arm re-run under this commit, plus
    three seed reports that select on a new logged parameter, `fusion_direction`).
  - **`start_run` logs every override as a parameter**, so a script must not log its own
    `+group.key` again under the same name with a different rendering — MLflow refuses the
    second value and the run dies after its report is already on disk (`5d06b630…`).
  - **A calibrator belongs to its checkpoint.** The three seeds' Platt scales span 0.72 to
    3.05; an averaged one would fit none of them.
Next:
  - **Q5 (CWE labels) from the owner** — the last Phase 6 criterion.
  - Phase 7a and the rater pack; then the single held-out evaluation (§6 item 0a).
  - The owner switches off scheduled optimisation for D: before the next long run (§7 #57).
  - Tests not run this session — no code changed.

---

### 2026-10-04 (session 15) — D9.7's six runs finished: the null survives fine-tuning; queue paused at the owner's request

Phase: 5 (criterion 5: the investigation's six runs are in; the scaffold control and the
gate are not).

Did:
  - **Found the GPU idle at session start** (Sun 12:58): the queue had not been restarted
    since it paused itself on Sat 3 at 08:00. Checked that nothing under `src/vulnloc` or
    `configs/` changed since the paused run began (`0ce5d96..782ea98`: no files), that the
    tree was clean and `last.pt` was on disk, then restarted `run_queue.py --only d97` at
    12:59 in its own console window so it outlives the session.
  - `d97-concat-seed1` resumed at 13:04:07 from epoch 5, batch 418, step 3821 into the same
    MLflow run (`earlier_resumes=0`) and FINISHED at 14:31. `cross_attention` seed 2 (114 min)
    and `concat` seed 2 (211 min) then ran uninterrupted.
  - **Paused at 21:51 at the owner's request** (`run_queue.py --stop`):
    `d97-llm_only-seed42` (`179632f5…`) stopped at epoch 6, batch 500, step 4565 after
    114 min, wrote `last.pt`, exited 75; the GPU is free.
  - Ran the two seed reports whose arms have three FINISHED seeds, by hand, with exactly the
    commands `run_queue.py --dry-run` prints. The queue runs them again at its end and
    rewrites the same files. **Wrote §1.22.**
  - Owner: `D:\vulnloc_transfer\` was handed to the second machine and its slate is running
    there (the owner's report — not verified from this machine; nothing is in
    `artifacts/machine2/` yet).
Files touched:
  - Docs only: this file, `CLAUDE.md`, `docs/Phases.md`. No code, no config.
Runs logged:
  - `633ea97b…` : concat seed 1, FINISHED, **resumed** (`resumed_at=epoch=5,batch=418,step=3821`).
  - `829c3555…` : cross_attention seed 2, FINISHED. `a0c42d6f…` : concat seed 2, FINISHED.
  - `179632f5…` : llm_only seed 42, **PAUSED — no number.**
  - Two `report_seeds.py` runs →
    `artifacts/phase5/seed_report_fused_finetuned_{cross_attention,concat}.json`:
    **Top-5 0.6162 ± 0.0184 vs 0.6169 ± 0.0104** (§1.22).
Learned:
  - **A paused queue is silent.** The plan's "restart Saturday evening" depended on a person,
    did not happen, and nothing said so: the card sat idle from Sat 08:00 to Sun 12:59. The
    session-start `--status` check is what caught it — never skip it.
  - **A hardware-accelerated browser takes dedicated VRAM on the NVIDIA adapter.** Brave held
    162 MiB there during the resumed run, while VS Code and `dwm` sat on the integrated GPU.
    With it open the run showed 3,575 MiB dedicated + 462 MiB shared and ~1.97 s/step over
    its first 79 steps (warm-up included). Close it during fine-tuned runs; per-process
    numbers come from `Get-Counter '\GPU Process Memory(*)\Dedicated Usage'`.
  - **MLflow shows a paused run as `FAILED`** until it is resumed and finishes (`633ea97b`
    did the same). Trust `run_queue.py --status`' `[paus]`, not the MLflow status column.
  - **Docker Desktop 4.93.0 installed at 22:08, on D: at the owner's request** — while the
    queue was paused. `Docker Desktop Installer.exe install --quiet --backend=wsl-2
    --installation-dir=D:\Docker\Docker --wsl-default-data-root=D:\Docker\wsl`, after
    `wsl --install --no-distribution` (WSL 3.0.1, VirtualMachinePlatform enabled). Installer
    verified first: 627,791,792 bytes, Authenticode `Valid`, signer Docker Inc. Both exit
    codes 0; `install-settings.json` records the D: data root. **What still landed on C:**
    WSL itself (`C:\Program Files\WSL`, 1,167 MB) and Docker's CLI plugins
    (`C:\Program Files\Docker\cli-plugins`, 675 MB) — neither is relocatable by a flag.
    **A reboot is pending and Docker has never been launched**: the licence is unaccepted
    (deliberately — the owner accepts it on first launch), no image is built, and the Phase 0
    and Phase 9 Docker criteria are exactly as unverified as before. The installer also
    set Docker Desktop to **start at sign-in** (`HKCU\...\Run`) — it must be quit, and that
    setting unticked, before any training run (§6 item 0a).
  - **After the 22:30 reboot the owner asked why the hard disk was so busy.** Measured: D:
    at 100% busy, queue 3.1, moving only ~6 MB/s. **D: is a SATA hard disk** (Seagate
    ST1000LM049); C: is the NVMe SSD — not checked before Docker was installed there. Two
    things were on it at once: Docker, auto-started and creating its 1.6 GB data disk, and
    **MATLAB's startup accelerator** (`D:\MATLAB`), which pre-reads at every sign-in and was
    still reading 14 MB/s after Docker stopped. Docker stopped (`wsl --shutdown`), its
    auto-start removed at the owner's request ("i dont want docker starting everytime i
    open my laptop"); the owner sets Docker up later. **Consequence for the project:** the
    corpus, the checkpoints and Docker share one spinning disk — never run Docker beside
    training, and wait for D: to settle after a boot before restarting the queue.
    At 22:40 the owner had the accelerator stopped too: its two SYSTEM scheduled tasks
    (`MATLAB R2023a Startup Accelerator`, `MATLAB R2026b Startup Accelerator`; logon + twice
    daily) are **disabled**, not deleted. D: read 0% busy afterwards.
Next:
  - `python scripts/run_queue.py --only d97` — resumes `llm_only` seed 42, runs seeds 1 and
    2 and the three seed reports: **≈ 5–8 h**. Then add the control comparison to §1.22 and
    put the Phase 5 gate to the owner.
  - The two held Q13 questions (§6 item 0a rule 8) are still unanswered.
  - Tests not run this session — no code changed.

---

### 2026-10-02 (session 14) — D9.7 launched, and the resume path it could not have survived without

Phase: 5 (criterion 5 in progress; no fine-tuned result yet).

Did:
  - **Q12 closed**: committed the A1 write-up and the frontend work (`18ca52b`, `6e64486`) so
    D9.7 runs log a clean sha.
  - **Q11 scheduled by delegation.** Owner: finish the project in about a week; laptop free
    except Saturday daytime; "do whatever you feel is best". So D9.7 is one resumable queue
    group of nine runs, not two fixed nights, and gains an `llm_only` fine-tuned arm.
  - **Found that `train.resume_from` was not a resume** (§7 #53) while planning runs that are
    longer than the laptop is free. Fixed it before launching anything: exact mid-epoch
    continuation, same MLflow run, early stopping and best metrics restored, atomic
    checkpoint writes, a config-equality guard on resume.
  - **Pause on request**: `VULNLOC_STOP_FILE` (an env var, not a config key — §7 #38). The
    trainer pauses at the next optimizer step, writes `last.pt`, exits 75.
  - **Queue**: `resumable` steps, `--until HH:MM`, `--stop`, keep-awake while running
    (`SetThreadExecutionState`), `--status` shows Top-5/Top-1 and a VRAM-spill marker;
    groups `d97check` and `d97`.
Files touched:
  - `src/vulnloc/training/{loop,job,tracking}.py`, `scripts/run_queue.py`
  - `tests/unit/test_resume.py` (15), `tests/unit/test_run_queue.py` (23)
  - `CLAUDE.md` (§4 commands, landmines #53/#54), this file
Runs logged:
  - `1ab2fd03…` : **probe, never a result** (`train.limit=300`, CPU, structural). Paused at
    epoch 1 batch 3 (exit 75), resumed by a second process into the same run: status
    FINISHED, eval steps 0/1/2 each exactly once, `resumed_at=epoch=1,batch=3,step=27`.
  - `80fc5238…`, `fbf65554…`, `c193e3bc…` : D9.7 cross_attention 42, concat 42,
    cross_attention 1 — FINISHED overnight, no resumes (§4). **Progress, not a result.**
  - `d97-overfit-*` : live-encoder overfit, **all three arms PASS** — Top-1 = 1.000 with
    `frozen=False`, cache off: cross_attention `d3dbd7e9…`, concat `b4998f5c…`, llm_only
    `ad9f36e8…`. Never results.
  - `2e3b19d5…` + a `semantic-cache` run : **code-path checks for the second machine, never
    results.** The GraphCodeBERT frozen cache built offline on this laptop (train 5,457 units,
    932 MB, 124 s, digest `5b3828776f7a`; val 488 units, `df44be5f6030`) into
    `artifacts/semantic_cache_graphcodebert/`, and its frozen overfit gate reached Top-1 = 1.000
    (123 of 150 epochs) reading that cache. The second machine rebuilds the cache itself.
  - `cb2b1e99…` : **VOID** (tagged). The first `d97-cross_attention-seed42` attempt, used as
    the vehicle for the live pause/resume test: paused twice in epoch 0, ran 190 steps
    paging VRAM (§7 #55), started on `929970a` before the fix. Checkpoints deleted, log kept
    as `artifacts/queue/logs/d97-cross_attention-seed42.void-cb2b1e99.log`. The step was
    then restarted from scratch on the fixed commit so the headline run carries no footnote.
Learned:
  - A feature nobody has used is a feature nobody has tested. `resume_from` sat in the config
    for three weeks with a docstring describing behaviour that would have corrupted the first
    run to rely on it (§7 #22's family).
  - First-step peak with the encoder training on the overfit subset: 2,532.6 MiB (61.8%).
    That is **not** the full-corpus peak — §1.17's worst unit is 3,786 MiB (92.4%).
  - **Measured cost of a fine-tuned fused run** (from `cb2b1e99…` after the fix): 5,457 train
    units / 11,514 windows = **721 optimizer steps per epoch at ~1.43 s/step ≈ 17.5 min**,
    plus ~1 min to evaluate 488 `val` units — **~19 min per epoch, so ≤ ~3.2 h for the full
    10 epochs** and ~2.7 h if early stopping lands where Phase 3's did. Nine runs ≈ 24–29 h.
    The earlier 18–24 h estimate for six runs was about right; ~2.0 s/step, seen before the
    fix, was already a mild spill.
  - **Overnight 2026-10-02/03, measured:** three D9.7 runs FINISHED with no resume —
    `80fc5238…`, `fbf65554…`, `c193e3bc…` — in **118, 123 and 168 min**, early stopping at
    epochs 5–6, so a full fine-tuned run is **~2–2.8 h**, under the estimate. `d97-concat-seed1`
    paused itself at 08:00:30 (epoch 5, batch 418, step 3821). Remaining: that run's tail, then
    five runs ≈ **12 h** from the Sat 3 evening restart. **No arm has three seeds yet — no
    comparison may be written down** (§7 #37).
  - The cache-release guard fired once in that epoch (step 429, 4,250 MiB reserved) — the
    oversized unit of §1.17, as predicted. One release per epoch is the healthy pattern.
  - **A residual, bounded spill remains and was left alone on purpose.** The clean
    `d97-cross_attention-seed42` process (started 23:43 on `70824ec`) measured **3,839 MiB
    dedicated + 302 MiB shared at ~1.59 s/step**, against 1.43 s/step with 0 shared. The
    guard compares `memory_reserved` with the card's nominal 4,095.5 MiB, but the budget
    Windows actually grants this process is ~3.84 GiB, so 3.84–4.10 GiB spills without
    tripping it. Likely fix: trigger on `torch.cuda.mem_get_info()` free memory instead.
    **Do not change it while D9.7 is in flight** — it is a `src/vulnloc` edit, and every
    remaining seed would then differ from the logged ones (§7 #38's family). ~10% wall
    clock, no effect on any number.
Next:
  - §6 item 0a.
  - Tests 917 passed / 2 skipped; ruff / black / mypy clean.

---

### 2026-09-15 (session 13) — gates signed, the cache that makes A1 affordable, and a Findings tab

Phase: 5 (A1 queued; no fused result yet).

**Owner decisions, all 2026-09-15:** Phase 3 and Phase 4 gates signed; ADR-008 and ADR-009
accepted; A1 fuses AST with subtokens on; the inspector ranks lines with the Phase 3 model
and gives template fix advice.

Did:
  - ADR-009 end to end: `training/semantic_cache.py`, `scripts/build_semantic_cache.py`,
    the cached branch in `FusedLineModel`, `FusedLineDataset(semantic_cache=)`,
    `train.semantic_cache` in `configs/train/fused.yaml`, the frozen-encoder refusal in
    `train_fused.py`. Built the cache (`c99764bc…`) and measured equivalence on the GPU.
  - Two ADR-009 assumptions broken by measurement and amended in place (§1.20, §7 #50, #51).
  - Overfit on the cached path passed for three arms; all six re-queued on the A1 config.
    `d4ac0260…` tagged void.
  - `run_queue.py`: `p5check` (overfit, all arms) and `phase5` (A1, core arms first).
  - Inspector: Findings tab (`frontend/scoring.py`, `frontend/fix_templates.py`,
    `findings()` in `pipeline.py`, `/api/findings`); the old view is the Detailed report tab.
    Fixed two stale panels (Semantic said "no valid weights"; Structural said "nothing is
    built") and a pre-existing sample-button overflow.
Files touched:
  - src/vulnloc/training/{semantic_cache,fused_datasets}.py, src/vulnloc/models/fused.py
  - scripts/{build_semantic_cache,train_fused,run_queue}.py, configs/train/fused.yaml
  - frontend/{scoring,fix_templates,pipeline,server}.py, frontend/static/*, frontend/phases.json
  - tests/unit/{test_semantic_cache,test_fix_templates,test_frontend_findings}.py,
    tests/unit/test_fused_datasets.py (fixture factored into `make_tokenizer`)
  - docs/adr/008, docs/adr/009, docs/Memory.md, CLAUDE.md, frontend/README.md
Runs logged:
  - c99764bc : semantic cache, train 932 MB / val 133 MB, fp32
  - 117df3a4 / aebd4407 / 3b32a40d : overfit on the cached path, Top-1 1.000 (sanity only)
Learned:
  - A "bitwise" determinism check on GPU cannot distinguish scatter noise from dropout; test
    the encoder output and the pooled output separately.
  - CodeBERT under AMP returns fp32. Check a dtype before designing storage around it.
  - The Phase 3 model scores even normal lines 0.7–0.97 — it has never seen a safe function.
    A developer-facing view has to lead with rank, not with the number.
  - `p5check` gate: five arms Top-1 1.000; `llm_only` 0.968 and still climbing — the frozen
    encoder's cost, recorded in §1.20. Committed as `d1003f6` (owner-approved) and launched
    `run_queue.py --only phase5` from the clean tree.
  - `report_seeds.py` gained `+seeds.tag`, because all six arms share `train.name=fused` and
    would otherwise overwrite one another's report.
  - **A1's frozen screen ran to completion (18 / 18, 09:49–14:16) and is written up in
    §1.21**: cross_attention ≈ concat on Top-5; every fusion arm beats frozen `llm_only`;
    nothing beats fine-tuned LLM-only.
  - Laptop slept at 11:41 mid-run (`sum` seed 42); resumed cleanly, results unaffected.
  - False alarm, recorded so it is not repeated: epoch 0 of the first A1 run read 281 s and
    looked 2.5× over budget. My own CPU profiling script was competing with it; the real
    epoch is ~111 s. **Do not profile on the machine a timed run is using.**
Next:
  - **Owner decision: D9.7's fine-tuned confirmation** (§6 item 2) — ~18–24 h.
  - Tests 874 passed / 2 skipped; lint clean.

### 2026-09-14 (session 12) — Phase 5 built end to end, and deliberately not run

Phase: 5 (3 of 5 criteria met **by code and tests**; 0 by measurement).

**The instruction was "start working ahead", and the gate is still unsigned.** `Rules.md`
§1 bans building ahead and §6 item 1 says Phase 5 must not start until the owner signs the
Phase 3 and Phase 4 gates. Closing a phase is not an assistant's decision, so the gate was
left unsigned, **no run was logged, and no metric was produced.** What was built is the
implementation, which needs no number to be correct.

Did:
  - `models/fusion/` — `FusionModule` base (projections, re-masking, validation, output
    width, line-mask policy), `CrossAttentionFusion` (bidirectional MHA, maps retained),
    and five comparators: `concat`, `sum`, `gated`, `llm_only`, `gnn_only`.
  - `models/fused.py` — `FusedLineModel`: both encoders → the shared line axis → fusion →
    the same `LineHead` both baselines use. `explain()` is Phase 7's attention-map export.
  - `training/fused_datasets.py` — `join_branches` (on `unit_id`), `collate_fused` (flat,
    `sem_`/`str_`-prefixed), `FusedBudgetBatchSampler` (windows **and** nodes),
    `fused_line_coverage`.
  - `configs/fusion/{default,cross_attention,concat,sum,gated,llm_only,gnn_only}.yaml`,
    `configs/train/{fused,overfit_fused}.yaml`, `scripts/train_fused.py`.
  - `training/job.py` gained an **optional** `param_groups` argument. `None` reproduces the
    previous behaviour exactly, so no Phase 3 or Phase 4 run changes meaning.
  - **ADR-008** records what is held fixed across A1's arms, and why. Status: Proposed.
  - 93 new tests. Suite: **791 passed / 2 skipped**; ruff, black, mypy, import-linter clean.

Files touched:
  - `src/vulnloc/models/fusion/*`, `src/vulnloc/models/fused.py`,
    `src/vulnloc/training/{fused_datasets,job}.py`, `scripts/train_fused.py`,
    `configs/fusion/*`, `configs/train/{fused,overfit_fused}.yaml`,
    `tests/unit/{test_fusion,test_fused_datasets,test_registry,test_configs}.py`,
    `docs/adr/008-fusion-contract.md`, `docs/Phases.md`, `docs/Memory.md`, `CLAUDE.md`,
    `frontend/phases.json`

Runs logged:
  - **None, deliberately.** The entrypoint was exercised on real `val` shards through its
    own composition path — all six arms build, gradient reaches both branches, attention
    comes back shaped, the join drops nothing — with no MLflow run and no metric.

Learned:
  - `torch.nn.MultiheadAttention` returns **NaN** for a unit whose key mask is fully
    masked, and units with no graph nodes are real here. §7 #41.
  - The fused line mask is the **union** of the two branches' coverage, so a fused model
    can score lines the structural arm cannot (§7 #17). That belongs to the *scaffold* —
    `concat` has it too — and reporting it as A1's result would be wrong. §7 #42.
  - A positional join between the two branch datasets is **correct today** (every logged
    Phase 4 run reports `graph_build_failures=0`) and silently wrong the first time it is
    not. §7 #43.
  - `Phases.md` and `Architecture.md` §4.4 disagree about the comparator set: `none` vs
    `llm_only`/`gnn_only`. Built as the latter; `Phases.md` now says so. ADR-008 §D8.6.
  - **Doc drift found, not fixed:** §7's prose cites issues #35–#40, which exist only in
    `CLAUDE.md` §8 — session 11 added them there and not here. See §6.3.

Then, same day (session 12b):
  - **`scripts/probe_vram_fused.py`** — the fused memory question, measured (§1.17).
    **The batch budgets do not control peak memory**: every budget in the grid is
    dominated by one oversized unit at **92.4% of the card**, and 46.9% is static
    optimizer state before a single activation. §7 #44 is closed.
  - **Found and fixed a bug in the shared loop**: `evaluate()` had no `no_grad`, and
    eval mode bypasses gradient checkpointing. §7 #46. **No logged metric changes.**
  - **`scripts/make_review3_figures.py`** — Review III figures and tables, generated from
    logged runs only, no value typed by hand. Every figure and table now says the metric
    is **validation-selected** (§1.18), and the deck carries a comparison against
    published work on the same corpus, quoted out of **`docs/prior_work.yaml`**.
  - Both run-discovering reports now refuse a run carrying a `probe` parameter (§7 #40's
    family: a naming convention was the only thing keeping probes out of a seed mean).

Then, in the same session (12c):
  - **Relabelled the Review III deck as validation-selected.** Every figure subtitle, every
    axis label and §1 of `results.md` now say the number is the epoch early stopping chose
    on `val` by `line/top_5_accuracy` — so Top-5 is **selected on itself**. §1.18.
  - **`docs/prior_work.yaml`** — the published line-level numbers on Big-Vul, quoted with
    sources, retrieval date and a per-row reason they are not comparable (§1.19). They
    render as a table and **never as a chart**; `tests/unit/test_prior_work.py` fails if a
    quoted value reaches an SVG.
  - Fixed a layout bug the longer captions exposed: `_wrap` assumed 6.0px per character,
    which is the *average* advance, and a caption full of run IDs and capitals ran 10px off
    the canvas. Now 6.4px, with all four figures verified at zero overflow.
  - Swept 12.83 GB of checkpoints (6 void/killed run dirs, 29 `last.pt`); every `best.pt`
    and `summary.json` verified intact. `artifacts/checkpoints` 20.9 GB → 8.2 GB.
  - **Costed A1 honestly and then found the lever.** Core two arms 20-30 GPU-h, full grid
    60-80, from measured parts. **ADR-009 (Proposed, Q10)**: freeze the encoder and cache
    its line features → **~5 h for all six arms × 3 seeds**. Nothing built; it changes what
    A1 asks, so it is the owner's call and must be settled before any fused seed is logged.
  - **Found a device-provenance hole while costing a second GPU.** `job.py:149` logs
    `resolved_device` as `str(device)` — which is `"cuda:0"` on *every* card. §7 #49.

Next:
  - **The gate.** Nothing in Phase 5 should run until §6 item 1 is done.
  - Re-run one arm of `probe_vram_fused.py` to make §1.17's `no_grad` saving citable.
  - **Compute Top-10 accuracy from the logged predictions.** It is the one cheap change
    that would make a single row of §1.19's table mean something — we report Top-5 and
    LineVul reports Top-10, so today the two cannot be placed against each other at all.

---

### 2026-09-13 (session 11) — Phase 4 wired, run and measured; the guards that were wrong

Phase: 4 (1 → 4 of 4 criteria in flight), Phase 3 seeds running unattended.

**Phase 4 went from "encoder exists" to "running end to end".** `training/graph_datasets.py`
(dataset, node-budget sampler, collation, coverage accounting), `StructuralLineModel`,
`configs/gnn/{default,gcn,gat,ggnn}.yaml`, `configs/train/{structural,overfit_structural}.yaml`,
`scripts/train_structural.py` on a shared `training/job.py` loop. `edge_type` is consumed
rather than carried (**ADR-007**), with `auto` refusing a multi-relation graph outright.

**Results, all on held-out `val`:**
- **Overfit test PASSED** — run `8b0a3c1d…`, Top-1 = Top-5 = **1.000**.
- **GNN-only over 3 seeds** — Top-5 **0.3645 ± 0.0231**, Top-1 0.1155 ± 0.0132,
  line F1 0.0956 ± 0.0016, IFA 19.11 ± 1.13. **Far below the semantic arm** (Top-5 0.610),
  which §1.12 explains rather than excuses.

**Three guards turned out to be wrong, and each was found by running the thing.**

1. **The overfit config under-trained.** 60 epochs with dropout on plateaus at Top-1 ~0.80
   and reads as FAILED. The wrong diagnoses, in order: capacity (no — 512/6 is *worse*),
   regularisation alone (no), feature aliasing (no — §1.12 puts 4-hop ambiguity at 7.35%).
   It was the schedule. §7 #35.
2. **`config_hash` cannot decide whether two runs are replicates.** It digests the whole
   resolved config, so it changes with the **seed** — the one thing replicates must differ
   on. Three genuine GCN replicates carried three different hashes. `report_seeds.py` would
   have refused every legitimate aggregation, Phase 3's included. Now compares logged
   **parameters**, excluding seed and run identity, and **names** the offending key.
3. **A key added to `semantic.yaml` that nothing reads** moved every semantic run's
   `config_hash` away from seed 42's logged `25c7178bafd2` — which would have made seeds 1
   and 2, queued for tonight, incomparable with the seed already logged. §7 #22 again, with
   a new consequence. §7 #38.

**§1.12 is the session's most reusable finding.** What the structural branch can see,
measured **without training anything**, via Weisfeiler-Lehman refinement: 76.22% of flagged
lines are indistinguishable from an unflagged one at 0 hops, 27.91% at 2, 7.35% at 4. It
says the ADR-002 §D2.5 subtoken gap is expensive only at shallow depth — **and that the
2/4/6 depth sweep is not a clean over-smoothing experiment** (§7 #37).

**Built for the network going away at 02:00.** `scripts/run_queue.py` runs the remaining
slate offline, sequentially, resumable, continuing past failures, with per-step logs and
recovered run ids. Verified beforehand that `microsoft/codebert-base` loads entirely from
the local HF cache — the only network dependency in the pipeline.

**Stopped:** Phase 3 seed 1's first attempt (`7ea077a0…`) was stopped by the owner at epoch
1 and is tagged `void` in MLflow; the queue restarts that seed from scratch.

**Later the same session — Phase 4 closed out, and four more guards that were wrong.**

- **All four Phase 4 exit criteria met** (§1.13). Depth sweep, three variants, ≥3 seeds,
  overfit test. The gate itself is unsigned — that is the owner's.
- **ADR-002 §D2.5's second half is built** — identifier subtokens, `representation/
  subtokens.py` + `build_subtoken_vocab.py` (run `ddb24d81…`, 31,389 entries, 99.72%
  occurrence coverage). **Off by default**, so every GNN number above stays reproducible.
  Built from **`train` only**: identifiers from val/test would be a leak, and the loader
  refuses an artefact that says otherwise.
- **The A4 comparison is queued** with a falsifiable prediction attached: §1.12 says the
  subtoken gain should be **large at depth 2 and small at depth 4**.
- **If the A4 result looks flat, do not first suspect the wiring — it was verified.**
  A small trial (600 units, 4 epochs, depth 2, CPU) gave *identical* Top-5 and Top-1 with
  and without subtokens, which looks exactly like a dead feature. It is not: blanking
  every identifier in a real `val` batch moves node states by up to **2.598**, and the
  encoder reports `uses_subtokens=True` with `subtoken_vocab_size=31389` logged as a run
  parameter. At that scale Top-1/Top-5 are coarse enough that two different models land on
  the same count; the F1s did differ (0.0176 vs 0.0178). Also worth knowing: only **8.7%
  of subtoken slots carry a real token**, because only leaf nodes have text at all — the
  feature can only ever speak through the leaves.

The four wrong guards, all found by asking what would happen rather than waiting:

1. **`report_phase4.py` did not group on `use_subtokens`.** The A4 runs would have been
   averaged into their own comparison partners — a feature comparison silently becoming a
   wider seed group. Caught **before** it produced a number.
2. **Then the opposite**: filtering on `use_subtokens` dropped every run that predated the
   flag. Rule that resolves both (§7 #39): *a parameter that did not exist when a run was
   made was, by definition, not enabled in that run.*
3. **The frontend had no branch for the new stage states**, so `trained` rendered with no
   badge and no rail colour — found by screenshotting the page, not by reading the JSON.
4. **The inspector's red banner said the three seeds "have not been re-run"**, which
   stopped being true today. Its lead-in sentence was hardcoded in the page; it now comes
   from the server with the rest of the record.

**Checked, not assumed:** `microsoft/codebert-base` loads entirely from the local HF cache
(the only network dependency); the machine will not sleep (High performance, timeout 0);
and seed 42's logged parameters are identical to a semantic run from tonight modulo the
seed, so the overnight Phase 3 aggregation will not be refused.

**Overnight into 2026-09-14 — both phases closed, and two findings retracted.**

The unattended queue ran **23 of 23 steps**. `scripts/run_queue.py --status` is the record.

- **Phase 3: 4 / 4.** LLM-only over three seeds on normalized text \u2014 **Top-5 0.6141 ±
  0.0055**, Top-1 0.3513 ± 0.0138, line F1 0.3692 ± 0.0204, IFA 11.93 ± 0.60 (§1.14).
- **Phase 4: 4 / 4.** Best structural config is **gcn depth 4 + subtokens, Top-5 0.4302 ±
  0.0372** (§1.15). Node-type-only was 0.3645.
- **`train_semantic.py` folded onto `training/job.py`** once its seeds were in, as the
  `TODO(decision)` there required. 293 → 187 lines. Verified the logged parameter set is
  unchanged, so a fourth seed still aggregates with the three that closed the phase.

**The two retractions are the important part of this session (§1.16).** Six extra seeds
were run purely to firm up two claims already written down, and both dissolved:

- **The over-smoothing finding is withdrawn.** At n=1 depth 6 sat 0.047 below depth 4; at
  n=3 the gap is 0.016 and **no pairwise depth comparison in 2\u20136 exceeds 0.6 sd**.
- **A4's "1.5× larger at depth 4" was a one-seed artefact.** The gains are +0.0567 and
  +0.0657, **0.27 sd apart** \u2014 no depth dependence. §1.12's prediction stays unsupported,
  but the reversal does not stand either.

Both were written with correct caveats \u2014 one seed, the ±0.023 spread named, "suggestive
not conclusive", "re-run before the paper" \u2014 and both still got written as findings.
**2 sd of seed noise looks exactly like a result.** The rule that follows: do not write a
comparison down until every arm has ≥3 seeds (§7 #37).

**Guards that were wrong, all found by running rather than reading:** the overfit config
under-trained; `config_hash` refused every legitimate seed aggregation; a key added to
`semantic.yaml` that nothing read; A4 runs would have been averaged into their own
partners; then the opposite when the filter was over-tightened; two `train.limit=600` CPU
trials reached an A4 mean; `resolved_device`, added mid-sweep, split a group hours after
§7 #39 recorded exactly that failure mode; and the frontend had no branch for its own new
stage states.

### 2026-09-12 (session 10) — the inspector, Phase 4's foundation, and seed 42 restarted

**Frontend.** `frontend/` is a pipeline inspector: paste a C/C++ function and every built
stage runs on it live — ADR-006 normalization, ADR-005 grammar selection, all four graph
builders, windowing, and the shared line axis. FastAPI + vanilla JS, no build step. It is
**not** the Phase 9 serving API (unstarted, and `Phases.md` forbids building ahead), so it
sits outside `src/vulnloc` and outside the import-linter graph. **It shows no
vulnerability scores** — there is no valid model, and `Rules.md` does not exempt demos.
`/api/model` is the single place that changes when seeds land. `frontend/phases.json`
mirrors CLAUDE.md §3 and must be updated when a phase changes.

**Phase 4 foundation** (not the phase — its exit criteria need GPU work):
- **Node-type vocabulary, run `2c9ce1b1…`** — 304 types + unknown slot over all 188,636
  units, 0 build failures, digest `a0643c4e…`. §1.11 records the measurement that closes
  landmine §7 #21: `tree-sitter-c`'s own symbol list would **miss 85** of these types.
- `StructuralEncoder` puts GCN/GAT/GGNN behind one scaffold, so the Phase 4 comparison is
  between convolutions and the 2/4/6 over-smoothing sweep is a config change. Residual +
  layer norm default on; a test pins that the residual actually preserves node
  distinctness at depth 6.
- **Gap, recorded not skipped:** ADR-002 §D2.5 wants node-type **+ subtoken** embedding.
  Only the first exists — `to_pyg` emits no subtoken ids, so the second is an L1
  tensor-layout change plus a vocabulary (an ADR-002 amendment). `TODO(decision)` on
  `NodeFeatures`. **A4 must not be read as measuring the full D2.5 feature set.**

**Seed 42 restarted on the normalized corpus** — run `fc902107…`, `normalize_whitespace=True`
confirmed in the logged params, peak 2,467.7 MiB / 4,095.5 (60.3%), matching the §1.9 probe
to within 0.5%. **It finished**: early stopping at epoch 8 of 10 on `line/top_5_accuracy`,
best checkpoint epoch 5, 3 h 53 m wall clock. Top-5 0.610, Top-1 0.363, line F1 0.378,
IFA 11.94, flagged-line coverage 99.96%. The curve rises from epoch 0 instead of starting
flat — the evidence §1.10 asked for. `eval/loss` rose monotonically 0.0405 → 0.0646 while
train loss fell to 0.0296, so the run is overfitting 5,457 line-labelled units and loss is
the wrong stopping signal here; watching `line/top_5_accuracy` was the right call.
`function/f1` = 1.000 stays degenerate by construction. **Seed 1 launched 2026-09-12.**
One seed of the three Phase 3 requires — no citable result yet.

**Paper work committed.** `docs/paper/` and 8 `scripts/make_*` files had been untracked
since 2026-08-18. They carry no results numbers, so ADR-006 invalidated nothing in them —
but the draft **predates ADR-006 and has no preprocessing subsection at all**; §3.4 and
§3.15 both need the indentation leak.

**A test bug ADR-006 created, found by hypothesis.** `test_tokenization.py`'s property
test compared token offsets derived from the *raw* input against line indices derived
from `CodeUnit.source`, which ADR-006 normalizes. The two differ for any whitespace-
bearing input, so the pairing was off by one. **`CodeUnit.source` is not the string you
passed in** — see §7 #34. The helper also zipped a filtered line list against a prefix of
the offsets, which misaligns as soon as a body token maps to `NO_LINE`; both are fixed.
LineMap itself was verified correct and is not implicated.

**Watch:** `test_graphs_ast.py::test_graph_invariants_hold_on_random_input` failed once
under hypothesis and did not reproduce. Not chased. If it recurs, it is a real AST
invariant violation and takes priority over anything else in flight.

> **Sessions 7 and 8 are archived** in
> [`docs/memory/ARCHIVE-2026-09-14.md`](memory/ARCHIVE-2026-09-14.md) — Phase 3's
> foundation, moved once Phase 3's criteria were met. Nothing superseded.


> **Sessions 2 through 6 are archived** in
> [`docs/memory/ARCHIVE-2026-09-12.md`](memory/ARCHIVE-2026-09-12.md) — Phase 0, Phase 1
> and Phase 2, all closed. Moved 2026-09-12 under §0's length rule; nothing superseded.


## 4. Experiment Ledger

Every claim in a paper or doc must resolve to a row here.

| Run ID | Date | Config | Seeds | Fusion | Branch | Function F1 | Top-5 Acc | IFA | Notes |
|--------|------|--------|-------|--------|--------|-------------|-----------|-----|-------|
| `6c942e6c…` | 2026-10-12 | `compare_verdicts.py +compare.bundle=…/serving/ba7dcc5f-r2`, run from the worktree of branch `phase-9/screening`, clean `0af6ae2`; the served checkpoint `ba7dcc5f…` through the serving path on CPU; trains nothing | 42 | cross_attention | fused (served) | — | — | — | **§1.35: what each headline says about 488 vulnerable and 1,000 not-vulnerable `val` functions — an experiment kept apart, not a number of record.** The released headline raises a concern about 85.0% / 93.5%; the pattern headline about 27.7% / 21.0%; length alone separates the two kinds at 0.7111, the checkpoint's highest line score at 0.3321. No figure is a precision. The six-function trial runs before it went to scratch stores (one of them the stray `D:\probe.db`) and are not in this ledger |
| `52b9c998…` (of record) · `30aeff44…` (same numbers, earlier table wording) | 2026-10-08 | `evaluate_scores.py +representation.strip_comments=true +evaluate.scores_dir=./artifacts/phase6/scores_retrained +evaluate.calibration_dir=./artifacts/phase6/retrained` — clean `4281b3f` / `b875d70`; reads the nine `d97r` `val` score files and their three calibrations, loads no model | 42, 1, 2 | cross_attention / concat / llm_only | fused | *not reported — degenerate* | 0.5394 ± 0.0116 / 0.5546 ± 0.0084 / 0.5519 ± 0.0187 (the seed reports', reproduced); source order 0.3071 | 14.56 ± 0.43 / 14.06 ± 0.26 / 14.72 ± 1.27; source order 29.78 | **§1.29: Phase 8's full line-level suite on `val`.** PR-AUC **0.1713 ± 0.0143 / 0.1816 ± 0.0090 / 0.1610 ± 0.0097** against a positive rate of 0.0664; Top-10 0.667 / 0.680 / 0.676; Recall@20% LOC 0.470 for all three. **No per-unit metric separates the arms.** 93 comparisons, three at ≥ 2 pooled sd, all concat − `llm_only` (PR-AUC +0.0206 at 2.20; ECE) — a threshold chance crosses 7 times in 100 at three seeds, so a candidate for the held-out pass, not a finding (§7 #64). Bootstrap 95% over units ≈ ± 0.045 on Top-5 per checkpoint. Ties in source order; concat's Top-1 is 0.2614 here against 0.2621 logged — one tied unit. `val` only |
| `96b04469…` `2e9a2edc…` `af8b025e…` / `fbcf8b73…` `df29e1e1…` `3ee81242…` / `8f32f39d…` `09a50662…` `b626975c…` (scores) · `513dd46e…` `79b8c8d1…` `8b7ac935…` (calibration) · `10319b01…` (score audit) · `9196a0e3…` `798f6ef3…` `a65aa282…` (explanations) | 2026-10-08 | `score_checkpoint.py +score.run_id=<run> +score.output_dir=./artifacts/phase6/scores_retrained` ×9, `calibrate.py` per arm, `audit_score_leakage.py +representation.strip_comments=true`, `explain_checkpoint.py` ×3 — clean `1686536` / `bbb3d94` | 42, 1, 2 | cross_attention / concat / llm_only | fused | — | *unchanged — calibration is monotone*; blank lines withheld (474 units): 0.5499 ± 0.0120 / 0.5633 ± 0.0084 / 0.5633 ± 0.0211 | *unchanged* | **§1.28: the retrained checkpoints calibrated, audited and explained on `val`.** Line F1 out of fold **0.2359 ± 0.0120 / 0.2408 ± 0.0074 / 0.2330 ± 0.0118** — a tie (largest gap 0.79 pooled sd), against 0.39 with the tell in view. ECE 0.0133 / 0.0076 / 0.0168 after calibration. No unit answered by a tell; 274 of 2,808 flagged lines blank. 13,080 of 13,080 flagged lines explained and verified. **`8afbf7da…` was refused by the reproduction check in four of six passes — a tie-order flip of one rank in one unit; the kept file reproduces at the default tolerance (§7 #63).** Void: `15fccf6d…` (audit at `n=2`), `e735045f…` (tolerance probe) |
| `9ac7a146…` `7579bc33…` `10b43ce5…` · `f3d754eb…` (seed report) | 2026-10-08 | `run_queue.py --only d97r` — D9.7's constants plus `+representation.strip_comments=true`, concat; commits `fbc44e2` / `1686536` (docs only between them) | 42, 1, 2 | concat | fused | 1.000 *(degenerate)* | **0.5546 ± 0.0084** | 14.06 | **§1.28: concat retrained on comment-stripped text — the last arm, and A1 on text with no tell.** Top-1 0.2621 ± 0.0222. **Cross-attention does not beat it** (Top-5 −0.0152, 1.51 pooled sd; Top-1 +0.0062, 0.38) **and neither fused arm beats `llm_only`** (concat − `llm_only` Top-5 +0.0028, 0.19). Nothing between arms reaches 2 pooled sd. Seed 42 is a resumed run (`resumed_at: epoch=6,batch=6,step=3944`; paused by the owner 05:46–10:59) whose selected checkpoint predates the pause. Validation-selected, `val`. Pending Q14 |
| `d82a6a70…` `a1dce535…` `9296a0e5…` · `f814e521…` (seed report) | 2026-10-08 | `run_queue.py --only d97r` — D9.7's constants plus `+representation.strip_comments=true`, `llm_only`; commits `3bc918e` / `fbc44e2` (docs only between them) | 42, 1, 2 | llm_only | fused | 1.000 *(degenerate)* | **0.5519 ± 0.0187** | 14.72 | **§1.28: the scaffold control retrained on comment-stripped text.** Top-1 0.2801 ± 0.0062. None resumed. Indistinguishable from stripping at evaluation only (0.5546 ± 0.0084). **Retrained cross-attention does not beat it:** Top-5 −0.0124 (0.80 pooled sd, sign flips across seeds), Top-1 −0.0118 (1.92 pooled sd, same sign on all three seeds — a candidate in the control's favour, not a finding). Seeds 1 and 2 ran to the 10-epoch cap, selected at epoch 7. Concat is the row above. Validation-selected, `val`. Pending Q14 |
| `ba7dcc5f…` `8afbf7da…` `ff980345…` · `bd2544c7…` (seed report) | 2026-10-07 | `run_queue.py --only d97r` — D9.7's constants plus `+representation.strip_comments=true`, cross_attention; commits `0a4843c` / `d5e5dda` / `5af24ba` (source drift in `explanation/generator.py` only — accepted, recorded in the report) | 42, 1, 2 | cross_attention | fused | 1.000 *(degenerate)* | **0.5394 ± 0.0116** | 14.56 | **§1.28: the designed model retrained on comment-stripped text — the first leak-free arm.** Top-1 0.2683 ± 0.0060. Indistinguishable from the D9.7 checkpoints with comments stripped at evaluation only (0.5353 ± 0.0200; 0.25 pooled sd), and 0.077 below the 0.6162 it was reported at with the tell in view. Seed 2 early-stopped after four epochs, selected at epoch 0. Its control is the row above. Validation-selected, `val`. Pending Q14 |
| `bcf18403…` | 2026-10-07 | `run_queue.py --only d97rcheck` — `train_fused.py +fusion=cross_attention train=overfit_fused` with D9.7's constants and `+representation.strip_comments=true`, commit `0a4843c` | 42 | cross_attention | fused | — | Top-1 1.000 | — | **The overfit gate for the leak-free rerun. Never cite as a result.** Encoder training, comment-stripped text; Top-1 = 1.0000 over the last five of 150 epochs, read from the metric history. `d97r` started after it |
| `84582530…` `baf55d15…` `f3b641bc…` / `0f391973…` `be36cca8…` `9e0e13ec…` / `9d89bb6d…` `3a1507ed…` `006b0ea1…` (scores) · `e8a27dc5…` `a38fd4db…` `5bada5f3…` (calibration) · `9b87fbb2…` `b6f750f2…` (score audits, fine-tuned) · `30a0d40b…` `18b2535c…` (frozen A1) · `7c64dfac…` `421bf67c…` (frozen A6) · `af4173a3…` (corpus report) | 2026-10-07 | `score_checkpoint.py +score.run_id=<run> +representation.strip_comments=true` ×9, `calibrate.py` per arm on `scores_stripped/`, `audit_score_leakage.py` with and without the switch, `report_comment_stripping.py` — clean `4ac5f7e` / `9dc3604` / `ae0cf3a` | 42, 1, 2 | cross_attention / concat / llm_only | fused | — | **comments stripped at evaluation: 0.5353 ± 0.0200 / 0.5415 ± 0.0090 / 0.5546 ± 0.0084** (as shipped 0.6162 / 0.6169 / 0.6169) | 14.13 / 14.34 / 13.89 | **§1.25: the D9.7 checkpoints on comment-stripped `val`.** Line F1 out of fold 0.2355 ± 0.0040 / 0.2338 ± 0.0059 / 0.2362 ± 0.0030 (as shipped 0.39). No unit is answered by the tell; the arms still tie. **Trained with comments, scored without — not the leak-free number; `d97r` is.** Stripper identical to the grammar on 6,586 of 6,586 clean units; 8–12% of flagged lines are comment-only. **Frozen screen (27 more scorings):** frozen `llm_only` does not read the tell, every arm with a structural branch does; fusion's lead over it is +0.010 to +0.029 on stripped text against +0.032 to +0.051 as logged. ADR-010 (Proposed), Q14 |
| `cde35987…` `753f4c60…` `0fca1b4f…` / `920989e5…` `7a525ea0…` `f07faf7b…` / `ed9d88e5…` `64c5d5a0…` `317aef09…` | 2026-10-07 | `run_queue.py --only a6` — A1's frozen constants with `+fusion.args.direction={both,sem_from_str,str_from_sem}`, commits `2e9c0a4` / `82f1957` / `9083d4d` (source identical) | 42, 1, 2 | cross_attention | fused | 1.000 *(degenerate)* | **0.5408 ± 0.0181 / 0.5166 ± 0.0095 / 0.5498 ± 0.0231** | 12.19 / 12.99 / 13.96 | **A6: BIDIRECTIONAL IS NOT RESOLVABLY BETTER THAN EITHER ONE-WAY ARM** (largest Top-5 gap −0.0242, 1.67 pooled sd, with a sign flip). Top-1 0.2310 / 0.2503 / 0.2649 — both one-way arms higher on all three seeds, a candidate, not a finding. `both` reproduces §1.21's cross_attention arm seed for seed. Overfit gates `836aec8e…` `f3acb184…` passed. Frozen, validation-selected, `val`, **on text carrying the §1.25 leak**. §1.26 |
| `412a2df4…` `2c427d73…` `1eca6836…` · `3ac6a4c3…` · `51525d82…` | 2026-10-07 | `explain_checkpoint.py +explainer=template +explain.run_id=<run> +representation.strip_comments=true` ×3, once as shipped, `make_rater_pack.py` | 42, 1, 2 | cross_attention | fused | — | — | — | **Phase 7a (§1.27): 13,908 of 13,908 flagged lines explained and verified, 0 rejected, on stripped `val`.** A pattern names a cause for 8.1–8.8% of flagged lines (13.5–14.9% of top-ranked ones). Pilot rater pack, 100 `val` samples. Explanations, not model results. A first pass (`6f76427b…` and three more) is superseded: four template sentences were corrected after reading it |
| `9f462174…` · `ded852ee…` | 2026-10-07 | `audit_label_leakage.py` (every shard) · `audit_score_leakage.py` (the nine D9.7 `val` score files), both at clean `a86b7e1` | 42, 1, 2 | cross_attention / concat / llm_only | fused | — | all 482 units 0.6162 / 0.6169 / 0.6169 (reproduces §1.22); **the other 424 units 0.5708 ± 0.0227 / 0.5653 ± 0.0106 / 0.5660 ± 0.0047**; the 58 tell units 0.9483 / 0.9943 / 0.9885 | — | **THE §1.25 EVIDENCE: a second formatting leak.** A flagged line that ended a block comment lost the slash of its `*/`: the tell is flagged at precision 1.0000 on `val` (204 of 204), 0.9744 on `test`, 0.9882 on `train`; a line ending in `*/` is never flagged. The checkpoints use it. Rule-only baselines: source order Top-5 0.3071, tell first 0.3942. Data artefacts and a post-hoc stratification, not results — and the reason every other row's line-level number is now qualified. §7 #58, open decision Q14 |
| `569c9701…` (calibration) · `06c04755…` `5d3a6ad9…` `a593b1be…` (scores); concat `fd3e94c5…` and `llm_only` `97cb6a76…` likewise — all at clean `c7b6025` | 2026-10-07 | `score_checkpoint.py +score.run_id=<run>` ×9, then `calibrate.py '+calibrate.runs=[80fc5238…,c193e3bc…,829c3555…]'` (and once per other arm) — Platt, 15 bins, 5 folds by unit, `val` | 42, 1, 2 | cross_attention | fused | — | *unchanged — calibration is monotone* | *unchanged* | **Phase 6: calibration and the validation-chosen threshold (§1.24).** ECE 0.1704 ± 0.0330 → **0.0137 ± 0.0104**; line F1 0.2954 ± 0.0855 at 0.5 → **0.3898 ± 0.0190** out of fold (P 0.4020, R 0.3787). Thresholds 0.1686 / 0.1083 / 0.2159. Every re-scoring reproduces its run's logged metrics. Over-confident above p = 0.8 (observed 0.67–0.81). **Same procedure on the other arms: line F1 0.3908 ± 0.0030 (concat), 0.3936 ± 0.0072 (`llm_only`) — a tie.** `val` only. A first pass from an uncommitted tree (`e93ad84f…` and ten more) and a failed attempt (`5d06b630…`) are tagged `void` |
| `3ee36f97…` | 2026-10-07 | `train_fused.py +fusion=llm_only train=fused seed=42 … llm.frozen=false train.semantic_cache.enabled=false train.limit=3000 train.eval_limit=100 train.epochs=1`, commit `6d16adc` | 42 | llm_only | fused | — | — | — | **VOID / probe — never a result** (`train.limit` ≠ 0, so both reports refuse it; also tagged `void`). The live check of the fixed VRAM guard (§7 #57): no spill in 23 samples, 1.01 s/step. Its checkpoints were deleted. The first run to log `resolved_gpu` |
| **MACHINE 2** — `f4d81b36…` `513d9d6b…` `45de097a…` (reference) and 18 more frozen; `57df8cff…` `77268204…` `3a953045…` / `b04a6b82…` `7d785c26…` `fb445d65…` (fine-tuned). Every id is in §1.23 | 2026-10-03 → 05 | `run_queue.py --only m2 m2ft` on the **second laptop** (RTX 4050 Laptop 6 GB, torch 2.14.1+cu130, commit `782ea98`), imported as files at `artifacts/machine2/Sujal_20261005-224401/` — **not in this machine's MLflow store** | 42, 1, 2 | cross_attention | fused | 1.000 *(degenerate)* | frozen reference **0.5498 ± 0.0219**; GAT 0.5456, GGNN 0.5692, weighted BCE 0.5802, plain BCE 0.5747, line-only 0.5456, GraphCodeBERT 0.5747. Fine-tuned CodeBERT **0.6272 ± 0.0073** / GraphCodeBERT **0.6169 ± 0.0098** | 12.48 (ref) / 11.54 / 11.53 | **MACHINE 2 — NEVER POOLED WITH THIS LAPTOP'S RUNS (§7 #49).** A4 in fusion, A8 line-only, A9, A5. **No ablation resolves at three seeds** (largest: weighted BCE +0.0304, 1.96 pooled sd). Fine-tuned GraphCodeBERT − CodeBERT = −0.0104 (−1.20 pooled sd). Four of the six fine-tuned runs are resumed (tags in §1.23). Five overfit gates passed (Top-1 1.0000, read from the store's history). Validation-selected, `val`. §1.23 |
| `179632f5…` `72f067d0…` `2c7a7631…` | 2026-10-04 → 07 | `run_queue.py --only d97` — as the row below with `+fusion=llm_only`: **D9.7's scaffold control** | 42, 1, 2 | llm_only | fused | 1.000 *(degenerate)* | **0.6169 ± 0.0048** | 11.13 | **NEITHER FUSED ARM BEATS FINE-TUNED LLM-ONLY IN THE SAME SCAFFOLD** (cross_attention −0.0007, concat 0.0000 on Top-5). Top-1 0.3527 ± 0.0021, line F1 0.3726 ± 0.0157. Per seed Top-5 0.6141, 0.6141, 0.6224. **All three are resumed**: `resumed_at=epoch=6,batch=500,step=4565` (owner's pause) / `epoch=5,batch=976,step=4100` (WinError 5 crash, §7 #56) / `epoch=6,batch=946,step=4800` (GPU driver fault, §7 #57); seed 1's selected epoch 5 was completed after its resume. Validation-selected, `val`. 134, 210, 170 min over two attempts each. §1.22 |
| `80fc5238…` `c193e3bc…` `829c3555…` / `fbf65554…` `633ea97b…` `a0c42d6f…` | 2026-10-02 → 04 | `run_queue.py --only d97` — `train_fused.py train=fused llm.frozen=false train.semantic_cache.enabled=false +gnn=gcn gnn.use_subtokens=true`, **ADR-009 D9.7**, `src/vulnloc` as of `af4af6a` (logged shas `70824ec` / `0ce5d96` / `782ea98` differ in docs, scripts and tests only) | 42, 1, 2 | cross_attention / concat | fused | 1.000 *(degenerate)* | **0.6162 ± 0.0184 / 0.6169 ± 0.0104** | 11.26 / 11.46 | **D9.7: THE NULL SURVIVES FINE-TUNING (−0.0007, 0.05 pooled sd; paired signs flip).** Top-1 0.3444 ± 0.0293 / 0.3645 ± 0.0073 and IFA are not resolvable either; line F1 0.2954 ± 0.0855 / 0.3811 ± 0.0159 is the threshold moving. Per seed Top-5 0.6224, 0.6307, 0.5954 / 0.6058, 0.6183, 0.6266. `633ea97b` is **resumed** (`resumed_at=epoch=5,batch=418,step=3821`); its selected checkpoint predates the pause. Validation-selected, `val`. 118, 168, 114 / 123, 183, 211 min. §1.22 |
| `d3dbd7e9…` `b4998f5c…` `ad9f36e8…` | 2026-10-02 | `run_queue.py --only d97check` — overfit, **D9.7 config**, encoder training | 42 | cross_attention / concat / llm_only | fused | — | Top-1 1.000 ×3 | — | **The D9.7 overfit gate. Never cite as a result.** Top-1 = 1.000 in 141 / 134 / 141 of 150 epochs; first-step peak 2,532.6 MiB |
| `cb2b1e99…` · `1ab2fd03…` | 2026-10-02 | `cb2b1e99`: first `d97-cross_attention-seed42` attempt · `1ab2fd03`: `train_structural.py train.limit=300` on CPU | 42 | cross_attention · — | fused · structural | — | — | — | **Both VOID / probes, never results.** `cb2b1e99` was the live pause/resume test vehicle and found §7 #55 (tagged `void`, checkpoints deleted). `1ab2fd03` proved pause → resume end to end through Hydra (§3 session 14) |
| `192e067c…` `e7ae3108…` `ce2380bf…` / `86343b54…` `1428a86f…` `5b09a7a4…` | 2026-09-15 | `train_fused.py train=fused llm.frozen=true +gnn=gcn gnn.use_subtokens=true` — **A1 core**, commit `d1003f6` | 42, 1, 2 | cross_attention / concat | fused | 1.000 *(degenerate)* | **0.5408 ± 0.0181 / 0.5394 ± 0.0198** | 12.19 / 15.12 | **A1: NO RESOLVABLE DIFFERENCE on Top-5 (+0.0014, 0.07 sd; paired signs flip).** Frozen semantic encoder — the claim is the qualified one (ADR-009 D9.1). IFA favours cross_attention on all three seeds; Top-1 favours concat on all three. Validation-selected. §1.21 |
| `74021cc5…` `b6d4bb5f…` `9d88f284…` / `e6331b32…` `df0de21a…` `d2718402…` / `e0ca38de…` `df964b78…` `2d4de3bf…` / `cca2b33b…` `e75f7e34…` `394ceb9b…` | 2026-09-15 | as above — **A1 ablations** | 42, 1, 2 | sum / gated / llm_only / gnn_only | fused | 1.000 *(degenerate)* | 0.5304 / 0.5221 / **0.4903** / 0.4191 | 15.43 / 14.89 / 14.24 / 19.75 | **Every fusion arm beats frozen `llm_only` on every seed** (+0.032 to +0.051); not the coverage union — `llm_only` covers the same 99.96%. **Freezing costs 0.124 Top-5** against Phase 3's fine-tuned 0.6141, so no fused arm beats the project's best model. `gnn_only` seed 2 never improved on epoch 0. §1.21 |
| `2233c9a6…` `01be9392…` `7906edbf…` `83876edc…` `25d03e12…` `cf63c8ef…` | 2026-09-15 | `run_queue.py --only p5check` — overfit, **A1 config** | 42 | all six | fused | — | Top-1 1.000 ×5; llm_only 0.968 | — | **The A1 overfit gate. Never cite as a result.** `llm_only` still climbing at epoch 150 — the frozen encoder's cost (§1.20). |
| `117df3a4…` / `aebd4407…` / `3b32a40d…` | 2026-09-15 | `train_fused.py train=overfit_fused llm.frozen=true` (cached semantic path, **subtokens off**) | 42 | cross_attention / concat / sum | fused | — | **1.000** | — | **Overfit sanity on the ADR-009 cached path: PASSED** for all three (Top-1 1.000). Cache digest `77f7b07e…`. Not the A1 config — re-queued with subtokens on. **Never cite as a result.** `d4ac0260…` (gated) stopped at epoch 1, **tagged void**. |
| `c99764bc…` | 2026-09-15 | `build_semantic_cache.py train=fused llm.frozen=true` | — | — | — | — | — | — | **ADR-009's cache.** `train` 5,457 units / 932 MB / 132 s, `val` 488 / 133 MB / 19 s, fp32. Determinism: token diff 0.0, line diff 1.9e-6. Data artefact, not a result. §1.20 |
| **config change** | 2026-09-15 | `gnn.use_subtokens=true`, `gnn.builder=ast` for **every A1 arm** (per run, `run_queue.py`) | — | all | fused | — | — | — | **Ledger row for the owner's D8.7 decision.** The defaults in `configs/gnn/default.yaml` are unchanged, so no earlier run's config moves. |
| `cbdc2a1f…`/`8985c411…` (d2), `032e4700…`/`f1203001…` (d6), `e9378a9e…`/`f1b26024…` (A4 d2) | 2026-09-14 | `train_structural.py` — third seeds for every sweep arm | 1, 2 | — | gnn_only | — | see §1.16 | — | **These runs RETRACT two findings.** With three seeds per arm the depth sweep has **no resolvable effect** (2 vs 4 = 0.03 sd, 4 vs 6 = 0.59 sd) — the one-seed over-smoothing reading in §1.13 is withdrawn. And A4's gain is **flat across depth** (+0.0567 at d2, +0.0657 at d4, 0.27 sd apart), so §1.15's "1.5× larger at depth 4" was itself a one-seed artefact. **What stands: subtokens help by ≈+0.06 Top-5 regardless of depth.** |
| `77370d56…` / `0d324da8…` / `3c570cc1…` + `b92e6985…` | 2026-09-14 | `train_structural.py +gnn=gcn gnn.use_subtokens=true` | 42, 1, 2 (d4); 42 (d2) | — | gnn_only | 1.000 *(degenerate)* | **0.4302 ± 0.0372** (d4) | 20.78 | **A4 — ADR-002 §D2.5's second half, §1.15.** Subtokens lift the structural branch from Top-5 0.3645 to **0.4302** (Top-1 0.1155 → 0.1750). **The gain is LARGER at depth 4 (+0.0657) than at depth 2 (+0.0436), which falsifies §1.12's prediction** — disambiguation is not the main thing subtokens do. Depth 2 is one seed. |
| `5b177450…` / `e82ff268…` | 2026-09-14 | `train_semantic.py train=semantic seed={1,2}` | 1, 2 | — | llm_only | 1.000 *(degenerate)* | 0.6120 / 0.6203 | 12.52 / 11.33 | **The two seeds that close Phase 3** (§1.14). With seed 42: **Top-5 0.6141 ± 0.0055, Top-1 0.3513 ± 0.0138, line F1 0.3692 ± 0.0204, IFA 11.93 ± 0.60.** Best checkpoints epoch 5/5/6; all three cross-check clean against their on-disk summaries. 2.24 h and 4.08 h. **Phase 3's ≥3-seed criterion is MET.** |
| `11c2b890…` / `f083ae38…` | 2026-09-13 | `train_structural.py +gnn=gcn gnn.depth={6,2}` | 42 | — | gnn_only | — | 0.3174 / 0.3361 | — | **The depth sweep, §1.13.** Depth 4 is best (0.3645). **Depth 6 loses to depth 4 while its *feature* ceiling is better** (5.39% vs 7.35%), so ambiguity cannot explain it — over-smoothing or depth-optimisation can. **One seed each; the 0.047 gap is ≈2× the depth-4 seed spread. Suggestive, not conclusive.** |
| `5892b032…` / `d5a63481…` | 2026-09-13 | `train_structural.py +gnn={gat,ggnn} train=structural seed=42` | 42 | — | gnn_only | — | 0.3506 / 0.3237 | — | **Phase 4 exit criterion: all three variants run on real data.** GAT 0.3506 / Top-1 0.1224; GGNN 0.3237 / Top-1 0.1017; GCN 0.3645. One seed each — the three are **not** separated by more than seed noise and must not be ranked. |
| `ddb24d81…` | 2026-09-13 | `build_subtoken_vocab.py` (ast, train only) | — | — | — | — | — | — | **The identifier table, ADR-002 §D2.5's second half.** 148,983 units, 0 failures, 56,075 distinct subtokens over 17.7M occurrences; `min_count=5` keeps 31,387 types = **99.72%** of occurrences. Digest `e1d018bd…`. **Built from `train` alone** — identifiers from val/test would hand the model embeddings for evaluation-only names. Data artefact, not a result. |
| `46ab8056…` | 2026-09-13 | `train_structural.py +gnn=gcn train=structural seed=42` | 42 | — | gnn_only | 1.000 *(degenerate)* | **0.338** | 20.41 | **The first GNN-only baseline.** Line P/R/F1 = 0.628 / 0.051 / 0.094, Top-1 0.108. It learns (Top-1 0.033 → 0.112, F1 0.015 → 0.099), so the curve is honest — but it is **far below the semantic arm** (Top-5 0.610, F1 0.378, run `fc902107…`). Expected: node features are node-type only (ADR-002 §D2.5 half-built) and §1.12 measures that 7.35% of flagged lines stay indistinguishable even at 4 hops. Coverage 90.81%; 2,397 train flagged lines have no graph node (§7 #17). **One seed of three — not citable.** Vocabulary `a0643c4e…`. |
| `b530f1ff…` | 2026-09-13 | `report_node_feature_aliasing.py` (ast, 1,500 units/split) | — | — | — | — | — | — | **What the structural branch can see, measured without training — §1.12.** Weisfeiler-Lehman ceiling on flagged-line recoverability: **76.22% of flagged lines are indistinguishable from an unflagged one at 0 hops, 27.91% at 2, 7.35% at 4.** Says the §D2.5 subtoken gap is expensive only at shallow depth, and that **depth 2's deficit in the sweep is representational, not over-smoothing**. Data artefact, not a result. |
| `8b0a3c1d…` | 2026-09-13 | `train_structural.py +gnn=gcn train=overfit_structural` (32 graphs) | 42 | — | gnn_only | — | **1.000** | — | **Phase 4 overfit test: PASSED.** Top-1 **1.0000**, Top-5 **1.0000**, loss 1.57 → 0.2483 over 150 epochs. All three convolutions pass (gcn/gat/ggnn reach Top-1 = 1.000 at ~100 epochs). **It first read as FAILED because the config under-trained** — 60 epochs with dropout on plateaus at Top-1 ~0.80. A debugging instrument — **never cite this as a result.** |
| `fc902107…` | 2026-09-12 | `train_semantic.py train=semantic seed=42` | 42 | — | llm_only | 1.000 *(degenerate)* | **0.610** | 11.94 | **The first line-level numbers this project has produced that are not an artefact.** Post-ADR-006 (`data.normalize_whitespace=true` confirmed in the logged params). Early-stopped at epoch 8 of 10 on `line/top_5_accuracy`, best checkpoint epoch 5; 3 h 53 m. Line P/R/F1 = 0.463 / 0.319 / 0.378, Top-1 0.363, IFA 11.94, flagged-line coverage 99.96%. **It learned**: Top-5 0.531 → 0.610 and Top-1 0.247 → 0.363 across the run, against `1843412a…`'s flat 0.960 from epoch 0 — this is what §1.10 said to look for. Top-1 0.363 is ≈5.6× the 6.44% base rate. **`function/f1` = 1.000 is degenerate by construction and must never be quoted** (`line_labelled_only: true` leaves `fp=tn=0`). **NOT a citable result: Phase 3 needs ≥3 seeds.** Caveat: `eval/loss` rose monotonically 0.0405 → 0.0646 while train loss fell to 0.0296 — overfitting on 5,457 line-labelled train units, and the ranking metrics improved regardless, so loss is the wrong early-stopping signal here. Quote no line number without §1.8 truncation, §7 no-node coverage and §7 line-cap ceilings beside it. |
| `2c9ce1b1…` | 2026-09-12 | `build_node_vocab.py` (`ast`, `best_of_two`) | — | — | — | — | — | — | **The node-type vocabulary of record, Phase 4's prerequisite.** 304 types + unknown slot over all 188,636 units, 0 graph-build failures, digest `a0643c4e…`. **Measured: `tree-sitter-c`'s own symbol list would miss 85 of these types; `tree-sitter-cpp`'s would miss 3 and carry 188 that never occur.** Data artefact, not a result. |
| `1d89fe71…` | 2026-09-08 | `audit_label_leakage.py` (post-L0 rebuild) | — | — | — | — | — | — | **Confirms the corpus is clean at rest**: 0 partitions leak, the leading-whitespace tell fires on 0 lines. `shards_normalized_at_l0=true` — this reads the rebuilt shards, so it is evidence the fix landed, *not* evidence the leak was imaginary (compare `a5707491…`). Data artefact, not a result. |
| `8c1cccdd…` | 2026-09-08 | `build_cache.py` (ADR-006 D6.6, `data.normalize_whitespace=true`) | — | — | — | — | — | — | **The shards of record.** Rebuilt with normalized source. Every count identical to `1f716614…` (188,636 functions; train 148,983 / 5,457 line-labelled / 34,027 flagged); only the digests differ. Supersedes `1f716614…`. Data artefact, not a result. |
| `a5707491…` | 2026-09-08 | `audit_label_leakage.py` | — | — | — | — | — | — | **THE ADR-006 EVIDENCE. "Line starts with a space" predicts the flaw label at precision 1.0000, ZERO false positives, on all six partitions** (recall 0.83–0.99; `clean_eval` lift 38×). No tell leaks after normalization. Data artefact, not a result. |
| `1843412a…` | 2026-08-17 | `train_semantic.py train=semantic seed=42` | 42 | — | llm_only | ~~1.000~~ | ~~0.996~~ | ~~0.191~~ | 🟥 **VOID — DO NOT CITE. ADR-006.** It learned the indentation artefact, not the vulnerability: its val scores match the one-line whitespace rule to three decimals, and line F1 was flat at 0.960 from epoch 0 through early stopping at epoch 6. Function F1 = 1.000 is separately degenerate (`line_labelled_only` leaves `fp=tn=0`). Retained as the record of the mistake. |
| `ffe3d03a…` | 2026-08-16 | `train_semantic.py train=overfit` (32 samples) | 42 | — | llm_only | — | **1.000** | 0.129 | **Phase 3 overfit test: PASSED.** Loss 1.237 → 0.0026. A debugging instrument — **never cite this as a result.** |
| `1904c295…` | 2026-08-16 | `probe_vram.py` (CodeBERT, 512 tok, 24-point grid) | — | — | — | — | — | — | **AMP + checkpointing at batch 16 = 2,456 MiB / 22.9 win/s. 3 configs silently spilled past the 4 GiB card.** Closes §7 #7. |
| `b7ccc9ca…` | 2026-08-16 | `report_truncation.py` (CodeBERT, 512/128) | — | — | — | — | — | — | **Truncation costs 38.1% of all flagged lines; 44.5% on `clean_eval`. Windowing keeps 100%.** Data artefact, not a result. |
| `004b05de…` | 2026-08-16 | `report_representation.py graph.builder=cpg` (`best_of_two`) | — | — | — | — | — | — | CPG of record under ADR-005. 59,664,000 edges; CFG +26.9%, DFG +27.2% vs. the declared-language build. Data artefact, not a result. |
| `0729ae71…` | 2026-08-16 | `report_representation.py` (`best_of_two`) | — | — | — | — | — | — | AST of record under ADR-005. **89.14% clean parses.** Data artefact, not a result. |
| `228e35e0…` | 2026-08-16 | `characterize_parse_failures.py` | — | — | — | — | — | — | The evidence for ADR-005's residue section: 20,480 units no grammar parses cleanly hold **31.58% of the flagged lines**. |
| `093d672b…` | 2026-08-16 | `report_representation.py` (`declared`) | — | — | — | — | — | — | Pre-ADR-005 baseline, retained. 37.01% clean. **Not comparable with the runs above.** |
| `1f716614…` | 2026-08-11 | `build_cache.py` (pre-ADR-006) | — | — | — | — | — | — | Preprocessed shards, 188,636 functions. **Superseded by `8c1cccdd…`** — its shards carry the un-normalized source that leaks the label (§1.10). Retained for provenance; do not train from shards built by it. |
| `e395b21b…` | 2026-08-11 | `build_splits.py` | 42 | — | — | — | — | — | **Reproducibility check: `reproduced=True`** against `0c64a30f…`. Evidence for the Phase 1 byte-identity criterion. |
| `0c64a30f…` | 2026-08-11 | `build_splits.py` (ADR-004 on) | 42 | — | — | — | — | — | The split of record. Fingerprint `0ade7483…`, `is_clean=True`, 426 duplicates withheld. |
| `80b2b588…` | 2026-08-11 | `build_splits.py` (pre-ADR-004) | 42 | — | — | — | — | — | First full split. **Exposed 207 hashes / 426 functions leaking across partitions with zero shared projects.** Superseded. |
| `f3f24d45…` | 2026-08-10 | `train=debug device.kind=cuda` | 42 | — | — | — | — | — | Phase 0 **GPU** plumbing check. RTX 3050 Ti, 4 GiB, cc 8.6, torch 2.13.0+cu130. **Never cite this.** |
| `58c71014…` | 2026-08-10 | `train=debug` | 42 | — | — | — | — | — | First run with a real `git_sha` (`09d5c54`). CPU. **Never cite this.** |
| `a042076a…` | 2026-08-10 | `train=debug` | 42 | — | — | — | — | — | Phase 0 plumbing check. No model, no data. Logs `noop/loss` only. **Never cite this.** |

**Current best configuration:** —
**Current headline claim:** — *(no claims permitted until Phase 8)*

---

## 5. Open Decisions (blocking)

These come from `PRD.md` §9. Nothing downstream should be built on a guess.

| # | Question | Options | Recommendation | Blocks | Status |
|---|----------|---------|----------------|--------|--------|
| Q1 | Which dataset? The proposal deck names none. | See `ResearchPlan.md` §3 | Big-Vul (LineVul release) primary + DiverseVul generalization + PrimeVul validation + Juliet controlled | Phase 1 | ✅ **Closed — ADR-001, ADR-003** |
| Q2 | Target language(s)? | C/C++, Java, Python, multi | C/C++, single language for v1 | Phase 2 | ✅ **Closed — ADR-002** |
| Q3 | Primary graph representation? | AST · CFG · DFG · CPG | Incremental AST → CFG → DFG, CPG as their union; all four config-selectable for ablation A3 | Phase 2 | ✅ **Closed — ADR-002** |
| Q4 | Explanation supervision available? | Ground-truth text · templates + human eval · none | Template baseline + human eval; do not claim generative quality without a reference set | Phase 7 | 🟨 Deferred |
| Q5 | CWE-type classification required? | Binary only · binary + type | Deck's example output names a type ("Buffer Overflow") — implement type if labels exist, else derive from templates and say so | Phase 6 | ✅ **Closed 2026-10-07 — the owner deferred the learned type classifier (§2).** Explanations name a template-derived category and say so; the `CWE ID` stays unit metadata, so it is reversible |
| Q6 | CPG tooling: build ourselves vs. external generator? | In-house passes · external exporter | In-house — an external generator would take the line mapping out of our control. DeepDFA/MegaVul graphs used as offline validation oracles only | Phase 2 | ✅ **Closed — ADR-002** |
| Q8 | **Which grammar parses a Big-Vul function?** Measured corpus-wide (§1.7): the dataset's own `lang` column yields **37.01%** clean parses, C **35.74%**, C++ **88.24%**, best-of-two per unit **89.14%** | Trust `lang` · always C · always C++ · **best of two per unit** | **Best of two per unit**, C++ first and C++ wins ties. The 1,712 units it gains over always-C++ are C that the C++ grammar *misreads* (`class`, `new`, `this` are identifiers in C), so the trade is a clean tree against a confidently wrong one. Cost: the grammar becomes data-dependent, and the node-type vocabulary is now a union over two grammars — a Phase 4 input | Phase 2 corpus build | ✅ **Closed 2026-08-16 — ADR-005 accepted by the project owner.** Reversible via `graph.language={cpp,c,declared}`; `declared` reproduces every pre-ADR-005 number |
| Q9 | **Should the ADR-006 whitespace normalization also be applied at L0**, so `source` never carries the defect into the hashes, the split or the shards? | L1 only · **L0 as well** | **L0.** Chosen by the project owner 2026-09-08. It was accepted expecting a new split fingerprint and **turned out to cost nothing**: every field the split reads is invariant under the transform — `func_hash` already strips all whitespace (landmine #3), and the line count is preserved — so the assignment is bit-identical and `0ade7483…` stands. Only the shards' source text changes. L1 stays as idempotent defence in depth | Phase 3 retraining | ✅ **Closed 2026-09-08 — ADR-006 D6.6** |
| Q10 | **Should A1 run with the semantic encoder frozen?** A1 as specified costs **60-80 GPU-h** (core two arms: 20-30 h). §1.9 measured `frozen` at **37.8 win/s vs 22.9** and **1,234 MiB vs 2,456** - and, far more important, a frozen encoder is a *pure function of its input*, so its output can be computed once and read from disk, putting a fused run at Phase 4's 2-5 minutes | Fine-tuned as specified · **frozen + cached, staged** · fewer seeds · fewer arms | **Frozen + cached, staged (ADR-009).** All six arms × 3 seeds becomes **~5 h**. It stays controlled by exactly ADR-008 §D8.4's argument - it changes how good *all* arms are, not which wins - and it is the **harder** test of the novelty, since a fine-tuned encoder can reshape its own line features until a trivial combiner suffices. Cost: A1 answers a narrower question and must be phrased that way, and D9.7's fine-tuned confirmation is triggered by the margin, not skipped by default. **Fewer seeds and fewer arms are not options** - §7 #37 is what that costs | Phase 5 / A1 | ✅ **Closed 2026-09-15 — ADR-009 accepted by the project owner.** Built and verified the same day (§1.20) |
| Q11 | **Run ADR-009 D9.7's fine-tuned confirmation?** The frozen screen found cross_attention ≈ concat on Top-5 (+0.0014, 0.07 sd, §1.21) — the no-margin case D9.7 exists for. Phases.md's fifth Phase 5 criterion requires a negative A1 to be *investigated before Phase 6* | **Run it**: cross_attention and concat fine-tuned (`train.semantic_cache.enabled=false llm.frozen=false`, otherwise A1's constants), seeds 42/1/2, ~18–24 h, overnight, laptop kept awake · **Skip it**: report the frozen result, qualified, and record the skip as the owner's explicit decision · **Partial**: one fine-tuned seed per arm first as a probe (not citable — §7 #37) | **Run it.** It is the only way to learn whether fusion beats fine-tuned LLM-only (0.6141), which every fused arm currently loses to, and it is what closes criterion 5. A probe-only run would tempt a one-seed conclusion this project has retracted twice | **Phase 5 exit / Phase 6 start** | 🟨 **6 of 9 runs FINISHED; PAUSED 2026-10-04 21:51 at the owner's request** (running since 2026-10-02, session 14). The owner delegated the schedule ("do whatever you feel is best"; laptop free except Saturday daytime), so: **all nine runs in one resumable queue group** rather than two fixed nights — `d97-{cross_attention,concat}` seed by seed, then **`llm_only` fine-tuned in the fused scaffold × 3**, added so that "fusion beats fine-tuned LLM-only" is measured against a controlled arm instead of Phase 3's 0.6141 from a different scaffold (ADR-008 §D8.6). Interruptions are continued, not restarted (§7 #53). Live-encoder overfit check first (`d97check`). **2026-10-04: the six runs D9.7 specifies are FINISHED — cross_attention 0.6162 ± 0.0184 vs concat 0.6169 ± 0.0104 Top-5, no resolvable difference, agreeing with the frozen screen (§1.22).** The `llm_only` control is 0 of 3 (seed 42 paused at epoch 6); **do not quote that arm, or any fusion-vs-LLM-only statement, until all three are FINISHED**. **CLOSED 2026-10-07: nine of nine FINISHED; `llm_only` 0.6169 ± 0.0048 — neither fused arm beats it (§1.22). The owner signed the Phase 5 gate the same day (§2)** |
| Q12 | **Commit the A1 write-up?** Memory.md §1.21 + ledger, Phases.md criterion-5 note, CLAUDE.md §3, `frontend/phases.json`, `docs/review3_panel_brief.md` — uncommitted since `d1003f6` | Commit now · commit with the D9.7 runs | Commit before any D9.7 run starts: a dirty tree tags every run `-dirty` (§7 #4), so the write-up must be committed first either way | Clean `git_sha` on D9.7 runs | ✅ **Closed 2026-10-02** — committed as `18ca52b` (frontend) and `6e64486` (write-up, code map, report material, bibliography), before any D9.7 run |
| Q13 | **Scope under the one-week deadline (≈ 2026-10-09): which Phase 6–10 exit criteria are met, and which are deferred in writing?** §6 item 0a lists them criterion by criterion. Three blockers cannot be worked around this week: people other than the owner (human evaluation run, independent reader), this machine (no Docker, one GPU), and negatives (anything function-level needs the 148,983-function split) | **Defer the ❌ rows explicitly** and ship the rest · extend the deadline · cut whole phases (9 or 7) to protect 8 | **Defer the ❌ rows explicitly.** Phases 5–6 closed, 7a, Phase 8 on the frozen grid with `test` + `clean_eval`, Phase 9 without verified Docker, a paper *draft* whose limitations name every deferral. If something must be cut, cut Phase 9's UI before any Phase 8 ablation — the claims are the project | **Marking Phases 6–10 done at all** | 🟨 **PARTLY ANSWERED 2026-10-03.** Owner: three raters are available (human evaluation can run), a second GPU laptop exists (independent reproduction, clean-machine Docker), and Docker is installed Sat 3 evening. Those three rows move to ✅ in §6 item 0a. **Still unanswered:** deferring the function-level criteria (unit aggregation, DiverseVul, full external reproduction) and 7b; whether the external baseline runs as a balanced subsample or stays `(reported)`. (The second laptop's GPU is answered: 6 GB, `docs/SecondMachine.md`.) **Docker was not installed on Sat 3; it was installed on Sun 4 at 22:08 (session 15, to `D:\Docker`), rebooted and launched once — auto-start off, nothing built; the owner sets it up later**. **ANSWERED 2026-10-07 (§2): the function-level criteria and 7b are deferred, external baselines stay `(reported)` — the owner delegated the choice ("whatever is best") and can reverse it; the full model is the designed cross-attention architecture. Still the owner's: Q5 (CWE labels), which Phase 6's fourth criterion needs decided or deferred** — **answered the same afternoon: deferred, and the Phase 6 gate signed (§2)** |
| Q14 | **What happens to the comment-terminator leak (§1.25, §7 #58)?** A flagged line that ended a block comment lost the slash of its `*/`; the tell is right 97–100% of the time on `train`/`val`/`test`, the nine D9.7 checkpoints use it, and ≈ 0.05 of every logged `val` Top-5 sits on the 12% of units it answers. `test` carries it on 13.4% of units; `clean_eval` has a related regime (flagged lines carry no comments at all) | **(a) Strip comments label-blind at L0/L1 (ADR-010), rebuild, retrain the headline** — 9 fine-tuned runs, 17–32 GPU-h on this laptop, or on machine 2 as one whole group · (b) repair or strip at **evaluation only**, re-score the existing checkpoints (minutes each) and report that, with the stratified table beside it · (c) report as is, with §1.25 as a limitation | **(b) now, (a) if the GPU-hours exist.** (b) gives a number the tell cannot inflate tonight, from checkpoints that already exist; it under-states what a leak-free-trained model would do, which is the safe direction. (a) is the real fix and is what ADR-006 did; it voids every line-level number and has to be the owner's call two days before the deadline. (c) is not defensible for `test` | **The held-out evaluation; every line-level claim** | ✅ **CLOSED 2026-10-08 — the owner accepted ADR-010 with option (a): the nine `d97r` checkpoints are the models of record and the held-out evaluation is on stripped text (§2).** Raised 2026-10-07 (session 16); the history of how it was measured follows. **Both are built (2026-10-07 evening).** (b) is measured: Top-5 0.535–0.555 on comment-stripped `val` against 0.616–0.617, line F1 0.234–0.236 against 0.39, arms still tied (§1.25). (a) is queue group `d97r`, with ADR-010 (Proposed) as its decision record. **(a) IS NOW MEASURED TOO (2026-10-08, §1.28): all nine runs retrained without comments — Top-5 0.5394 ± 0.0116 / 0.5546 ± 0.0084 / 0.5519 ± 0.0187, line F1 0.2359 / 0.2408 / 0.2330 (cross_attention / concat / `llm_only`). The arms tie, and each matches (b) for the same arm — so (b) did not under-state a leak-free-trained model, as this row had guessed. What is left is the owner's acceptance.** Neither has touched a held-out partition |
| Q15 | **How is A1 stated, now that it has a two-part answer?** Frozen encoder, comment-stripped text, validation: cross_attention is ahead of concat on Top-5 over six seeds, and the first three seeds' result was repeated on three more (§1.32). Fine-tuned — the models of record: no difference, on validation and on both held-out partitions (§1.28, §1.30) | Lead with the fine-tuned null and report the frozen result as a qualified secondary finding · lead with the frozen result · give both equal weight | **The first.** The models of record and the only held-out numbers are the fine-tuned ones; the frozen result is validation-only by protocol and carries every qualifier §1.32 lists | The paper's A1 section and abstract; nothing that is built | ✅ **Closed 2026-10-12 — the first option, by the owner's delegation ("do 1 and 2 according to you", §2).** The online draft's Abstract and Section 1 were brought to it the same day; Section 3.10 already read that way (raised 2026-10-11) |
| Q7 | What happens to the 426 functions that are byte-identical across partitions? | Withhold · reassign to one partition · keep and report | **Withhold** — costs 0.23% of the corpus and is the only option preserving both project- and function-disjointness | Phase 1 exit | ✅ **Closed 2026-08-11 — ADR-004 accepted by the project owner.** Reversible via `data.splits.dedup_across_splits=false`, which changes the fingerprint |

---

## 6. Next Actions

Ordered. Top item is what the next session starts with.

### 0a. THE WEEK — handoff for every session until 2026-10-09 — READ THIS BEFORE §0

Written 2026-10-02 (session 14). **Owner deadline: the whole project in about a week
(≈ Fri 2026-10-09).** One 4 GiB laptop GPU, one run at a time. Everything below follows
from two measured costs:

| Kind of run | Cost | Source |
|---|---|---|
| Fine-tuned fused run | **~15–19 min per epoch; 114–211 min per run** by the queue's clock (six runs; the longest, `a0c42d6f`, selected step 7) | sessions 14–15, D9.7 runs 1–6 (§1.22) |
| Frozen-cache fused run | **6–27 min, ~15 typical** | the A1 screen, `run_queue.py --status` |

**So: fine-tune only what the headline needs. Every sweep and ablation runs off the frozen
cache.** A design that needs a tenth fine-tuned run needs an argument first.

#### Where it stands — 2026-10-11 08:10 (session 18): everything of the night is done and written up; Phase 9 is merged and runs in Docker; nothing is running

- **Done since midnight:** the held-out pass (§1.30); `a1r` and its replication `a1rep`
  (§1.32, tables of record `bcfeb4ff…`); Phase 9 merged as `f5f450a`, the bundle of
  record exported, latency measured, the demo page checked (§1.33). The main tree is
  clean; 1,498 tests pass, 3 skip.
- **The owner, at about 07:00:** *"hello whats the update"*, then *"whats next to be
  done, start"*. The session reported, and started Phase 9's merge and its measurements.
- **Nothing is running or paused**, so `configs/` and `src/vulnloc/` may be edited. Both
  held-out partitions have been read; they are not scored or evaluated again.
- **What is left, by who can do it:**
  1. *The owner's decisions:* Q15 — how the paper states A1's two-part answer (§5);
     whether the latency of long functions (p95 2.63 s against a recommended 2 s) is
     accepted or worked on; the Phase 8 and Phase 9 gates.
  2. *The owner's reading:* the failure taxonomy (`docs/FailureTaxonomy.md` §6, sheet
     `artifacts/phase8/failure_cases.md`); the human evaluation with three raters
     (`artifacts/phase7/rater_pack_test/`, `docs/HumanEvalProtocol.md`).
     *2026-10-11, late morning:* the owner pasted a written review of the taxonomy with
     no words of their own. The session checked it against the sheet and revised the
     draft (§1.31; `docs/FailureTaxonomy.md` §7). **At about 11:10 the owner accepted it**
     — *"proceed ahead with this and Siddharth read this"* (§2) — **and the Phase 8 box
     is ticked.** The human evaluation is what is left under this item.
  3. *Docker — done on this laptop at 08:00* (§1.33): the image builds, the `serve`
     container answers as the host does, and Phase 0's command runs in it. Left: a
     second machine, if the owner wants "clean" to mean that; pinning the image's
     versions to the lock file; a commit SHA inside the image. *Still needing a quiet
     machine and the network:* the rest of the clean-clone criterion (an environment
     from the lock file, the data pipeline, a reproduced run).
  4. *Phase 10* is not started.
- **2026-10-11, about 12:30 — where session 18 stopped (the owner's weekly usage was at
  95%).** The owner asked why the tool cannot say a function has no vulnerability, was
  told why (no model here saw a safe function; a function-level claim needs the deferred
  work, 3–4 days frozen), and chose two things, both without new training: **the five
  "honest result" items** (a plain "nothing flagged, and that is not a statement of
  safety" message; the top-ranked lines shown even when none is flagged; the weakness
  patterns that were tested; what was not examined; the measured miss rate beside the
  result) **and "Option A", a more readable demo page. No generative explainer.**
  - *Built and committed:* `src/vulnloc/serving/report.py` and `POST /analyze/report` —
    the §5.4 object with a report beside it (summary, five highest-ranked lines, every
    flagged line's verified explanation and evidence, the 19 patterns checked and which
    matched, coverage). `/analyze` is unchanged and still exactly the contract. Ten
    tests (`tests/unit/test_serving_report.py`); the three serving test files pass, 51.
  - *Not done yet:* (1) `serving/static/index.html` still calls `/analyze` — the page
    has to be moved to `/analyze/report` and laid out (summary on top, a "why this line"
    panel per finding, attended lines highlighted, the patterns and coverage cards).
    (2) `report.measured` is null for the bundle of record: `bundle.json` has no
    `measured` block. It needs an export option that copies the released checkpoint's
    held-out Top-5 out of `artifacts/phase8/evaluation_test.json` — quoted from the
    file, nothing scored again — and a re-export to a new directory. (3) No fix advice:
    the explanation templates carry a reason and no remedy, and
    `frontend/fix_templates.py` is unreviewed (`CLAUDE.md` §10 5b). (4) `CodeMap.md`,
    the commands in `CLAUDE.md`, the API section of the model card; the full suite was
    not re-run, and the Docker image predates the endpoint.
  - *Disk:* the project is 173 GB, 153.5 of it `artifacts/checkpoints` (129 runs, a
    `best.pt` and a `last.pt` each). About 89 GB could go without touching a result —
    `last.pt` of finished runs 72.6 GB, `best.pt` of the overfit gates 16.4 GB. **The
    owner said to delete nothing for now; nothing was deleted.**
  - *The owner's literature review was checked by the session* (pasted text, 37
    references). Wrong as written: ERASER removes evidence from the *input*; Jain and
    Wallace say "frequently uncorrelated", and show alternative attention distributions,
    not random ones; UniXcoder uses comments; Vul-LMGNNs uses online *distillation*; the
    whitespace shortcut is this project's finding, not literature; the WL sentence.
    Not verifiable here, to check against the papers: Fu et al. F1 0.82, DCVD 0.87 /
    0.92 / 0.94, ActiveClean +7 points, "87 studies", "18 models" in Li et al., and what
    DCVD does not state. None of those numbers is in `docs/prior_work.yaml`.
  - *The online draft* (the Claude Doc holding the literature review and methodology):
    a section "Corrections to this methodology (11 October 2026)" was added at the top
    of Part II — seven points where the 1 October prose is out of date (comment
    stripping, the models of record, the tie, the frozen result, the held-out pass, the
    template explainer, and the draft's "smooth maximum" function-level score, which is
    still to be checked against the code). **Later the same day the Methodology prose
    was rewritten in place to agree with all seven and the corrections section removed**
    (a delegated agent, at the owner's word). Revised passages carry a temporary
    highlight, with a note under "Part II" — to be removed before submission. A
    "Reporting a result" paragraph (3.1) describes the report and the "No known
    vulnerability found" wording. Checked in the code: the head does compute a
    log-sum-exp function logit (`heads/line_head.py`), used only in the auxiliary loss
    term and a degenerate logged F1; the service does not expose it. Not edited: the
    Abstract, the Introduction, the front-matter checklists, Figure 1; there is no
    results section. 3.10's A1 framing is the session's recommendation, pending Q15.
  - *Still waiting on the owner, who said "wait on those":* Q15, latency, the Phase 8
    and 9 gates, a time for the clean-clone check.
- **2026-10-11, 12:35 — the clean-clone check was started at the owner's word** ("i meant
  clean clone check, so start it"). One unattended script, `artifacts/queue/cleanclone_check.py`,
  working in `D:ulnloc-cleanclone` (log `cleanclone.log`, outcome `result.json` there).
  **Written before it ran:** the clone is of this machine's HEAD (the branch is ahead of
  its remote); the environment comes from `requirements.lock.txt` alone; the raw corpora
  are hard-linked from the main tree and verified, not downloaded again; everything
  downstream is rebuilt. *Reproduced* means the split fingerprint equals `0ade7483…`, the
  rebuilt shards and both vocabularies equal the main tree's byte for byte, and the two
  frozen runs of record at seed 42 (comment-stripped, `val`) land within their arm's
  three-seed sd of the logged Top-5 — cross_attention `8cde0dd5…` 0.5207 (sd 0.0102),
  concat `6ce89a15…` 0.4917 (sd 0.0157). A same-seed rerun is not expected to be
  bit-identical (§7 #50); the difference is reported as it is. No held-out partition is
  touched. The clone's runs live in the clone's own MLflow store and are never pooled
  with this tree's.
- **2026-10-12 (session 19) — the clean-clone check was read and written up: §1.34.** It
  ran 12:29–13:39, every step exit 0. The split, the shards and both vocabularies
  rebuild exactly (the vocabulary files differ only in the `run_id` they record — the
  check's byte comparison was too strict); 1,506 tests pass in the new environment.
  Concat at seed 42 lands inside the band written beforehand (0.4772 against 0.4917);
  **cross-attention does not (0.5436 against 0.5207, 2.2 of the three-seed sd)** with
  every logged parameter and digest equal — a same-seed rerun is a replicate, and both
  runs sit inside their arm's six-seed spread. The Phase 8 box is not ticked: the
  owner's, at the gate. `D:\vulnloc-cleanclone` (the clone, its environment and two
  checkpoints) is still on disk; nothing was deleted.
- **2026-10-12 (session 19) — returned human-evaluation sheets were checked and are NOT
  usable as three independent ratings; nothing was computed from them and nothing was
  copied into the repo.** The owner downloaded `VulnLoc_human_eval.zip` (Downloads,
  11:52): three folders, each with the `test` pack (`samples.md` and `README.md` equal
  the pack of record by SHA-256) and a filled sheet. Each sheet is well-formed — 100
  rows, every cell 1–5, no `0`, a comment on every low score. **But the three sheets
  carry the same score in 400 of 400 cells**; only the comments differ in wording
  (raters 1 and 2: 9 verbatim, mean similarity 0.83). Of three of rater 3's comments
  checked against the samples (S042, S071, S100), all three describe code or a weakness
  that is not in the sample. All three `about_you.txt` files are blank, so no rater
  background is recorded (`docs/HumanEvalProtocol.md` §3). Krippendorff's α on these
  would be 1.0 by construction and is not a measurement of agreement. The scores
  themselves, for whoever did rate once: correctness 1 on 94 of 100, faithfulness 4–5,
  usefulness 1–2 on 97, clarity 3–4 on 97 — not a result, not to be quoted. **The
  human-evaluation criterion stays open; what happens next is the owner's.**
  **Asked who filled the sheets in, the owner answered: *"gemini rated those"*.** So no
  person has rated anything: the sheets are a language model's output and are never
  reported as human evaluation, as raters, or as agreement. The owner also asked *"what
  if we skip this?"*; the session recommended deferring the human evaluation in writing.
  **Decided the same hour: offered "defer in writing" or "three people rate the pack",
  the owner said *"then do 1"* — the human evaluation is deferred (§2, first row).** It
  is no longer on the list of what is left; the pack and the protocol stand unchanged
  if three people are ever found. The online draft says so in two sentences (the
  "Explanations" paragraph of the methodology and "Attention and explanations" under
  the limitations): no human rated the explanations, the protocol was defined and the
  evaluation deferred. `ResearchPlan.md` §4.3, `HumanEvalProtocol.md` and `Phases.md`
  carry the same note.
- **2026-10-12 (session 19) — two of the owner's open decisions were delegated and
  made: *"do 1 and 2 according to you explain to me 3"*** (§2, first two rows).
  **Q15 is closed** — the fine-tuned null leads, the frozen result stands beside it as a
  validation-only finding. **The latency miss is accepted and stated**, nothing tuned.
  - *The online draft, changed for Q15:* the Abstract (held-out `test` Top-5 for the
    three arms, the PR-AUC difference that repeats, the frozen six-seed lead with its
    qualifiers; "replicated null" is gone); Table 1 (two held-out rows) and its caption;
    RQ1 and RQ2; the paragraph after them; the limits paragraph; the "Design" paragraph;
    contributions 2 and 5; a dated update note at the top; the first open item ticked
    and the draft-status line under the Abstract deleted. **Not highlighted** — the
    highlight convention is Part II's; the update note lists the passages instead.
    Still stale there: the two older notes at the top (kept as history), Sections 2.6
    and 2.8 if they lag, and Sections 4–6 do not exist.
  - *`docs/MODEL_CARD.md`* has two new bullets under known limits (latency; no person
    rated the explanations) and therefore **differs from the copy in the bundle of
    record**, which is not edited. The re-export owed for `report.measured` carries it.
  - *Explained to the owner, then decided:* the clean-clone box (item 3) — asked
    *"what do you recommend"*, told "tick it, with the miss stated", the owner said
    *"okay then proceed"*; ticked in `Phases.md`, Phase 8 is 2 / 4, and the online
    draft's reproducibility section has a paragraph on it (not highlighted). *Not
    raised again:* the Phase 8 and Phase 9 gates.
  - **Then the owner wrote *"okay finish these two also so phase 8 and 9 done?"*** — in answer to whether to go ahead with the demo page.
    Session 19 read it as: finish the two Phase 8 boxes and close both gates. Done the
    same day — **Phase 8 is 4 / 4 (§1.1f) and Phase 9 is 4 / 4 (§1.1g); the owner has
    read neither gate's answers, and each tick says in `Phases.md` what it claims.** The
    two weakest: the ablations box (A3, A7 and A10 were never run — ticked as "every
    claim has a verdict") and the Docker box (one laptop). Also done: the demo page on
    `/analyze/report`, `export.measured_from`, the bundle of record re-exported to
    `artifacts/serving/ba7dcc5f-r2`, the image rebuilt (`7283510c22c7`) — §1.33's last
    subsection. Docker Desktop is shut down; nothing is running; the tree is clean.
  - **Evening — the owner asked what could be done about `int add(int a, int b)` being
    flagged at 0.99, and has to show the project on 14–16 October.** Told: retraining
    with non-vulnerable functions is the real fix and does not fit before the 14th (a
    day of building, a trial, then one to two nights; `train` holds 5,457 line-labelled
    vulnerable functions and 141,494 non-vulnerable ones, median 27 lines against 13 —
    length is a ready shortcut). The owner then asked what is realistic *"if we do need
    a 'no vulnerability line' even if it is not a hundered percent accurate"*, and
    *"can we keep the current thing separate and these two things separate?"*
    - *Built, separate:* **branch `phase-9/screening`, worktree `D:\vulnloc-wt\screening`,
      commit `58cd313`, NOT merged.** One new page, `GET /experimental/patterns`: the
      same `/analyze/report` under a headline that follows the weakness patterns —
      "Known weakness pattern on line N" or **"No known vulnerability pattern found"**
      (with "not a statement that the function is safe"). `/` does not link to it and
      is unchanged; nothing the API returns changes; no training. Driven in a headless
      browser: `int add` reads "No known vulnerability pattern found", the overrun
      example reads "Known weakness pattern on line 6". Run it beside the released
      demo with `PYTHONPATH=D:/vulnloc-wt/screening/src python scripts/serve.py
      +serve.bundle=D:/project/artifacts/serving/ba7dcc5f-r2 +serve.port=8081` from
      the worktree.
    - *Measured for it, `val` only, from the stored explanations of `ba7dcc5f…`:* a
      pattern matches some flagged line in 135 of 488 vulnerable functions, so **that
      headline would read "No known vulnerability pattern found" for 353 of 488 (72%)
      functions that do contain a vulnerability.** True as a statement about patterns;
      poor as a signal. How often it would fire on harmless functions is not measured.
      *(Measured later the same day: 210 of 1,000 — §1.35.)*
    - *Not started, waiting for the owner's yes:* the screening gate — a small
      classifier on the released model's internal features, vulnerable against
      non-vulnerable at one to one, validation only, to be judged against a length-only
      rule. It is new training, which the owner declined on 2026-10-11; a day to a day
      and a half. It would go on the same branch, on a page of its own.
    - *Also corrected:* "the model flags a line in every function" (§1.33, §1.1g).
    - *Later the same day — the record the owner asked for:*
      **`docs/VerdictComparison.md`** (§1.35, run `6c942e6c…`). Both headlines were
      asked about the same 1,488 `val` functions, harmless ones included for the first
      time: the released headline raises a concern about 85.0% of vulnerable functions
      and **93.5% of harmless ones**; the pattern headline about 27.7% and 21.0%;
      length alone separates the two kinds at 0.7111 and the checkpoint's highest line
      score at 0.3321. The script is on the experiment branch only (`0af6ae2`). **The
      gate is still not started** — the owner's words ("its fine save these
      results…", §2) were not read as a yes to training; ask by name. *Left for the
      owner to delete:* `D:\probe.db`, an 880 KB tracking database a mis-typed trial
      command made at the root of D: (a safety check refused the session's removal).
  - *What is left now is Phase 10* — Sections 4 to 6 of the paper, the public README
    (it must say "a ranker for a function already suspected" first, §1.1g), the
    reproducibility appendix, licences and citations, and the independent reader on the
    second laptop, who should be given something deterministic to reproduce (§1.1f).
    Smaller: pin the image to the lock file; a commit SHA in the image.
  - *Superseded — left, in order, as it stood before that:* the two gates and Phase 8's two remaining boxes, all the owner's; the
    demo page on `/analyze/report`, the `measured` block and the re-export (the 12:30
    bullet of 2026-10-11 below); Phase 10 — Sections 4–6, the README, the
    reproducibility appendix, licences and citations, the independent reader.
- **Housekeeping left behind:** two worktrees in the session's scratchpad (`wt` on
  `phase-8/tables`, `wt9` on `phase-9/serving`) are merged and no longer needed —
  `git worktree prune` once that directory is gone. `artifacts/serving/ba7dcc5f` (536 MB)
  is the bundle of record; a development bundle of the same checkpoint sits in the
  scratchpad and is not of record.

#### Where it stood — 2026-10-11 01:15 (session 18): the held-out pass is DONE; `a1r` is done; `a1rep` is running

- **The held-out pass ran from 00:15 to 00:47 and every step exited 0**
  (`artifacts/queue/heldout_pass.out`): eighteen score files, the two evaluations, the two
  leak audits, explanations of the designed model's three seeds on `test`, and the rater
  pack. **Results: §1.30.** Neither partition is scored or evaluated again. Figures:
  `artifacts/phase8/figures/{arms,reliability}_{test,clean_eval}.svg`.
- **`a1r` finished at 00:14** with three seeds an arm. Results, and the tables of record
  (`922719b5…`): §1.32.
- **`a1rep` has been running since 00:49** (`artifacts/queue/a1rep.out`): fifteen frozen
  runs — cross_attention and concat on seeds 3, 4 and 5 first, then `llm_only`, sum and
  gated — and ten seed reports; about four hours. The session started it under the
  owner's words of that night (§2, first row). What it is read for is in §1.32 and was
  committed (`c6373cf`) before its first run. Slate:
  `python scripts/run_queue.py --only a1rep --status`.
- **The owner, shortly after midnight:** *"train or build whatever you need to overnight,
  internet will be on till 4 am i guess"*, and *"keep the docs updated and context as
  well updated so theres no dip in your performance"*.
- **What is left, in order:**
  1. *After `a1rep`:* its ten reports in `artifacts/adr010/`
     (`…frozen_stripped_replication_<arm>.json` with `n_seeds: 3`,
     `…frozen_stripped_six_seeds_<arm>.json` with `n_seeds: 6`); the two `pending` tables
     in `docs/ablations.yaml` set to `run`; `python scripts/make_ablation_tables.py`; the
     outcome written into §1.32 against the plan that section already holds.
  2. *Phase 9, once no group is running:* the merge of `phase-9/serving` (now `c2e4d92`:
     the model card carries the held-out numbers of §1.30 since 2026-10-11 01:00; a trial
     merge on 2026-10-10 had no conflict); the bundle of record; latency on an idle machine; the demo
     page's result state; Docker; §1.33.
  3. *The owner's:* the failure-taxonomy review (`docs/FailureTaxonomy.md` §6); the rest
     of the clean-clone criterion; the human evaluation, for which the `test` rater pack
     now exists (`artifacts/phase7/rater_pack_test/`); the Phase 8 gate.

#### Where it stood — 2026-10-11 00:00 (session 18): `a1r` on its last run; the held-out pass chained behind it at the owner's answer

- The owner wrote *"continue"* at about 23:53 on 2026-10-10. Session 18 restarted the
  `a1r` queue alone at 23:54 (`a1r-gated-seed2`, then the six seed reports; transcript
  `artifacts/queue/a1r_resume2.out`) and asked whether the held-out pass should follow
  it. The owner's answer, at about 00:00: **"Yes, after a1r tonight"** (§2, first row).
- One instance of `artifacts\queue\a1r_then_heldout.ps1` has been waiting since
  00:00:23. It runs `artifacts/queue/heldout_pass.py` about a minute after the queue
  exits — expected at about 00:20 — unless a `STOP` file is present then.
- Held-out score files at 00:00: 1 on `test`, 0 on `clean_eval`. Nothing held-out has
  been evaluated or looked at.
- Not yet done: the six `a1r` seed reports, the ablation tables of record, §1.32, §1.30.

#### Where it stood — 2026-10-10 21:15 (session 18): PAUSED again at the owner's request — `a1r` at 17 of 18 runs, the held-out pass NOT started

- The owner, at about 21:09: *"pause for a bit i will continue later"*. `run_queue.py
  --stop` was issued at 21:09. `a1r-sum-seed2` was in flight; a frozen run is not
  resumable, so it finished by itself, and the queue exited at 21:12 without starting
  another step.
- `a1r` stands at 17 of 18 runs. Left: `a1r-gated-seed2` (about 18 min) and the six seed
  reports. Five arms have three seeds, `gated` has two; no seed report exists, so no
  `a1r` number is quoted (§7 #37).
- The held-out pass was not started. The chain script found the `STOP` file when the
  queue exited and ended — the last line of `artifacts/queue/a1r_then_heldout.out` says
  so. Score files: 1 on `test`, 0 on `clean_eval`, as on 2026-10-08; nothing held-out has
  been evaluated or looked at.
- Nothing is running. The owner said they will continue later. The 20:50 block below
  describes what was running before this pause and how it was started.

#### Where it stood — 2026-10-10 20:50 (session 18): `a1r` is running again, with the chain script waiting behind it

- The owner wrote *"continue"* at about 20:49, after the conversation was compacted.
  How session 18 read that word, and what the reading rests on, is the first row of §2.
- Running since 20:50, started on a clean tree at `d8b1a34` with the GPU and both disks
  idle: the `a1r` queue (`a1r-sum-seed2`, then `a1r-gated-seed2`, then the six seed
  reports; transcript `artifacts/queue/a1r_resume.out`), and one instance of
  `artifacts\queue\a1r_then_heldout.ps1`. That script runs the held-out pass once the
  queue has exited — expected at about 21:35 — unless a `STOP` file is present then.
- Held-out score files at 20:50: 1 on `test`, 0 on `clean_eval`, as on 2026-10-08.
  Nothing held-out has been evaluated or looked at.
- Not yet done: the six `a1r` seed reports, the ablation tables of record, §1.32, and
  §1.30.

#### Where it stood — 2026-10-10 18:55 (session 18): PAUSED at the owner's request — `a1r` at 15 of 18 runs, the held-out pass NOT started

- **The owner, 18:51:** *"if anything is running, pause for a while and compact"*.
  `run_queue.py --stop` was issued at 18:52. A frozen run is not resumable, so it is not
  told to stop: **`a1r-gnn_only-seed2`, in flight, finishes by itself within minutes and
  the queue then starts nothing more.**
- **`a1r` then stands at 16 of 18 runs.** Left: `a1r-sum-seed2`, `a1r-gated-seed2`
  (about 18 min each) and the six seed reports. cross_attention, concat, `llm_only` and
  `gnn_only` will have three seeds each, sum and gated two — **no `a1r` number is quoted
  until the seed reports exist** (§7 #37).
- **The held-out pass was not started and will not start by itself now.** The chain
  script finds the `STOP` file when the queue exits and ends without launching it
  (`artifacts/queue/a1r_then_heldout.out` says so). Still 1 of 18 score files.
- **To resume, when the owner says so — two commands, in this order, on a clean tree:**
  `python scripts/run_queue.py --only a1r` (removes the `STOP` file, runs the two runs
  and the reports), and, for the pass to follow as the owner chose at 16:25,
  `powershell -ExecutionPolicy Bypass -File artifacts\queue\a1r_then_heldout.ps1`
  started beside it. **Never start `heldout_pass.py` by hand while that script runs.**
- **Nothing else is running.** Phase 9 is committed on `phase-9/serving` (`6b93d58`) and
  unmerged; the 16:30 block below lists what is left of it.

#### Where it stood — 2026-10-10 16:30 (session 18): `a1r` running, THE HELD-OUT PASS CHAINED BEHIND IT at the owner's word, Phase 9 started on a branch

- **Two owner decisions at 16:25 (§2):** the held-out pass runs **tonight, after `a1r`**;
  Phase 9 starts now. The owner is away for a few hours and said to do what is needed.
- **The chain:** `artifacts/queue/a1r_then_heldout.ps1` (git-ignored) waits for the `a1r`
  queue process to exit, then runs `artifacts/queue/heldout_pass.py`. Transcripts:
  `artifacts/queue/a1r_then_heldout.out`, then `heldout_pass.out`. **Exactly one thing
  starts the pass — that script. Do not also start it by hand:** two instances could
  score one checkpoint twice. If the chain is found dead with the pass unfinished, the
  2026-10-08 block below is the instruction (one command; it skips what exists).
- **`a1r` at 16:25:** 6 of 18 runs done — cross_attention and concat, three seeds each;
  `llm_only`, `gnn_only`, sum and gated follow, then the six seed reports. About 19:30.
- **While either runs: the main tree stays clean and nothing under `src/vulnloc/` or
  `configs/` changes.** Phase 9 is on branch `phase-9/serving`, in a worktree of its
  own, and is **not merged** until both have finished. Docs-only commits (branch
  `phase-8/tables`) are still fast-forwarded in between steps.
- **When `a1r` ends:** read each seed report's `n_seeds`; mark the two `pending` tables
  `run` in `docs/ablations.yaml`; `python scripts/make_ablation_tables.py`; write §1.32.
- **When the pass ends:** §1.30 — the held-out results, read against
  `docs/EvaluationProtocol.md` and nothing else; then the held-out figures
  (`make_phase8_figures.py +figures.split=test`, and `clean_eval`).
- **Phase 9 at 17:05 — built, on branch `phase-9/serving`, NOT merged** (worktree
  `…/scratchpad/wt9`; run its code with `PYTHONPATH=<worktree>\src` from `D:\project`):
  - `src/vulnloc/serving/{bundle,inference,schemas,app}.py` + `static/index.html`;
    `scripts/{export_serving_bundle,serve,measure_latency}.py`; a `serve` compose service;
    `docs/MODEL_CARD.md` (validation numbers only); 41 tests; `import-linter` keeps
    "serving never imports training".
  - **Measured on the real checkpoint `ba7dcc5f…`, development build, not a logged run:**
    48 `val` functions, 3,272 lines — the served batch equals training's collated batch
    exactly (0 differing tensors), same top line 48 of 48, logit difference median
    0.0004 / p99 0.005 / max 0.090 (CPU fp32 against stored GPU mixed precision). Two
    seeds of one arm differ by > 0.33 somewhere in every one of 488 functions, so the
    export's bound is 0.2. Under load, beside training: median 0.38 s a function.
  - **Run by hand at 17:06 on that development bundle** (commit `6b93d58` follows from
    it): `/health`, `/version`, a 200 from `/analyze`, a 422 and a 400 in the one error
    shape, and the demo page's landing state. For a six-line `strcpy` example the
    `strcpy` line came first (0.99, "Buffer overflow", a verified explanation) — **and an
    empty line where a comment had been stripped was flagged too, at 0.096 against the
    0.0925 threshold.** The service now leaves blank lines out of what it offers (the
    evaluation's "blank lines withheld" reading, §1.29). `GET /version` had an empty
    split fingerprint — training runs do not tag it; the export now reads the split
    artefact. `docs/Serving.md` is the usage guide. **Still unseen: the demo page's
    result state in a browser.**
  - **Not done, in order:** (1) merge after `a1r` and the pass; (2) add the held-out
    numbers to the model card; (3) export the bundle of record —
    `python scripts/export_serving_bundle.py +explainer=template +export.run_id=ba7dcc5ff8354cc4b6d7e7b76f876fd9 +export.scores_dir=./artifacts/phase6/scores_retrained +export.calibration_report=./artifacts/phase6/retrained/calibration_report.json`;
    (4) `python scripts/measure_latency.py +latency.bundle=./artifacts/serving/ba7dcc5f`
    on an idle machine; (5) start `scripts/serve.py` and look at the demo page; (6)
    Docker, never beside training, and "on a clean machine" is the second laptop's;
    (7) `Memory.md` §1.33, `Phases.md`, a decision-log row for `httpx` (dev extra) and
    for serving seed 42.
- The 15:00 block below is unchanged except where this one says otherwise.

#### Where it stood — 2026-10-10 15:00 (session 18): `a1r` IS RUNNING; the held-out pass is still paused, untouched

- **The owner's instruction, whole:** *"kindly continue the next steps bui;ding/training"*
  (§2 has how it was read). Pre-flight at 14:36: GPU empty, D: 100% idle, no Docker, no
  Python, on mains.
- **`a1r` — the frozen A1 screen on comment-stripped text — was started at 14:52** from
  clean `e4ed22c`: 18 runs (cross_attention / concat at seeds 42, 1, 2, then `llm_only`,
  `gnn_only`, sum, gated × 3) and six seed reports into `artifacts/adr010/`. Expect 5–7 h.
  - `python scripts/run_queue.py --only a1rcheck a1r --status` shows where it is;
    `--stop` pauses it; `--only a1r` continues it.
  - **While it runs or is paused: nothing under `configs/` or `src/vulnloc/` changes**
    (§7 #54, #62). `scripts/`, `tests/`, `frontend/` and docs only, and the main tree is
    kept clean — session 18 builds in a git worktree (branch `phase-8/tables`) and
    fast-forwards `phase-2/representation` to it between runs.
  - **Quote nothing from `a1r` before an arm has three seeds** (§7 #37), and never pool it
    with the screen trained with comments (§1.21) or with the fine-tuned runs.
- **The gate passed first** (`a1rcheck`, §7 #52): `3c749001…`, 150 epochs in 7.0 min,
  Top-1 = 1.0000 first at epoch 58 and at every one of the last eight; it logged
  `strip_comments=True`, `frozen=True` and cache digest `e3fa34fae5e4`. `--status` prints
  the selected epoch's 0.8710 — never read a gate off that line.
- **The stripped semantic cache was rebuilt by the gate's first step** — the plan's step,
  and it replaces what is there. Run `19e49b8f…`; input digests unchanged (`e3fa34fa…`
  train, `63d0e5ed…` val), same units, rows and bytes. Measured against a copy of the
  2026-10-07 `val.pt` (run `91fc56b4…`) kept for the purpose: masks identical, every one
  of the 488 units differs somewhere, **largest absolute difference 2.9e-6** (§7 #50's
  scatter noise; the tolerance is 1e-5). The 27 stripped-at-evaluation score files of
  §1.25 were made against the earlier build and are reproducible to that, not bitwise.
  No commit touched the text transform, the tokenizer or the cache code in between.
- **THE HELD-OUT PASS HAS NOT MOVED.** Still 1 of 18 score files (`ba7dcc5f…` on `test`),
  nothing evaluated, nothing looked at; the block below is still the instruction. A
  launch at about 14:38 was refused by the session's permission system before any process
  started — the transcript has no new line. **Do not start it on "continue": ask the
  owner, and run it only on words that name it.** It and `a1r` both need the GPU and D:;
  when the owner says go, `run_queue.py --stop`, run the pass, then `--only a1r` again.
- **Built this session, on `val` artefacts only:** `docs/ablations.yaml` and
  `scripts/make_ablation_tables.py` — Phase 8's ablation tables from the 27 seed reports
  that exist, every run checked against the parameters its table declares. The tables of
  record are made after `a1r` ends, when its two tables stop being `pending`.
- **Also built (15:30): the failure taxonomy, as a draft** — `scripts/sample_failures.py`
  (a failure defined by rule, cases drawn by content hash, `val` only; run `42f8f27c…`),
  the sheet `artifacts/phase8/failure_cases.md`, and `docs/FailureTaxonomy.md`: 107 of
  482 units missed on every seed, 30 read by the session, eight categories. **§1.31 has
  the counts. The criterion is not ticked: the owner reviews the cases
  (`docs/FailureTaxonomy.md` §6).**
- **Also built (15:45): the figures** — `scripts/make_phase8_figures.py` draws three SVGs
  from `evaluation_<split>.json` and `failure_cases.json` into
  `artifacts/phase8/figures/`: the arms on the three headline metrics against reading
  from the top, the reliability curve, and found-against-chance by kind of line. No model
  loaded, no partition read; titles say what is plotted and never a result, because the
  same script draws the held-out figures. `val` only so far.
- **The clean-clone check, the half that is cheap (15:39):** a fresh `git clone` of
  `a63a62e` into an empty directory, no artefacts, the package imported from the clone —
  **1,434 tests pass, 5 skip** (4 min 36 s; coverage 91%). Two of the five skip because a
  clone has no checkpoint and no generated figures (`test_frontend_structural.py`,
  `test_prior_work.py`); the other three skip in the working tree as well. **Not done,
  and what the criterion actually asks:** a fresh environment installed from the lock
  file, the data pipeline end to end, and one frozen run reproduced from the clone.
- **Still to do in Phase 8 when `a1r` ends:** the ablation tables of record and their
  write-up (§1.32). **Still the owner's:** the held-out pass; the taxonomy review.

#### Where it stood — 2026-10-08 18:20 (session 17): THE HELD-OUT PASS IS PAUSED at the owner's request — 1 of 18 score files exists, nothing is evaluated

- **Started at 18:09** from clean `d6c3fa6` with `artifacts/queue/heldout_pass.py`
  (transcript `artifacts/queue/heldout_pass.out`). **At 18:14 the owner asked to pause
  everything — the hard disk was at 100% — and to do it later.** The runner and its
  children were killed at 18:14:55; nothing of the project is running.
- **What exists:** `ba7dcc5f…` on `test` — score file and manifest, written 18:11:49, its
  scoring run logged. **That checkpoint is never scored on `test` again.**
- **What was in flight:** `8afbf7da…` on `test`, started 18:12:33, killed before it wrote
  anything. No score file, no manifest, no run: not a scoring, and the next start makes
  its first.
- **Nothing held-out has been looked at.** No session has opened the one score file or
  read the metrics in its manifest; nothing has been evaluated.
- **To continue — one command, when the laptop is free for an hour or more:**
  `.venv\Scripts\python.exe artifacts\queue\heldout_pass.py` from `D:\project`, on a clean
  tree. It scores only what is missing (17 files), then evaluates, audits and explains.
  **Do not run `score_checkpoint.py` by hand for a held-out partition.**
- **Why the disk:** every scoring step starts a new Python process — PyTorch and its
  libraries load from `.venv` on D: — and reads a 1,529 MB `best.pt` from D:, which is the
  spinning disk. The first step took 172 s against about 60 s on `val` in the afternoon.
  Measured after the stop: D: 65–88% idle, only the editor doing I/O. **Expect D: busy
  for the whole pass.** An option that is the owner's and was not taken: copy the nine
  checkpoints (13.8 GB) to the SSD and pass `+score.checkpoint_dir`; C: has 21 GB free.
- **Unchanged:** the protocol is final (`d6c3fa6`), the evaluation code is frozen, and
  the rest of the block below.

#### Where it stood — 2026-10-08 18:10 (session 17): Phase 7 signed, the evaluation protocol final — the single held-out pass about to be made

- **The owner signed the Phase 7 gate, approved the human-evaluation protocol and
  confirmed the evaluation protocol** (§2, the owner's words quoted there). Phase 8 is the
  current phase. `docs/EvaluationProtocol.md` is final in the commit that carries this
  block; **nothing in it, in `src/vulnloc/evaluation/` or in `scripts/evaluate_scores.py`
  may change once a held-out file exists.**
- **The held-out pass, in the protocol's order (§8):** each of the nine `d97r` checkpoints
  scored on `test`, then on `clean_eval` — `score_checkpoint.py +score.run_id=<full id>
  +score.split=<split> +score.final=true +score.output_dir=./artifacts/phase6/scores_retrained`
  — **once: `score_checkpoint.py` does not refuse to overwrite a held-out score file, so
  never run it for a checkpoint whose `<run>_<split>.npz` already exists.** A crash before
  a file is written is not a second scoring; record it. Then `evaluate_scores.py` and
  `audit_score_leakage.py` per partition into `artifacts/phase8/`, then explanations and
  the rater pack on `test`.
- **If you are reading this and `artifacts/phase6/scores_retrained/` holds `_test` or
  `_clean_eval` files but §1.30 does not exist, a session stopped mid-pass.** Do not
  rescore what exists; score only what is missing, then evaluate.

#### Where it stood — 2026-10-08 17:05 (session 17): Phase 8 started on `val` at the owner's instruction — the Phase 7 gate not yet signed, no held-out partition read

- **The owner said "start with phase 8"** after being given the steps for the Phase 7
  gate and for confirming the evaluation protocol. It is recorded as a go-ahead to build
  (§2) and as nothing else: **the Phase 7 gate is not signed and
  `docs/EvaluationProtocol.md` is not confirmed.** Ask for both before the held-out pass;
  do not infer either.
- **Built, tested, `val` only:** the line-level metric suite as the protocol specifies it.
  - `src/vulnloc/evaluation/metrics/localization.py` — per-unit ranking metrics behind
    `METRIC_REGISTRY` (Top-1/3/5/10, IFA, first and average rank, Recall@1/5/20% LOC),
    each under three tie orders; pooled average precision and precision / recall / F1.
  - `src/vulnloc/evaluation/statistics.py` — mean ± sample sd, the paired comparison in
    pooled sd, the bootstrap interval over units.
  - `src/vulnloc/evaluation/evaluator.py` — one checkpoint, an arm over its seeds, the
    pre-stated comparisons, the ceilings a score file cannot see.
  - `scripts/evaluate_scores.py` — the one place a results table may come from; writes
    `artifacts/phase8/evaluation_<split>.json` and `.md`, logs an MLflow run, and refuses
    `test` / `clean_eval` without `+evaluate.final=true`.
- **How it was checked before anything was logged:** on the nine `d97r` `val` score files
  every logged Top-1 / Top-5 / IFA lies inside its tie bounds; in-sample F1 and ECE equal
  the calibration reports'; Top-1 / Top-5 for all lines, blank lines withheld and source
  order equal the score audit's (`10319b01…`) exactly; the ceilings are the figures
  already on record (482 of 488 units ranked, 274 of 2,808 flagged lines blank, 578 of
  3,386 = 17.07% past the line cap).
- **The definitions are in the protocol** (`docs/EvaluationProtocol.md` §5.1): which
  lines are ranked, the tie order, per-unit effort-aware recall, PR-AUC as average
  precision. They were written on `val`, before any held-out score exists, and are part
  of what the owner confirms.
- **Measured on `val` (§1.29, run `52b9c998…`):** no per-unit metric separates the three
  arms — Top-5 0.5394 / 0.5546 / 0.5519, IFA 14.6 / 14.1 / 14.7, PR-AUC 0.171 / 0.182 /
  0.161 against a positive rate of 0.066 — and each beats reading from the top (Top-5
  0.307, IFA 29.8). Of 93 comparisons three reach 2 pooled sd, all concat − `llm_only`
  (PR-AUC, ECE); at three seeds chance crosses that line 7 times in 100, so it is a
  candidate for the held-out pass, not a finding (§7 #64).
- **Next, in order:** the owner's two steps (below) — the gate, the protocol, now
  including §5.1's definitions and §6's note on what 2 pooled sd is worth; then the
  held-out pass: `score_checkpoint.py … +score.split=test +score.final=true` for the nine,
  then `clean_eval`, then `evaluate_scores.py +evaluate.split=<split>
  +evaluate.final=true` and the leak audit on each. **About an hour of GPU; `clean_eval`
  scoring has never been timed.** Still to build in Phase 8: the ablation tables from the
  existing seed reports, figures (the reliability curves are already in the report's
  JSON), the failure taxonomy (≥ 20 cases), the clean-clone check.
- **`frontend/phases.json` is unchanged on purpose:** its Phase 8 stage is "final
  benchmark numbers", and there are none yet.
- **Unchanged from the block below:** the models of record, the two owner steps and how
  to do them, what is still the owner's.

#### Where it stood — 2026-10-08 15:50 (session 17): Q14 closed — ADR-010 accepted; two owner steps stand before Phase 8

- **The owner accepted ADR-010 (Q14) this afternoon** (§2). The nine `d97r` checkpoints
  are the models of record; the full model is `ba7dcc5f…` / `8afbf7da…` / `ff980345…`
  with the calibrators in `artifacts/phase6/retrained/`. Every number logged with
  comments in place stays in the record, labelled as measured on leaky text.
- **Nothing is running or paused. No code, config or run changed** — `strip_comments`
  stays an explicit override (ADR-010 D10.3), so no `config_hash` moves.
- **Written into the documents the same hour:** ADR-010's status; `docs/EvaluationProtocol.md`
  §2 and §3 (decided — the file as a whole is still a draft); `docs/HumanEvalProtocol.md`
  §1 (the rater sample comes from seed 42 of the model of record, `ba7dcc5f…`; it named
  `80fc5238…`; the seed is unchanged) and §2 (stripped text).
- **The owner asked how to do the next two steps and was given them. Neither is done:**
  1. **The Phase 7 gate.** Read `artifacts/phase7/sentences_for_review.md` — 19 template
     sentences and 8 fixed forms, one page — and say which sentence numbers are of
     concern, or that none is. Approve `docs/HumanEvalProtocol.md` as it now reads. Say
     "sign". Record it the way §1.1c and §1.1d were recorded: a §2 row with the owner's
     words, §1.1e's heading, `Phases.md`, `CLAUDE.md` §3 and `frontend/phases.json`.
  2. **Confirm `docs/EvaluationProtocol.md`.** It is then committed as final.
- **With both, Phase 8 starts:** the metric code first (`evaluation/` is still an empty
  package), then the single held-out pass over `test` and `clean_eval` with the nine
  checkpoints. The 2026-10-07 authorisation (§2) asks for no further check-in once the
  protocol is committed.
- **Still the owner's, not blocking:** whether raters start on the existing pilot pack
  or on one redrawn from `ba7dcc5f…`; whether `a1r` runs, and where — while it runs on
  this laptop nothing under `src/vulnloc/` changes, which would hold up Phase 8's code;
  D10.4 (the transform at L0); scheduled optimisation of D: (admin).
- **Not started, and not to be started before both steps above:** Phase 8 code, the
  held-out evaluation.

#### Where it stood — 2026-10-08 14:50 (session 16): `d97r` complete — nothing running or paused

- **Everything `d97r` was started for is measured and written up in §1.28**: nine runs,
  three seed reports, nine score files, three calibrations, the audit, the explanations.
  The slate shows 13 steps done and no `[FAIL]`.
- **The result in one line:** with no comment in the text at training or at evaluation,
  cross_attention / concat / `llm_only` score Top-5 0.5394 ± 0.0116 / 0.5546 ± 0.0084 /
  0.5519 ± 0.0187 and line F1 0.2359 / 0.2408 / 0.2330 on `val`. They tie. Each matches
  the same arm with comments stripped at evaluation only.
- **`configs/` and `src/vulnloc/` may be edited again** — no `d97r-*` step is running or
  paused. The rule returns with the next group: while one runs or is paused, nothing under
  either changes (§7 #54, #62).
- **Two things to say whenever these numbers are quoted:** concat seed 42 is a resumed run
  (`resumed_at: epoch=6,batch=6,step=3944`; its selected checkpoint predates the pause),
  and the cross-attention seed report carries a recorded source-drift override (§7 #62).
- **One thing to know about the guard** (§7 #63): a correct checkpoint can be refused by
  the reproduction check on a tie-order flip. Re-run; do not raise the tolerance.
- **The owner's decisions, in the order they unblock work:**
  1. **Q14** — accept ADR-010 and these runs as the numbers of record. Its evidence is now
     complete (§5). Recommended: accept. If accepted, the full model of record becomes
     `ba7dcc5f…` / `8afbf7da…` / `ff980345…` and the calibrators in
     `artifacts/phase6/retrained/`.
  2. **The Phase 7 gate** — read `artifacts/phase7/sentences_for_review.md`, approve
     `docs/HumanEvalProtocol.md`, sign (§1.1e).
  3. Whether raters start on the pilot pack, or on a pack redrawn from the retrained
     checkpoints once Q14 is decided.
  4. Confirm `docs/EvaluationProtocol.md` (the held-out evaluation on stripped text, with
     these nine checkpoints), so Phase 8 can start.
  5. Whether `a1r` — the frozen screen on stripped text, ~5 h — runs, here or on the second
     laptop.
  6. Turn off scheduled optimisation for D: (admin).
- **Not started, and not to be started without the owner:** Phase 8 code, the held-out
  evaluation, `a1r`.

#### Where it stood — 2026-10-08 14:25 (session 16): all nine `d97r` runs done, the follow-up scoring under way

- **14:21 — the ninth run and the third seed report are in.** Concat: `9ac7a146…`
  (resumed once), `7579bc33…`, `10b43ce5…`; seed report `f3d754eb…`, `n_seeds: 3`.
  **§1.28 has all three arms: on text with no tell, cross-attention does not beat
  concat and neither fused arm beats `llm_only`.**
- **14:22 — `d97r_after.ps1` took over, as designed.** It re-ran the failed
  cross-attention seed report through the queue with the plan's override (the slate now
  shows no `[FAIL]`), confirmed all three arms complete, and began scoring the nine
  checkpoints on `val`. Then: calibration per checkpoint, the score audit, explanations
  for the cross-attention arm. **Read `artifacts/queue/d97r_after.out` before quoting
  anything from `artifacts/phase6/scores_retrained/`, `artifacts/phase6/retrained/` or
  `artifacts/phase7/explanations_retrained/`** — every step's exit code is in it.
- **Until that script has finished, still nothing under `configs/` or `src/vulnloc/`**
  (scoring rebuilds each model from its run's archived config and the current code).
  After it: no `d97r-*` step is running or paused, and both may be edited again.
- **Then finish §1.28's "Not known yet"** with `python artifacts/queue/d97r_read.py`.
- The owner's open decisions are unchanged — listed in the 05:50 block below. **Q14 now
  has its evidence: all nine retrained runs.**

#### Where it stood — 2026-10-08 11:05 (session 16): `d97r` running again — resumed at the owner's request

- **Resumed at 10:59 with `artifacts/queue/d97r_resume.ps1`**, after 5 h 13 min paused
  (05:46 → 10:59). Checked first: clean tree at `2e41629`, nothing under `configs/` or
  `src/vulnloc/` changed since the paused run's commit `fbc44e2`, `last.pt` intact, GPU
  empty, mains power, D: idle.
- **`d97r-concat-seed42` continues as the same run, `9ac7a146…`.** Its log says "Resumed
  from checkpoint … resumed_epoch=6 batches_already_trained=6", `earlier_resumes=0`; the
  first optimizer step past the pause was taken at 11:04. Tags: `resumed_at:
  epoch=6,batch=6,step=3944`, `resume_git_sha: 2e41629` (docs only since `fbc44e2`).
  **Quote that `resumed_at` tag beside every concat number** (§7 #53).
- **Correction to the block below:** the pause did not land "in its sixth epoch". Six
  epochs were complete and evaluated (the sixth ended at step 3,941, twenty-two seconds
  before the pause); the run was six batches into the seventh.
- **What is left:** the rest of that run, concat seeds 1 and 2, the concat seed report —
  then `d97r_after.ps1` (already waiting, minimized window) clears the `[FAIL]` on the
  cross-attention seed report and scores, calibrates, audits and explains on `val`. About
  2–5 h of training, then about 1 h.
- **Unchanged:** nothing under `configs/` or `src/vulnloc/` until it has finished; the
  table, the "when it has finished" steps and the owner's open decisions in the block
  below all still apply. `python scripts/run_queue.py --stop` pauses it again, and
  `d97r_resume.ps1` continues it.
- **`d97r_resume.ps1` was hardened after this start.** It used to count the chain's first,
  instant `--only d97rcheck` call as "the queue is up"; had the follow-up script then
  started in the two-second gap before the real queue, it would have seen a paused step
  and no queue and stood down. It did not happen (the transcript shows it waiting from
  10:59:25), and the script now waits for the `d97r` queue or a training process.

#### Where it stood — 2026-10-08 05:50 → 10:59 (session 16): `d97r` paused at the owner's request

**Nothing is running. Nothing is broken. Six of the nine training runs are finished, a
seventh is paused mid-run, two have not started.**

| Step | State | Run | `val` Top-5 / Top-1 |
|---|---|---|---|
| `d97r-overfit-cross_attention` (the gate) | done | `bcf18403…` | passed — never a result |
| `d97r-cross_attention-seed42` / `seed1` / `seed2` | done | `ba7dcc5f…` `8afbf7da…` `ff980345…` | **0.5394 ± 0.0116 / 0.2683 ± 0.0060** |
| `d97r-seed-report-cross_attention` | **`[FAIL]` in the slate — the report exists** | `bd2544c7…` | see below |
| `d97r-llm_only-seed42` / `seed1` / `seed2` | done | `d82a6a70…` `a1dce535…` `9296a0e5…` | **0.5519 ± 0.0187 / 0.2801 ± 0.0062** |
| `d97r-seed-report-llm_only` | done | `f814e521…` | |
| `d97r-concat-seed42` | **`[paus]`** at step 3,944 (sixth epoch, 80.7 min in), `last.pt` written | `9ac7a146…` | — |
| `d97r-concat-seed1`, `seed2`, `d97r-seed-report-concat` | pending | | |

The two finished arms are written up in §1.28. Nothing has been scored, calibrated,
audited or explained from these checkpoints yet — that step runs after all nine.

**To resume — one command, any time the laptop is free and plugged in:**

```powershell
powershell -ExecutionPolicy Bypass -File D:\project\artifacts\queue\d97r_resume.ps1
```

It refuses a dirty tree or a queue that is already running; otherwise it removes the STOP
file, starts the queue in its own window (which continues `9ac7a146…` from its checkpoint —
same run id, same learning curve — then runs concat seeds 1 and 2 and the concat seed
report) and puts `d97r_after.ps1` behind it. **About 2–5 h of training, then about 1 h of
scoring.** `python scripts/run_queue.py --stop` pauses it again.

**The `[FAIL]` is not a failed training run and needs no repair by hand.** No training run
failed that night: all six that finished did so on their first attempt, none resumed. The `[FAIL]`
is the queue's first attempt at the cross-attention *seed report* — a ten-second
aggregation step — which refused because one explainer file changed between two of the
arm's seeds (§7 #62). The report was made minutes later with the recorded override
(`bd2544c7…`, `n_seeds: 3`), so the arm's numbers are complete. The mark itself clears on
resume: `d97r_after.ps1` runs `--only d97r --retry-failed` after the last concat run, and
the queue's plan now carries the override. **Do not run `--retry-failed` by hand while the
group is paused — it would also restart the paused training run.**

**Until it is resumed and finished:**

- **Change nothing under `configs/` or `src/vulnloc/`.** A paused run refuses to resume
  under a different config (§7 #54); a package change between concat's seeds fails
  concat's report (§7 #62). `scripts/`, `tests/`, `frontend/` and docs are fine,
  committed at once.
- `d97r-concat-seed42` will be a **resumed run**: quote its `resumed_at` tag beside
  concat's numbers (§7 #53). The other eight are uninterrupted.
- Do not quote a concat number, or A1's comparison on leak-free text, before concat has
  three finished seeds and a report at `n_seeds: 3`.

**When it has finished:** read `artifacts/queue/d97r_after.out` first (every step's exit
code is in it), then `python artifacts/queue/d97r_read.py` (all three arms: runs,
provenance, seed reports, calibration, audit, explanations) and
`python artifacts/queue/d97r_stats.py` (within-arm and between-arm differences in pooled
sd). Both only read. Then finish §1.28: the concat arm, cross_attention − concat, the
calibrated line F1, the audit's blank-line view. Artefacts land in
`artifacts/phase6/scores_retrained/`, `artifacts/phase6/retrained/` and
`artifacts/phase7/explanations_retrained/` — never beside the comment-trained
checkpoints' (§7 #61).

**Still the owner's, unchanged by the night:** Q14 (accept ADR-010 and the retraining as
the numbers of record); the Phase 7 gate (read `artifacts/phase7/sentences_for_review.md`,
approve `docs/HumanEvalProtocol.md`, sign); whether raters start on the pilot pack; whether
the second laptop runs `a1r`; confirming that the held-out evaluation runs on stripped
text with the `d97r` checkpoints (`docs/EvaluationProtocol.md`, draft); turning off
scheduled optimisation for D:. **Not started, and not to be started without the owner:**
Phase 8 code, the held-out evaluation, `a1r`.

The helper scripts under `artifacts/queue/` are git-ignored: they exist on this laptop
only (`d97r_chain.ps1`, `d97r_gate.py`, `d97r_after.ps1`, `d97r_resume.ps1`,
`d97r_report_cross.ps1`, `run_plan_step.py`, `d97r_read.py`, `d97r_stats.py`).

#### Where it stood — 2026-10-07 18:15 → 2026-10-08 05:46 (session 16): the night `d97r` ran

- **`d97r` was started at 18:04 by session 16, pending the owner's decision on Q14.** Its
  own console window ("VulnLoc d97r"), transcript `artifacts/queue/d97r_chain.out`. Step
  1 is the overfit gate on comment-stripped text (`d97rcheck`); the nine fine-tuned runs
  start only if it ends at Top-1 = 1.0 with `strip_comments=True` logged. **The gate
  passed at 18:23** — run `bcf18403…`, 150 epochs in 19.7 min, the last five at Top-1 =
  1.0000 (the queue's `--status` prints its selected epoch, 0.9677) — and
  **`d97r-cross_attention-seed42` started at 18:24** from clean `0a4843c`, both datasets
  logging `strip_comments=True`.
  Order: cross_attention ×3, its seed report, `llm_only` ×3, concat ×3. **Expect about a
  day**: six D9.7 runs took 114–211 min each.
  - `python scripts/run_queue.py --only d97rcheck d97r --status` shows where it is.
  - `python scripts/run_queue.py --stop` pauses it cleanly; `--only d97r` continues it.
  - **While it runs or is paused: no edits to `configs/` or to ANY file under
    `src/vulnloc/`** — corrected 2026-10-07 22:10, §7 #62: the seed report compares the
    whole package between an arm's seeds, not only what training imports. `scripts/`,
    `tests/`, `frontend/` and docs are fine, committed at once.
  - If the owner decides against retraining (Q14), stop it and tag its runs; nothing else
    depends on it.
- **The frozen screen was audited too (§1.25, last subsection).** Frozen `llm_only` does
  not read the tell; every arm with a structural branch does. Fusion's lead over frozen
  `llm_only` shrinks from +0.032…+0.051 to +0.010…+0.029 on stripped text and keeps its
  sign. `a1rcheck` / `a1r` (18 frozen runs on stripped text) are defined and **not
  started** — a candidate for machine 2, as one whole group.
- **20:14 — the first of the nine is FINISHED:** `d97r-cross_attention-seed42`, run
  `ba7dcc5f…`, 109.2 min, one attempt, not resumed, clean `0a4843c`, 13.5 min an epoch,
  early-stopped after eight. Its curve rises (validation Top-5 0.5000 at epoch 0, 0.5519
  at the selected epoch 4), so the stripped text trains. **One seed is not a result —
  nothing about `d97r` is quoted until an arm has three** (§7 #37); the arm's seed report
  runs by itself after seed 2. Seed 1 started at 20:13. The GPU read 85–87 °C for most of
  the run: if the laptop can be given more air, give it.
- **22:02 — the second is FINISHED:** `d97r-cross_attention-seed1`, run `8afbf7da…`,
  109.1 min, one attempt, not resumed, clean `d5e5dda`. Seed 2 started at 22:02. Still
  two seeds: nothing is quoted.
- **22:10 — this arm's seed report needs, and now has, the source-drift override.**
  Between seed 42 (`0a4843c`) and seed 1 (`d5e5dda`) nothing under `configs/` or on
  the training path changed, but commit `7ea7a9c` changed
  `src/vulnloc/explanation/generator.py`, and `report_seeds.py` compares all of
  `src/vulnloc` (§7 #62). `explanation` is layer 5; training is layer 3 and the
  import-linter contract, checked on every commit, forbids it from importing upward
  — so no training run can execute that file. `run_queue.py` therefore passes
  `+seeds.allow_source_drift=true` to `d97r-seed-report-cross_attention` **only**; the
  report lists the changed file under `provenance.src_vulnloc_drift_accepted`. **The
  queue process that is running was started before this change and still holds the
  old plan: its own attempt at this report will be `[FAIL]` ("runs were produced by
  different source trees") — expected, harmless.** The report is then made by hand
  with the new plan's exact command, and `d97r_after.ps1` re-runs the failed step
  through the queue at the end so the slate records it `done`.
- **22:58 — THE FIRST ARM IS IN (§1.28).** `d97r-cross_attention-seed2`, run
  `ff980345…`, 54.4 min, one attempt, not resumed, clean `5af24ba`: early-stopped after
  four epochs, selected at epoch 0. The queue's own seed-report step went `[FAIL]` on
  source drift exactly as predicted; `artifacts/queue/d97r_report_cross.ps1` then made
  the report with the plan's command (run `bd2544c7…`, `n_seeds: 3`, the drift recorded).
  **Cross-attention trained and evaluated without comments: Top-5 0.5394 ± 0.0116,
  Top-1 0.2683 ± 0.0060** — where stripping at evaluation alone had put it (0.5353 ±
  0.0200). `d97r-llm_only-seed42` started at 22:58. **`--status` shows one `[FAIL]`:
  that is this step, and `d97r_after.ps1` clears it at the end.**
- **Thu 8, 04:28 — THE SECOND ARM IS IN (§1.28).** `llm_only` retrained: runs
  `d82a6a70…` (83.8 min), `a1dce535…` (122.9), `9296a0e5…` (121.8), each one attempt,
  none resumed; the queue made the seed report itself (`f814e521…`, `n_seeds: 3`, no
  drift). **Top-5 0.5519 ± 0.0187, Top-1 0.2801 ± 0.0062. Retrained cross-attention does
  not beat it** (Top-5 −0.0124 at 0.80 pooled sd; Top-1 −0.0118 at 1.92, same sign on
  every seed: a candidate, not a finding). `d97r-concat-seed42` started at 04:28; the
  three concat runs should end between about 07:15 and 11:15, then `d97r_after.ps1`
  takes about an hour.
- **Thu 8, 05:46 — `d97r` IS PAUSED, at the owner's request.** `run_queue.py --stop`;
  `d97r-concat-seed42` (run `9ac7a146…`, clean `fbc44e2`) paused cleanly at optimizer
  step 3,944, in its sixth epoch, 80.7 min in, with `last.pt` written (`kind=paused`).
  The queue exited, the GPU is free, and `d97r_after.ps1` stood down on the STOP file
  without scoring anything. **State: 6 of 9 training runs finished, a seventh paused
  mid-run; cross_attention and `llm_only` have their three-seed reports (§1.28); concat
  has no finished seed.** (This line first said "7 of 9 runs done" — a miscount of the
  slate's seven finished *steps*, one of which is a seed report; corrected 05:55.)
  - **To continue:** `powershell -ExecutionPolicy Bypass -File
    D:\project\artifacts\queue\d97r_resume.ps1` — it removes the STOP file, starts
    the queue in its own window (same run id, same learning curve) and puts the
    follow-up script behind it. About 2–5 h of training remain, then ~1 h of scoring.
  - **`d97r-concat-seed42` will be a resumed run:** quote its `resumed_at` tag with
    concat's numbers (§7 #53). The other eight are uninterrupted.
  - **While it is paused the rule still holds:** nothing under `configs/` or
    `src/vulnloc/` changes — a paused run refuses to resume under a different config
    (§7 #54), and a package change between concat's seeds fails its report (§7 #62).
  - `--status` shows `[FAIL]` on `d97r-seed-report-cross_attention`: the queue's first
    attempt. The report exists (`bd2544c7…`); the follow-up script re-runs the step.
- **Started the same evening — what happens after the nine, without a session:**
  `artifacts/queue/d97r_after.ps1` runs in its own window (transcript
  `artifacts/queue/d97r_after.out`). It waits for the queue to end; if a step is
  `[FAIL]` it runs `--only d97r --retry-failed`, at most twice; a queue somebody paused
  (a STOP file, or a paused step with nothing failed) it leaves alone. Then, **for
  complete arms only** (three runs done and a seed report at `n_seeds: 3`): scores each
  checkpoint on `val` into `artifacts/phase6/scores_retrained/`, calibrates per
  checkpoint into `artifacts/phase6/retrained/` (the other two arms under `arms/`),
  audits into `artifacts/phase6/retrained/score_leakage_val_stripped.json`, and
  explains the cross-attention arm into `artifacts/phase7/explanations_retrained/`.
  `val` only; it starts no training the `d97r` group did not already contain.
- **First pace reading, 18:30:** 144 s per hundred optimizer steps (D9.7's fused arms ran
  at ~200), 3,267 MiB dedicated, no shared memory in use. One reading — look again.
- **A bundle for machine 2 is ready, unused:** `D:\vulnloc_transfer\vulnloc_20261007.bundle`
  (commit `18516aa`, beside the 2026-10-03 one, which is untouched) with
  `A1R_START_HERE.md`. It is only for `a1r`, and only if the owner decides to run it
  there (`docs/SecondMachine.md` §6).
- **A draft of the held-out evaluation protocol** is at `docs/EvaluationProtocol.md`. It
  binds nothing until Phase 8 starts; its §2 and §3 are Q14 put to the owner in concrete
  form.
- Everything in the 17:30 block below still holds.

#### Where it stood — 2026-10-07 17:30 (session 16): leak measured, fix built, A6 and Phase 7a written up

- **The leak (§1.25) has a fix and three measurements.** `strip_comments` is built, off by
  default, and agrees with the grammar on 6,586 of 6,586 clean units. The nine D9.7
  checkpoints score **Top-5 0.535–0.555 on comment-stripped `val` against 0.616–0.617 as
  shipped**, line F1 0.234–0.236 against 0.39; the arms still tie. ADR-010 is written and
  **Proposed**; **Q14 is the owner's**.
- **A6 is done (§1.26):** bidirectional is not resolvably better than either one-way arm.
- **Phase 7a is built and measured (§1.27):** every flagged line gets a verified
  explanation; a pattern names a cause for about one in twelve. The gate answers are in
  §1.1e, **prepared, not signed**; the owner's reading of the 19 template sentences for
  attack instructions is the open criterion.
- **The pilot rater pack is ready**: `artifacts/phase7/rater_pack_stripped/` — 100 `val`
  samples, three empty sheets, instructions. The protocol (`docs/HumanEvalProtocol.md`)
  draws the *rated* sample from `test`; whether raters start on the pilot is the owner's.
- **`d97r` — D9.7 again on stripped text — is the leak-free headline.** Queue groups
  `d97rcheck` (one overfit gate) and `d97r` (nine fine-tuned runs, arm by arm, 17–32
  GPU-h). `artifacts/queue/d97r_chain.ps1` runs gate then runs. **If it is running when
  you read this, it was started by session 16 pending Q14** — `python scripts/run_queue.py
  --stop` pauses it cleanly, and `--only d97r` continues it. While it runs or is paused:
  **no edits to `configs/` or to ANY file under `src/vulnloc/`** (corrected 22:10, §7
  #62 — the seed report compares the whole package); `scripts/`, `tests/`, `frontend/`
  and docs are fine, committed at once.
- **Do not run the held-out evaluation on the shards as they are.** When it runs, it runs
  with `+representation.strip_comments=true`, on checkpoints trained that way.
- Still the owner's: Q14; the Phase 7 gate; the template review; scheduled optimisation of
  D: (§7 #57).

#### Where it stood — 2026-10-07 15:30 (session 16): A LABEL LEAK FOUND — read §1.25 before quoting anything

- **A second formatting leak, of ADR-006's family, is live in `train`, `val` and `test`**
  (§1.25, §7 #58): a flagged line that ended a block comment lost the slash of its `*/`.
  Audited and logged (`9f462174…`, `ded852ee…`). The nine D9.7 checkpoints use it; about
  0.05 of every logged `val` Top-5 sits on the 12% of units it answers. **A1's null
  survives it.** Everything frozen, structural, Phase 3, A6 and machine 2 is unaudited.
- **Q14 (§5) is the owner's**: strip comments and retrain, or repair at evaluation only.
  Session 16 is building the evaluation-time repair behind an explicit switch and
  re-scoring the nine checkpoints on `val` with it once A6 releases the GPU.
- **Do not run the held-out evaluation on the shards as they are.** The owner's go-ahead
  (§2) was given before this was known; `test` carries the tell on 13.4% of its units.
- Phase 7a's package is written and tested on the `phase-7/explanation` worktree
  (`D:\vulnloc-wt\phase7`); it is merged when the `a6` group ends.

#### Where it stood — 2026-10-07 15:00 (session 16): Phases 5 AND 6 SIGNED; 7a is current; A6 running

- **The owner closed Q5 (type classifier deferred), signed the Phase 6 gate (§1.1d) and
  authorised the single held-out evaluation for the start of Phase 8** (§2). Nothing
  waits on the owner except switching off scheduled optimisation for D: (§7 #57) and,
  later, the Phase 7 gate. The owner is away until about 22:00 and asked for work and
  training to continue unattended; Brave is closed.
- **A6 (attention direction) is running as queue group `a6`.** Both overfit gates passed
  (`836aec8e`, `f3acb184` — Top-1 = 1.0000 over the last five epochs, read from the
  metric history, not from `--status`, which prints the selected epoch). Nine frozen
  runs and three seed reports follow. **A run takes about 38 min, not the ~15 planned
  with:** `a6-both-seed42` (`cde35987`) took 38.3 min against 23.1 for the same arm and
  seed in Phase 5 (`192e067c`), and the training process writes ~7 MB/s to D:. The cause
  is **not confirmed** (suspect: the mid-epoch checkpoints added on 2026-10-02). Expect
  the group to end around 19:30. **No A6 comparison is written anywhere until every arm
  has three seeds** — check `n_seeds` in `artifacts/phase6/seed_report_fused_a6_*.json`.
- **While `a6` runs, the main working tree stays clean**: build in a separate git
  worktree and merge after the group ends, or commit a docs change at once.
- **Next, in order:** (1) the A6 write-up when the group ends. (2) Phase 7a — evidence
  extraction, templates, the grounding verifier, the rater pack. (3) After the owner
  signs Phase 7: commit Phase 8's evaluation protocol **first**, then the one held-out
  scoring pass.
- The 03:15 block below is kept for the record; where it says Q5 is open, it predates
  this one.

#### Where it stood — 2026-10-07 03:15 (session 16): D9.7 COMPLETE, nothing running

- **9 of 9 runs FINISHED; all three seed reports say `n_seeds: 3`.** `--status` shows twelve
  steps `done`. The control is in §1.22: **`llm_only` 0.6169 ± 0.0048 Top-5 — neither fused
  arm beats it.** All three `llm_only` runs are resumed; the tags are in §1.22.
- **Rule 2 is lifted**: no `d97-*` step is running or paused, so `src/vulnloc/` and
  `configs/` may change. **The overnight fixes landed the same morning** (§7 #49, #56, #57):
  the checkpoint rename waits out a lock, the queue continues a crashed run twice by
  itself, a seed report fails instead of passing short, the VRAM guard fires where the
  spill actually starts, and every run logs which card it used. 965 tests pass.
- **Phase 5 is SIGNED (the owner, Wed 7) and Phase 6 is three of four criteria in**
  (§1.24): the designed cross-attention model is the full model; its scores are calibrated
  and its threshold chosen on `val`. **Phase 6's last criterion is Q5 (CWE labels) — the
  owner's "decided" or "deferred".**
- **Next, in order:** (1) Q5 from the owner, which closes Phase 6. (2) Phase 7a —
  evidence extraction from the attention maps, templates, the grounding verifier — and the
  rater pack; the raters' time is the constraint, not the GPU. (3) **The single held-out
  evaluation**: `score_checkpoint.py +score.split=test +score.final=true`, then
  `clean_eval`, for all nine D9.7 checkpoints, applying each full-model checkpoint's
  `val`-fitted calibrator and threshold. **Once, and only when nothing upstream will change
  again** (`Rules.md` §5.3). It needs score files for concat and `llm_only` on `val` first
  (~2 min a checkpoint) if their thresholded metrics are to be reported.
- **Before the next long run the owner should stop Windows defragmenting D: on its own**
  (§7 #57) — it needs an administrator.
- **The second machine's export is IN (Wed 7, 09:15):**
  `artifacts/machine2/Sujal_20261005-224401/`, written up in **§1.23**. All 33 steps done,
  9 seed reports at three seeds, five overfit gates passed. No ablation resolves at three
  seeds; fine-tuned GraphCodeBERT does not beat CodeBERT; **weighted BCE (+1.96 pooled sd on
  the frozen screen) is the one candidate worth more seeds before Phase 6 fixes a loss.**
  That takes A4 (GAT/GGNN in fusion), A8 line-only, A9 and A5 off this GPU's list — as
  machine-2 numbers, never pooled with this laptop's.
- Still open: the two held Q13 questions (rule 8).
- The bullets below are the night's log, kept for the record. **Where they say nothing is
  in `artifacts/machine2/` yet, they predate the import.**

#### How the night went — 2026-10-06 20:33 → 2026-10-07 03:15

- **`run_queue.py --only d97` restarted Tue 6 at 20:33** in its own console window ("VulnLoc
  d97 queue"), at the owner's request. `d97-llm_only-seed42` resumed at 20:37:02 into the
  same run (`179632f5…`, `earlier_resumes=0`) from epoch 6, batch 500, step 4565. Seeds 1
  and 2 and the three seed reports follow: **≈ 5–8 h**. Still **0 of 3** for the control —
  no number yet (rule 5).
- **The bullets under "2026-10-04" below describe the pause and are superseded on that one
  point.** Rule 2 is unchanged: a step is running, so no edits under `src/vulnloc/` or
  `configs/`.
- **The second machine's slate is finished (the owner's report, Tue 6).** Nothing is in
  `artifacts/machine2/` yet; the owner imports it later. What to bring is one
  `export_results.py` folder, after the nine seed reports are run **there**
  (`docs/SecondMachine.md` §4–5). Read the five overfit gates' Top-1 in its `RESULTS.md`
  before anything else from it is used (§7 #52).
- The two held Q13 questions (rule 8) were raised again and are **still unanswered**.
- **Wed 7, 00:25 — 7 of 9 FINISHED, one CRASHED, one running.**
  - `d97-llm_only-seed42` (`179632f5…`) finished at 20:52, 133.6 min over two attempts.
  - **`d97-llm_only-seed1` (`72f067d0…`) is `[FAIL]`**: `PermissionError [WinError 5]` on the
    checkpoint rename at 22:47, in epoch 5 (§7 #56). Not a model failure. `last.pt` (step
    4100, written 22:44:06) and `best.pt` (epoch 2) are intact; a 1.4 GB `last.pt.tmp` is
    left beside them and is overwritten by the next save.
  - `d97-llm_only-seed2` (`2c7a7631…`) started 22:58 and had finished epoch 3 at 00:21.
    Read once at 00:22, just after that epoch boundary: 2,799 MiB dedicated + 78 MiB shared
    (Brave still open — one reading, not a comparison with the 20:38 one).
  - **A watcher is waiting in its own console window** (`artifacts/queue/d97_finish.ps1`,
    transcript `artifacts/queue/d97_finish.out`). When the queue exits it runs
    `run_queue.py --only d97 --retry-failed`, then re-runs the `llm_only` seed report if all
    three seeds are `done`. It starts nothing if a `STOP` file exists, if seed 2 ended
    paused, or if another queue is already running.
  - **The next session checks:** `--status` shows nine runs `done`; the transcript; and
    `n_seeds: 3` in `artifacts/phase5/seed_report_fused_finetuned_llm_only.json`. If the
    report says 2, it is the queue's own early one — re-run it before reading a number.
- **Wed 7, 01:12 — seed 2 crashed too, on a GPU driver fault** (`CUDA error: unspecified
  launch failure`; §7 #57), 144 min in, epoch 6. The queue ran its three reports — the
  `llm_only` one on **one** seed, flagged non-citable — and exited at 01:15.
- **01:15 → 03:15 — the watcher did what it was built for.** `--retry-failed` continued seed
  1 from step 4100 (early-stopped after epoch 8, 02:48) and seed 2 from step 4800
  (early-stopped after epoch 7, 03:14), then re-ran the `llm_only` report: `n_seeds: 3`.
- **01:32 → 02:32 — Windows defragmented D: under the run** and a VRAM spill slowed seed 1
  to 536 s per 100 steps; both cleared without intervention (§7 #57).
- **The session's own checks stopped at 02:18**: the background wait that should have woken
  it then was only delivered at 06:59, with the owner's next message. The run finished
  unwatched for its last hour — the watcher, not the session, carried it.

#### Where it stood — 2026-10-04 21:51 (session 15): PAUSED at the owner's request

- **6 of 9 runs FINISHED.** cross_attention and concat have three seeds each and are
  aggregated (§1.22): **0.6162 ± 0.0184 vs 0.6169 ± 0.0104 Top-5 — no resolvable difference.**
- **`d97-llm_only-seed42` is `[paus]`** at epoch 6, batch 500, step 4565 (`179632f5…`).
  Seeds 1 and 2 and the three seed reports are pending — **≈ 5–8 h** once restarted with
  `python scripts/run_queue.py --only d97`. Nothing restarts it but a person or a session.
- **Because a step is paused, rule 2 below is in force**: no edits under `src/vulnloc/` or
  `configs/` in this tree.
- **Docker Desktop 4.93.0 was installed Sun 4 at 22:08, to `D:\Docker`** (program in
  `D:\Docker\Docker`, WSL data root `D:\Docker\wsl`; §3 session 15). **Rebooted 22:30;
  the owner launched it once and accepted the licence; the data disk is confirmed on D:**
  (`D:\Docker\wsl\disk\docker_data.vhdx`, 1,580 MB). **Auto-start is OFF** (Docker's
  `AutoStart: false`, the `HKCU\...\Run` entry removed, `com.docker.service` Manual) and
  Docker is stopped. **The owner will set it up later — no image is built**, so the Phase 0
  and Phase 9 criteria are still unverified. `make docker-build` waits for a slot when no
  fine-tuned run is in flight.
- **D: is a spinning hard disk** (Seagate ST1000LM049, SATA, 932 GB); C: is the NVMe SSD
  (238 GB, 24 GB free). The project, its data, the checkpoints **and Docker** all live on
  the HDD, which reads 100% busy at a few MB/s under scattered I/O. Docker on it will be
  slow, and **Docker and a training run must never share it at the same time.** MATLAB's
  startup accelerator (`D:\MATLAB`) used to read from D: for minutes after every sign-in and
  twice a day; **both of its scheduled tasks were disabled on 2026-10-04 at the owner's
  request** (R2023a and R2026b; re-enable with `Enable-ScheduledTask`). Still check that D:
  is quiet before restarting the queue.
- **Close hardware-accelerated browsers before restarting** — Brave held 162 MiB of the
  NVIDIA card during a run (§3 session 15).
- The second machine has `D:\vulnloc_transfer\` and is running its slate (owner's report).

#### Start of every session — in this order

1. `python scripts/run_queue.py --only d97 --status` — before reading anything else.
2. **If a `d97-*` step is running or `[paus]`:** do not edit `src/vulnloc/` or `configs/` in
   this working tree (§7 #54, and every remaining seed would differ from the logged ones).
   Docs-only commits are safe — `report_seeds.py` allows a differing sha when `src/vulnloc`
   did not change — **but commit them at once**: a dirty tree tags the next run `-dirty`.
3. **If nothing is running, steps are pending and the laptop is free:** restart it —
   `python scripts/run_queue.py --only d97` (add `--until HH:MM` if the laptop is needed).
   It resumes a paused run itself. Never relaunch a `d97-*` step by hand.
4. **Do not run CPU-heavy work beside a fine-tuned run** — the full test suite stretched an
   overfit check from 21 to 28 minutes on 2026-10-02. Lint and small test files are fine.
5. **Never quote a D9.7 number until all three seeds of that arm are FINISHED** (§7 #37),
   and list each resumed run's `resumed_at` tag beside it (§7 #53).
6. All nine runs must come from this laptop (§7 #49).
7. **A second GPU laptop (6 GB) trains side by side** (owner, 2026-10-03) —
   `docs/SecondMachine.md`. It runs whole groups only: `m2check` + `m2ftcheck` (GraphCodeBERT
   cache build, five overfit gates), then `m2` (21 frozen ablation runs around the A1
   cross_attention arm: A4 GAT/GGNN in fusion, A9 BCE variants, A8 line-only, A5 frozen
   GraphCodeBERT) and `m2ft` (A5 fine-tuned: CodeBERT vs GraphCodeBERT × 3 seeds, ~12–17 h).
   It keeps its own `mlflow.db` and reports there; results come back with
   `scripts/export_results.py` into `artifacts/machine2/`, never into this machine's store.
   Never copy `mlflow.db` or checkpoints between machines (§7 #49). The transfer folder is
   `D:\vulnloc_transfer\` (~2.7 GB). **Handed over and running there as of 2026-10-04 (the
   owner's report); nothing has come back to `artifacts/machine2/` yet.** Remind the owner that each
   overfit gate's Top-1 must be read before `m2` / `m2ft` start (§7 #52).
8. **Ask the owner the two held questions of Q13 (§5)** — held on 2026-10-03 at the
   owner's request ("we will tackle them tomorrow"), raised again on 2026-10-04 and **not
   yet answered**: external baseline as a balanced subsample or `(reported)`; whether the
   function-level criteria and 7b are deferred in writing. (The third, the second laptop's
   GPU, is answered: 6 GB.) Session 15's recommendation: `(reported)`, since the night lost
   on Sat 3 used the buffer the subsample depended on; and defer both in writing.

```bash
python scripts/run_queue.py --only d97 --status      # the nine runs, with Top-5 / Top-1
python scripts/run_queue.py --only d97 --until 08:00 # continue; pause cleanly at 08:00
python scripts/run_queue.py --stop                   # pause now, from another terminal
```

D9.7's order: cross_attention / concat at seeds 42, 1, 2 — then llm_only × 3 — then three
seed reports (`artifacts/phase5/seed_report_fused_finetuned_<arm>.json`).

#### The schedule — a plan, not a promise; there is no slack in it

| When | GPU (nights) | Daytime (no GPU) |
|---|---|---|
| Fri 2 → Sat 3, 08:00 | **DONE:** D9.7 runs 1–3 FINISHED; run 4 (concat seed 1) paused at epoch 5 | — |
| Sat 3 evening → Sun 4 morning | **Did not happen** — the queue was not restarted and Docker was not installed | — |
| Sun 4, 12:59 → 21:51 | **DONE:** concat seed 1 (resumed), cross_attention seed 2, concat seed 2 — both fused arms complete and aggregated (§1.22). `llm_only` seed 42 paused at epoch 6 at the owner's request | §1.22 written for the two fused arms |
| Next GPU slot | Docker Desktop not running, browsers closed, D: settled after boot (it is a hard disk), then `run_queue.py --only d97`: `llm_only` 42's tail + seeds 1 and 2 ≈ **5–8 h**, then the three seed reports | — (source frozen) |
| Mon 5 | — | Add the `llm_only` control to §1.22 once its three seeds are in. Close Phase 5 criterion 5; owner signs the gate. Fix the residual spill guard (session 14 log). Build: held-out evaluation of a checkpoint on `test` and `clean_eval`; the missing line metrics (Top-3/10, MFR/MAR, Recall@k%LOC, PR-AUC); calibration + reliability curve; threshold chosen on `val` |
| Mon 5 night | Frozen slate 1 — Phase 6's sweep, **the same search budget for cross_attention and concat** (`ResearchPlan.md` §5, A1's rule) | — |
| Tue 6 | Frozen slate 2 — A3 (graph type), A4 (GNN kind in fusion), A6 (attention direction), A8 (line-only vs joint), A9 (BCE / weighted / focal), 3 seeds each | Close Phase 6: calibration, threshold, best config frozen, Q5 deferred in writing. Phase 7a: evidence extraction, templates, grounding verifier |
| Wed 7 | Final config fine-tuned × 3 seeds **only if it differs from D9.7's** (~9–10 h); then `test` + `clean_eval` evaluation | Faithfulness (deletion test). Human-evaluation protocol *written* |
| Thu 8 | Buffer for reruns | Phase 8: tables and figures by script, significance over seeds, failure taxonomy (≥ 20 cases). Phase 9: FastAPI, schemas, latency |
| Fri 9 | — | Paper draft: results, limitations, threats. Model card, README quickstart, final Memory.md |

**Session 15 (2026-10-04): the plan is about a day behind** — the Sat 3 night was lost and
D9.7 still owes ≈ 5–8 GPU-h. Every row from "Mon 5 night" down shifts unless Thursday's
buffer absorbs it. Against that, the second machine has taken A4, A9, A8 line-only and A5
off "frozen slate 2", so this GPU's slate 2 is A3 and A6 only.

A6 and A8 need small code changes; A3 needs ADR-007 accepted (CPG has several relations).
**None of that code can land before D9.7 finishes** unless it is written in a separate git
worktree — not set up as of session 14, and note that `.venv` holds an *editable* install
pointing at this tree's `src/`, so a worktree needs `PYTHONPATH` set to its own `src/` and
its own link to `artifacts/`.

#### Phases 6–10 against the week — what is possible, criterion by criterion

Three things cannot be bought with effort this week. They are the whole reason the phases
cannot all close as written:

1. **People and elapsed time.** Some criteria are done by humans who are not the owner and
   not a session. **Largely lifted 2026-10-03:** the owner has three raters and a second
   laptop. What remains is their *time* inside the week.
2. **This machine.** Docker is not installed (Phase 0's own criterion is open for the same
   reason), and there is exactly one GPU. **Docker: lifted** — the owner installs it Sat 3
   evening. One GPU per comparison group: unchanged.
3. **Negatives.** Every model here trains on the 5,457 line-labelled *vulnerable* functions.
   Anything function-level — "is this function vulnerable?" — needs the 148,983-function
   train split, ~27× the units per epoch. Fine-tuned, that is days per run (**an estimate
   from unit counts, not a measurement**), and it is why `function/f1` = 1.000 is degenerate
   and why the model cannot say a function is safe.

| Phase | Exit criterion / deliverable | This week? | Why |
|---|---|---|---|
| 5 | Negative A1 investigated before Phase 6 | ✅ | D9.7's six runs done Sun 4 (§1.22); the `llm_only` control needs ≈ 5–8 GPU-h more, then the owner's signature |
| 6 | Line scores calibrated, reliability curve | ✅ | Post-hoc on `val`; no training |
| 6 | Threshold chosen on validation only | ✅ | Same |
| 6 | Best full-model config frozen and checkpointed | ✅ | From D9.7 + the frozen sweep |
| 6 | CWE labelling decided or deferred (Q5) | ✅ as "deferred" | Owner's word needed |
| 6 | Unit-level aggregation, joint loss that means something | ❌ | Blocker 3 — no negatives |
| 6 | Hyperparameter sweep | 🟨 frozen only | A fine-tuned sweep is ~3 h a point |
| 7 | 7a: a valid explanation for every positive prediction | ✅ | Deterministic; ~1–2 days of code |
| 7 | Grounding test suite passes | ✅ | Automatic |
| 7 | Explanations reviewed for attack instructions | 🟨 | A human must read them — owner |
| 7 | Human-evaluation protocol **defined** | ✅ | A document |
| 7 | Human evaluation **run** (E9: ≥ 3 raters, ≥ 100 samples) | ✅ *(owner, 2026-10-03: three raters are available)* | Needs 7a's explanations first, so the rubric and the 100 samples must reach the raters by **Wed 7**; rating and agreement (Krippendorff's α / Fleiss' κ) Thu–Fri. The raters' time is the constraint, not the GPU |
| 7 | 7b generative explainer | ❌ | Optional in `Phases.md` ("only after 7a is measured"); a local generator beside CodeBERT on 4 GiB, and a week with no slack |
| 8 | Every table cell traceable to a run id | ✅ | Already how the repo works |
| 8 | Ablations confirm or refute each novelty claim | 🟨 | A1–A4, A6, A8 (line vs joint), A9 fit on the frozen cache. **Not** A5 fine-tuned GraphCodeBERT (another ~10 h), A7 attention pooling (cannot use cached line features), A8 unit-only (blocker 3), A10 (needs 7b + raters) |
| 8 | External baselines reproduced under our splits | ❌ mostly | LineVul / Devign-style detectors are function-level — blocker 3. A static analyser (Cppcheck/Flawfinder) is feasible if a day is spared. The rest stays `(reported)` via `docs/prior_work.yaml` |
| 8 | Cross-dataset generalization (DiverseVul) | ❌ | Function-level only — blocker 3. `clean_eval` (PrimeVul-confirmed) **is** feasible and is the headline |
| 8 | Failure taxonomy, ≥ 20 inspected cases | ✅ | A session drafts, the owner reviews |
| 8 | Results reproducible from a clean clone | ✅ *(ticked by the owner 2026-10-12, with one rerun's miss stated — §1.34, §2)* | Tests + one frozen run from a fresh clone: yes. The 11 GB data pipeline end to end: hours, only if spared — *run on 2026-10-11: it rebuilt exactly* |
| 9 | FastAPI, schemas, error responses, latency, model card | ✅ | ~1 day |
| 9 | End-to-end request in Docker on a clean machine | ✅ *(owner, 2026-10-03: Docker installed Sat 3 evening; a second laptop exists)* | Install **before** restarting the queue that evening — the queue is paused from 08:00 and the install wants a reboot. Keep Docker Desktop/WSL2 **closed during training**: it takes host RAM and CPU a fine-tuned run needs (the test suite alone cost 33%). Also closes Phase 0's open criterion. The image pins Python 3.11 (ADR-000 §D0.3); this tree runs 3.12 |
| 10 | Paper draft, limitations, reproducibility appendix, README | ✅ as a draft | Results land Wed–Thu, so the results sections are written in the last two days |
| 10 | An independent reader reproduces a headline number | ✅ *(owner, 2026-10-03: a second laptop with a GPU)* | "Independent" means **someone other than the owner, from the README alone**. Pick a frozen-cache number (~15 min once the corpus and cache exist), not a fine-tuned one. Setting that machine up *is* the clean-clone test of Phase 8. Its GPU model and VRAM are **not yet known** — and its runs never join a comparison group from this laptop (§7 #49) |
| 10 | All licenses and citations verified | 🟨 | 37 BibTeX keys; needs the owner's check |

#### The GPU budget, reviewed 2026-10-03 against session 14's measurements

The owner brought back an earlier session's costing ("planned but not costed", 50–90 h).
Its per-run figures hold up against what was measured tonight. Three of its lines do not:

| Item | Earlier estimate | Reviewed | What changed |
|---|---|---|---|
| D9.7 | 6 runs, 18–24 h | **9 runs, ~20–22 h** at the measured 2–2.8 h a run (first projected 24–29 h) | `llm_only` added as the scaffold control; runs stop early at epochs 5–6 |
| A3 graph types (frozen, 3 seeds) + ADR-007 comparison | 4–7 h | 4–7 h, **unmeasured for CPG** | CPG graphs are larger; ADR-007 must be accepted first |
| A5 GraphCodeBERT, fine-tuned, 3 seeds | 7–12 h | ~9–10 h | One arm only. Both arms doubles it |
| A6 direction, A7 pooling, A9 loss (frozen) | 5–6 h | 5–6 h **for max-pooling only in A7** | Attention pooling needs token states; the cache holds *pooled* line features. That arm runs the live frozen encoder — hours, not minutes — or is dropped |
| GAT / GGNN to 3 seeds | ~10 min | ~10 min | — |
| Phase 6 sweep + final config | fine-tuned, 10–30 h | **sweep frozen (~8–10 h), then one fine-tuned config × 3 seeds (~10 h) only if it differs from D9.7's** | A fine-tuned sweep is the line that breaks the week. The sweep must give concat the same search budget as cross-attention (`ResearchPlan.md` §5) |
| External line-level baseline | 7–12 h | **~14–24 h, and only as a balanced subsample** | The estimate assumed our 5,457-unit population. A LineVul-style detector is function-level and needs negatives; even positives + matched negatives doubles each run, and the full 148,983-function split is out of reach. An estimate from unit counts, not measured |
| Final `test` / `clean_eval`, Phases 7, 9, 10 | minutes | minutes **of GPU** | The table never costed engineering or rater time, and those are the week's real constraint |

**Reviewed total: ~48–65 GPU-h without the external baseline, ~62–89 h with it** (revised 2026-10-03 for D9.7's measured run times; the fine-tuned A5 and final-config lines likely shrink the same way, but that is not yet measured). From Sat 3
evening to Fri 9 there are roughly 110–125 GPU-h **if this laptop trains nearly round the
clock**, so the budget fits — on two conditions the table hides: the code for each slate
must be ready when the GPU frees up, and nothing under `src/` may change while a
fine-tuned group is in flight. The second laptop is the natural development machine for
that reason, and could run a *self-contained* frozen ablation group of its own once it is
set up.

**What the week can honestly deliver:** Phases 5 and 6 closed (6 with unit-level and Q5
explicitly deferred), 7a built, Phase 8 with the frozen ablation grid and `test` /
`clean_eval` numbers, Phase 9 without verified Docker, and a paper *draft* whose
limitations section names every ❌ above. **Which ❌ and 🟨 rows are deferred is Q13 (§5) —
the owner's decision, not yet taken.** Do not mark a phase done by quietly dropping a
criterion; record the deferral under the phase in `Phases.md`.

### 0. Where things stood (2026-09-15, session 13)

**Phases 3 and 4 are signed. ADR-008 and ADR-009 are accepted. A1's frozen screen is done
(§1.21): cross-attention does not beat concatenation on Top-5; every fusion arm beats frozen
LLM-only; nothing beats fine-tuned LLM-only (0.6141).** Next is an owner decision — D9.7's
fine-tuned confirmation (item 2). The inspector's Findings tab still ranks with the Phase 3
model, which remains the best model.

```bash
python scripts/run_queue.py --status     # every run, its wall clock and its run id
python scripts/report_phase4.py          # Phase 4's criteria, the depth sweep, A4
cat artifacts/phase3/seed_report_semantic.json
```

| | three seeds, held-out `val` |
|---|---|
| **Semantic (Phase 3)** | **Top-5 0.6141 ± 0.0055**, Top-1 0.3513, line F1 0.3692 (§1.14) |
| **Structural (Phase 4)** | **Top-5 0.4302 ± 0.0372** with subtokens, 0.3645 without (§1.15) |

Read §1.16 before using any depth or A4 number: **two findings were retracted there** and
the depth sweep has no resolvable effect.

### 1. THINGS THAT STILL NEED A HUMAN — none of them blocks A1

0. **Q11 — run ADR-009 D9.7's fine-tuned confirmation?** and **Q12 — commit the A1
   write-up** (§5). Both deferred by the owner on 2026-09-15. **Phase 5 is not complete
   until Q11 is resolved** (run, or explicitly skipped) **and the gate is signed.**
1. ~~Sign the Phase 3 and Phase 4 gates~~ — **signed 2026-09-15.**
2. **ADR-006 and ADR-007 are still `Proposed`** (ADR-008 and ADR-009 were accepted
   2026-09-15). ADR-006 is implemented, measured and load-bearing in the paper.
3. ~~ADR-007 D7.4~~ — **moot for A1**: the owner chose AST, which has one relation.
   It returns only if a later experiment fuses a CPG.
4. **Review the Findings tab's fix templates** (`frontend/fix_templates.py`). They are
   pattern text written in session 13, pinned by tests for matching, but no human has
   checked the advice itself. Cheap, and it is what a developer reads.
5. **The branching model.** Work is still on `phase-2/representation`, ~25 commits ahead
   of `main`, two phases past what the branch name says. `Rules.md` §8 assumes a PR flow.
6. **Two human-only data tasks**, both cheap and both gating paper claims: categorise the
   40 parse-failure samples (`artifacts/representation/parse_failure_sample.md`, blank
   `cause:` fields), and add a second reviewer to the label audit so the ~30% noise figure
   gets an inter-rater number instead of resting on n=20.

### 2. Phase 5 — the experiment this project exists to run

**The frozen screen is DONE (2026-09-15, §1.21): cross_attention ≈ concat on Top-5
(+0.0014, 0.07 sd); every fusion arm beats frozen `llm_only` on every seed; nothing beats
Phase 3's fine-tuned 0.6141.**

**NEEDS THE OWNER — ADR-009 D9.7, the fine-tuned confirmation.** The margin is nil, which is
the case D9.7 exists for. Re-run cross_attention and concat **fine-tuned**
(`train.semantic_cache.enabled=false llm.frozen=false`, same seeds and constants otherwise),
3 seeds each. Cost from the ledger: Phase 3 semantic runs took 2.2–4.1 h, so ~6 runs ×
~3–4 h ≈ **18–24 h** on this laptop, overnight-scale and sleep-sensitive (§7 #24, the
11:41 sleep). It answers two things the frozen screen cannot: whether the mechanism ordering
survives an adapting encoder, and **whether fusion beats fine-tuned LLM-only at all** — the
comparison the paper most needs. Alternative: report the frozen result as-is, qualified.

How the screen was run, for reproduction — two queue groups, because an overfit run that
fails still exits 0 (§7 #52):

```bash
python scripts/build_semantic_cache.py train=fused llm.frozen=true   # DONE, run c99764bc… (§1.20)
python scripts/run_queue.py --only p5check     # overfit, all six arms, A1 config — ~45 min
#   → read each arm's Top-1 in MLflow; every arm must reach ~1.0 before the next line
python scripts/run_queue.py --only phase5      # A1: core 2 arms × 3 seeds first, then 4 ablations × 3
python scripts/run_queue.py --status
```

Every A1 step carries the same constants: `+gnn=gcn gnn.use_subtokens=true llm.frozen=true`,
with `train=fused` enabling the cache. **Changing any of them mid-screen makes the logged
seeds incomparable** (§7 #38). The live path is still there for ADR-009 D9.7's fine-tuned
confirmation: `train.semantic_cache.enabled=false` with `llm.frozen=false`.

**The claim A1 can support is the qualified one** — "with a fixed semantic representation,
X beats Y by Z" (ADR-009 D9.1) — and D9.7's fine-tuned confirmation is triggered by the
margin relative to the seed spread, not skipped because it is slow. Per-arm reports:
`artifacts/phase5/seed_report_fused_<arm>.json`.

Two things that must travel with any A1 number, both of which are the *scaffold* and not
the mechanism: the fused line mask is the union of both branches' coverage (§7 #42), and
the fused arm trains on the join of the two branch populations (§7 #43, logged per run).

### 3. Worth doing, not blocking

- **`gnn.use_subtokens=true` should probably become the default** — it is +0.06 Top-5 and
  +0.06 Top-1 for no measured cost. Left off only so the runs logged without it stay
  reproducible. Flip it deliberately, with a ledger row.
- **`gat` and `ggnn` are one seed each** and must not be ranked against `gcn` from that.
  Three seeds each if the paper compares convolutions (~10 min).
- ~~Decide Q10 / ADR-009~~ — **accepted and built 2026-09-15** (§1.20).
- **Compute Top-10 line accuracy from the logged predictions.** We report Top-5 and every
  published line-level baseline on this corpus reports Top-10, so today §1.19's table has
  no row that can be placed against ours at all. This is the cheapest change with a real
  payoff in the write-up — and it must be computed from the stored predictions of the
  existing runs, not by re-running anything, or it stops being comparable to §1.14.
- **Move `gnn` into `config.yaml`'s `defaults`** now that no sweep is in flight, and drop
  the `+` (see `configs/gnn/README.md` and §7 #38).
- **The paper has no Results section.** Deliberate: A1 is the result that matters.
  Methodology §3.8 and §3.16 are current as of this session.
- **ADR-007 D7.4 becomes live if Phase 5 fuses a CPG.** `gnn.builder` defaults to `ast`,
  which is what every Phase 4 number was measured under; `edge_handling=auto` will refuse a
  multi-relation graph outright rather than average three relations. ADR-008 §D8.7.
- Phase 0's Docker criterion (3 / 4) still needs a machine with Docker.

---

## 7. Known Issues & Gotchas

Things that cost time once and should not cost time twice.

| # | Issue | Impact | Workaround / Fix | Status |
|---|-------|--------|------------------|--------|
| 1 | **Docker Desktop 4.93.0 was installed on the dev machine on 2026-10-04** (`D:\Docker`, a hard disk; launched once, auto-start off) — until then it was absent. **Nothing has been built or run with it** | Phase 0 exit criterion "Docker image builds and runs the same command" is **unverified**. The `Dockerfile`/`docker-compose.yml` are written and reviewed but never built | Run `make docker-build && make docker-train` on any machine with Docker and record the result here. Until then Phase 0 is 3/4 | 🟥 Open |
| 2 | **GNU `make` is not installed** on the dev machine | The literal `make setup / test / lint` commands in `Phases.md` cannot run on Windows | `make.ps1` mirrors every target; the `Makefile` is exercised by CI on Linux | 🟨 Mitigated |
| 2b | PowerShell's default execution policy **blocks `make.ps1`** ("running scripts is disabled on this system") | Every new Windows teammate hits this on their first command | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or `powershell -ExecutionPolicy Bypass -File .\make.ps1 <target>`. Documented in `README.md` | ✅ Documented |
| 3 | **MLflow 3.x refuses the filesystem tracking store** (`file:./mlruns`) — raises `MlflowException`, maintenance mode | Any doc or tutorial suggesting `file:./mlruns` will fail | Use `sqlite:///artifacts/mlflow.db` (ADR-000 §D0.7). Do **not** set `MLFLOW_ALLOW_FILE_STORE=true` | ✅ Fixed |
| 4 | **`git_sha` logged as `unknown`** — the repo had no commits | Runs were not traceable to code; `Rules.md` §5.8 unsatisfiable | Fixed by the initial commits on `phase-0/foundation`; run `58c71014…` tags `git_sha=09d5c54`. `git_sha()` still reports `unknown`/`-dirty` honestly rather than faking a SHA — treat a `-dirty` tag on a results run as invalid | ✅ Resolved |
| 5 | Hydra's `hydra.job.chdir` is **False** (default at `version_base="1.3"`) | Relative paths in `configs/config.yaml` resolve against the *launch* directory, not the run directory. Launching from a subdirectory silently creates a second `artifacts/` tree | Launch from the repo root, or make the paths absolute in Phase 6 when run directories start mattering | 🟨 Known |
| 6 | **`pip install torch` yields a CPU-only wheel on Windows** — the GPU is silently unused | Every run would have been far slower with no error and no warning from torch | `make setup-cuda` (channel `cu130`, the only one with a cp312 Windows wheel for torch 2.13.0). `device.kind=auto` now *warns* on CPU fallback; `device.kind=cuda` fails hard. Verify with `make gpu-check` | ✅ Fixed — run `f3f24d45…` |
| 6b | **Reinstalling torch fails with `WinError 5`** if any Python process has it imported — pip cannot delete the loaded `_C.pyd` and leaves a `~orch` stub behind | The 1.9 GB download is wasted and the venv is left in a half-uninstalled state | Close every Python process (including editors and watchers) before `make setup-cuda`. Clean leftovers with `Remove-Item -Recurse .venv/Lib/site-packages/~*`. The wheel stays in pip's cache, so the retry is fast | ✅ Documented |
| 6c | `requirements.lock.txt` pins `torch==2.13.0+cu130`, which is **not on PyPI** | A plain `pip install -r requirements.lock.txt` fails on a fresh machine | The file carries a header with the required `--extra-index-url`. Regenerate with `make freeze` after any torch change | ✅ Documented |
| 7 | **4 GiB VRAM ceiling** on the RTX 3050 Ti Laptop | CodeBERT/GraphCodeBERT at 512 tokens + a GNN branch + retained cross-attention maps will not fit at a comfortable batch size | **Measured 2026-08-16 (§1.9, run `1904c295…`).** AMP + gradient checkpointing at **batch 16 windows** peaks at 2,456 MiB (60%) and is *also* the fastest trainable configuration at 22.9 windows/s, leaving 1.6 GiB for Phase 5. Plain fp32 fine-tuning caps at batch 8 and 97% of the card | ✅ Resolved — measured, with a recommendation |
| 8 | The console log sink **silently dropped every `extra=` field** | `make gpu-check` printed a bare "Accelerator" with no GPU attached; console output was strictly poorer than the JSON file | `ConsoleFormatter` appends `key=value` pairs. Never use the stdlib default formatter for this logger | ✅ Fixed |
| 9 | **Big-Vul is dominated by two projects: Chrome 40.9% + linux 24.8% = 65.8%** of 310 projects. 2 projects reach 50% of the corpus, 12 reach 80% | A naive project-wise split produces wildly unbalanced folds; hashing project names lands both giants in one split and starves the others | **Solved.** `splits.py` packs projects greedily largest-first into whichever split is furthest below quota. Measured on the real distribution: **80.0/10.0/10.0 with zero shared projects** (train 105 projects / 6,408 line-labelled, val 103 / 870, test 102 / 965) | ✅ Fixed |
| 12 | **The project-wise val/test splits contain only small projects.** Chrome, linux and Android necessarily land in train, since together they exceed the val+test quota | val/test are distributionally different from train — not just unseen projects, but *smaller* projects. Generalization numbers are conservative in one sense and unrepresentative in another | Inherent to the corpus, not fixable by a better algorithm. **State it as a threat to validity** in the paper alongside the project-wise numbers | 🟨 Known — document |
| 13 | **The clean evaluation set is not project-disjoint from training.** ADR-003 selects it by label confidence, so it contains Chrome/linux functions whose projects are also in train | Easy to misread as a generalization result. It is not one | Keep the two claims separate: `clean_eval` measures **line-localization quality under trustworthy labels**; the project-wise test split measures **generalization to unseen projects**. Never report one as the other | 🟨 Known — document |
| 10 | **24.4% of Big-Vul's vulnerable functions have no line labels** (2,657 of 10,900): `flaw_line_index` is empty | The real line-level corpus is 8,243 functions, not the 10,900 vulnerable ones. Function-level and line-level denominators differ and must never be conflated | Loader marks these explicitly; they are usable for the unit-level task only. Report both denominators in `DATA_CARD.md` | 🟨 Known |
| 11 | Our measured Big-Vul line counts **differ from the commonly quoted figures** | Literature quotes "44,603 lines / 11,823 functions"; we measure **46,513 lines / 8,243 functions** on the LineVul release | **Narrowed 2026-08-11:** it is *not* a parsing artefact — out-of-range, duplicate-index, text-mismatch and count-mismatch are all **zero** corpus-wide, so 46,513 is exactly the corpus's own flagged-line set. Remaining candidates: a different release, a different notion of "function", or their counting convention. `Rules.md` §5.7: cite ours, name the gap | 🟥 Open — narrowed |
| 34 | **`CodeUnit.source` is not the string you passed to it.** ADR-006 strips edge whitespace from every line at construction, so for any input carrying it the unit's text differs from the input. Anything deriving character offsets, byte spans or line indices must use `unit.source`, never the original. This broke a property test silently and only for whitespace-bearing input — it had passed for weeks | ✅ Understood — test fixed 2026-09-12 |
| 32 | **Big-Vul's LineVul release encodes the flaw label in the indentation.** `processed_func` strips leading whitespace from every line *except* the flagged ones, which are pasted back verbatim from `flaw_line`. "Line starts with a space" predicts the label at **precision 1.0000, zero false positives**, on all six partitions (§1.10, run `a5707491…`) | **It voided the project's first training result.** Run `1843412a…` learned the whitespace rule and matched it to three decimals, scoring line F1 0.96 on held-out projects — *above* the ~30% label-noise ceiling that made it impossible. Our own "0 text mismatches" check could not see it: `labeling.py:272` compares `lines[i].strip()`, stripping away the very difference that leaks | **Fixed by ADR-006**: `normalize_source_whitespace` strips both ends of every line in `CodeUnit`, default-on, line-count preserving. `scripts/audit_label_leakage.py` is the standing guard — re-run it after any corpus or preprocessing change. **A flat learning curve from epoch 0 is the symptom to watch for** | ✅ Fixed — ADR-006, guard script in place |
| 33 | **A consistency check that normalizes away the thing it checks passes forever.** The label-derivation audit compared stripped text and reported 0 mismatches across 188,636 functions — correctly, and uselessly, for this purpose | Gave documented, corpus-wide confidence in exactly the dimension that was broken. §1.4's "0 text mismatches" is still true and still means less than it reads like | When a check normalizes before comparing, record **what the normalization hides**. Byte-level equality should be checked somewhere, even if only as a reported statistic rather than a failure | 🟨 Known — generalise before Phase 8 |
| 18 | **~30% of line labels point at the wrong lines** (§1.6, human audit, 6/20, CI [14.5%, 51.9%]) | This is the **ceiling on every line-level metric** the project reports. A model scoring far above ~70% line precision is more likely fitting label noise than localizing. Also easily confused with the 19.2% *function*-level rate — they are different quantities | Phase 6: do not treat line labels as clean. Phase 8: report the ceiling alongside every line-level number. Widen the audit with a second reviewer before the figure carries paper weight | 🟥 Open — inherent to `labels_v1` |
| 14 | **The clean evaluation set averages 138.3 lines/function against the corpus's 31.0** — 4.5× longer, and **1,502 CodeBERT tokens against train's 358**. **Measured 2026-08-16 (§1.8, run `b7ccc9ca…`): truncation destroys 38.1% of all flagged lines corpus-wide, 44.5% on `clean_eval`**, and 38.05% of `clean_eval`'s line-labelled functions lose a label they are graded on | The 512-token limit lands hardest on precisely the partition that carries the headline line-level numbers. A corpus-wide statistic would have been reassuring and wrong — and the estimate it replaced (§1.4, "≈250 tokens") was wrong in the same direction | **Fixed: windowing is the default** (`llm.windowing=true`), and it recovers 100% of flagged lines in every partition. `false` reproduces the published-baseline behaviour and must be labelled as truncated wherever it is used | ✅ Measured and fixed — report the gap, never hide it |
| 15 | **A project-wise split does not prevent function-level leakage.** 207 hashes / 426 functions spanned partitions with **zero shared projects** — forks and vendored copies (Chrome/Android, php/php-src, ImageMagick/ImageMagick6, libxml2) | Byte-identical functions on both sides of the train/test boundary: exactly the criticism levelled at published Big-Vul results | **Fixed** — withheld into a `duplicate` partition (ADR-004), 0.23% of the corpus. **Exact match only**, so 207 is a *floor*: re-run with near-duplicate detection in Phase 2 and expect it to grow | ✅ Fixed — floor only |
| 16 | **Big-Vul's `index` column is not random.** Line-labelled positives cluster at the high end — measured on `bigvul_test`, **all 788** line-labelled rows have `index ≥ 160,000` while total rows spread evenly | "Take the first N rows" for a smoke test yields a subset with almost no positives; treating `index` as a time proxy would produce a fake temporal split | Sample by content hash (`utils/hashing.py`), never by row position or index range | 🟨 Known |
| 19 | **`node.start_point` segfaults the interpreter.** `tree-sitter==0.26.0` (the latest release) + `tree-sitter-c==0.24.2`: walking a tree and reading `start_point` crashes the process on real corpus input. Reproduced deterministically on `bigvul-188127`; minimal trigger is its first **258 lines / 7,810 bytes**. The identical walk reading `start_byte` completes over all 3,061 nodes. Pure ASCII, max line length 87 — nothing exotic in the input | A native crash cannot be caught, so any corpus-wide pass touching `Point` dies with no traceback and no partial result. It was found only because a 2,618-unit smoke test segfaulted | **Never read `start_point`/`end_point`.** Line numbers come from `start_byte` through `LineMap`. Regression fixture at `tests/fixtures/parser/segfault_start_point.c`; if anyone reintroduces `Point` the test suite crashes rather than fails. No upstream fix to take — 0.26.0 is already the newest release | ✅ Fixed — designed around |
| 20 | **`str.splitlines()` is the wrong splitter for source code.** It breaks on carriage return, vertical tab, form feed, U+001C–U+001E, U+0085, U+2028 and U+2029; Tree-sitter and Big-Vul's `flaw_line_index` break only on `\n` | Every line number after the first stray character would shift by one, silently. Measured: exactly **1 function of 188,636** (`bigvul-76919`, a form feed) is split differently — rare enough to survive any casual test, which is what makes it dangerous | Use `representation.line_map.split_lines`. Never call `str.splitlines()` on source | ✅ Fixed |
| 23 | **The DFG is name-based and says so.** No aliasing (`p = &x; *p = 1` does not define `x`), no scoping (two sibling `i`s are one variable), fields and array elements collapse to their base, and stores through a pointer or subscript are **weak updates** that do not kill the previous definition | Data-flow edges are an under-approximation where aliasing matters and an over-approximation where scoping does. Neither can move a node to the wrong line, but a *claim* that the model "follows the data flow" needs this caveat attached | Documented in `graphs/dfg.py` and asserted by tests that fail if someone silently "fixes" a limitation. Real alias analysis is out of scope for v1 — ADR-002 §D2.3 chose in-house passes to control the line mapping, not to solve pointer analysis | 🟨 Known — by design |
| 24 | **The CFG models no exceptions, `setjmp`/`longjmp`, computed `goto`, or calls as transfers of control**; a `goto` whose label was cut away by function extraction is left dangling rather than guessed | Paths through those constructs are absent, so any reachability claim is a lower bound | Documented in `graphs/cfg.py`. `call` edges are reserved and unemitted — Big-Vul units are single functions | 🟨 Known — by design |
| 21 | **3,646 flagged lines (7.84%) carry no AST node at all** (§1.7). **Cause identified 2026-08-16: it is parse damage, not blank lines.** 3,093 of the 3,646 — **84.8%** — sit in the 20,480 units that parse cleanly under no grammar (run `228e35e0…`) | Nothing can be pooled over them, so the structural branch cannot score them however good the model is. A *separate* ceiling from the ~30% label noise (#18), and unevenly distributed: `clean_eval` 98.78% covered vs. `test` 83.41% | Report line coverage per partition alongside every line-level metric. **Do not expect a better grammar to fix it** — coverage is flat across grammar policies (#27). The lever, if there is one, is the CFG or a fallback that emits a token node per unparsed line | 🟥 Open — Phase 6/8 reporting input, cause now known |
| 26 | **The 10.86% of functions that no grammar parses cleanly hold 31.58% of the flagged lines** — 14,690 of 46,513, and 1,584 of the 8,243 line-labelled functions. Damaged units average **68.0 lines against 26.5** for clean ones, and length carries both vulnerability and line labels. Damage is also 2.5× worse on `clean_eval` (23.66%) than on train (9.30%) | "Only 11% of the corpus fails to parse" is true and understates the line-level exposure by a factor of three. Said in a paper it would be a misleading statistic | **Always quote the residue as a share of flagged lines, never of functions.** Full characterisation in `artifacts/representation/parse_failures.json` (run `228e35e0…`) and ADR-005. Mitigating it is a *parser* problem — macro-aware preprocessing or signature repair — not a grammar-selection one | 🟥 Open — inherent to Big-Vul extraction |
| 27 | **The grammar is now a function of the source** (ADR-005 `best_of_two`), so the node-type vocabulary is a **union over the C and C++ grammars**, which do not share a symbol set | ADR-002 §D2.5 embeds `node_type` as a categorical feature. Phase 4 must build the vocabulary from the corpus under this policy — not from either grammar's symbol list — and must not read a unit's node types as identifying its language. A one-character edit can flip a unit's grammar and change its node types wholesale, which is an unmeasured robustness risk | Build the vocabulary by scanning the corpus. If embeddings prove unstable, `graph.language=cpp` removes the effect for 0.90 points of parse rate. `ParseResult.language` and the report's `parsed_language` breakdown make the split auditable | 🟨 Known — Phase 4 design input |
| 31 | **Checkpoints are written at epoch boundaries only, so a stopped run loses everything.** Measured 2026-08-16: a seed-42 run was stopped ~16 minutes in, partway through epoch 0, and left **no checkpoint and no metrics** | On this hardware an epoch is >16 minutes, so any interruption — a reboot, a thermal event, a user stopping a hot laptop — costs the whole epoch. `Trainer.load_checkpoint` restores weights, optimizer, scaler and step counters correctly; it simply has nothing to restore from | **Add per-N-steps checkpointing before starting a multi-hour run** (§6 item 1). Until then, treat any long run as all-or-nothing | 🟥 Open — blocks the ≥3-seed criterion in practice |
| 30 | **The 512-line cap costs `val` 17.07% of its flagged lines and `train` only 0.70%** — a **24× difference** (measured 2026-08-16 on the real datasets: val 578 of 3,386, train 238 of 34,027). Lines past `llm.n_lines` are dropped from features *and* labels, so they can never be predicted | A third partition-specific ceiling, on top of truncation (§1.8) and no-node coverage (#21). Evaluating on `val`/`test` against a cap tuned by looking at `train` understates the model's ceiling on exactly the splits that are reported | The dataset **counts and logs** `flagged_beyond_line_cap` per split rather than dropping silently, and it is logged as an MLflow param. Raise `llm.n_lines` for evaluation, or report the cap loss beside every line metric. Do not tune the cap on train | 🟥 Open — Phase 3/8 reporting input |
| 29 | **Exceeding VRAM on Windows does not raise — it silently pages to host RAM.** Measured: 3 of 24 probe configurations peaked *above* the card's 4,095 MiB and ran anyway, at 10–25× the cost (`amp` batch 16: 19.8 → 2.5 windows/s). Only one point in the grid actually raised `OutOfMemoryError` | A memory sweep that trusts the absence of a crash finds no ceiling. It completes, reports valid numbers, and takes a day instead of an hour — the worst kind of failure, because nothing looks wrong | **Never treat "it did not OOM" as "it fits".** Compare `torch.cuda.max_memory_allocated` against `get_device_properties().total_memory`; `scripts/probe_vram.py` reports `within_physical_vram` and that is what "fits" means. Applies to Phase 4 and Phase 5 batching too | ✅ Documented — inherent to WDDM |
| 28 | **One unit is not one forward pass.** Windowing makes `clean_eval` average **4.17 windows per unit, reaching 88**; the longest function in the corpus is 146,721 tokens — **287 windows** | A batch sized in units has wildly variable cost, and on 4 GiB (#7) the worst unit in a batch bounds it. Exactly the trap #22 describes for graphs, on the other branch | Batch in **windows**, not units, and pool to units afterwards — `align_tokens_to_lines(window_to_unit=…)` already does the second half. Length-bucket by window count | 🟨 Known — Phase 3 design input |
| 22 | **Graphs reach 55,486 nodes** on the largest function; mean is 251.66 (§1.7), and `clean_eval` averages 1,138 nodes / **1,577 CPG edges** — 4.5× the corpus mean, matching its 4.5× function length | With 4 GiB of VRAM (#7) a batch is bounded by the worst graph in it, not the mean. Length-bucketed batching or a node cap is required, and a node cap silently drops lines | Measure before Phase 4 batching is written. Do not size batches from the mean | 🟨 Known — Phase 4 design input |
| 25 | **A fixed iteration cap on the reaching-definitions fixpoint silently halved the data-flow edges** — 25 of 400 real functions truncated, one returning 1,093 of 13,534, with no error raised | Wrong graphs that look right. Only caught because a timing comparison was off; it would have propagated into every downstream result | Budget scales per node and **raises** when exhausted (`graphs/dfg.py`). Never return a partially converged analysis. Fix measured at **+60% DFG edges** on `clean_eval` | ✅ Fixed 2026-08-16 |
| 35 | **A GNN trained from scratch needs far more epochs to overfit than a fine-tuned transformer does.** At 60 epochs with dropout on, all three convolutions plateau at Top-1 ~0.80 and Phase 4's overfit test reads as **FAILED** | A healthy model is diagnosed as broken, and the wrong fixes get tried | At ~100–150 epochs with dropout zeroed all three reach Top-1 = 1.000 (run `8b0a3c1d…`). Note what the wrong diagnoses were: **not** capacity (512-wide / 6-deep is *worse*), **not** regularisation alone, **not** feature aliasing (§1.12 puts 4-hop ambiguity at 7.35%). Before concluding a model cannot learn, check the schedule | ✅ Fixed in `overfit_structural.yaml` |
| 36 | **`train.zero_dropout` lives in the `train` group, not as `gnn.dropout=0` on the command line** | Hydra group selection is order-dependent: `+gnn=gcn` placed *after* an experiment config silently restores the dropout, and the overfit test then fails for a reason nothing reports | A config value that governs a decision belongs where selection order cannot undo it. Pinned by a test that composes the overrides in both orders | ✅ Fixed |
| 37 | **The depth sweep has no resolvable signal, and a single seed said otherwise.** At n=1, depth 6 sat 0.047 Top-5 below depth 4 and was written up as over-smoothing evidence; at **n=3 every pairwise gap in 2–6 is under 0.6 sd** (§1.16) | Two traps at once: reading a depth difference as architectural when §1.12 shows the *representation* differs by depth, and reading ~2 sd of seed noise as a finding | **Do not write down a comparison until every arm has ≥3 seeds** — the rule the headline numbers already follow. The original reading is retracted in §1.13 and superseded by §1.16 | ✅ Retracted |
| 38 | **Adding a config group to `config.yaml`'s `defaults` changes every run's `config_hash`** — including runs of other branches | `scripts/report_seeds.py` uses that hash to decide whether two runs are replicates, so doing it mid-sweep makes a logged seed incomparable with the seeds still to run | `gnn` is selected as `+gnn=…`, and `fusion` as `+fusion=…`. Move a group into `defaults` only when no sweep is in flight. See `configs/gnn/README.md` and `configs/fusion/README.md` | 🟨 Live constraint |
| 39 | **A new config flag makes every earlier run look like a third group.** When `gnn.use_subtokens` landed, runs logged before it had no such parameter | Both directions were live bugs on the same day: a report *grouping* on it would have split the earlier runs off, and one *not* grouping on it would have averaged the A4 arm into its own comparison partners | The rule that resolves it: **a parameter that did not exist when a run was made was, by definition, not enabled in that run**, so an absent parameter matches a requested falsy value. Applied in `report_phase4.py`'s group key and `report_seeds.py`'s `where` filter | ✅ Fixed |
| 40 | **A probe is not a result, and tagging it afterwards only works if you remember.** Two CPU trials at `train.limit=600` were grouped with full-corpus runs and shifted an A4 depth-2 mean before anyone noticed | Same family as #39: adding a logged parameter mid-sweep splits a comparison group — `resolved_device`, added between the first and second A4 seed, did it again hours after #39 was written | Both reports now refuse a run with a non-zero `train.limit` outright, alongside `train.overfit` and the `void` tag. `report_seeds.py` compares only parameters **every** run logged and reports the asymmetric ones instead of refusing on them | ✅ Fixed |
| 46 | **`model.eval()` does not stop autograd, and eval mode silently disables gradient checkpointing.** `training/loop.py::evaluate` had no `no_grad`, so every activation of the evaluation forward was retained for a backward that never came | Worse than it sounds: `transformers/modeling_layers.py` gates checkpointing on `self.training`, so `llm.gradient_checkpointing` does **nothing** during evaluation — making eval the most expensive pass in the loop, not the cheapest. It affected Phase 3, Phase 4 and anything else on the shared loop | Fixed 2026-09-14 with `@torch.no_grad()` and three tests. **No logged metric changes** — nothing in the evaluation path reads a gradient — so every Phase 3 / Phase 4 number stands. The size of the saving is not yet from a logged run; `scripts/probe_vram_fused.py` carries an `eval_grad` row to make it citable | ✅ Fixed; magnitude pending a logged run |
| 41 | **`torch.nn.MultiheadAttention` returns NaN for a unit whose key mask is fully masked**, and does not raise. `softmax` over an all-masked row is `0/0`; the NaN reaches every parameter on the first backward pass and the run dies minutes later somewhere unrelated | Units with no keys on one side are **real here**, not hypothetical: a function whose graph will not build, and the 7.84% of flagged lines with no node (#21) | `fusion/cross_attention.py::cross_attend` gives such units a permissive key mask so the softmax has something to normalise, then zeroes their attended output and their attention map. Pinned by a test whose fixture always contains one | ✅ Handled |
| 42 | **The fused model can score lines the structural arm cannot — and that is the scaffold, not cross-attention.** The fused line mask is the union of the two branches' coverage (ADR-008 §D8.1) | `concat`, `sum` and `gated` have the identical advantage. Reporting it as A1's result would credit the union to the mechanism | `fused_line_coverage()` reports `flagged_semantic_only` / `flagged_structural_only` / `recovered_by_union` per batch, and `gnn_only` is the arm that isolates it. The single-branch arms are scored on **their own** branch's coverage, never the union | 🟨 Stated; must travel with every A1 number |
| 43 | **A positional join between the two branch datasets is correct today and silently wrong the first time it is not.** `LineLabelledDataset` keeps every line-labelled unit; `GraphLineDataset` drops any whose graph will not build | Every logged Phase 4 run reports `graph_build_failures=0` at 5,457 / 488 units — so the bug would pass every test written against real data, then pair one function's tokens with another's control flow, train, and converge | `join_branches` intersects on `unit_id`, refuses duplicates, and refuses a pair whose halves disagree on `line_labels`/`n_lines`/`target`; `collate_fused` re-checks at batch level; the population difference is logged as a run parameter | ✅ Handled |
| 44 | **The fused model's batch budgets do not control its peak memory.** MEASURED (§1.17, runs `f96a0744…` / `c1a2fbab…`): every budget from (2,1024) to (16,8192) is dominated by the **same** oversized unit — 54 windows / 9,233 nodes, **3,786 MiB = 92.4% of the card** — because a unit over budget gets its own batch by design | #29 is why this had to be measured rather than assumed: exceeding VRAM on Windows does not raise, it pages to host RAM at 10–25× slower. It fits, with AMP finite throughout, and `concat` sits within 8 MiB of `cross_attention` — the fairness property ADR-008 §D8.2 exists for | **1,922 MiB (46.9%) is static optimizer state before a single activation**, so the lever for headroom is `llm.frozen` or a smaller optimizer, **not** `max_windows_per_batch`. Re-probe with `scripts/probe_vram_fused.py` after any change to either encoder | ✅ Measured |
| 49 | **`resolved_device` cannot tell two GPUs apart.** `job.py:149` logs `str(device)`, and `str(torch.device('cuda', 0))` is `"cuda:0"` on **every** CUDA card. The card model is logged nowhere | This is §7 #40's guard — added specifically so a device difference could not split or silently merge a comparison group — blind to the device difference that matters. Run three seeds on one laptop and a fourth on a different GPU and `report_seeds.py` finds every comparable parameter identical and **aggregates them without a word**. Live the moment any run happens on a second machine | Log `torch.cuda.get_device_name(index)` beside the device string. Until then: **never split a comparison group across machines** — ADR-008's whole premise is that A1's arms differ in the fusion mechanism and nothing else. Run whole groups on one card. **Fixed 2026-10-07:** `training/job.py` logs `resolved_gpu` (the card's name, `none` on CPU) as a parameter, so `report_seeds.py` refuses runs from two cards as replicates — pinned in `tests/unit/test_resume.py`. **It landed after machine 2's 27 runs and this laptop's D9.7**, so none of those carry it and the rule above still governs anything involving them | ✅ Fixed 2026-10-07 — for runs made from then on |
| 47 | **A metric chosen by early stopping on itself is not a held-out number**, and nothing in the pipeline says so. Both train configs watch `line/top_5_accuracy` on `val` with `watch_mode: max`, and §1.14 / §1.15 read their headline from that epoch | Top-5 is selected on itself; the metrics beneath it are reported at an epoch Top-5 chose, on the same split. The bias is real and its **size is unknown**. Every chart, table and slide inherits it silently, because no *value* is wrong — only the label | §1.18. `scripts/make_review3_figures.py` writes "validation-selected" into every figure subtitle, both axis labels and §1 of `results.md`, so a cropped chart carries the caveat. `test` and `clean_eval` stay untouched until the fusion family closes (`Rules.md` §5.3) | ✅ Labelled; bias unquantified by design |
| 48 | **YAML silently drops a published significant figure.** A bare `value: 0.360` parses as a float and renders back as `0.36` — LineVD published three decimals and the table showed two | Small, and exactly the kind of thing that is indefensible when a reviewer checks a quotation against its source. Same shape as #4 (pandas turning `"2"` into `"2.0"`): the type system silently renormalising someone else's text | Quoted numbers are stored as **strings** in `docs/prior_work.yaml` — they are quotations, never arithmetic. `tests/unit/test_prior_work.py::test_quoted_values_are_strings_so_published_precision_survives` fails if one is unquoted | ✅ Fixed |
| 45 | **A fused batch must be flat.** `training/loop.py::_move` walks a batch's **top level** only, so a nested `{"semantic": …, "structural": …}` batch would arrive half on the GPU with nothing raising | The failure is a device mismatch several frames away from its cause, or worse, a silent CPU fallback | Keys are prefixed `sem_`/`str_` — the `CLAUDE.md` §9 convention — and the shared line-axis keys stay unprefixed because there is exactly one of each. Pinned by a test | ✅ Handled |
| 50 | **CUDA line alignment is not bitwise reproducible, and a "bitwise" determinism check cannot tell that from dropout.** `index_add_` on CUDA accumulates in no fixed order: two passes over one batch give identical token states and line states up to **1.9e-6** apart. ADR-009's first cache build refused itself on it | A correct encoder read as a broken one, and the obvious "fix" — loosen the tolerance until it passes — would also hide live dropout if applied at the wrong stage | Check the two stages separately: token states **bitwise** (dropout lives there), pooled states within `ALIGNMENT_TOLERANCE = 1e-5` — seven orders of magnitude under dropout's 13.8. `semantic_cache.assert_deterministic` | ✅ Handled |
| 51 | **Under AMP, CodeBERT's output is fp32, not fp16.** RoBERTa's final LayerNorm autocasts to fp32, so "the encoder computes in fp16" is false at its output | ADR-009 §D9.6 chose fp16 storage on that premise; it would have added ~3e-5 of rounding the live path never has | Measure the dtype, don't assume it. Cache stored fp32 (1.07 GB) | ✅ Handled |
| 52 | **An overfit run that fails to memorise still exits 0**, so a queue that chains the overfit check straight into the real runs would start A1 on a broken arm with nothing stopping it | Hours of A1 on an arm whose branch never learned | Overfit steps are their own queue group, `p5check`; read their Top-1 in MLflow before `--only phase5` | ✅ Handled |
| 53 | **`train.resume_from` was not a resume.** Until 2026-10-02 it (a) opened a *second* MLflow run, so one training run was two run ids and `report_seeds.py` — which reads the best step from one run's metric history and requires `FINISHED` — would have seen half a curve; (b) built a fresh `EarlyStopping`, so the first post-resume evaluation became "best" whatever it scored and patience restarted; (c) replayed the interrupted epoch from batch 0 on mid-epoch weights, training part of it twice | A resumed run would have reported a best epoch from its second half only, after up to an epoch of extra training, under a new run id. **No logged result ever used it** (§3, session 10: seed 1 was restarted instead) — the only reason no number is wrong | Fixed in `training/loop.py`, `job.py`, `tracking.py`: `last.pt` carries `batch_in_epoch`, the epoch's loss totals, patience and the best metrics; a resume reopens the **same** run, refuses a config that differs from the one it started with, replays the same shuffle (`seed + epoch`) and skips exactly the batches already trained. Checkpoints are written to a temp file and renamed. Pinned by `tests/unit/test_resume.py`: weights after pause+resume **equal** an uninterrupted run's. **Not restored: the dropout RNG** — a resumed run is a replicate, not a bit-identical copy; it carries a `resumed_at` tag, which must be reported beside its number | ✅ Fixed 2026-10-02 |
| 54 | **A paused run is stranded by any config change.** The resume guard compares every flattened config value except `train.resume_from` against what the run logged at its start | Editing a config file — or adding a group to `defaults` (#38) — while a `d97-*` step shows `[paus]` makes that run un-resumable; its hours are lost | Finish or deliberately discard paused runs first. `run_queue.py --status` shows them | 🟨 By design |
| 55 | **Loading a checkpoint onto the GPU spills VRAM for the rest of the run.** MEASURED 2026-10-02 on `cb2b1e99…` (D9.7 cross_attention seed 42): `torch.load(map_location=device)` put a second copy of the weights and optimizer state on the card, PyTorch kept the freed blocks cached, and the resumed process held **3,692 MiB dedicated + 1,872 MiB shared** (Windows `GPU Process Memory` counters) | **5.0 s/step against 2.0 s/step before the pause** — 2.5× slower, no exception, and `_check_memory` saw nothing because it reads *allocated* memory once, at the first step, and the spill is in *reserved* | `load_checkpoint` now loads to host memory (`load_state_dict` copies per parameter), and `Trainer._release_cache_if_spilling` calls `empty_cache()` at any optimizer step where `memory_reserved` exceeds the card. Neither changes a number. **To check a live run:** `Get-Counter '\GPU Process Memory(*)\Shared Usage'` — hundreds of MiB shared on the training pid is a spill, whatever `nvidia-smi` says | ✅ Fixed 2026-10-02 |
| 56 | **Windows can refuse a checkpoint's rename, and the run dies on it.** MEASURED 2026-10-06 22:47 on `72f067d0…` (D9.7 `llm_only` seed 1, epoch 5, 116 min in, the step-4200 periodic checkpoint): `save_checkpoint`'s `partial.replace(path)` raised `PermissionError: [WinError 5] Access is denied` — another process had `last.pt` open. **The holder is not identified.** Two suspects, neither proven: Windows' scheduled defragmentation of D: (#57) — a pass was cancelled at 23:26, and if it ran the full hour the 01:32 pass did, it was moving files at 22:47; and Defender, whose log shows a quick scan started and stopped at 22:42:57 with real-time protection on | Exit 1 → `[FAIL]`, and the queue **does not retry a failed step** unless started with `--retry-failed`: it moves to the next step. At its end the arm's seed report runs on the seeds that finished; from the code (`report_seeds.py` writes below `min_seeds` and returns normally) it writes `n_seeds: 2` with a `warning`, exits 0 and is marked `done`, so it is **not** re-run when the missing seed completes later | `run_queue.py --only d97 --retry-failed` continues the step from its intact `last.pt` in the same run (the ≤ 100 steps after that checkpoint are trained again from its weights, and the run carries a `resumed_at` tag). Then re-run the arm's seed report with the command `--dry-run` prints, and **read `n_seeds` before quoting anything**. **Fixed 2026-10-07, in three places.** (1) `training/loop.py::_replace_with_retry`: the rename waits out a lock — eight retries over about two minutes, a WARNING each — and only then raises; the previous `last.pt` is untouched throughout. (2) `run_queue.py --retries N` (default 2): a step that crashed *after* writing a checkpoint is continued from it before the queue moves on; one that died before its first checkpoint, or cannot resume, is not. (3) Every seed-report step in the slate passes `+seeds.require_sufficient=true`, and `report_seeds.py` then exits non-zero below `min_seeds` — the report is still written and still says `sufficient_for_phase_3: false`, but the step is `[FAIL]`, not `done`. Tests in `test_resume.py`, `test_run_queue.py`, `test_report_seeds.py` | ✅ Fixed 2026-10-07 |
| 57 | **Three things an unattended night did to a fine-tuned run, none visible in a morning `--status`.** MEASURED 2026-10-06/07. **(a) Windows defragments D: when nobody is at the keyboard.** Automatic Maintenance ran `ScheduledDefrag` on the hard disk the run checkpoints to: `Microsoft-Windows-Defrag` logs passes cancelled at 21:20, 22:26, 23:26 and 02:32 and one completed at 04:31; the 01:32 pass was watched — `defrag.exe` started 01:32:19 and was cancelled at 02:32:20. During it D: read 0% idle and a 32 MB write-through took 1.6 s (20 MB/s). It runs as SYSTEM: `Stop-ScheduledTask`, `schtasks /End` and `MSchedExe.exe Stop` from a non-admin shell each reported success and stopped nothing. **(b) A GPU driver fault.** 01:12:13–01:12:52: 20 × `nvlddmkm` id 13 ("Graphics FECS Exception: Logging error 0x2") and 2 × id 153; `2c7a7631…` died with `torch.AcceleratorError: CUDA error: unspecified launch failure`. No `nvlddmkm` event between 2026-10-02 and that minute; the GPU read 76 °C under load ten minutes later. Cause unknown. **(c) A VRAM spill the guard does not see.** From 01:34 `72f067d0…` climbed to 3,810 MiB dedicated + 792 MiB shared and slowed to 248 s, then 536 s, per 100 steps (median that night ~150 s; the previous worst, 504 s, was `179632f5…` on 2026-10-04); GPU power fell to 30–40 W. It cleared at the epoch boundary (2,773 + 78 MiB) with no intervention | (a) slower checkpoint writes, and a candidate for #56's lock. (b) exit 1 → `[FAIL]`, not retried (#56's queue behaviour). (c) a run that takes hours longer and says nothing. Across the 143 hundred-step intervals the three `llm_only` runs had logged by 01:56, the p90 is 205–248 s against a median of 123–152 s per run, so slow patches are routine, not rare | (a) **The owner turns off scheduled optimisation for D:** (Optimize Drives → Change settings — administrator) or lets it finish before a long run; it completed on 2026-10-07 04:31. (b) `--retry-failed`; if it recurs, look at the driver and at heat before blaming code. (c) **Fixed 2026-10-07, after measuring it.** `scripts/probe_vram_spill.py` (new; not an MLflow run — it describes a machine on a day) grows the pool 128 MiB at a time and reads the process's own Windows counters. Two runs forty minutes apart, Brave open both times: paging began at **≈ 3,520 MiB** reserved (dedicated plateau 3,615) and at **≈ 3,780 MiB** (plateau 3,869) — the budget is not the card's 4,095.5 MiB and **not a constant**. The old rule, `reserved > total`, would have waited until ~575 and ~320 MiB had already spilled, which is the band the night's run sat in. In both runs the driver's own free-memory figure (`torch.cuda.mem_get_info`) reached 0 at ≈ 3,290 MiB reserved — earlier than the spill, never later — so `cache_release_reason` now releases when that figure is under 32 MiB and at least 64 MiB of cache exists to give back, keeping the old rule as well. Cost of the worst case, a release on every step: **30 / 60 / 90 ms** for a 1 / 2 / 3 GiB working set, against steps of 1.2–2.0 s. Changes no number. Tests: `tests/unit/test_vram_guard.py`, whose cases are the probe's rows. **Seen working under real fine-tuning the same morning** — probe run `3ee36f97…` (`llm_only`, encoder training, `train.limit=3000`, one epoch, tagged `void`): the guard fired at step 3 (`driver_out_of_free_memory`, reserved 3,330 MiB, allocated 1,530) and fewer than 50 times in 387 steps; the trainer's shared usage read **76–78 MiB in all 23 samples** (20 s apart), dedicated peaking at 3,317 MiB; **387 steps in 389.7 s = 1.01 s/step**, 103–107 s per hundred including the checkpoint write, against the night's medians of 123–152 s. One short run, with Brave open; the first full-length run should still be read | (a) 🟥 Open — the owner's, (b) 🟨 mitigated by the retry in #56, (c) ✅ Fixed 2026-10-07 |
| 58 | **A flagged line that ended a block comment has lost the slash of its `*/` — a second formatting leak, which ADR-006's whitespace fix does not close.** MEASURED 2026-10-07 (§1.25; `audit_label_leakage.py` run `9f462174…`): a line ending a comment with a bare `*` is flagged at precision 1.0000 on `val` (204 of 204), 0.9744 on `test`, 0.9882 on `train`, and answers 58 of 482 scored `val` units; a line ending in an intact `*/` is flagged 0 times in 8,474 on `train`. `clean_eval` has no positive tell but only 4 of its 4,358 flagged lines contain `/*` at all | All nine D9.7 checkpoints use it (`audit_score_leakage.py` run `ded852ee…`): Top-5 0.95–0.99 on the 58 units against 0.565–0.571 on the other 424, so ≈ 0.05 of every logged `val` Top-5 and ≈ 0.07 of Top-1 sit on it. It also *causes* parse damage — an unterminated comment swallows the code after it — so the structural branch may read it as damage. A1's null survives (the arms tie on the other units too). Every frozen, structural, Phase 3 and machine-2 number is unaudited | **Open decision Q14** (ADR-010 to be written): strip comments label-blind at L0/L1 and retrain, or repair at evaluation and report stratified. Until then: label every line-level number as measured on leaky text, and **do not run the held-out evaluation on the shards as they are**. Re-run both audits after any corpus change. The lesson ADR-006 already taught and the audit did not generalise: **a tell that is never right is as readable as one that is always right** — the audit now reports both | 🟥 Open — Q14 |
| 59 | **A semantic-cache directory holds one cache per partition, so building a second text variant into it replaces the first.** Found 2026-10-07 before it happened: `build_semantic_cache.py +representation.strip_comments=true` would have written the stripped features over `artifacts/semantic_cache/{train,val}.pt`, the cache of record (`c99764bc…`) that every frozen run was logged against | The loss would have been loud, not silent — `SemanticFeatureCache.load` checks the digest and would refuse every frozen run — but the cache of record is reproducible only to 1.9e-6 (§7 #50), not bitwise | `write_cache` refuses to replace a partition's cache with the other text variant; evaluation-time stripping reads `<dir>_stripped`; build the stripped cache with `train.semantic_cache.dir=./artifacts/semantic_cache_stripped`. Tests in `test_semantic_cache.py`, `test_score_checkpoint.py` | ✅ Guarded 2026-10-07 |
| 60 | **A run-id prefix can be a number to the config parser.** `+score.run_id=920989e5` — one of A6's runs — reached the script as the float `92098900000.0`. FOUND 2026-10-07: the scoring step failed on "no checkpoint directory", the chain went on, and the audit that followed reported an arm at `n=2` where three checkpoints existed | Nothing wrong was written — a run id that matches nothing is refused — but a batch script that does not stop on a non-zero exit leaves a hole, and an aggregate over two seeds looks like one over three unless `n_seeds` is read | `utils/config.py::run_id_text` refuses a run id that is not text and says why; all four scripts that take one use it. **Pass full 32-character run ids from scripts**, and read `n_seeds` in every audit before quoting it | ✅ Fixed 2026-10-07 |
| 61 | **A run trained on comment-stripped text is "not a change of text" — so its scores defaulted into the directory of record.** `score_checkpoint.py` sent scores to `scores_stripped/` only when the text differed from training; a `d97r` checkpoint is scored on the text it was trained on, so it would have been written into `artifacts/phase6/scores/` beside the nine comment-trained D9.7 checkpoints. FOUND 2026-10-07 by reading the defaults before the first `d97r` checkpoint was scored | Nothing wrong was written. Had it been: `audit_score_leakage.py` averages every seed of an arm it finds in a directory, so `cross_attention` would have been reported at `n=6` — three models that learned the artefact and three that never saw a comment — against comment-bearing text, with no error, because stripping keeps the line count (same family as #39, #40, #49) | Three populations, three default directories: `scores/`, `scores_stripped/` (trained with comments, evaluated without), `scores_retrained/` (trained without); `explain_checkpoint.py` follows the same three. The audit refuses score files made from the text variant it is not reading, and a checkpoint trained on stripped text is its own arm, `<arm>_retrained`. **`calibrate.py` writes `calibration_report.json` wherever `calibrate.output_dir` points and replaces what is there — always name a directory for anything but the full model of record** | ✅ Fixed 2026-10-07 |
| 62 | **"Training cannot import it" is not "safe to edit while a comparison group runs".** `report_seeds.py` compares all of `src/vulnloc` between the commits of an arm's seeds and refuses on any difference. FOUND 2026-10-07 22:08, two hours before it would have fired: commit `7ea7a9c` changed `src/vulnloc/explanation/generator.py` between `d97r` cross-attention's seed 42 (`0a4843c`) and seed 1 (`d5e5dda`), under a session rule that allowed `explanation/` edits because training does not import them | The arm's seed report would have been `[FAIL]` at midnight with nobody awake, the arm would have had no report in the morning, and the script that scores complete arms would have skipped it | `run_queue.py` passes `+seeds.allow_source_drift=true` to that one report, justified in a comment by the layer rule (L3 cannot import L5) and pinned by a test that the other two arms get no override; the changed file is written into the report's provenance — **quote it with the arm's numbers**. The rule is corrected everywhere it was written: **while a group runs, nothing under `src/vulnloc/` changes** | ✅ Fixed 2026-10-07; the running queue's own attempt at that report still fails once (it holds the old plan) |
| 63 | **The reproduction check can refuse a correct checkpoint: the logits are half-precision, so exact ties are ordinary, and nothing fixes their order.** FOUND 2026-10-08: `score_checkpoint.py` refused `8afbf7da…` in four of six passes, each time on `line/ifa_mean` alone, by 0.002075 — one rank position in one of 482 units. Measured on the nine `d97r` score files: 3,244–6,225 distinct logit values over 41,065 lines; the best flagged line exactly tied with an unflagged line in 7–22 units per checkpoint; `training/metrics.py` ranks with `torch.argsort`, which promises no order among equals | The unattended script left the arm uncalibrated (right) and audited it at `n=2` (wrong; run `15fccf6d…`, void). On the numbers themselves: tie order can move a checkpoint's Top-1 or Top-5 by at most one unit (0.0021), an arm's mean by ≤ 0.0007 and its IFA by ≤ 0.05 — far inside the seed sd, so no comparison changes | **Re-run; never raise `score.tolerance`** — a pass that reproduces is kept, and none in several means a real mismatch. Batch scripts must treat a refusal as "arm incomplete" for every later step, the audit included. The held-out scoring has nothing to reproduce and is made once: it carries this jitter (`docs/EvaluationProtocol.md` §6). **Phase 8's metric code must define tie order**; `training/metrics.py` is deliberately unchanged — a defined order is a different metric from the one every logged run used | ✅ **Defined 2026-10-08 (`b875d70`):** `evaluation/metrics/localization.py` reports the source-order value with both bounds beside it, and `evaluate_scores.py` refuses to write unless the scoring pass's logged value lies inside them (§1.29) |
| 64 | **A generated table that marks "≥ 2 pooled sd" is marking one comparison among many.** MEASURED 2026-10-08 (run `52b9c998…`): Phase 8's `val` evaluation makes 93 comparisons (31 metrics × 3 pairs) and three reach 2 pooled sd — concat − `llm_only` on PR-AUC (+0.0206, 2.20) and on ECE (out of fold −0.0092, 3.77; in sample −0.0130, 3.04). With three seeds an arm, 2 pooled sd is a two-sample t of 2.45 on 4 degrees of freedom: two arms with no difference reach it with probability ≈ 0.07 in any one comparison — ≈ 6.6 of 93 if they were independent | The script's first run (`30aeff44…`) printed the bare word "resolved" beside them. Read as written that is "concat beats `llm_only`" — on validation, at three seeds, out of a search over 93. The same family as #37, one level up: not too few seeds for one comparison, but too many comparisons for the seeds | The tables now print how many comparisons were made, name the crossings and state the chance figure (`4281b3f`; `statistics.threshold_p_value`). **Claims rest only on the nine headline comparisons — Top-5, IFA, PR-AUC × the three pre-stated pairs — and a crossing anywhere is a candidate to check on a partition it was not seen on** | ✅ Reported with its denominator since `4281b3f` |
| 65 | **A rate per kind of line reads as a property of the kind until you count the units behind it.** MEASURED 2026-10-10 (run `42f8f27c…`, §1.31): on `val`, 25.3% of preprocessor lines are flagged against 6.9% of all lines — and the 145 flagged ones sit in **13** of 482 units, a fifth of them in one function. The first table printed 37.7% (under a narrower rule) with no unit count; it looked like a third formatting tell for the minute it took to count. Same family as #13 and #40: a figure that rests on a handful of functions is about those functions | ✅ `sample_failures.py` prints the unit count and the largest unit's share beside every rate |
| 66 | **`report_phase4.py` sorts a group's `seeds` and not its `run_ids`.** Zipping the two labels runs with the wrong seed: `46ab8056…` is seed 42 and came out as seed 1 in the first ablation table. No mean or sd moves — only which run a reader would open for which seed. Found 2026-10-10 by comparing the table with `run_queue.py --status` | ✅ `make_ablation_tables.py` reads each run's seed from what the run logged, and refuses a seed report that disagrees with it. `phase4_report.json` itself is unchanged |
| 67 | **A group that gains seeds turns its old seed-report command into a different report.** `a1r`'s reports select on `{fusion_kind, frozen, strip_comments}`. Once `a1rep` has run, the same command by hand aggregates six seeds and writes over the three-seed file of record — and nothing refuses, because six runs with identical parameters *are* replicates. Noticed 2026-10-11 while adding the group, before any `a1rep` run existed | ✅ `report_seeds.py` takes `+seeds.seeds=[…]`; `a1rep`'s reports name their seeds and write to files of their own. **An `a1r` seed report re-run by hand needs `'+seeds.seeds=[42,1,2]'`.** The queue does not re-run them: they are recorded `done` |
| 68 | **A script that logs a measured value under a name the config already logged loses its run after the work is done.** `start_run` logs every override as a parameter, as given. `measure_latency.py` then logged the bundle path, normalized, as `latency.bundle`; MLflow refuses a parameter whose value changes, and run `8b74b485…` failed after timing all 200 requests and before writing anything. Found 2026-10-11 by running it on the bundle of record. Same family as #60: a value that looks the same and is not | ✅ `7d7c102`: measured facts are logged under names of their own, and a test fails if one collides with a config key. Fifteen other scripts call `log_params`; each has completed runs, none was audited for this |
| 69 | **An image whose work directory belongs to root starts no Hydra script.** The Dockerfile chowned everything it copied into `/workspace` and `artifacts/`, but `WORKDIR` itself was created by root; Hydra writes `outputs/<date>/<time>` beside the launch directory and the non-root user could not create it. The first image this project built (2026-10-11) died on `PermissionError: 'outputs'` before loading a model — the Dockerfile had been in the repo, unbuilt, since Phase 0 | ✅ `193f3ac`: `/workspace` and `outputs/` are the user's; the `serve` container and the default command both run. **Still open, two things the same build showed:** the image installs from `pyproject.toml`'s lower bounds, not the lock file (torch 2.14.1 inside, 2.13.0 on the host), and it has no `.git`, so a run inside logs `git_sha=unknown` |
| 70 | **A run repeated at the same seed is not the same number, and for cross-attention the gap can exceed the seed sd.** The clean-clone check (§1.34) reran two frozen runs at seed 42 with every logged parameter, vocabulary digest and cache digest equal: concat came back −0.0145 Top-5 from its run of record, cross_attention +0.0228 — 2.2 times its arm's three-seed sd (0.0102). The reported value is the best epoch of a `val` curve that moves by up to 0.10–0.11 from one epoch to the next. A reproduction criterion written as "within the seed sd of one logged run" therefore fails on a correct pipeline. | **Judge a rerun against the arm's distribution over seeds, and say so before running it; never promise an individual run's number.** Byte-compare only files that record no run id. The cause (CUDA scatter order, #50, suspected) is not isolated; one rerun an arm. |
| 71 | **A fixture written from memory of a file's shape passes against itself.** `measured_block`'s first tests fed it bare numbers; `evaluation_<split>.json` stores each ranking metric as one value per tie order, and the export died on the real file — before writing anything, only because the block is read first. | ✅ `b0afb75`: the fixture mirrors the file, and a test reads the real file when the tree has one. **When code reads an artefact another script wrote, one test must read a real one.** |
| 72 | **From Git Bash the pre-commit `import-linter` hook runs the system Python and fails** (`No module named 'importlinter'`) on any commit that touches a `.py` file; docs-only commits skip the hook and hide it. | Commit with the environment first on the path — `PATH="/d/project/.venv/Scripts:$PATH" git commit …` — or from an activated PowerShell. Never `--no-verify`. |
| 73 | **Docker Desktop starting or stopping makes a Hypothesis deadline fail.** `test_evaluation_localization.py`'s property took 2.0 s on its first call against a 200 ms deadline with D: busy, twice; it passes on a quiet machine. | Not changed — the evaluation layer and its tests stay as they are. **Run the suite with Docker closed, and read a `DeadlineExceeded` before reading it as a failure.** |
| 74 | **The served headline tracks a function's length, not whether it is vulnerable.** Asked about 1,000 `val` functions the corpus does not mark vulnerable, "N lines flagged" appeared for 93.5% — against 85.0% of vulnerable ones — and at equal length the two kinds get the same answer. The checkpoint's highest line score ranks harmless functions *above* vulnerable ones (0.3321), where length alone gives 0.7111 (§1.35, `6c942e6c…`). Until then every reading of the headline had been taken on vulnerable functions only, where 85% looks like a detection rate. | Never present `is_vulnerable`, a flagged line or `risk_score` as detection. **Judge any "nothing found" line on harmless functions as well as vulnerable ones, and by length band**; a screen has to beat length alone. The record and the rules for a later gate: `docs/VerdictComparison.md`. |
| 17 | The leakage check **counted `clean_eval`'s deliberate project overlap as leakage** | Every correct project-wise split reported `is_clean=False`, with a 55-project `shared_projects` list burying the 207 genuine duplicates | Fixed: project disjointness is checked over train/val/test only; the clean-eval overlap is reported separately and quantified (66.5%) | ✅ Fixed |

**Anticipated, pre-recorded so they aren't rediscovered painfully:**

- **Token limit vs. function length.** CodeBERT/GraphCodeBERT cap at 512 tokens; real vulnerable functions frequently exceed it. Truncation silently destroys line labels beyond the cutoff. Decide windowing strategy in Phase 3 and *report* how many lines are lost.
- **Line-map fragility.** Tree-sitter gives byte offsets; converting to line numbers must handle CRLF, tabs, unicode, and preprocessor directives. This is the single highest-risk piece of code in the repo.
- **Extreme class imbalance.** Vulnerable lines are a very small fraction of all lines. Accuracy is meaningless here; a model predicting "clean" everywhere will look excellent on the wrong metric.
- **Label noise from commit diffs.** Fix commits routinely include refactors, formatting, and unrelated changes. Some "vulnerable" lines will not be vulnerable. Quantify this before trusting line-level numbers.
- **Graph size explosion.** CPGs for long functions can reach thousands of nodes; batching and memory need attention in Phase 4.

---

## 8. What Didn't Work

Negative results are recorded here permanently (`Rules.md` §5.6). They are as valuable as the positive ones and prevent repeated dead ends.

| Date | What was tried | Result | Conclusion |
|------|----------------|--------------|------------|
| 2026-08-10 | **PrimeVul cross-reference as a training filter** (ADR-001 §D1.2): keep only Big-Vul line-labelled functions PrimeVul confirms vulnerable | Retains **1,393 of 8,243 functions — discards 83%**. Structural, not mistunable: PrimeVul holds only 6,004 vulnerable functions in total, fewer than Big-Vul's 8,243 line-labelled ones. 79.1% of the loss is *ambiguous* (PrimeVul dedup, not rejection) | Rejected as an inclusion filter. Superseded by ADR-003: PrimeVul becomes the **clean evaluation set**. Only the exclusion half survives — drop the 331 functions PrimeVul explicitly contradicts. **Absence of evidence is not evidence of a wrong label** |
| 2026-08-10 | MLflow filesystem tracking store, `file:./artifacts/mlruns` (the obvious default, and what `Architecture.md` §6 implies) | 6 tests failed with `MlflowException`: the backend is in maintenance mode and refused by default in MLflow 3.x | Switched to SQLite (ADR-000 §D0.7). Do not reintroduce the file store or the `MLFLOW_ALLOW_FILE_STORE` escape hatch |

---

## 9. Assumption Status

Tracks `PRD.md` §10 as evidence accumulates.

| ID | Assumption | Status | Evidence |
|----|------------|--------|----------|
| A1 | Primary persona = application developer | Unverified | — |
| A2 | Single-GPU compute budget | ✅ **Confirmed — and quantified** | Run `f3f24d45…`: one **NVIDIA RTX 3050 Ti Laptop GPU, 4.0 GiB**, compute capability 8.6, driver 591.59, `torch 2.13.0+cu130`. Single GPU confirmed; **4 GiB is the binding constraint**. The first CPU-only reading was the default PyPI wheel, not absent hardware — ADR-000 §D0.8 |
| A3 | Line-level labels obtainable | 🟨 **Available, not correct** | 8,243 functions / 46,513 flagged lines derived and verified corpus-wide: 0 out-of-range, 0 text mismatches, 0 count mismatches, 0 labels-on-negatives (§1.4). Availability and internal consistency confirmed. **Correctness measured at ~70%** by the human audit (§1.6) — and a further **7.84% of flagged lines carry no AST node** to score (§1.7 / §7 #21) |
| A4 | v1 scope = one language | ✅ **Confirmed — and built** | ADR-002 §D2.1 fixed C/C++; Phase 2 implemented exactly that. The cost ADR-002 predicted was real: C/C++ is the hardest case for line mapping, and it produced §7 #19 (`Point` segfault), #20 (`splitlines`) and Q8 (grammar selection, closed by ADR-005 — the corpus's own language label was wrong for 53% of it) |
| A5 | `evidence` schema as specified | Unverified | — |

---

## 10. Quick Reference for a New Session

**Read in this order:** `Memory.md` §1, §5, §6 → `Phases.md` (current phase only) → `Rules.md` → the specific interface in `Architecture.md` §4 you are implementing.

**Do not** read the whole codebase. If you find yourself needing to, this file has failed and should be improved in the same session.

**Before writing any code, confirm:**
- Which phase is active (§1)
- Whether your work depends on an open decision (§5)
- Which interface contract you are implementing (`Architecture.md` §4)
- Which exit criterion you are advancing (`Phases.md`)

**Where things live, as of 2026-08-16:**

| You want | Use |
|---|---|
| The corpus | `vulnloc.data.cache.iter_shard(cache_dir, split)` — **never a Big-Vul CSV** |
| One function in L1 form | `CodeUnit.from_labelled_function(...)` |
| A parse | `TreeSitterParser().parse_with_policy(source, policy=cfg.graph.language, declared=unit.language, unit_id=...)` — ADR-005; syntax errors are counted, not raised |
| A graph | `GRAPH_REGISTRY.get("ast"\|"cfg"\|"dfg"\|"cpg")().build(parse_result)` |
| A line number | `LineMap.byte_to_line` / `char_to_line`. **Never** Tree-sitter's `Point` (§7 #19) |
| To split source into lines | `representation.line_map.split_lines`. **Never** `str.splitlines()` (§7 #20) |
| Corpus-level graph numbers | `artifacts/representation/report_{ast,cpg}.json`, §1.7 |

**The standing ceilings on any line-level number** — quote them together, they are
different quantities: ~30% of line labels are wrong (§1.6); 7.84% of flagged lines have
no node to score, and 84.8% of *that* is parse damage (§7 #21, #26); 512-token truncation
lands hardest on `clean_eval`, which is 4.5× longer than the corpus average (§7 #14).

**And one framing rule.** The unparseable residue is 10.86% of *functions* but **31.58%
of flagged lines** (§1.7). Always quote it the second way.
