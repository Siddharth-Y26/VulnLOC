# Training on a second machine, side by side

Added 2026-10-03 (session 14). The owner's second laptop (**6 GB GPU**) takes training load
off the first one **without its runs ever entering a comparison with the first one's**. A
session started on the second laptop reads this file first, then `docs/Memory.md` §6 item 0a.

## The one rule

**A comparison group lives on exactly one machine.** `resolved_device` logs `cuda:0` on every
card (`Memory.md` §7 #49), so `report_seeds.py` cannot tell two GPUs apart: a seed from each
machine in one group would be averaged without a word. The guard is physical:

- The second machine keeps its **own** `artifacts/mlflow.db`. **Never copy `mlflow.db`,
  `artifacts/checkpoints/` or `artifacts/queue/` between machines**, in either direction.
- It runs **whole queue groups** — every arm and every seed — each with its own reference arm.
- Its seed reports run **on it**. Its results come back as a folder (`scripts/export_results.py`)
  and are read beside the first machine's, named by machine, never pooled.
- **Always pass `--only`.** On a fresh state file the queue refuses to start without it.

## What it runs

Four groups, in this order. `check` groups are overfit gates: they exit 0 even when they fail,
so **read their Top-1 before starting the group after them** (§7 #52).

| Group | What | Runs | Estimate on a GPU like the first laptop's |
|---|---|---|---|
| `m2check` | Build GraphCodeBERT's frozen feature cache; overfit gates for GAT, GGNN and GraphCodeBERT in the fused model, frozen | 1 + 3 | ~30 min |
| `m2` | Frozen ablations around the A1 cross_attention arm, seeds 42/1/2 | 21 | ~15 min each, ~5 h |
| `m2ftcheck` | Overfit gates with the encoder **training**: CodeBERT and GraphCodeBERT | 2 | ~45 min |
| `m2ft` | **A5 fine-tuned**: CodeBERT vs GraphCodeBERT, cross_attention, seeds 42/1/2 | 6 | 2–2.8 h each, ~12–17 h |

About 19–24 h in all; a 6 GB card is likely faster, but that is unmeasured. Fine-tuned steps
pause and resume exactly like the first machine's (`--until`, `--stop`, `Memory.md` §7 #53).

The variants, each changing exactly one thing against its group's reference:

| Step | Ablation | Change |
|---|---|---|
| `m2-reference-seed*` | — | The frozen A1 cross_attention arm, re-run here |
| `m2-gat-seed*`, `m2-ggnn-seed*` | A4 in fusion | GAT / GGNN instead of GCN |
| `m2-bce_weighted-seed*` | A9 | Weighted BCE instead of focal |
| `m2-bce_plain-seed*` | A9 | Unweighted BCE |
| `m2-line_only-seed*` | A8 | `train.function_weight=0` |
| `m2-graphcodebert-seed*` | A5, frozen | GraphCodeBERT's cached line features instead of CodeBERT's |
| `m2ft-codebert-seed*` | A5, fine-tuned — reference | D9.7's cross_attention config, re-run here |
| `m2ft-graphcodebert-seed*` | A5, fine-tuned | GraphCodeBERT instead of CodeBERT |

Two of these double as reproductions, read side by side and never pooled:
`m2-reference` against the first machine's frozen A1 cross_attention (0.5408 ± 0.0181 Top-5,
`Memory.md` §1.21), and `m2ft-codebert` against D9.7's cross_attention arm. Expect close, not
identical (§7 #50).

**Verified on the first machine before shipping (2026-10-03):** every command composes; argument
order does not change any resolved config; GraphCodeBERT shares CodeBERT's tokenizer, so the
line axis is unchanged. The GraphCodeBERT cache build and its frozen overfit gate were also
run on the first machine, offline, as a code-path check — see `Memory.md` §3, session 14.

## 1. What to transfer — one folder, ~2.7 GB

The first machine prepares `D:\vulnloc_transfer\`. Copy that whole folder to the second laptop
(USB drive or network share):

| In the folder | What it is | Size |
|---|---|---|
| `vulnloc.bundle` | The whole repository at the commit to run | small |
| `COMMIT.txt` | That commit's id, to check against | — |
| `artifacts\data\cache\` | Preprocessed shards — everything training reads | 202 MB |
| `artifacts\data\splits\` | The split (fingerprint `0ade7483…`) | 3.5 MB |
| `artifacts\representation\` | Node and subtoken vocabularies | < 1 MB |
| `artifacts\semantic_cache\` | CodeBERT's frozen line features | 1.0 GB |
| `hf\models--microsoft--codebert-base\` | CodeBERT weights and tokenizer | 0.95 GB |
| `hf\models--microsoft--graphcodebert-base\` | GraphCodeBERT weights and tokenizer | 0.48 GB |
| `START_HERE.txt` | The commands below | — |

Not included, on purpose: `mlflow.db`, `checkpoints\`, `queue\`, `logs\`, and the 13 GB raw
corpus (training never reads it). The GraphCodeBERT cache is not shipped either; `m2check`
builds it on the second machine.

The copy does not have to be trusted: each semantic cache refuses itself unless the split
fingerprint, shard digest, encoder revision and every other input match (ADR-009 D9.4), and
every run logs its vocabulary digests.

## 2. Setup on the second laptop

PowerShell. `<T>` is where the transfer folder landed; the project can live anywhere.

```powershell
git clone <T>\vulnloc.bundle project
cd project
git checkout phase-2/representation
git rev-parse --short HEAD          # must equal <T>\COMMIT.txt

py -3.12 -m venv .venv              # install Python 3.12 first if `py -3.12` is missing
powershell -ExecutionPolicy Bypass -File .\make.ps1 setup        # needs the internet once
powershell -ExecutionPolicy Bypass -File .\make.ps1 setup-cuda   # plain pip gives CPU-only torch on Windows
powershell -ExecutionPolicy Bypass -File .\make.ps1 gpu-check    # must name the 6 GB GPU

robocopy "<T>\artifacts" ".\artifacts" /E
robocopy "<T>\hf" "$env:USERPROFILE\.cache\huggingface\hub" /E
```

## 3. Run

```powershell
$q = ".\.venv\Scripts\python.exe scripts\run_queue.py"

iex "$q --only m2check m2ftcheck"     # ~75 min: cache build + five overfit gates
iex "$q --only m2check m2ftcheck --status"
```

**Read the five overfit Top-1 values** in MLflow (`.\.venv\Scripts\mlflow.exe ui
--backend-store-uri sqlite:///artifacts/mlflow.db`, metric `eval/line/top_1_accuracy`, last
epochs). Every one must reach ~1.000. If one does not, stop there and open a session.
Then, unattended:

```powershell
iex "$q --only m2 m2ft"               # ~17–22 h; frozen first, fine-tuned after
iex "$q --only m2 m2ft --until 08:00" # the same, pausing cleanly at 08:00
iex "$q --stop"                       # pause now, from another terminal
iex "$q --only m2 m2ft --status"      # progress, with Top-5 / Top-1
```

Re-running the same command resumes a paused fine-tuned run where it stopped. Do not edit
`src\` or `configs\` on this machine while a group is running.

## 4. Seed reports — on the second machine, once each variant has three FINISHED seeds

Each filter must select exactly one variant; if it selects two, the comparability guard
refuses, which is the correct outcome.

```powershell
$r = ".\.venv\Scripts\python.exe scripts\report_seeds.py train=fused +seeds.output_dir=./artifacts/m2"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, gnn_kind: gcn, loss: focal, train.function_weight: 0.3, model_name: microsoft/codebert-base}' +seeds.tag=m2_reference"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, gnn_kind: gat}' +seeds.tag=m2_gat"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, gnn_kind: ggnn}' +seeds.tag=m2_ggnn"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, loss: bce, train.auto_pos_weight: True}' +seeds.tag=m2_bce_weighted"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, loss: bce, train.auto_pos_weight: False}' +seeds.tag=m2_bce_plain"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, train.function_weight: 0}' +seeds.tag=m2_line_only"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, model_name: microsoft/graphcodebert-base}' +seeds.tag=m2_graphcodebert"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: False, model_name: microsoft/codebert-base}' +seeds.tag=m2ft_codebert"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: False, model_name: microsoft/graphcodebert-base}' +seeds.tag=m2ft_graphcodebert"
```

Checked 2026-10-03 with `--cfg job`: the filters parse to the strings MLflow logs (`True`,
`0.3`, `0`), dotted keys included. They were written before any `m2` run existed, so if one
matches nothing, compare it with a run's parameters in MLflow.

## 5. Getting the results back

On the second laptop — any time, as often as wanted; each export is a new timestamped folder
and nothing is ever overwritten:

```powershell
.\.venv\Scripts\python.exe scripts\export_results.py --groups m2check m2 m2ftcheck m2ft --out <USB>\results
```

It writes `<USB>\results\<host>_<timestamp>\` containing:

| File | What |
|---|---|
| `RESULTS.md` | One table: every step's status, run id, minutes, attempts, Top-5 / Top-1 / line F1 / IFA, plus the GPU, commit, Python and torch versions |
| `manifest.json` | The same, machine-readable |
| `runs\<run_id>\summary.json` | Each finished run's best-epoch record |
| `reports\` | The seed reports from step 4 |
| `logs\`, `queue_state.json` | The queue's own record |
| `mlflow_<host>.db` | A copy of the store, for `mlflow ui` only |

Add `--with-checkpoints` only when a model has to move (about 1.5 GB per fine-tuned run, e.g. to
evaluate it on `test` on the first machine).

On the **first** machine, copy that folder to `D:\project\artifacts\machine2\` — inside
`artifacts\`, so git ignores it, and **outside** `artifacts\mlflow.db` and
`artifacts\checkpoints\`, so nothing merges. To browse its runs:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///artifacts/machine2/<folder>/mlflow_<host>.db --port 5001
```

A session writing results up cites second-machine numbers as **"machine 2 (<GPU from
RESULTS.md>)"** with their run ids, beside — never inside — the first machine's.

## 6. A second slate, proposed 2026-10-07 — the frozen A1 screen on comment-stripped text

**Not started on either machine. The owner decides whether it runs, and where** (open
decision Q14, `Memory.md` §5). It exists because of the label leak found that day
(`Memory.md` §1.25): every frozen arm with a structural branch reads it, and frozen
`llm_only` does not, so "fusion beats frozen `llm_only`" has to be re-measured on text
without it. It is 18 frozen runs — about the size of this machine's first slate — and one
whole comparison group, so it must run on **one** machine from start to finish.

It needs the repository at a commit that has `strip_comments` (2026-10-07 evening or
later), so a new `vulnloc.bundle`; nothing else in the transfer folder changes. Then:

```powershell
git pull <T>\vulnloc.bundle phase-2/representation     # in the existing clone
$q = ".\.venv\Scripts\python.exe scripts\run_queue.py"

iex "$q --only a1rcheck"      # builds the stripped CodeBERT cache in its OWN directory
                              # (artifacts\semantic_cache_stripped), then one overfit gate
iex "$q --only a1rcheck --status"
```

**Read the overfit Top-1** as in step 3 — last epochs, ~1.000 — before going on. Then:

```powershell
iex "$q --only a1r"           # six arms x three seeds, then six seed reports in artifacts\adr010\
.\.venv\Scripts\python.exe scripts\export_results.py --groups a1rcheck a1r --reports-dir artifacts\adr010 --out <USB>\results
```

The stripped cache never goes into `artifacts\semantic_cache\`: a directory holds one
cache per partition, and the script refuses to replace one text variant with the other.

## While both machines work

- The first machine's `d97` group must not have source code change under it (`Memory.md` §6
  item 0a). Development can happen on the second machine's own clone; commit it there and
  bring it back with another bundle only between fine-tuned groups.
- Run each second-machine group from one commit, or its seed reports will show source drift.
- Setting this machine up is not Phase 10's independent reproduction. That needs someone other
  than the owner, working from the README alone.
