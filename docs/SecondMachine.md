# Training on a second machine, side by side

Added 2026-10-03 (session 14). The owner has a second laptop with a GPU. This file is how it
takes training load off the first one **without ever putting runs from two machines into
one comparison**. A session started on the second laptop should read this file first, then
`docs/Memory.md` §6 item 0a.

## The one rule

**A comparison group lives on exactly one machine.** `resolved_device` logs `cuda:0` on every
card (`Memory.md` §7 #49), so `report_seeds.py` cannot tell two GPUs apart: a seed from each
machine in one group would be averaged without a word. The guard is physical:

- The second machine keeps its **own** `artifacts/mlflow.db`. **Never copy `mlflow.db`,
  `artifacts/checkpoints/` or `artifacts/queue/` across**, in either direction.
- It runs **whole queue groups** — every arm and every seed — and each group carries its own
  reference arm.
- Its seed reports are run **on it**. Results meet the first machine's only side by side in a
  table, with the machine named, and are never pooled.

## What it runs

| Group | What | Runs | Cost at the first laptop's frozen pace |
|---|---|---|---|
| `m2check` | Overfit gate for GAT and GGNN inside the fused model | 2 | ~10 min each |
| `m2` | Frozen ablations around the A1 cross_attention arm, seeds 42/1/2 | 18 | ~15 min each, ~4.5 h |

The `m2` variants, each changing exactly one thing against `reference`:

| Step | Ablation | Change |
|---|---|---|
| `m2-reference-seed*` | — | The frozen A1 cross_attention arm, re-run here |
| `m2-gat-seed*`, `m2-ggnn-seed*` | A4 in fusion | `+gnn=gat` / `+gnn=ggnn` instead of GCN |
| `m2-bce_weighted-seed*` | A9 | Weighted BCE instead of focal |
| `m2-bce_plain-seed*` | A9 | Unweighted BCE |
| `m2-line_only-seed*` | A8 | `train.function_weight=0` |

`m2-reference` is the same configuration as the first laptop's frozen A1 cross_attention arm
(0.5408 ± 0.0181 Top-5, `Memory.md` §1.21). Setting the two side by side is a cross-machine
reproducibility check — expect close, not identical (§7 #50).

**Not here (yet):** anything fine-tuned. A fine-tuned fused run peaked at 3,786 MiB on its
worst unit (§1.17), so it needs a card of at least 4 GB, and ideally 6 GB or more to avoid the
spill of §7 #55. Once the second laptop's GPU is known, A5 (GraphCodeBERT, fine-tuned) and the
external baseline are the candidates to move here (§5 Q13).

## Setup

PowerShell, from the parent folder of where the project should live. Every path on the
second machine is relative to its own clone, so it does not need to be `D:\project`.

### 1. The code — at the first machine's commit

On the **first** machine (no push needed):

```powershell
git -C D:\project bundle create E:\transfer\vulnloc.bundle --all
git -C D:\project rev-parse --short HEAD   # write this down
```

On the **second**:

```powershell
git clone E:\transfer\vulnloc.bundle project
cd project
git checkout phase-2/representation
git rev-parse --short HEAD                 # must match what you wrote down
```

### 2. The environment

Python 3.12 (3.11 also satisfies `pyproject.toml`, but the first machine runs 3.12). Needs
the internet once.

```powershell
py -3.12 -m venv .venv
powershell -ExecutionPolicy Bypass -File .\make.ps1 setup
powershell -ExecutionPolicy Bypass -File .\make.ps1 setup-cuda   # plain pip gives a CPU-only torch on Windows
powershell -ExecutionPolicy Bypass -File .\make.ps1 gpu-check    # must name the GPU, not the CPU
```

### 3. The data — about 2.2 GB, copied, not rebuilt

Training reads only the preprocessed shards, the split, the vocabularies and the semantic
cache. The 13 GB raw corpus is **not** needed.

On the first machine, with the transfer drive at `E:\transfer`:

```powershell
$src = "D:\project\artifacts"; $dst = "E:\transfer\artifacts"
robocopy "$src\data\cache"      "$dst\data\cache"      /E
robocopy "$src\data\splits"     "$dst\data\splits"     /E
robocopy "$src\representation"  "$dst\representation"  /E
robocopy "$src\semantic_cache"  "$dst\semantic_cache"  /E
robocopy "$env:USERPROFILE\.cache\huggingface\hub\models--microsoft--codebert-base" `
         "E:\transfer\hf\models--microsoft--codebert-base" /E
```

On the second, from inside `project`:

```powershell
robocopy "E:\transfer\artifacts" ".\artifacts" /E
robocopy "E:\transfer\hf\models--microsoft--codebert-base" `
         "$env:USERPROFILE\.cache\huggingface\hub\models--microsoft--codebert-base" /E
```

**Do not copy** `mlflow.db`, `checkpoints\`, `queue\`, `logs\` or `data\raw\`.

The copy does not have to be trusted. The semantic cache refuses itself unless the split
fingerprint, shard digest, encoder revision and every other input match its digest
(ADR-009 D9.4), and each run logs the node and subtoken vocabulary digests. Compare those two
against a first-machine run's parameters once, by hand.

### 4. Check, then run

```powershell
.\.venv\Scripts\python.exe scripts\run_queue.py --only m2check
```

**Read each arm's Top-1 before going on** — an overfit run that fails still exits 0 (§7 #52).
Both must reach ~1.000. Then:

```powershell
.\.venv\Scripts\python.exe scripts\run_queue.py --only m2                  # add --until HH:MM if needed
.\.venv\Scripts\python.exe scripts\run_queue.py --only m2 --status         # progress, with Top-5 / Top-1
```

Write the second machine's GPU name (`nvidia-smi --query-gpu=name --format=csv`) into
`Memory.md` beside every number it produces.

### 5. Report — on the second machine

One seed report per variant, after all three of its seeds are FINISHED. Each filter must
select exactly one variant; if it selects two, the comparability guard refuses, which is the
correct outcome.

```powershell
$r = ".\.venv\Scripts\python.exe scripts\report_seeds.py train=fused +seeds.output_dir=./artifacts/m2"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, gnn_kind: gcn, loss: focal, train.function_weight: 0.3}' +seeds.tag=m2_reference"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, gnn_kind: gat}' +seeds.tag=m2_gat"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, gnn_kind: ggnn}' +seeds.tag=m2_ggnn"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, loss: bce, train.auto_pos_weight: True}' +seeds.tag=m2_bce_weighted"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, loss: bce, train.auto_pos_weight: False}' +seeds.tag=m2_bce_plain"
iex "$r '+seeds.where={fusion_kind: cross_attention, frozen: True, train.function_weight: 0}' +seeds.tag=m2_line_only"
```

Checked 2026-10-03 with `--cfg job`: the filters parse to the strings MLflow logs (`True`,
`0.3`, `0`), dotted keys included. They were written before any `m2` run existed, so if one
matches nothing, compare it with a run's parameters in MLflow (`mlflow ui
--backend-store-uri sqlite:///artifacts/mlflow.db`).

## While both machines work

- The first laptop's `d97` group must not change source code under it (§6 item 0a). The
  second machine has its own clone, so development can happen there. Commit it, and merge
  into the first machine's tree only between fine-tuned groups.
- Run the `m2` slate from one commit. If the second machine's `src/vulnloc` changes in the
  middle of the slate, its seed reports will show source drift.
- Setting this machine up is not Phase 10's independent reproduction. That needs someone other
  than the owner, working from the README alone.
