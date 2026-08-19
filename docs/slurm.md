# Running ATOM on Slurm (many datasets in parallel)

ATOM runs one dataset per invocation of `atom run`. To train models for **many
datasets** on an HPC cluster, run them as a **Slurm job array** — one array task
per dataset, all scheduled in parallel across the cluster.

This guide covers the idiomatic pattern, the three things you must get right on a
cluster (thread bounding, offline compute nodes, and the shared meta-KB), and
ready-to-run example scripts in [`scripts/slurm/`](../scripts/slurm/).

---

## TL;DR

```bash
# 1. On the LOGIN node (has network + shared FS): build the dataset packages
bash scripts/slurm/prepare.sh          # pack CSVs / fetch Kaggle -> $DATA/pkgs

# 2. Submit one array task per dataset (N = number of data lines in the manifest)
N=$(($(wc -l < scripts/slurm/datasets.tsv) - 1))
sbatch --array=1-"$N"%16 scripts/slurm/atom_array.sbatch

# 3. (optional) After the array finishes, merge the per-task meta-KBs
sbatch --dependency=afterany:<ARRAY_JOBID> scripts/slurm/merge_kb.sh
```

Each dataset lands in `$RESULTS/<name>/<name>-<timestamp>/` with a deployable
ONNX model package + provenance.

---

## Why a job array

- **One `atom run` per dataset** — a job array (`--array=1-N`) gives each dataset
  its own task, its own resource allocation, and its own log, all scheduled
  independently. This is exactly what Slurm's array feature is for.
- **Throttle with `%`** — `--array=1-200%16` keeps at most 16 tasks running at
  once, which protects the shared filesystem and meta-KB from a thundering herd.
- **Isolation** — array tasks don't share memory or CPUs, so ATOM's per-run
  determinism and budget honesty are preserved.

The alternative — a single job that loops over datasets sequentially — is fine
for a handful of small datasets (§6) but wastes the cluster for more than that.

---

## The manifest

Drive the array from a plain TSV, one line per dataset. Example
[`scripts/slurm/datasets.tsv`](../scripts/slurm/datasets.tsv):

```
name         package                target        task          time_budget  max_rows
credit       pkgs/credit             default        classification 300          2000000
housing      pkgs/housing            SalePrice      regression     300          2000000
segments     pkgs/segments                          clustering     180          2000000
sensors      pkgs/sensors                                          600          10000000
```

- Tab-separated, with a header row (skipped).
- Empty `target` → unlabeled data (the `--target` flag is omitted).
- Empty `task` → let ATOM infer the family (the `--task` flag is omitted).
- Array index `k` runs the dataset on data line `k` (`SLURM_ARRAY_TASK_ID`).

---

## The three cluster essentials

### 1. Bound ATOM's threads to the allocation

ATOM uses BLAS (numpy/scikit-learn) and joblib (`RandomForest`/`ExtraTrees`/
`IsolationForest` run with `n_jobs=-1`). By default these libraries grab **all
cores on the machine**, not just the ones Slurm gave your task — which
oversubscribes a shared node. Pin them to `--cpus-per-task` in the batch script:

```bash
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK:-1}   # bounds joblib/n_jobs=-1
```

This is the intended division of labor: **Slurm owns resource limits; ATOM uses
what the job is given.** ATOM does not (and should not) self-limit threads.

### 2. Compute nodes are usually offline — prepare packages first

`atom fetch kaggle:<slug>` needs the network; most compute nodes don't have it.
Do all `fetch`/`pack` on the **login node** as a prep step, writing ADPs to
shared storage. The compute-node array then runs fully offline (`atom run` needs
no network). See [`scripts/slurm/prepare.sh`](../scripts/slurm/prepare.sh).

### 3. The shared meta-KB — give each task its own, merge later

ATOM's meta-KB (`--kb`) is an append-only `records.jsonl` that warm-starts future
runs from similar past ones. Many array tasks appending to the **same** file
concurrently can interleave and drop records. Two safe options:

- **Per-task KB (recommended):** `--kb "$KBROOT/task-$SLURM_ARRAY_TASK_ID"`.
  No write contention. After the array finishes, concatenate them into the
  shared KB (records are versioned JSONL, so `cat` merges cleanly) — see
  [`scripts/slurm/merge_kb.sh`](../scripts/slurm/merge_kb.sh). Future batches then
  warm-start from the merged history.
- **Read-only shared KB:** point every task at a pre-built shared KB and discard
  in-batch writes. Simplest when you don't need cross-dataset learning within a
  batch.

Also give each dataset its **own `--out`** (`--out "$RESULTS/<name>"`); run dirs
are timestamped, so this only keeps results tidy.

---

## The array script

[`scripts/slurm/atom_array.sbatch`](../scripts/slurm/atom_array.sbatch):

```bash
#!/bin/bash
#SBATCH --job-name=atom-array
#SBATCH --output=logs/atom_%A_%a.out     # %A=array job id, %a=task id
#SBATCH --error=logs/atom_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# submit with:  sbatch --array=1-<N>%16 scripts/slurm/atom_array.sbatch
set -euo pipefail

# --- paths (edit for your cluster) ---
ATOM_ROOT=${ATOM_ROOT:-$HOME/atom}
MANIFEST=${MANIFEST:-scripts/slurm/datasets.tsv}
RESULTS=${RESULTS:-$HOME/atom/runs}
KBROOT=${KBROOT:-$HOME/atom/kb-array}
export ATOM_HOME=${ATOM_HOME:-$HOME/atom/home}
export PATH="$ATOM_ROOT/bin:$PATH"        # installer launcher; or `source .venv/bin/activate`

# --- bound threads to the allocation ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK:-1}

# --- pick this task's dataset (data line = header + task id) ---
# Parse with `cut`: `IFS=$'\t' read` collapses adjacent tabs and would drop the
# empty target/task fields of unlabeled/clustering rows.
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
[ -n "$line" ] || { echo "no manifest line for task $SLURM_ARRAY_TASK_ID"; exit 2; }
name=$(cut -f1 <<<"$line"); package=$(cut -f2 <<<"$line"); target=$(cut -f3 <<<"$line")
task=$(cut -f4 <<<"$line"); budget=$(cut -f5 <<<"$line"); maxrows=$(cut -f6 <<<"$line")

outdir="$RESULTS/$name"
if [ -f "$outdir/.done" ]; then echo "$name already done — skipping"; exit 0; fi

# --- optional flags ---
targ=(); [ -n "${target:-}" ] && targ=(--target "$target")
tsk=();  [ -n "${task:-}"   ] && tsk=(--task "$task")

echo "[$name] package=$package budget=${budget}s cpus=${SLURM_CPUS_PER_TASK:-?}"
atom run "$package" "${targ[@]}" "${tsk[@]}" \
    --time-budget "${budget:-120}" --max-rows "${maxrows:-2000000}" \
    --out "$outdir" --kb "$KBROOT/task-$SLURM_ARRAY_TASK_ID" \
    --seed 0 --yes

touch "$outdir/.done"    # idempotency marker for re-runs / requeue
echo "[$name] complete -> $outdir"
```

Notes:
- `--yes` skips the confirm gate (required for non-interactive jobs; ATOM also
  auto-proceeds when stdin isn't a TTY, but be explicit).
- The `.done` marker makes the array **safe to resubmit** — completed datasets
  are skipped, so a partial batch (preemption, timeout) can be requeued.
- Size `--cpus-per-task`, `--mem`, and `--time` to your largest dataset. `--time`
  must comfortably exceed `time_budget` (ATOM's stated budget governs *search*;
  final full-fidelity refits + ONNX export happen after and add time —
  budget for roughly 1.5–2× on large/multiclass data).

---

## Merging the per-task meta-KBs

After the array completes, fold the per-task KBs into the shared one so the next
batch warm-starts from everything learned. [`scripts/slurm/merge_kb.sh`](../scripts/slurm/merge_kb.sh):

```bash
#!/bin/bash
#SBATCH --job-name=atom-kb-merge
#SBATCH --output=logs/kb_merge_%j.out
#SBATCH --cpus-per-task=1 --mem=2G --time=00:05:00
set -euo pipefail
KBROOT=${KBROOT:-$HOME/atom/kb-array}
SHARED=${SHARED:-${ATOM_HOME:-$HOME/atom/home}/metakb}
mkdir -p "$SHARED"
cat "$KBROOT"/task-*/records.jsonl >> "$SHARED/records.jsonl" 2>/dev/null || true
echo "merged $(cat "$KBROOT"/task-*/records.jsonl 2>/dev/null | wc -l) records into $SHARED"
```

Chain it after the array with a dependency:

```bash
JID=$(sbatch --parsable --array=1-"$N"%16 scripts/slurm/atom_array.sbatch)
sbatch --dependency=afterany:"$JID" scripts/slurm/merge_kb.sh
```

---

## Collecting results

Each dataset's deployable model and metrics:

```
$RESULTS/<name>/<name>-<timestamp>/
├── model/pipeline.onnx     # deployable AMP
├── manifest.json           # signature, parity, deployable flag, lineage
├── metrics.json            # locked-test metrics
└── provenance/             # task, budget, phases, trials, seed
```

Roll up a summary table across the batch:

```bash
for m in "$RESULTS"/*/*/metrics.json; do
  python3 - "$m" <<'PY'
import json, sys, os
m = json.load(open(sys.argv[1]))
name = os.path.basename(os.path.dirname(os.path.dirname(sys.argv[1])))
print(name, m["primary_metric"], round(m["val_score_oriented"], 4),
      {k: round(v, 4) for k, v in m["test"].items()})
PY
done
```

---

## Common pitfalls

| Symptom | Cause / fix |
|---|---|
| Jobs oversubscribe / thrash a node | thread env vars not set — add the `OMP/OPENBLAS/MKL/LOKY` exports (§1) |
| `fetch` fails on compute node | no network there — pack/fetch on the login node first (§2) |
| Meta-KB records missing / garbled | many tasks writing one `--kb` — use per-task KB + merge (§3) |
| Job killed at time limit | `--time` too low for `time_budget` + finalize/export — raise `--time` (≈1.5–2× budget) |
| `atom: command not found` in job | add the launcher to `PATH` or activate the venv in the script |
| Re-submitting reruns everything | rely on the `.done` marker (or delete it to force a rerun) |

---

## 6. Simple sequential alternative (small N)

For a few small datasets, one job that loops is simpler than an array:

```bash
#!/bin/bash
#SBATCH -c 8 --mem=16G -t 02:00:00 --output=logs/atom_seq_%j.out
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK
export ATOM_HOME=$HOME/atom/home; export PATH="$HOME/atom/bin:$PATH"
# `IFS= read -r line` keeps the raw line; `cut` then preserves empty fields.
tail -n +2 scripts/slurm/datasets.tsv | while IFS= read -r line; do
  [ -n "$line" ] || continue
  name=$(cut -f1 <<<"$line"); pkg=$(cut -f2 <<<"$line"); target=$(cut -f3 <<<"$line")
  task=$(cut -f4 <<<"$line"); budget=$(cut -f5 <<<"$line"); maxrows=$(cut -f6 <<<"$line")
  targ=(); [ -n "$target" ] && targ=(--target "$target")
  tsk=();  [ -n "$task" ]   && tsk=(--task "$task")
  atom run "$pkg" "${targ[@]}" "${tsk[@]}" --time-budget "${budget:-120}" \
      --max-rows "${maxrows:-2000000}" --out "$HOME/atom/runs/$name" --yes
done
```

A shared `--kb` is safe here because runs are sequential (no concurrent writes),
so each dataset warm-starts from the previous ones in the same job.
