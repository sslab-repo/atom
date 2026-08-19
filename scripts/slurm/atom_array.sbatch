#!/bin/bash
# Run ATOM on many datasets as a Slurm job array — one array task per dataset.
#
#   N=$(($(wc -l < scripts/slurm/datasets.tsv) - 1))      # data lines (skip header)
#   sbatch --array=1-"$N"%16 scripts/slurm/atom_array.sbatch
#
# See docs/slurm.md for the full guide.
#SBATCH --job-name=atom-array
#SBATCH --output=logs/atom_%A_%a.out      # %A = array job id, %a = task id
#SBATCH --error=logs/atom_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail

# --- paths (override via env at submit time; edit defaults for your cluster) ---
ATOM_ROOT=${ATOM_ROOT:-$HOME/atom}
MANIFEST=${MANIFEST:-scripts/slurm/datasets.tsv}
RESULTS=${RESULTS:-$HOME/atom/runs}
KBROOT=${KBROOT:-$HOME/atom/kb-array}
export ATOM_HOME=${ATOM_HOME:-$HOME/atom/home}
# Put the launcher on PATH (installer layout). For a pip/venv install instead:
#   source /path/to/.venv/bin/activate
export PATH="$ATOM_ROOT/bin:$PATH"

# --- bound library threads to the allocation (Slurm owns resource limits) ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK:-1}   # bounds joblib / n_jobs=-1

# --- pick this task's dataset (data line = header row + array task id) ---
# Parse with `cut` (not `IFS=$'\t' read`, which collapses adjacent tabs and so
# drops empty target/task fields for unlabeled/clustering datasets).
task_id=${SLURM_ARRAY_TASK_ID:?run via: sbatch --array=1-N atom_array.sbatch}
line=$(sed -n "$((task_id + 1))p" "$MANIFEST")
[ -n "$line" ] || { echo "no manifest line for task $task_id"; exit 2; }
name=$(   cut -f1 <<< "$line")
package=$(cut -f2 <<< "$line")
target=$( cut -f3 <<< "$line")
task=$(   cut -f4 <<< "$line")
budget=$( cut -f5 <<< "$line")
maxrows=$(cut -f6 <<< "$line")

outdir="$RESULTS/$name"
if [ -f "$outdir/.done" ]; then
    echo "[$name] already complete — skipping (delete $outdir/.done to rerun)"
    exit 0
fi

# --- optional flags: empty target => unlabeled; empty task => inferred ---
# `${arr[@]+"${arr[@]}"}` expands safely when empty (bash 3.2 + `set -u`).
targ=(); [ -n "${target:-}" ] && targ=(--target "$target")
tsk=();  [ -n "${task:-}"   ] && tsk=(--task "$task")

echo "[$name] package=$package budget=${budget:-120}s cpus=${SLURM_CPUS_PER_TASK:-?} host=$(hostname)"
atom run "$package" ${targ[@]+"${targ[@]}"} ${tsk[@]+"${tsk[@]}"} \
    --time-budget "${budget:-120}" --max-rows "${maxrows:-2000000}" \
    --out "$outdir" --kb "$KBROOT/task-${task_id}" \
    --seed 0 --yes

touch "$outdir/.done"    # idempotency marker: safe to resubmit the array
echo "[$name] complete -> $outdir"
