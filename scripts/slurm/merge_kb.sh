#!/bin/bash
# Merge the per-task meta-KBs written by the array into the shared meta-KB, so
# the next batch warm-starts from everything the batch learned. Records are
# versioned append-only JSONL, so concatenation is a safe merge.
#
# Chain after the array:
#   JID=$(sbatch --parsable --array=1-N%16 scripts/slurm/atom_array.sbatch)
#   sbatch --dependency=afterany:$JID scripts/slurm/merge_kb.sh
#SBATCH --job-name=atom-kb-merge
#SBATCH --output=logs/kb_merge_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail

KBROOT=${KBROOT:-$HOME/atom/kb-array}
SHARED=${SHARED:-${ATOM_HOME:-$HOME/atom/home}/metakb}
mkdir -p "$SHARED"

n=$(cat "$KBROOT"/task-*/records.jsonl 2>/dev/null | wc -l | tr -d ' ')
cat "$KBROOT"/task-*/records.jsonl >> "$SHARED/records.jsonl" 2>/dev/null || true
echo "merged ${n:-0} records into $SHARED/records.jsonl"
