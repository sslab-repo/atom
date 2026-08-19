#!/bin/bash
# Login-node prep: build ATOM Dataset Packages BEFORE submitting the array.
# Compute nodes are usually offline, so do all network + packing here.
#
#   bash scripts/slurm/prepare.sh
#
# Edit the two blocks below for your data. Packages land under $PKGS, which the
# array manifest (datasets.tsv) points at.
set -euo pipefail

ATOM_ROOT=${ATOM_ROOT:-$HOME/atom}
export PATH="$ATOM_ROOT/bin:$PATH"        # or: source /path/to/.venv/bin/activate
PKGS=${PKGS:-pkgs}
mkdir -p "$PKGS"

# --- 1. Local CSVs -> ADP -----------------------------------------------------
# atom pack <csv> --target <col> --name <name> --out $PKGS
atom pack data/credit.csv   --target default   --name credit   --out "$PKGS"
atom pack data/housing.csv  --target SalePrice --name housing  --out "$PKGS"
atom pack data/segments.csv                    --name segments --out "$PKGS"   # unlabeled

# --- 2. Kaggle datasets -> ADP (needs network + [kaggle] extra) ---------------
# atom fetch kaggle:<owner/dataset> --target <col> --name <name> --out $PKGS
# atom fetch kaggle:uciml/iris --target Species --name iris --out "$PKGS"

echo "prepared packages:"; ls -1 "$PKGS"
echo "next: N=\$(( \$(wc -l < scripts/slurm/datasets.tsv) - 1 )); sbatch --array=1-\$N%16 scripts/slurm/atom_array.sbatch"
