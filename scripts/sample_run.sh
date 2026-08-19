#!/usr/bin/env bash
# Sample ATOM run — end-to-end smoke test on a tiny synthetic dataset.
# Generates data, packs it, inspects it, trains a model, and shows the result
# (metrics + the deployable ONNX package). Safe to run repeatedly.
#
#   bash scripts/sample_run.sh
#   ATOM=/path/to/atom bash scripts/sample_run.sh    # use a specific launcher
set -euo pipefail

ATOM="${ATOM:-atom}"                       # override: ATOM=~/atom/bin/atom
command -v "$ATOM" >/dev/null 2>&1 || { echo "cannot find '$ATOM' — 'source ~/.bashrc' (or ~/.zshrc), or set ATOM=/path/to/atom"; exit 1; }

WORK="$(mktemp -d 2>/dev/null || mktemp -d -t atom-sample)"
trap 'rm -rf "$WORK"' EXIT
BOLD=$(printf '\033[1m'); GREEN=$(printf '\033[32m'); NC=$(printf '\033[0m')

echo "${BOLD}1) generate a small classification dataset${NC}"
python3 - "$WORK/sample.csv" <<'PY'
import csv, random, sys
rng = random.Random(7)
with open(sys.argv[1], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["feature_a", "feature_b", "feature_c", "label"])
    for _ in range(600):
        c = rng.random() < 0.5
        w.writerow([rng.gauss(3 if c else 0, 1.2),
                    rng.gauss(0, 1),
                    rng.choice(["red", "green", "blue"]),
                    "positive" if c else "negative"])
print("  wrote", sys.argv[1])
PY

echo "${BOLD}2) pack CSV -> ATOM Dataset Package${NC}"
"$ATOM" pack "$WORK/sample.csv" --target label --name sample --out "$WORK/pkgs"

echo "${BOLD}3) inspect the package${NC}"
"$ATOM" inspect "$WORK/pkgs/sample"

echo "${BOLD}4) train (30s budget) -> model package${NC}"
"$ATOM" run "$WORK/pkgs/sample" --time-budget 30 --yes --out "$WORK/runs"

echo
onnx="$(ls "$WORK"/runs/*/model/pipeline.onnx 2>/dev/null | head -1 || true)"
if [ -n "$onnx" ]; then
  echo "${GREEN}${BOLD}SAMPLE RUN OK${NC} — deployable model package produced:"
  echo "  ONNX graph : $onnx"
  echo "  manifest   : $(dirname "$(dirname "$onnx")")/manifest.json"
  echo "  metrics    : $(dirname "$(dirname "$onnx")")/metrics.json"
else
  echo "note: run completed but no ONNX graph (model may be parity-gated); see native/model.pkl"
fi
echo
echo "(temporary files under $WORK are removed on exit)"
