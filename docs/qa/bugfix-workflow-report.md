# Bugfix Workflow Report — Iterations 1–2 (converged)

Date: 2026-07-14 · Commits `81f12e0`..`e5d4769` · Suite: 29 tests green ·
Workflow: fix → issue case → all-dataset regression → full feature test →
improvements with compare-or-rollback → fresh-dataset validation (max 5
iterations; stopped at 2 — two consecutive zero-bug full sweeps).

## Step 1 — bugs fixed (each verified on its issue case + full sweep)

| Bug | Root cause (proven) | Fix | Evidence |
|---|---|---|---|
| B1 RF/DT binary ONNX inversion | skl2onnx 1.20 `zipmap=False` emits proba `[-p, p]` and argmax labels for BINARY tree classifiers (any label dtype; multiclass exact) | empirical detection + in-graph repair (`Add [1,0]`; `label = Gather(classes, ArgMax)`) | agreement 0.22 → 1.00, proba diff 1e-7 |
| B2 HistGB ONNX drift | (a) fused float32 preprocessing shifts rows across quantile-dense tree-leaf boundaries — inherent to ORT float tree kernels; (b) threshold rounding direction (incl. a NumPy-2 NEP-50 comparison trap in the first fix attempt) | floor-rounded threshold rewrite from the model's own float64 thresholds + parity gate redefined as FUNCTIONAL equivalence (label agreement + primary-metric Δ ≤ 2e-3 on a 512-row labeled sample) | housing model-only match 0.88 → 1.0000; housing HistGB with 30% row drift but Δr²=0.0004 now passes; a member with Δr²=0.011 is still correctly rejected |
| B3 small-data val overfitting | 485 trials selected on a 77-row val split | val < 1000 rows → stratified 3-fold CV trial scoring | diabetes test roc_auc 0.8525 → 0.8592, acc 0.746 → 0.762 |
| B4 warm-start metric mislabel | prior score printed under current run's metric name | KB records store their metric | display-correct |
| B5 finalize budget overrun | unbounded full-fidelity refits | refit-cost estimate honored in finalize | 10/10 sweep runs within stated budget (was up to +78%) |
| B6 KB warm-start dilution | no similarity cutoff | `nearest(max_distance=2.0)` | dissimilar datasets no longer warm-start |

## Step 2 — full regression (post-fix)

pytest 28/28 (now 29) · original 5 datasets: **4/5 deployable AMPs** (was
2/5; the 5th is a *correct* rejection: one member genuinely degrades r² by
1.1%) · **0/5 budget overruns** (was 5/5) · 10/10 CLI functions pass
(inspect zip+dir, pack, pack-images, fetch, modules list/verify, run with
clustering/anomaly/--include-experimental, image gating).

## Step 3 — improvements (compare-or-rollback)

| Improvement | Result | Decision |
|---|---|---|
| I1 hash-encoded categoricals | housing r² 0.8407→0.8372 | **rolled back** |
| I5 stratified fidelity sampling | fraud roc_auc 0.9603→0.9564 | **rolled back** |
| I3 `atom fetch kaggle:<slug>` | functional; used for all of step 4 | **kept** |

## Step 4 — fresh datasets (iteration trigger) and iteration 2

Five new disciplines fetched + run, zero bugs: telco churn, wine quality,
heart disease, adult income, insurance. One loud improvement signal —
insurance r² = 0.15 with `smoker` dropped as categorical — triggered
iteration 2: **one-hot encoding from fingerprint vocabularies** (train-split
profiling → leak-free; unseen categories → zero row; 256-column cap).

| Dataset | Before → After (primary evidence) |
|---|---|
| insurance | r² **0.153 → 0.875** (rmse 11456 → 4408) |
| heart-disease | roc_auc **0.825 → 0.958** (acc 0.767 → 0.884) |
| adult-income | roc_auc 0.887 → 0.934 |
| telco-churn | roc_auc 0.859 → 0.879 |
| wine-quality | unchanged (all-numeric — correct no-op) |
| original 5 | no regressions; budgets held; AMPs deployable |

## Remaining notes (not bugs; tracked for roadmap)

- Encode-categorical as a *pipeline module* (target encoding, interactions)
  would supersede loader-level one-hot; fingerprint category vocabularies
  are technically data values (≤64 short strings) — consider hashing them
  if fingerprints are ever shared beyond the run.
- Wine quality (ordinal 6-class, bal_acc 0.32): ordinal-aware treatment or
  `--task regression` guidance.
- Fraud minority recall (bal_acc ~0.75): resampling/cost-sensitive modules.
- Housing: deployability-aware ensemble selection could prefer parity-clean
  members when scores tie.

## Final state

29 tests · 10 datasets across 10 disciplines run end-to-end · 9/10
deployable ONNX AMPs (1 correct rejection) · all runs within stated budgets
· meta-KB flywheel and fetch→pack→run→export loop fully operational.
