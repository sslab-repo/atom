# Bugfix Workflow — Iteration 5 (round-5 findings, converged)

Date: 2026-07-15/16 · Commits `5c19946`..`HEAD` · Suite: 29 → 36 tests green ·
Workflow: fix → issue case → all-dataset regression → full feature test →
improvements with compare-or-rollback → 5 fresh datasets (iteration trigger).
Stopped at 2 iterations — the second fresh-dataset sweep found no bugs.

## Iteration 1 — bugs fixed (each verified on its issue case + regression)

| Bug | Root cause (proven) | Fix | Evidence |
|---|---|---|---|
| BUG-1 dirty-numeric dropped | numeric columns polluted with string missing-markers ("N/A", "?") typed as string → dropped as high-cardinality | added those to MISSING_SENTINELS + a ≥95%-numeric coercion probe (flag `numeric-coerced:<col>`); loader coerces stragglers to NaN | stroke keeps `bmi` (auc 0.895); auto-mpg keeps `horsepower` (r² 0.873→0.896, rmse 2.84→2.57) |
| BUG-2 phantom NaN class | rows with a missing *target* counted as a class; ordinal advisory gated at >10 levels | drop unlabeled rows with a stated count (profiler flag `unlabeled-rows:<t>:<n>`); advisory now fires from 3 levels | cirrhosis classes 5→4, f1 0.316→0.392, "6 train rows dropped" surfaced |
| BUG-3 clustering 0-trials crash + busy-spin | silhouette scored on RAW features (NaNs) while pipelines impute internally; all-failed batches spun the whole budget; exit hint said "increase budget" | metrics score in the pipeline's transformed space (`Evaluator.metric_features`); breaker after 3 all-failed batches; error signatures surfaced in progress + exit message | cc-customers 0→289 trials, test silhouette 0.564, deployable AMP; error now named |

## Iteration 1 — Step 2 full regression (post-fix)

pytest 29→34 · **all 10 round-5 datasets end-to-end, 10/10 deployable**
(was 9/10; cc-customers was the failure) · 0 crashes · CLI feature sweep
11/11 (inspect zip+dir, pack, pack-images + image gating, fetch, modules
list/verify, run supervised/clustering/anomaly/`--max-trials`/
`--include-experimental`).

## Iteration 1 — Step 3 improvements (compare-or-rollback)

| Improvement | Result | Decision |
|---|---|---|
| I1 decision-threshold tuning (binary, report-only) | stroke test bal_acc 0.500→0.757; shoppers 0.794→0.867 (model/AMP untouched) | **kept** |
| I2 suspiciously-perfect verdict | fires on vg-sales & rice-type (near-perfect score after correlated-feature warnings) | **kept** |
| I3 CLI log hygiene | stroke run log ~700→62 lines (sklearn warnings muted at CLI only) | **kept** |
| I4 fetch/pack `--target` UX | raw ValueError → friendly error + difflib suggestion + header list | **kept** |
| I5 meta-KB prior display | prior score shown only for the same, scale-free metric | **kept** |
| (found in step 3) regression-CV stratification | small-data CV stratified regression targets → 429 sklearn warnings/run; stratify classification only | **fixed** (auto-mpg log 974→139) |

## Iteration 2 — trigger: BUG-4 from the first fresh-dataset sweep

The 5 fresh datasets (hepatitis-c, rice-type, beer-consumption, forest-fires,
drug-classification) surfaced one bug:

- **BUG-4 locale decimal comma.** beer-consumption-sao-paulo (Brazilian) dropped
  its 4 temperature columns + precipitation as high-cardinality strings —
  values use "," as the radix point ("27,3"). Fix: profiler votes a
  `^-?\d+,\d+$` value as numeric and marks `ColumnProfile.decimal_comma` when
  the fraction isn't always 3 digits (so a thousands separator "1,234" stays
  a string); the loader converts "," → "." for flagged feature and target
  columns. **Evidence: beer r² 0.326 → 0.670 (rmse 3.66 → 2.56)** — the
  temperature signal is now used. Verified on the issue case + 2 unit tests
  (decimal kept/parsed; thousands separator preserved).

Rolled back in Step 3 of iteration 2:

| Improvement | Result | Decision |
|---|---|---|
| multiclass leak screen (one-vs-rest correlation) | hepatitis-c & drug stayed unflagged (perfect scores are multivariate, not single-feature leaks); no false positives, but **no measurable change** on any test dataset | **rolled back** (compare-or-rollback) |

## Iteration 2 — full regression + fresh sweep (convergence)

Full regression: **all 15 datasets** (10 round-5 + 5 iteration-1 fresh)
re-run under the fixed build — no crashes, no unexpected column drops, scores
consistent, all gates firing. pytest 34→36.

5 new fresh datasets, **zero bugs, 5/5 deployable**:

| Dataset (discipline) | Task | Test result |
|---|---|---|
| students-exams (education) | reg | r² 0.854; categorical demographics one-hot |
| vehicle-fraud (insurance, imbal.) | clf | auc 0.943; threshold-tuned bal_acc 0.630→0.889 |
| car-msrp (automotive pricing) | reg | r² 0.989; string Model/Market-Category dropped, Make one-hot |
| ice-cream (retail) | reg | r² 0.982; **leak screen flagged `Temperature` \|r\|=0.990** (correct) |
| bank-churn2 (banking) | clf | auc 0.863; threshold-tuned bal_acc 0.680→0.780 |

No bugs and no kept improvements in the second sweep → **workflow converged.**

## Remaining notes (not bugs; tracked for roadmap)

- Multiclass perfect scores (hepatitis-c, drug: f1 1.000) are unscreened by
  design — the leak screen is single-feature/one-vs-rest and these are
  multivariate. A model-based "suspiciously-perfect regardless of correlation"
  flag would catch them but risks false alarms on legitimately-easy data
  (mushroom, iris); deferred pending a calibrated trigger.
- forest-fires r² < 0 (zero-inflated `area`): log-target transform is the
  known lever; a target-transform preprocessing axis is roadmap, not v1.
- HAR `subject` group leakage and imbalance thresholding beyond report-only
  tuning both map to existing deferred items (group-aware splits; class-weighted
  members / SMOTE).

## Final state

36 tests · 20 distinct Kaggle datasets exercised this workflow (10 round-5 +
5+5 fresh) · **all deployable ONNX AMPs, 0 crashes** · budgets held · every
iter-5 fix carries a regression test.
