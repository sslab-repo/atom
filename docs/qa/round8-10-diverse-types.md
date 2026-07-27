# Round-8 Validation — 10 New Datasets of Different Types (test-only, no fixes)

Date: 2026-07-27 · ATOM @ `6e79ec6` · Policy: report observations/issues/
improvements, fix nothing. Full data, `--time-budget 120`. Types deliberately
spread: extreme-imbalance, tiny multiclass, ordinal, text, clustering, wide
sensor, dates/categoricals, acoustic.

## Results

| Dataset (type) | Task | AMP | Test result |
|---|---|---|---|
| cc-fraud (285k, 0.17% fraud) | binary | ✅ | auc 0.965; threshold-tuned bal_acc 0.80→0.90 |
| zoo-animals (101 rows) | 7-class | ✅ | f1 0.760; ordinal advisory fired |
| wine-quality2 (white+red) | ordinal 7-class | ✅ | f1 0.440; `type` one-hot; ordinal + imbalance advisory |
| spam-text (text only) | binary | — | **exit 1 — no-features guard** (Message dropped; text needs M6) |
| mall-customers (unlabeled) | clustering | ✅ | silhouette 0.586; `Genre` one-hot |
| social-ads | binary | ✅ | auc 0.914, bal_acc 0.950 |
| pokemon (8% legendary) | binary | ✅ | auc 0.990 but bal_acc **0.707** (see OBS-1) |
| melbourne-housing (dates+cats) | reg | ❌ parity | r² 0.855; member Δr²=0.00215 > 0.002 gate (see OBS-2) |
| voice-gender (acoustic) | binary | ✅ | auc 0.998, bal_acc 0.988 |
| pump-sensor (220k × 52, timestamp) | 3-class | ✅ | f1 0.9995 (see OBS-4) |

**9/10 end-to-end, 8/9 deployable.** spam-text is a graceful hard-exit
(text-only). melbourne is a parity rejection. No crashes in the engine.

## What worked well

- **Extreme imbalance handled**: cc-fraud (0.17% positive) and pump-sensor
  (BROKEN ≈ 7 / 220k) both ran, deployable; cc-fraud threshold-tuned to
  bal_acc 0.90.
- **Categoricals one-hot** everywhere they fit (wine `type`, mall `Genre`,
  social `Gender`, pokemon `Type 1/2`, melbourne `Type/Method/Regionname`).
- **Ordinal advisory** fired on zoo + wine (numeric N-level targets).
- **Robust ingest**: all-NaN `sensor_15` dropped cleanly; high-card
  strings (Address, Name, animal_name) dropped with reason.
- **Leak screen**: zero false positives across all 10.
- **Budget honesty verified** (isolation): zoo at 120s → real 103.7s ≈
  ATOM elapsed 103s. (Batch wall-times were inflated 3-5× by CPU contention —
  see OBS-5 — but ATOM's own clock is accurate.)

## Observations / issues (NOT fixed)

- **OBS-1 — imbalanced binary underserved when val minority is tiny.**
  pokemon (8% legendary): auc 0.990 but test bal_acc 0.707 at the default
  0.5 cut. Threshold tuning **did not fire** — val has only 87 rows (~7
  legendaries), so no threshold beat the default by the >0.01 val margin.
  Also the imbalance *flag* trigger is 1% (rare-class), so 8% is never
  flagged. Two levers: relative imbalance flag (e.g. minority < 20-40%),
  and tune the threshold on pooled CV out-of-fold predictions (or use
  class-weighted members) so it survives a small val split.
- **OBS-2 — regression parity gate rejects a practically-equivalent
  ensemble.** melbourne member Δr² = 0.00215 vs the 0.002 functional cutoff
  → whole ensemble `deployable=False`, despite a 0.00215 r² difference being
  operationally negligible. The gate works as designed but the absolute
  2e-3 regression bound is tight; a relative tolerance (e.g. Δr²/|r²|) or a
  slightly looser regression bound would keep such models deployable.
- **OBS-3 — datetime columns never become features.** pump `timestamp`
  (220k unique) is dropped as high-card; melbourne `Date` (58 unique) is
  **one-hot expanded into 58 dummies** — neither yields year/month/day-of-week
  /epoch features. A datetime-expansion step is the clear lever (recurring:
  beer `Data`, pump `timestamp`, melbourne `Date`).
- **OBS-4 — pump-sensor f1_macro 0.9995 is suspiciously high** for a class
  with ~7 total instances. Almost certainly temporal autocorrelation (rows
  adjacent in time share a label) rather than genuine skill — a random split
  leaks. Argues for time-aware splits when a timestamp is present (already a
  deferred roadmap item).
- **OBS-5 — CPU oversubscription (~6.6× cores).** A single run drives
  user+sys ≈ 680s of CPU in 103s real on a 10-core box (unbounded BLAS /
  sklearn threads). Single-run budget honesty holds, but on the **shared lab
  server** concurrent runs (or ATOM beside other users) will thrash wall-time
  badly — this round's batch runs inflated to 380-540s. Bounding
  `OMP_NUM_THREADS` / sklearn `n_jobs` would make the budget robust under
  real multi-user load.
- **OBS-6 — text-only datasets hard-exit.** spam-text (one `Message` column)
  drops to the no-features guard and exits 1. Informative, but a common case;
  a minimal bag-of-words/hashing fallback (pre-M6) or an earlier "text
  modality unsupported" gate would be friendlier than a late hard error.

## Improvement ideas (ranked)

1. **Datetime feature expansion** (year/month/dow/epoch) — now seen 3×;
   self-contained, high-value.
2. **Imbalance v2** — relative-minority flag + small-val-robust threshold
   selection (pooled-CV or class weighting). Directly fixes pokemon.
3. **Relative regression parity tolerance** — stop rejecting Δr²≈0.002
   ensembles (melbourne).
4. **Thread/CPU bounding** for the shared-server deployment (OBS-5).
5. **Time-aware split** when a time column exists (OBS-4) — already roadmap.
6. **Minimal text vectorizer** pre-M6 so text datasets aren't a hard exit.

## Verdict

Engine is solid across a genuinely diverse type spread: 9/10 end-to-end, 0
crashes, extreme imbalance + clustering + wide-sensor + ordinal all handled,
leak screen clean. The two deployability gaps (pokemon imbalance thresholding,
melbourne tight parity bound) and the recurring datetime-feature loss are the
highest-value next levers.
