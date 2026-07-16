# Round-5 Validation — 10 New Datasets (test-only, no fixes)

Date: 2026-07-15 · ATOM @ `8f32393` · Policy: report bugs/observations, fix nothing.
Protocol identical to round-4: full data (`--max-rows 10000000`), `--time-budget 120`,
fresh KB (`r5-kb`), `atom fetch kaggle:<slug>` → `atom run`.

## Datasets & results

| Dataset (discipline) | Task | AMP | Elapsed/120s | Test result |
|---|---|---|---|---|
| stroke (epidemiology, 5.1k, ~5% pos) | clf | ✅ | 104s | auc 0.896 — but bal_acc **0.500** (see OBS-2) ⚠ `bmi` dropped (BUG-1) |
| water-potability (env. science, 3.3k) | clf | ✅ | 114s | auc 0.719 |
| shoppers (e-commerce, 12.3k) | clf | ✅ | 105s | auc 0.941 |
| har-smartphones (signal proc., 7.4k × 561) | 6-class | ✅ | ⚠️ 129s | f1 0.996 (see OBS-1: subject leakage) |
| cirrhosis (hepatology, 418) | clf "5-class" | ✅ | 112s | f1 0.316 (see BUG-2: NaN target became a class) |
| titanic (transportation, 891) | clf | ✅ | 105s | auc 0.853 |
| auto-mpg (automotive eng., 398) | reg | ✅ | 103s | r² 0.873 ⚠ `horsepower` dropped (BUG-1) |
| grad-admissions (education, 500) | reg | ✅ | 103s | r² 0.852 |
| vg-sales (market analytics, 16.6k) | reg | ✅ | 103s | r² 1.0000 — compositional leak; tiered warning fired (OBS-3) |
| cc-customers (marketing segm., 9k) | clustering | — | 102s | **exit 1 — 0 successful trials** (BUG-3) |

9/10 end-to-end, 9/9 deployable AMPs among completions, worst overrun +7.5%
(har-smartphones, finalize 18.8s of a 561-feature ensemble).

## Bugs found (NOT fixed)

- **BUG-1 — numeric columns with string missing-value sentinels are dropped
  wholesale.** stroke's `bmi` (`"N/A"`) and auto-mpg's `horsepower` (`"?"`)
  are typed object at pack time and then dropped as `high-cardinality string`
  — a clinically/physically important feature silently lost in both cases.
  Fix direction: numeric-coercion probe (e.g. ≥95% of non-null values parse
  as numbers → coerce, sentinel → NaN) before the categorical/drop decision.
- **BUG-2 — rows with missing *target* become a phantom class.** cirrhosis
  `Stage` ∈ {1,2,3,4} with 6 NaN rows → confirm gate reports `classes: 5`
  and the model trains/scores on the NaN class. Should drop unlabeled rows
  (with a stated count) at pack or load. Also: the ordinal-target advisory
  (added iter-4) did not fire on this numeric 1–4 target — check its
  cardinality threshold.
- **BUG-3 — all-trial-failure busy-spins the full budget, error unsurfaced.**
  cc-customers (CC GENERAL.csv, has NaNs in `CREDIT_LIMIT`/`MINIMUM_PAYMENTS`)
  failed every clustering trial: log shows ~1,250 repeats of `rung f=1: 0/9 ok`
  for 100s, then exit 1 "no successful trials within budget — increase
  --time-budget". Three sub-issues: (a) the underlying trial exception is
  never shown, so the failure is undiagnosable from the log (suspicion:
  NaN-intolerant clustering pipeline — unverified, run stopped here);
  (b) iter-3's busy-loop break covers zero-*admission* batches but not
  all-*failed* batches — deterministic failures should trip a breaker after
  N identical strikes; (c) the "increase --time-budget" hint is wrong advice
  for a deterministic failure.

## Observations / improvement notes (NOT implemented)

- **OBS-1 — group leakage: HAR `subject` (participant id 1–30) is both kept
  as a numeric feature and ignored for splitting** (`roles.group: None`);
  random row splits put the same subject in train and test, so f1 0.996
  overstates cross-subject generalization. This is the concrete case for the
  deferred group-aware split emission (status.md deferred item 4).
- **OBS-2 — imbalanced binary: degenerate hard predictions.** stroke
  (≈5% positive): auc 0.896 but balanced_accuracy 0.500 / f1_macro 0.493 —
  the exported classifier never predicts the positive class at the default
  0.5 threshold. Levers: threshold tuning on val, or class-weighted members
  (status.md deferred item 5).
- **OBS-3 — sub-threshold compositional leak trains to r²=1.0000.**
  vg-sales: tiered screen correctly warned (`NA_Sales` |r|=0.949,
  `EU_Sales` |r|=0.916 — regional sales sum to the target) but each feature
  is below the 0.98 hard gate, so the run proceeds and reports a perfect
  score with no post-run flag. The round-4 note reproduces exactly;
  candidate: multi-feature (linear-combination) leak probe or a post-run
  "suspiciously perfect + prior warnings" verdict.
- **OBS-4 — log hygiene.** sklearn `ConvergenceWarning` spam dominates run
  logs (hundreds of lines); `y_pred contains classes not in y_true` repeats
  on tiny CV folds (cirrhosis, 45-row val). Cosmetic: `finalize_s=-0.0s`
  (vg-sales).
- **OBS-5 — KB warm-start priors cross task scales.** grad-admissions
  (rmse ≈0.05 scale) warm-started from auto-mpg's `rmse≈3.241` prior; label
  is honest but the cost/benefit signal is meaningless across target scales.
- **OBS-6 — fetch UX.** A wrong `--target` surfaces as a raw
  `ValueError` traceback (grad-admissions: header is `Chance of Admit`, no
  trailing space in v1.1 — the friendly-error path added in iter-4 covers
  consent gating only).

## Verdict

The iter-4 fixes hold: parity gate passed 9/9 completed runs (no false
rejections), budget honesty is good (worst +7.5%), and the tiered leak
screen fired where it should. The engine's first hard failure since round-3
is at the edges again — trial-failure diagnosability and dirty-numeric
ingest — plus two evaluation-honesty gaps (group leakage, imbalance
thresholding) that both map to already-deferred roadmap items.
