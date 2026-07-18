# Improvement Workflow — Iteration 7: conditional-leak detection (converged)

Date: 2026-07-18 · Suite: 67 → 69 tests green · Scope: additive (~30 lines,
`_leak_screen`). Workflow (improvement mode): implement → compare → keep/rollback
→ regression → 5 fresh datasets.

## Motivation

The marginal leak screen measures each feature's correlation with the target in
isolation. A feature that reconstructs the target *conditional on another
column* passes silently. Confirmed on ds-salaries-2023: no feature correlates
> 0.5 marginally with `salary_in_usd`, yet the model scored r² = 0.995 — because
`salary` (local currency) is the target pre-conversion, exact within each
`salary_currency` group. A misleadingly "deployable" model with no warning.

## Improvement A (kept) — conditional-leak detector

Added to `_leak_screen`, purely additive: one-hot columns (`col=value`) are used
as ready-made group indicators; for each numeric feature not already marginally
flagged, compute its correlation with the target *within* each group with ≥50
members and flag `possible-conditional-leakage: '<feat>' within '<col>'` when the
best within-group |r| ≥ 0.98. The `suspiciously-perfect` verdict now also keys
off conditional flags.

**Compare (detection quality, not score — the change adds no training signal):**

| Dataset | Result | Verdict |
|---|---|---|
| ds-salaries23 | `'salary' within 'salary_currency' (\|r\|=1.000)` | ✅ real leak caught (was silent) |
| vg-sales | `NA_Sales / EU_Sales / Other_Sales within 'Platform'` (\|r\|≈1.0) | ✅ real (compositional: target = Σ regional sales) |
| 16 other regression datasets | no conditional flag | ✅ zero false positives |

## Improvement B (rolled back) — recommended threshold in model card

Added the val-tuned operating point to the pickled model card. Rolled back per
compare-or-rollback: the tuned threshold is **already** persisted, readable, in
`metrics.json` (`test.decision_threshold`); the pickle copy duplicated it with no
measurable benefit.

## Regression (all 18 labeled prior datasets)

Re-run under the new screen: scores unchanged (the screen is diagnostic only),
0 crashes, and conditional-leak flags fired on **exactly the two genuinely-leaky
datasets** (ds-salaries, vg-sales) and nothing else. pytest 67 → 69 (2 new:
conditional leak detected on the currency-within-group pattern; no false positive
on independent features across groups). `atom modules verify` 17/17.

## Step 4 — 5 fresh datasets (convergence)

| Dataset (discipline) | Task | Leak screen | Test |
|---|---|---|---|
| pima-diabetes (clinical) | clf | clean | auc 0.842; threshold-tuned bal_acc 0.703→0.740 |
| boston-housing (real estate) | reg | clean | r² 0.867 |
| loan-approval (credit) | clf | clean (no false leak despite income/amount features) | auc 0.981; tuned bal_acc 0.890→0.920 |
| paris-housing (synthetic) | reg | `squareMeters` \|r\|=1.000 + suspiciously-perfect | r² 1.000 (correct catch, marginal) |
| smoking-signal (clinical) | clf | clean | auc 0.881 |

**5/5 end-to-end, 0 false-positive leak flags on the 4 clean datasets, 1 correct
catch (paris squareMeters).** No new bug surfaced → improvement workflow converged.

## Net effect

Across 23 labeled datasets this iteration (18 regression + 5 fresh) the leak
subsystem now flags all three leak shapes — marginal (paris, vg-sales),
compositional (vg-sales), and **conditional/interaction (ds-salaries), previously
undetectable** — with zero false positives. tests 67 → 69.
