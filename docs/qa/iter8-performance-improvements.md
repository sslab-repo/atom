# Iteration 8 — Performance Improvement Loop (10-dataset suite)

Date: 2026-08-07 · Conditions: (1) compute is not a constraint — Slurm manages
resources; pursue the most performant approach. (2) all changes must be general,
never tuned to a specific dataset. Method: implement → run same 10 datasets →
compare → keep or roll back. Reported every iteration.

Baseline: round-8c (commit `4391258`). Suite = the same 10 datasets
(cc-fraud, zoo-animals, wine-quality2, spam-text, mall-customers, social-ads,
pokemon, melbourne-housing, voice-gender, pump-sensor).

## Iteration 1 — class-balanced training (KEPT) · `cfd887e`

Every classifier gains a `class_balance` search dimension (none/balanced);
FIT applies balanced `sample_weight` uniformly (works for HistGB too;
training-only, ONNX graph unchanged). General imbalance lever — the search
keeps it only when it wins the task metric.

| Dataset | metric | before → after |
|---|---|---|
| pump-sensor | f1_macro | **0.666 → 0.9996** |
| pokemon | bal_acc / f1_macro | 0.707 → 0.850 / 0.756 → 0.874 |
| cc-fraud | bal_acc (default cut) | 0.802 → 0.896 |
| wine-quality2 | f1_macro | 0.440 → 0.455 |
| balanced / regression / clustering | — | unchanged (search declines balancing) |

The single biggest win of the loop. Verified with a unit test.

## Iteration 2 — deployability-aware selection + expanded finalize pool (KEPT) · `0bd0707`

Finalize the full TOP_K (not a budget-shrunk 2). Selection ranks candidates by
val score and ships the best; only falls back to a parity-faithful lower
candidate when it is within a 2% (scale-free) margin — deployability is a
tie-break, never a reason to sacrifice score. (A first cut that always preferred
faithful tanked wine 0.45→0.30; corrected to the margin rule.)

No regressions; all iter-1 gains preserved. Set up the infrastructure that
iter-4 completes.

## Iteration 3 — ExtraTrees for model diversity (ROLLED BACK) · reverted

Added ExtraTrees (clf+reg). **Net negative**: at a fixed 120s budget a 5th
method dilutes the search — pokemon trials 409→309 and its pick generalized
worse (f1 0.874→0.790); wine's winner flipped to a val-lucky ExtraTrees that
scored worse on test (0.455→0.389). Lesson: under a fixed budget, adding
per-method cost causes winner's-curse dilution. Reverted per compare-or-rollback.

## Iteration 4 — full-pool finalize (KEPT) · `f1be21e`

Since compute is free, always refit the whole TOP_K (3×-budget anti-hang
ceiling only), so search overrun can't starve the finalize pool.

| Dataset | effect |
|---|---|
| wine-quality2 | finalize pool **1 → 5**, deployable **False → True**, performance unchanged (f1 0.4547, bal_acc 0.424→0.448) — the fuller pool gave selection a faithful near-equal fallback |
| other 9 | identical (already had full pools) |

Cost: wall-time on expensive refits (wine 218s, cc-fraud 152s vs 120s stated);
acceptable under the compute-free directive.

## Iteration 5 — log-target transform for regression (INVESTIGATED, REJECTED)

The biggest untapped *general* regression lever (heavy-tailed targets:
melbourne, housing, forest-fires). Prototyped the export path: skl2onnx has no
`TransformedTargetRegressor` shape calculator, so the inverse (`expm1`) must be
appended to the fused graph by hand. It converts fine, **but the `exp()` inverse
amplifies float32 tree-threshold drift** — measured parity on a HistGB log-target
model: max row rel-diff **0.50**, r² native 0.478 vs ONNX 0.517. Log-target tree
models would almost always fail the parity gate and ship non-deployable.
Rejected before implementation — wrong fit for a deployability-first design.
(A linear-only log-target would export cleanly but the gain is narrow.)

## Iteration 6 — default-config search seeding (ROLLED BACK)

Seed batch 1 with each method's library-default config (a known-good per-method
baseline) to cut random-sampling variance. **Mixed, with regressions on the
flagship imbalanced datasets** — the deterministic seeds consume exploration
budget: cc-fraud auc 0.966→0.961, social-ads auc 0.919→0.912, pokemon secondary
bal_acc 0.85→0.71; while melbourne r² 0.844→0.856 and zoo/pump improved. No
systematic win and it risks the imbalance gains, so reverted. (Same fixed-budget
dilution lesson as iter-3: on a time-bounded search, spending trials on fixed
baselines trades away exploration.)

## Cumulative result (current build vs round-8c baseline)

| Dataset | Task | round-8c | now | AMP |
|---|---|---|---|---|
| cc-fraud | binary | bal_acc 0.802* | **0.896*** / f1 0.908 | ✅ |
| zoo-animals | 7-class | f1 0.560 | 0.560 | ✅ |
| wine-quality2 | ordinal 7-class | f1 0.440, ❌ | **f1 0.455, ✅** | ✅ |
| spam-text | text | exit 1 | exit 1 (text guard) | — |
| mall-customers | clustering | silhouette 0.586 | 0.586 | ✅ |
| social-ads | binary | auc 0.914 | 0.919 | ✅ |
| pokemon | binary (8%) | bal_acc 0.707 | **0.850** / f1 0.874 | ✅ |
| melbourne-housing | reg | r² 0.844 | 0.844 | ✅ |
| voice-gender | binary | auc 0.999 | 0.999 | ✅ |
| pump-sensor | 3-class | f1 0.666 | **0.9996** | ✅ |

*default-threshold balanced accuracy. **9/9 deployable** (was 8/9 — wine
recovered). Big gains on every imbalanced/rare-class dataset; nothing regressed.
tests 69 → 71, modules 17/17.

## Observed issues (this loop)

- **Budget overrun on large/multiclass data** (wine 218s, cc-fraud 152s vs
  120s): full-fidelity refits of the full pool exceed the reserved tail. A
  deliberate performance-for-time trade under condition 1; the reported budget
  is effectively search-budget. If honesty of the stated number matters later,
  reserve finalize cost from the search deadline.
- **wine ordinal ceiling** (f1 ~0.455): 7 overlapping quality levels treated as
  flat multiclass — an ordinal task mode is the lever, not balancing.

## Deferred / needs architectural change (not achievable as a safe in-loop tweak)

- **Log/Box-Cox target transform** — rejected in iter-5 (exp amplifies float32
  parity drift). Would need a parity policy that scores the ONNX graph's own
  metric rather than native-vs-ONNX fidelity, or a float64 inference path.
- **Ordinal task mode** (wine, zoo, amazon quality/rating targets) — a new task
  family / metric, not a tweak.
- **Time-aware splits** (pump temporal leakage) — packager + split-policy work.
- **Minimal text vectorizer** (spam-text) — new preprocessing modality.
- **Probability calibration**, **stacked ensembling**, **more base learners** —
  all add per-trial/per-method cost that dilutes a fixed-budget search (iters 3
  & 6); only worthwhile with a larger trial budget or a smarter (non-random)
  search strategy (the ADR-0006 Search-registry work).

## Convergence

Six iterations: **three kept** (class-balancing, deployability-aware selection,
full-pool finalize), **two rolled back** with evidence (ExtraTrees, default-config
seeding — both fixed-budget dilution), **one investigated and rejected**
pre-implementation (log-target — breaks ONNX parity). The consistent finding:
class-balancing was the dominant available win, and under a fixed per-run search
budget every remaining *quality* lever trades exploration for its addition, or
trades deployability for its output. Further general gains now require an
architectural change — a smarter search strategy (so added models/CV don't
dilute) or a revised parity policy (so target transforms stay deployable) — not
another in-loop tweak. The loop has converged.

Net across the loop: large, general, deployment-preserving gains on imbalanced
data (pump f1 0.67→0.9996, pokemon bal_acc 0.71→0.85, cc-fraud 0.80→0.90), wine
recovered to deployable, **9/9 deployable, zero regressions**, tests 69→71.
