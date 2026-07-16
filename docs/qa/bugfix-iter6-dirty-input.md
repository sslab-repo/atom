# Bugfix Workflow — Iteration 6: generalized dirty-input parsing (converged)

Date: 2026-07-16 · Suite: 36 → 67 tests green · Scope approved (net +40
production lines: profiler + loader consolidation). Workflow: fix → issue
cases → all-dataset regression → improvements → 5 fresh datasets.

## Problem

Dirty-numeric handling had accreted as disjoint special cases (missing
sentinels, a 95%-coercion probe, decimal-comma), and the profiler
(`_classify_value`, decides whether a column is *kept*) and the loader
(`_to_float`, decides how a value is *read*) parsed independently. When they
disagreed a column was silently dropped — the shared root of BUG-1 and BUG-4.

## Fix — one exception-controlled parser, shared by profiler + loader

`parse_numeric(value, decimal_comma=False) -> float | None` in the ingest
layer, called by both the profiler's per-value vote and the loader. Every
parse is exception-guarded: an unparseable value becomes `None`/NaN, never
raises. Generalized cases (each a unit test), beyond the old N/A/`?`/comma:

| Class | Example → value |
|---|---|
| Currency symbols ($ € £ ¥ ₩ ¢) | `$1,234.56` → 1234.56 |
| Percent | `45%` → 0.45; `12.5%` → 0.125 |
| Thousands separators | `1,234,567` → 1234567; `1,234.56` → 1234.56 |
| Accounting negatives | `(50)` → −50 |
| Whitespace / leading `+` | `" 42 "`, `"+5"` → 42, 5 |
| Expanded missing markers (case-insensitive) | `-`, `--`, `#N/A`, `n.a.`, `missing`, `unknown`, `nil`, `.` → NaN |
| Locale decimal comma (column-flagged) | `27,3` → 27.3; `1.234,56` → 1234.56 |

Column-level ≥95%-numeric gate retained, so one stray value can't flip a text
column to numeric. Single-comma values (`27,3` vs `1,234`) are numeric either
way; the fraction lengths decide radix-vs-thousands at the column level.

**Deliberately NOT parsed** (unbounded / semantics-guessing): unit suffixes
(`5 kg`, `10 mph`) and slash composites (`120/80` blood pressure) — these fall
through to categorical/drop, verified on sleep-health.

## Step 2 — full regression (all 20 prior datasets)

Every round-5 (10) + step-4 iter-5 (5) + iter-5 fresh (5) dataset re-run under
the new parser: **identical column drops, consistent scores, 0 crashes** — the
parser is a strict superset (same on clean data, better on dirty). beer still
keeps its 4 temperature columns (r² 0.67). pytest 36 → 67 (31 new: parametrized
parse_numeric accept/reject, currency/%/thousands column fixtures, is_missing,
decimal-comma). `atom modules verify` 17/17.

## Step 3 — improvements

The generalization *is* the improvement (replaces 3 ad-hoc special-cases with
one tested parser). One minor hygiene add: mute ridge `LinAlgWarning`
(ill-conditioned matrix) at the CLI layer, matching the iter-5 filter. No
rollbacks.

## Step 4 — 5 fresh datasets (convergence)

| Dataset (discipline) | Task | AMP | Test result |
|---|---|---|---|
| ds-salaries23 (labor econ, 3.7k) | reg | ✅ | r² 0.995 (salary≈salary_in_usd, as round-4 noted) |
| cal-housing (real estate, 20k) | reg | ✅ | r² 0.840; `ocean_proximity` one-hot |
| sleep-health (clinical, 374) | clf | ❌ parity | acc 1.000 but AMP gated (see OBS-1); BP "120/80" correctly kept categorical |
| housing-prices (real estate, 545) | reg | ✅ | r² 0.630; yes/no one-hot |
| amazon-books (publishing, 550) | 14-class | ✅ | f1 0.483; ordinal advisory fired (User Rating) |

**5/5 end-to-end, 0 crashes, 4/5 deployable, no false coercions.** No
dirty-input bug surfaced → the dirty-input workflow has converged.

## Observations (NOT bugs; out of this change's scope)

- **OBS-1 — small-sample parity fragility.** sleep-health (22-row val) exports
  with `label_agreement 0.955`, `metric_delta_f1_macro 0.046` — a single
  boundary row flipping under float32 on a tiny parity sample fails the 0.98
  floor. The gate is **failing closed** (refuses a non-faithful AMP — the safe
  direction), unrelated to dirty-input parsing. Candidate for a future
  iteration: a minimum parity-sample size or a small-n score-correlation
  tolerance (same family as the deferred anomaly-parity item).
- **OBS-2 — amazon User Rating is ordinal** (14 numeric levels): advisory fires;
  same ordinal-target lever as student-G3 / wine-quality.

## Final state

67 tests · 25 distinct Kaggle datasets exercised across the whole iter-5/6
effort · dirty-input parsing unified and generalized with exception control ·
no regressions · every generalized case carries a test.
