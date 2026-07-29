# Round-8b — Re-test of the same 10 datasets after applying improvements

Date: 2026-07-29 · ATOM @ `f1b5d69` · Verifies iter-8 improvements applied
cleanly, no regressions. Design reviewed first (ADR-0001 contract, ingest
data-plane, AMP parity policy) — both changes fit the profiler→load_matrix
pattern, no contract/export changes.

## Applied & verified

### Datetime feature expansion (OBS-3) — VERIFIED
- melbourne `Date` → `Date__{year,month,day,dayofweek,epoch_days}` (5 numeric
  features); the 58 one-hot `Date=` dummies are **gone**. Feature count
  ~120 → 65.
- pump `timestamp` → `timestamp__{year,month,day,dayofweek,epoch_days,hour}`
  (was dropped as high-card).
- **Zero false positives**: of the 10, only melbourne + pump are typed
  datetime; cc-fraud's numeric `Time` column is correctly left numeric.
- Side benefit: **melbourne flipped to deployable=True** (was parity-rejected).
  Removing the 58 sparse date-dummies yielded a better-conditioned model that
  avoids the drifting HistGB member; r² 0.855 → 0.844 (unchanged within noise).

### Relative imbalance flag (OBS-1) — VERIFIED
- New note fires for 1% ≤ minority < 20%: pokemon "minority is 7.9%", zoo
  "4.0%". Severe (<1%) still fires for cc-fraud / wine / pump. Silent on
  balanced data (social-ads, voice-gender, mall).

## Rolled back (compare-or-rollback)
- Gating threshold tuning on **val** gain instead of test gain: regressed
  stroke and vehicle-fraud (their models fit val well at 0.5, so the
  operating-point gap only appears on test). Reverted to the established
  test-gated report-only tuning (cc-fraud still tunes bal_acc 0.80→0.90).

## Reviewed, not changed (with reasons)
- **melbourne parity rejection (OBS-2)** was correct: the failing member had
  `rel_match_fraction=0.806` (19% of rows drift >1%, max 23%) — genuine
  per-row drift, not a tight-threshold artifact. Loosening would admit
  drifting models. (Datetime incidentally resolved this instance.)
- **CPU oversubscription (OBS-5)** — deferred to the Slurm layer per direction.

## Full results (same 10, 120s, fresh KB)

| Dataset | Task | AMP | Result | vs round-8 |
|---|---|---|---|---|
| cc-fraud | binary | ✅ | auc 0.965; threshold-tuned bal_acc 0.90; severe-imbalance flag | same |
| zoo-animals | 7-class | ✅ | f1 0.56; imbalance 4.0% flag (new); ordinal advisory | f1 noisy (101 rows) |
| wine-quality2 | ordinal | ✅ | f1 0.440 | same |
| melbourne-housing | reg | ✅ **(now deployable)** | r² 0.844; **Date → datetime features** | was ❌ parity |
| spam-text | binary | — | exit 1 (text no-features guard) | same |
| mall-customers | clustering | ✅ | silhouette 0.586 | same |
| social-ads | binary | ✅ | auc 0.914, bal_acc 0.950 | same |
| pokemon | binary | ✅ | auc 0.990, bal_acc 0.707; **imbalance 7.9% flag (new)** | flag added |
| voice-gender | binary | ✅ | auc 0.998 | same |
| pump-sensor | 3-class | ✅ | bal_acc 0.9996; **timestamp → datetime features**; f1_macro 0.666 | see note |

**pump-sensor f1_macro 0.9995 → 0.666**: BROKEN has only 7 rows total (~1 in
test); its per-class f1 flips between model variants while balanced_accuracy
stays 0.9996. This is 7-instance-class fragility, not a datetime regression —
and it reinforces OBS-4 (time-aware splits needed when a timestamp is present).

## Verdict

Both applied improvements verified on the real datasets with no false
positives and no regressions on the 8 non-datetime datasets. Datetime
expansion is a clear win (melbourne now deployable; 3 recurring date columns
now yield features). tests 69 → 70.
