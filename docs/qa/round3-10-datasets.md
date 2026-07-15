# Round-3 Validation — 10 New Datasets (test-only, no fixes)

Date: 2026-07-14 · ATOM @ `b0ceb93` · Policy: report bugs/observations, fix nothing.

## Commands used (every test)

```bash
# fetch (one per dataset; szeged needed a header check, then explicit pack)
atom fetch kaggle:uciml/breast-cancer-wisconsin-data --target diagnosis --name bc-wisconsin
atom fetch kaggle:iabhishekofficial/mobile-price-classification --target price_range --name mobile-price
atom fetch kaggle:uciml/glass --target Type --name glass-id
atom fetch kaggle:uciml/mushroom-classification --target class --name mushroom
atom fetch kaggle:datasnaek/league-of-legends --target winner --name lol-matches
atom fetch kaggle:harlfoxem/housesalesprediction --target price --name kc-houses
atom fetch kaggle:uciml/sms-spam-collection-dataset --target v1 --name sms-spam        # FAILED (BUG-1)
atom fetch kaggle:sakshigoyal7/credit-card-customers --target Attrition_Flag --name bank-churn
atom fetch kaggle:dgomonov/new-york-city-airbnb-open-data --target price --name nyc-airbnb
atom fetch kaggle:budincsevity/szeged-weather --name szeged-weather
atom pack <cache>/weatherHistory.csv --target "Temperature (C)" --name szeged-weather

# run (identical settings for all)
atom run r3/<name> --time-budget 60 --max-rows 40000 --yes --kb r3-kb --out r3-runs
```

## Results

| Dataset (discipline) | Task | Result (test) | AMP | Elapsed/60s |
|---|---|---|---|---|
| bc-wisconsin (oncology) | binary clf, small-data CV | acc/f1/auc **1.000** | ✅ | 51s |
| mobile-price (electronics) | 4-class clf, small-data CV | f1 0.917 | ✅ | 60s |
| glass-id (forensic chem.) | 7-class clf, 214 rows | f1 0.323 (honest small-data) | ✅ | 59s |
| mushroom (mycology) | binary clf, ALL-categorical | acc/f1/auc **1.000** | ✅ | 54s |
| lol-matches (esports) | binary clf | acc 0.972, auc 0.998 | ✅ | ⚠️ 72s |
| kc-houses (real estate) | regression | r² 0.868 | ✅ | 54s |
| szeged-weather (meteorology) | regression | r² 1.000 (see OBS-1) | ✅ | ⚠️ **161s** |
| bank-churn (banking) | binary clf | auc 1.000 (see OBS-1) | ✅ | 52s |
| nyc-airbnb (hospitality) | regression | r² 0.173 | ❌ gated | ⚠️ 85s |
| sms-spam (NLP) | — | **failed at fetch/pack** | — | — |

9/10 end-to-end, 8/9 deployable AMPs. `mushroom` (100% categorical
features) scoring perfectly is a strong validation of the iteration-2
one-hot encoding — before it, this dataset would have had zero features.

## Bugs found (NOT fixed)

- **BUG-1 — packager crashes on non-UTF-8 CSVs.** `atom fetch/pack` on
  `uciml/sms-spam-collection-dataset` dies with `UnicodeDecodeError`
  (Latin-1 content). The parquet read path was hardened for this in the
  first sweep (CIC-IDS cp1252 bytes); the CSV *packager* path was not.
  Fix direction: encoding detection/fallback in `pack_csv`.
- **BUG-2 — budget overruns are back on some shapes.** szeged-weather
  **161s vs 60s stated (+168%)**, nyc-airbnb 85s (+42%), lol 72s (+20%).
  B5 covered finalize *refits*; the overrun here is outside admission
  control — suspicion: data loading (96k rows × string parsing), AMP
  export (ONNX conversion of several large members), and locked-test
  loading are all unbudgeted. Fix direction: bring load/export phases
  into the budget model, or state the budget as search-only explicitly.

## Observations / improvement notes (NOT implemented)

- **OBS-1 — no dataset-leakage detection.** Two "perfect" scores are
  dataset artifacts, not ATOM wins: szeged-weather includes
  `Apparent Temperature (C)` (≈ the target), and bank-churn ships two
  pre-computed `Naive_Bayes_*` classifier-output columns (a known flaw of
  that Kaggle upload). ATOM trained on the leak both times and reported
  1.000 without comment. Improvement: confirm-gate warning on
  near-perfect single-feature correlation with the target, and a
  post-run "suspiciously perfect" flag.
- **OBS-2 — nyc-airbnb deployable=False**: the regression metric-delta
  gate rejected drifting tree members (gate functioning as designed);
  also r² 0.17 reflects heavy-tailed prices and dropped text features —
  a log-target transform and text modality would be the levers.
- **OBS-3 — glass-id (7 classes, 214 rows)**: runs honestly but weakly;
  candidate for ordinal/nested-CV refinements and more search time.
- **OBS-4 — text columns are dropped silently** (sms-spam would have had
  zero features even if packed; airbnb `name`/`host_name` dropped) —
  text feature support is the M6 foundation-adapter scope.

## Verdict

No crashes in the engine itself across 9 runs; both new bugs live at the
edges (ingest encoding, budget accounting for non-search phases). Scores
are strong wherever the data is sound, and the two suspicious perfect
scores were traced to dataset leakage, arguing for a leakage screen at
the confirm gate.
