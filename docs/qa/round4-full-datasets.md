# Round-4 — Bugfix iter-3 + 10 New Datasets (FULL data, no row caps)

Commands (identical for all; targets vary):
```
atom fetch kaggle:<slug> --target <col> --out r4 --name <name>
atom run r4/<name> --time-budget 120 --max-rows 10000000 --yes --kb r4-kb --out r4-runs
atom run r4/country-data --task clustering ...   # unlabeled -> forced clustering
```
Fixes this round: BUG-1 encoding-tolerant packing (sms-spam now ingests);
BUG-2 export budgeting (weather 163s->82s; phases logged); BUG-3 admission
busy-loop; leakage screen + no-features guard (improvements, kept).

| Dataset (discipline) | Task | AMP | Elapsed/120s | Test result |
|---|---|---|---|---|
| credit-default (credit risk, 30k) | clf | ✅ | 106s | auc 0.789 |
| body-performance (sports sci, 13k) | 4-class | ❌ gated | 117s | f1 0.749 |
| bankruptcy (accounting, 6.8k, imbal.) | clf | ❌ gated | 107s | auc 0.949 |
| eu-bank-churn (banking, 10k) | clf | ✅ | 105s | auc 0.860 |
| body-fat (kinesiology, 252) | reg | ✅ | 102s | r² 0.9986 ⚠ leak flagged: Density (r=0.996 — Siri equation) |
| diamonds (gemology, 54k) | reg | ✅ | 125s | r² 0.9999 |
| student-grades (education, 649) | clf | ✅ | 125s | f1 0.459 (ordinal G3; weak, honest) |
| heart-failure (clinical, 299) | clf | ✅ | 103s | auc 0.861 |
| ds-salaries (labor econ, 607) | reg | ✅ | 103s | r² 0.963 (note: 'salary' local-currency feature r<0.98, not flagged) |
| country-data (dev. economics) | clustering | ❌ (KMeans export) | 102s | silhouette 0.721 |

10/10 end-to-end, 0 crashes, worst overrun +4% (was +168%), 7/10 deployable
(3 honest parity-gate rejections). Leak screen caught a real physical leak
(body-fat Density). Substitution: uciml/abalone-dataset requires Kaggle
auth consent -> replaced with shivam2503/diamonds.
Notes for next round: ordinal targets (student G3 classified at 21 levels);
sub-threshold leakage (ds-salaries r≈0.9); multiclass member parity deltas.
