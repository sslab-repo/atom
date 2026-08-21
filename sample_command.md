# ATOM — Sample Commands (try it yourself)

End-to-end examples on three sample datasets. Generate the datasets, then run any
command. Verified on macOS (Apple Silicon, MPS) with the torch-enabled build.

## 0. Setup

Point `atom` at the torch-enabled venv (so the deep models + GPU work), and go to
the sample folder:

```bash
alias atom="$HOME/Dev/sslab-git/atom/.venv/bin/atom"
cd ~/Download/Sample     # customers.csv (binary), plants.csv (multiclass), machines.csv (time-series)
```

Generate the three datasets (if not already present):

```bash
python3 - <<'PY'
import csv, random, os
D = os.path.expanduser("~/Download/Sample"); os.makedirs(D, exist_ok=True); rng = random.Random(42)
with open(f"{D}/customers.csv","w",newline="") as f:            # BINARY: churn
    w=csv.writer(f); w.writerow(["tenure_months","monthly_charges","support_tickets","contract","churn"])
    for _ in range(1200):
        c=rng.random()<0.35
        w.writerow([max(0,int(rng.gauss(12 if c else 36,10))), round(rng.gauss(80 if c else 60,20),2),
                    max(0,int(rng.gauss(4 if c else 1,2))),
                    rng.choice(["monthly","monthly","yearly"] if c else ["yearly","yearly","monthly"]),
                    "yes" if c else "no"])
with open(f"{D}/plants.csv","w",newline="") as f:               # MULTICLASS: 3 species
    w=csv.writer(f); w.writerow(["petal_len","petal_wid","sepal_len","sepal_wid","species"])
    C={"setosa":(1.5,0.3,5.0,3.4),"versicolor":(4.3,1.3,5.9,2.8),"virginica":(5.6,2.0,6.6,3.0)}
    for _ in range(900):
        sp=rng.choice(list(C)); pl,pw,sl,sw=C[sp]
        w.writerow([round(rng.gauss(pl,0.4),2),round(rng.gauss(pw,0.25),2),
                    round(rng.gauss(sl,0.4),2),round(rng.gauss(sw,0.35),2),sp])
with open(f"{D}/machines.csv","w",newline="") as f:             # TIME-SERIES: sensor sequences
    w=csv.writer(f); w.writerow(["machine_id","timestamp","temperature","vibration","status"])
    for mid in range(400):
        fail=rng.random()<0.4
        for t in range(24):
            w.writerow([f"M{mid:03d}",t, round(60+(t*0.9 if fail else 0)+rng.gauss(0,2),2),
                        round(0.5+(t*0.05 if fail else 0)+rng.gauss(0,0.15),3), "failing" if fail else "healthy"])
print("wrote 3 datasets to", D)
PY
```

## 1. Inspect the data & the method registry

```bash
atom pack customers.csv --target churn --name customers
atom inspect customers
atom modules list          # 15 classifiers + conv1d/lstm (torch present)
```

## 2. Binary classification (the main case)

```bash
atom run customers --time-budget 60 --yes --out runs
```

See which method won and rank the finalists:

```bash
python3 -c "
import json,glob
m=json.load(open(sorted(glob.glob('runs/customers-*/metrics.json'))[-1]))
print('optimized:', m['primary_metric'], '| test:', {k:round(v,4) for k,v in m['test'].items()})
for c in sorted(m['candidates'], key=lambda c:-c['val_score_oriented']):
    print(f\"  {c['val_score_oriented']:.4f}  {c['pipeline']['method']['name']}\")
"
```

## 3. Multiclass classification

```bash
atom pack plants.csv --target species --name plants
atom run plants --time-budget 60 --yes --out runs
```

## 4. Choose methods, or set the split ratio

```bash
atom run plants --methods neural-net-mlp,support-vector-machine,random-forest --time-budget 30 --yes --out runs
atom pack customers.csv --target churn --name cust_70   --split 0.7/0.15/0.15    # custom ratio
atom pack customers.csv --target churn --name cust_auto --split auto             # size-based
```

## 5. Binary "one class vs. the rest"

```bash
python3 - <<'PY'
import csv
TARGET, POSITIVE = "species", "setosa"                 # class of interest
rows=list(csv.DictReader(open("plants.csv")))
with open("plants_bin.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    for r in rows: r[TARGET]="positive" if r[TARGET]==POSITIVE else "rest"; w.writerow(r)
PY
atom pack plants_bin.csv --target species --name plants_bin
atom run plants_bin --time-budget 30 --yes --out runs      # optimizes ROC-AUC
```

## 6. Time-series — torch-free (works on any machine)

```bash
atom pack machines.csv --target status --type timeseries --time timestamp --group machine_id --name machines
atom run machines --time-budget 60 --yes --out runs
```

Groups rows into per-machine sequences, extracts summary features
(mean/std/min/max/last/slope per channel), classifies with the 15 classifiers.

## 7. Time-series — deep models (conv1d / LSTM) on the GPU

```bash
atom pack machines.csv --target status --type timeseries --time timestamp --group machine_id --ts-layout raw --name machines_raw
atom run machines_raw --methods conv1d-classifier,lstm-classifier --time-budget 60 --yes --out runs
```

The log prints `device: mps (torch 2.13.0)` and `raw sequences: 2 channels x 24 steps` —
the nets train on the Apple GPU. Force CPU instead:

```bash
ATOM_DEVICE=cpu atom run machines_raw --methods conv1d-classifier --time-budget 60 --yes --out runs
```

## Notes

- Outputs land in `runs/<name>-<timestamp>/`: `metrics.json` (scores),
  `model/pipeline.onnx` (deployable model + `manifest.json`), `provenance/`.
- `machines.csv` has a strong trend, so you'll see a **"suspiciously-perfect"**
  warning — that's the leak screen working; the models score near-perfectly on it.
- The deep sequence models currently save the native model (`deployable=False`);
  they run/predict fine — ONNX export for the torch tier is the next feature.
- Without the alias, `atom` from `scripts/install.sh` is **torch-free** (classical
  tier only). Use the alias above for the conv1d/lstm deep models.
