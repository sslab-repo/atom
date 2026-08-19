# ATOM — Command Manual

ATOM (**AuTO ai Machine**) is a command-line AutoAI platform. You point it at a
dataset and it profiles the data, infers the task, searches models under a time
budget, evaluates on a locked test split, and emits a **deployable model
package** (a fused ONNX graph verified to reproduce the trained model) plus full
provenance.

This manual covers using the `atom` command on **Linux and macOS**. It documents
every subcommand, flag, output, and the common workflows.

- Applies to: `atom` 0.1 (package `atom-ai`), Python ≥ 3.10.
- Runs identically on RHEL/Ubuntu/other Linux and macOS (Intel or Apple
  Silicon). ATOM is pure Python + portable wheels (numpy, pyarrow, scikit-learn,
  onnxruntime, skl2onnx); nothing platform-specific.

---

## 1. Installation

### 1.1 Self-contained installer (recommended for lab / shared machines)

```bash
git clone <atom-repo> && cd atom
bash scripts/install.sh
source ~/.bashrc     # Linux (bash)
source ~/.zshrc      # macOS (zsh)
atom modules verify  # self-test
```

Installs everything under `~/atom` (private virtualenv, launcher, config, data,
runs) with **no root on Linux**. See `docs/install.md` for the folder layout,
`--prefix`, `--no-rc`, and `--uninstall` options. Re-running upgrades in place
and preserves `~/atom/config/atom.env`.

### 1.2 pip install (for developers / embedding in another app)

```bash
python3 -m venv .venv && source .venv/bin/activate    # macOS/Linux
pip install -e .                    # core
pip install -e '.[kaggle]'          # + `atom fetch kaggle:<slug>`
pip install -e '.[dev]'             # + pytest, ruff
```

Core dependencies (installed automatically): `pyarrow`, `numpy`,
`scikit-learn`, `skl2onnx`, `onnxruntime`. Optional extras: `kaggle`
(kagglehub), `boosted` (xgboost/lightgbm — auto-registered if present),
`imbalanced` (imbalanced-learn).

Verify:

```bash
atom --help
atom modules verify     # expect "17/17 modules pass the smoke gate"
```

### 1.3 Platform notes

| | Linux (RHEL 10 / Ubuntu) | macOS (Intel / Apple Silicon) |
|---|---|---|
| Python | system `python3` (3.10+) | 3.10+, or `brew install python@3.12` |
| Privileges | user account only | admin only if Homebrew must install Python |
| Wheels | all deps ship manylinux wheels | universal2 / arm64 wheels |

First install needs network access (PyPI). After that, `atom run` works fully
offline; only `atom fetch` needs the network.

---

## 2. Concepts (30-second version)

- **ADP — ATOM Dataset Package**: a self-describing dataset (a folder or `.zip`)
  with a `manifest.json`, typed Parquet splits (train/val/test), and roles
  (target, id, group, time). You create one with `atom pack` / `atom fetch`, or
  bring your own.
- **Run**: `atom run <ADP>` → search → locked-test evaluation → outputs.
- **AMP — ATOM Model Package**: the run's deployable artifact — a fused ONNX
  graph (`model/pipeline.onnx`) that serves standalone via onnxruntime, plus a
  `manifest.json` (input/output signature, parity report, lineage). A native
  sklearn fallback (`native/model.pkl`) is always written too.

Typical flow: **pack/fetch → inspect → run → deploy the ONNX package**.

---

## 3. Command reference

Global form: `atom <command> [args] [options]`. Every command returns exit
code `0` on success, non-zero on error (usable in scripts and CI).

### 3.1 `atom pack` — CSV → ADP

Convert a loose CSV into a typed ATOM Dataset Package.

```bash
atom pack <csv> [--target COL] [--name NAME] [--out DIR]
```

| Option | Default | Meaning |
|---|---|---|
| `<csv>` | — | path to the source CSV (positional, required) |
| `--target COL` | none | label/target column; omit for unlabeled data (clustering/anomaly) |
| `--name NAME` | CSV stem | package name |
| `--out DIR`, `-o` | `.` | output directory |

Handles messy real-world CSVs automatically: dirty numerics (`"N/A"`, `"?"`,
`"1,234"`, `"$50"`, `"27,3"` locale decimals), datetime columns (expanded to
year/month/day/day-of-week/epoch features), categorical one-hot, and rows with a
missing target (dropped with a stated count). A wrong `--target` prints the
available columns and a suggestion.

```bash
atom pack sales.csv --target revenue --name sales --out packages/
# → wrote ADP: packages/sales
```

### 3.2 `atom fetch` — Kaggle dataset → ADP

Download a Kaggle dataset and pack it in one step. Requires the `kaggle` extra
and Kaggle credentials in `~/.kaggle/kaggle.json`.

```bash
atom fetch kaggle:<owner/dataset> [--target COL] [--file NAME.csv] [--name NAME] [--out DIR]
```

| Option | Default | Meaning |
|---|---|---|
| `<source>` | — | `kaggle:<owner>/<dataset>` (only `kaggle:` scheme supported) |
| `--target COL` | none | target column |
| `--file NAME` | largest CSV | pick a specific CSV inside the dataset |
| `--name NAME` | slug | package name |
| `--out DIR`, `-o` | `.` | output directory |

```bash
atom fetch kaggle:uciml/iris --target Species --name iris
```

Consent-gated datasets print a clear message telling you to accept the dataset's
terms on kaggle.com first.

### 3.3 `atom pack-images` — image folder → ADP

Class-per-subfolder image directory into an image ADP (data plane only; model
training for images lands with the foundation adapters).

```bash
atom pack-images <folder> [--name NAME] [--out DIR]
```

Expected layout: `folder/<class_name>/<image files>`.

### 3.4 `atom inspect` — profile a dataset

Fingerprint a package: modality, split sizes, roles, target distribution, data
quality flags, and per-column types.

```bash
atom inspect <pkg.zip|dir> [--json] [--sample-rows N] [--columns N]
```

| Option | Default | Meaning |
|---|---|---|
| `<package>` | — | ADP folder or `.zip` |
| `--json` | off | emit the full fingerprint as JSON (for tooling) |
| `--sample-rows N` | 50000 | rows sampled from train for profiling |
| `--columns N` | 12 | max columns to display |

```
$ atom inspect packages/sales
package    : sales  [atom-dataset-v1]
modality   : tabular / supervised
samples    : train=7,000  val=1,500  test=1,500
columns    : 14  (profiled 7,000 train rows)
roles      : id=sample_id, target=['revenue']
target dist: ...
flags      : numeric-coerced:price; datetime:order_date
--- columns (first 12) ---
  order_date   datetime  missing=0.0% distinct≈365
  ...
```

Use `--json` to feed the fingerprint into another program.

### 3.5 `atom run` — train & export a model

The core command. Profiles → confirm gate → budgeted search → locked-test
evaluation → writes the model package + provenance.

```bash
atom run <pkg> [--target COL] [--task FAMILY] [--time-budget S]
               [--max-trials N] [--min-trials N] [--max-rows N]
               [--out DIR] [--kb DIR] [--seed N] [--yes]
               [--include-experimental]
```

| Option | Default | Meaning |
|---|---|---|
| `<package>` | — | ADP folder or `.zip` |
| `--target COL` | manifest role | target column; overrides/completes the ADP's declared roles |
| `--task FAMILY` | inferred | force `classification` \| `regression` \| `clustering` \| `anomaly-detection` |
| `--time-budget S` | 120 | wall-clock **search** budget in seconds |
| `--max-trials N` | none | cap on trials (stops at whichever bound hits first) |
| `--min-trials N` | none | ensure at least N trials even if time is short |
| `--max-rows N` | 100000 | cap training rows loaded (spread across the file) |
| `--out DIR` | `runs` | where run outputs are written |
| `--kb DIR` | `$ATOM_HOME/metakb` or `~/.atom/metakb` | meta-knowledge base (warm-starts from similar past runs) |
| `--seed N` | 0 | RNG seed — runs are deterministic |
| `--yes`, `-y` | off | skip the interactive confirm gate (non-interactive/CI) |
| `--include-experimental` | off | let unpromoted modules join the search |

**The confirm gate.** Before spending budget, ATOM prints the inferred task
(family, target, metric, class count, and any advisories like class imbalance,
target leakage, or ordinal-target hints) and asks for confirmation. In a
non-interactive shell (pipe/cron/Slurm) it auto-proceeds; pass `--yes` to skip
it explicitly.

```bash
atom run packages/sales --time-budget 120 --yes
```

Task is inferred automatically: a declared/`--target` column → classification
or regression (by cardinality); no target → clustering/anomaly. Override with
`--task`.

**Terminal output** ends with:

```
=== result ===
  final    : single   trials: 204   elapsed: 118s
  val      : roc_auc=0.8712
  test     : accuracy=0.9310  balanced_accuracy=0.8820  f1_macro=0.8790  roc_auc=0.9120
  artifacts: runs/sales-20260819-101112
  AMP: deployable=True (1 ONNX graph(s), parity ok, selected single)
```

### 3.6 `atom modules` — inspect the algorithm registry

```bash
atom modules list      # every registered module: kind, name, version, lifecycle, families
atom modules verify    # contract smoke-test each module (returns non-zero on any failure)
```

`verify` is the health check — use it after install/upgrade and in CI.

---

## 4. Run outputs (the model package)

Each run writes a timestamped directory under `--out`:

```
runs/sales-20260819-101112/
├── manifest.json          # AMP: input/output signature, ONNX graph list,
│                          #   parity report, deployable flag, lineage
├── model/pipeline.onnx    # the deployable fused graph (the AMP)
├── native/model.pkl       # native sklearn pipeline (fallback / reference)
├── metrics.json           # primary metric, final kind, val + locked-test
│                          #   metrics, all candidate scores
├── provenance/
│   ├── run.json           # package id, task, budget, data shape, phase
│   │                      #   timings, leak warnings, seed
│   └── trials.jsonl       # one line per trial (config, score, cost)
└── README.md              # one-line human summary
```

`deployable: true` in `manifest.json` means the ONNX graph was verified to
reproduce the trained model's outputs (the parity gate). If a model can't be
faithfully exported, ATOM ships `deployable: false` and you use `native/` —
the run never fails for export reasons.

---

## 5. Serving a model package

The AMP is a standard ONNX graph — serve it anywhere onnxruntime runs
(Python, C++, C#, Java, Node), no ATOM or scikit-learn needed. The input/output
signature is in `manifest.json` under `signature`.

```python
import json, numpy as np, onnxruntime as ort

run = "runs/sales-20260819-101112"
sig = json.load(open(f"{run}/manifest.json"))["signature"]
feature_order = sig["input"]["features"]        # exact column order the graph expects

sess = ort.InferenceSession(f"{run}/model/pipeline.onnx")
X = np.array([[...]], dtype=np.float32)          # shape (n_rows, len(feature_order))
outputs = sess.run(None, {"X": X})
# classification -> outputs = [labels, probabilities];  regression -> [predictions]
```

- Feed features as `float32` in the order given by `signature.input.features`.
- Classification graphs output `label` then `probabilities`; the class order is
  `signature.label_map`. Regression graphs output `prediction`.
- For imbalanced binary tasks, `metrics.json` may include a
  `decision_threshold` — apply it to the positive-class probability instead of
  the default 0.5 to reproduce the reported balanced metrics.

The native fallback (for parity-gated models or quick Python use):

```python
import pickle
model = pickle.load(open(f"{run}/native/model.pkl", "rb"))
```

---

## 6. Configuration (environment variables)

| Variable | Effect |
|---|---|
| `ATOM_HOME` | base dir for the meta-KB and caches (default `~/.atom`); the meta-KB lives at `$ATOM_HOME/metakb` |

The installer keeps all user config in `~/atom/config/atom.env` (sourced by
your shell rc), so settings survive upgrades. You can also set variables inline:

```bash
ATOM_HOME=/scratch/$USER/atom atom run packages/sales --yes
```

Kaggle credentials for `atom fetch` live in `~/.kaggle/kaggle.json`
(0600 perms), per Kaggle's standard.

---

## 7. Workflows & recipes

### 7.1 Bring your own CSV → deployable model

```bash
atom pack mydata.csv --target label --out pkgs      # CSV → ADP
atom inspect pkgs/mydata                             # sanity-check profiling
atom run pkgs/mydata --time-budget 300 --yes        # train (5 min budget)
# → deployable ONNX at runs/mydata-<ts>/model/pipeline.onnx
```

### 7.2 Unlabeled data (clustering / anomaly)

```bash
atom pack sensors.csv --name sensors --out pkgs      # no --target
atom run pkgs/sensors --task clustering --yes         # or --task anomaly-detection
```

### 7.3 Non-interactive / scripted (CI, cron, Slurm)

Always pass `--yes` (or rely on non-TTY auto-proceed) and set an explicit
`--out`. ATOM does not manage compute itself — under Slurm, request cores in the
batch script; ATOM uses the cores the job is given.

```bash
#!/bin/bash
#SBATCH -c 8 --mem=16G -t 00:30:00
export ATOM_HOME=/scratch/$USER/atom
atom run "$DATA/pkgs/study" --time-budget 600 --max-rows 2000000 \
         --out "$RESULTS" --yes
```

### 7.4 Reproducibility

Runs are deterministic given the same package, code, and `--seed`. Provenance
(`provenance/run.json`, `trials.jsonl`) records the exact task, budget, data
shape, and every trial, so any result is auditable.

### 7.5 Faster iteration vs. thorough search

- Quick look: `--time-budget 30` (or `--max-trials 50`).
- Thorough: raise `--time-budget`; ATOM keeps searching and finalizes the best
  models it found. The budget governs **search**; final full-fidelity refits and
  export happen after and may add time.

---

## 8. Exit codes & troubleshooting

| Symptom | Cause / fix |
|---|---|
| `atom: command not found` | `source ~/.bashrc`/`~/.zshrc` (installer), or activate the venv (pip install) |
| `unrecognized arguments: --target X` | quote multi-word targets: `--target "Chance of Admit"` |
| `target column 'X' is not in <csv>` | ATOM lists the real columns + a suggestion; fix the name |
| `no usable features — every column was dropped` | text-only / all-high-cardinality data; text modality needs the foundation adapters |
| `no successful trials: N failed — <error>` | the underlying model error is surfaced; usually a data issue (e.g. NaNs a method can't take) |
| `'<slug>' requires a Kaggle login/consent` | accept the dataset terms on kaggle.com, then re-`fetch` |
| `AMP: deployable=False` | model couldn't be faithfully exported to ONNX; use `native/model.pkl` |
| kagglehub `ImportError` on `fetch` | `pip install 'atom-ai[kaggle]'` |

Health check any install: `atom modules verify` (expect `17/17 ... pass`).

---

## 9. Quick reference card

```
atom pack <csv> --target COL --out DIR         CSV  -> ADP
atom fetch kaggle:<owner/ds> --target COL       Kaggle -> ADP   (needs [kaggle])
atom pack-images <folder>                       image folder -> ADP
atom inspect <pkg> [--json]                     profile a dataset
atom run <pkg> --time-budget S --yes            train -> ONNX model package
atom modules list | verify                      registry / health check

deploy: runs/<name>-<ts>/model/pipeline.onnx    (signature in manifest.json)
```
