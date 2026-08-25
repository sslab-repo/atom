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

**Fresh machine** (also installs OS prerequisites via brew/dnf/apt — macOS,
RHEL 10, Debian 13):

```bash
git clone <atom-repo> && cd atom
bash scripts/setup.sh        # OS packages + ATOM + health check  (--yes for non-interactive)
source ~/.bashrc             # Linux (bash)  ·  ~/.zshrc on macOS
bash scripts/sample_run.sh   # sample end-to-end test
```

**Python 3.10+ already present** — install just ATOM (no OS packages):

```bash
bash scripts/install.sh
source ~/.bashrc     # Linux (bash)  ·  ~/.zshrc on macOS
atom modules verify  # self-test
```

Installs everything under `~/atom` (private virtualenv, launcher, config, data,
runs) with **no root on Linux**. See `docs/install.md` for the folder layout,
`--prefix`, `--no-rc`, `--no-system`, and `--uninstall` options. Re-running
upgrades in place and preserves `~/atom/config/atom.env`.

### 1.2 pip install (for developers / embedding in another app)

```bash
python3 -m venv .venv && source .venv/bin/activate    # macOS/Linux
pip install -e .                    # core
pip install -e '.[torch]'           # + deep tier (conv1d/lstm; CUDA/MPS if present)
pip install -e '.[kaggle]'          # + `atom fetch kaggle:<slug>`
pip install -e '.[dev]'             # + pytest, ruff
```

Core dependencies (installed automatically): `pyarrow`, `numpy`,
`scikit-learn`, `skl2onnx`, `onnxruntime`. Optional extras: `torch` (the
optional PyTorch deep tier — `conv1d`/`lstm` for raw time-series, auto-registered
when present), `kaggle` (kagglehub), `boosted` (xgboost/lightgbm — auto-registered
if present), `imbalanced` (imbalanced-learn). Without `torch`, ATOM runs the full
classical tier; with it, `atom modules verify` reports two extra modules.

Verify:

```bash
atom --help
atom modules verify     # expect "28/28 modules pass the smoke gate"
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
| `--split TRAIN/VAL/TEST` | `0.8/0.1/0.1` | split ratios, e.g. `0.7/0.15/0.15` (normalized); or `auto` (size-based) |
| `--type` | `tabular` | declared input type: `tabular`, `text`, or `timeseries` (ADR-0008; images use `pack-images`). Routes methods at run time |
| `--time COL` / `--group COL` | — | required with `--type timeseries`: the time (ordering) and sequence-id (grouping) columns |
| `--ts-layout` | `features` | time-series representation: `features` (per-sequence summary stats, torch-free) or `raw` (padded sequences for the deep conv1d/lstm models) |

**Time-series (`--type timeseries`)** groups rows by `--group` (one sequence per
entity), orders by `--time`, and packs one row per sequence. Two layouts
(`--ts-layout`):
- `features` (default, **torch-free, any machine**): per-sequence summary stats
  (mean/std/min/max/last/slope per channel) → the 15 classifiers run on them.
- `raw`: padded sequences (channel-major) so the **deep sequence models**
  `conv1d-classifier` / `lstm-classifier` (PyTorch tier) can learn temporal
  patterns; the tabular classifiers also run on them.

The split is per-sequence (no entity leaks). The deep models train on the
resolved device (cuda/mps/cpu) and are searched only when PyTorch is installed.

```bash
atom pack sensors.csv --target status --type timeseries --time ts --group machine_id
atom run sensors --time-budget 120 --yes                             # torch-free features

atom pack sensors.csv --target status --type timeseries --time ts --group machine_id --ts-layout raw
atom run sensors --methods conv1d-classifier,lstm-classifier --yes   # deep (needs [torch])
```

`--split` controls the train / validation / test partition (a deterministic
hash split). A **validation** split is always kept — ATOM selects the model on
it, so all three fractions must be > 0. `auto` picks by dataset size:
`< 1k rows → 70/15/15`, `1k–100k → 80/10/10`, `≥ 100k → 90/5/5`.

Handles messy real-world CSVs automatically: dirty numerics (`"N/A"`, `"?"`,
`"1,234"`, `"$50"`, `"27,3"` locale decimals), datetime columns (expanded to
year/month/day/day-of-week/epoch features), categorical one-hot, and rows with a
missing target (dropped with a stated count). A wrong `--target` prints the
available columns and a suggestion.

```bash
atom pack sales.csv --target revenue --name sales --out packages/
atom pack sales.csv --target revenue --split 0.7/0.15/0.15    # custom ratio
atom pack sales.csv --target revenue --split auto             # size-based
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
| `--methods A,B,C` | all | restrict the search to these methods (comma-separated); unknown names error with the available list |

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

**During the search**, each rung line names the winning algorithm and its exact
hyperparameters for that rung, so you can watch what is being tried:

```
rung f=0.1:  9/9 ok, best roc_auc=0.8408 extra-trees(n_estimators=300,max_depth=10,min_samples_leaf=12,class_balance=none)  [9s elapsed, ~26s left]
rung f=0.33: 3/3 ok, best roc_auc=0.8556 linear-discriminant-analysis(solver=svd)  [14s elapsed, ~22s left]
rung f=1:    1/1 ok, best roc_auc=0.8691 random-forest(n_estimators=265,max_depth=15,min_samples_leaf=1,class_balance=balanced)  [5s elapsed, ~30s left]
```

(`f=` is the fidelity — the fraction of training rows used at that rung; the
search promotes survivors from `0.1 → 0.33 → 1.0`.)

**At the end**, a full-field leaderboard lists every method with its **validation**
metrics (primary metric first, then accuracy / balanced-accuracy / f1) at that
method's best trial, `*` on the method(s) feeding the final model, and a `@f<...>`
tag on any method that never reached full data — then the result summary:

```
=== all 17 methods searched — validation metrics at each method's best trial (sorted by roc_auc; * = in the final model) ===
  * roc_auc=0.8691  acc=0.8066  bal_acc=0.7905  f1_macro=0.7895   random-forest            (233 trial(s))   -> final val 0.8827
    roc_auc=0.8637  acc=0.8272  bal_acc=0.7953  f1_macro=0.8044   neural-net-mlp           (160 trial(s))   -> final val 0.8645
    roc_auc=0.8583  acc=0.8080  bal_acc=0.7832  f1_macro=0.7872   linear-discriminant-analysis  (298 trial(s))
    ...
    roc_auc=0.7519  acc=0.6982  bal_acc=0.6564  f1_macro=0.6423   conv1d-classifier        (152 trial(s) @f0.1)
    skipped  support-vector-machine  — <most-common error line for that method>
    skipped  lstm-classifier         — not reached within the time/trial budget
(sorted by cross-validated search score; '-> final val' = the finalist re-scored on the held-out val split after a full-data refit — the number selection used.)
(* = chosen into the final model (greedy blend) by that held-out score, so it can rank differently from the leaderboard.)
=== result ===
  final    : single   trials: 204   elapsed: 118s
  val      : roc_auc=0.8712
  test     : accuracy=0.9310  balanced_accuracy=0.8820  f1_macro=0.8790  roc_auc=0.9120
  artifacts: runs/sales-20260819-101112
  AMP: deployable=True (1 ONNX graph(s), parity ok, selected single)
```

The same leaderboard is saved to `metrics.json` under `"leaderboard"` — one entry
per method with `status`, `best_score`, **`metrics`** (the full per-metric dict),
**`config`** (the winning hyperparameters, so you can reproduce or deploy that
method directly), `best_fidelity`, **`final_val`** (the finalist's held-out val
score, or `null` if the method wasn't finalized), `trials`/`ok`/`errors`,
`reason`, and `in_final`. Only the winner is scored on the locked test set — by
design, so the held-out estimate stays honest — so per-method numbers are
validation scores.

**Two rankings, on purpose.** The leaderboard is sorted by each method's
best **search** score — cross-validated on small data (val split < 1000 rows), so
it's stable but optimistic. The final model is chosen differently: ATOM re-fits
the top pipelines on the full training data, scores them **once on the held-out
val split** (`-> final val`), and greedily blends the combination that maximizes
that held-out score. So `*` (blend membership) follows the held-out numbers and
can differ from the leaderboard order — a method can top the search leaderboard
yet be re-scored lower at full fidelity (visible as a lower `-> final val`) and so
not make the blend.

### 3.6 `atom modules` — inspect the algorithm registry

```bash
atom modules list      # every registered module: kind, name, version, lifecycle, families
atom modules verify    # contract smoke-test each module (returns non-zero on any failure)
```

`verify` is the health check — use it after install/upgrade and in CI.

### 3.7 Algorithms searched

`atom run` searches every applicable method for the task and picks the most
accurate. `atom modules list` prints the live set; the stable built-ins are:

| Task family | Methods searched |
|---|---|
| classification (tabular) | logistic-regression, sgd-classifier, perceptron, linear/quadratic-discriminant-analysis, gaussian-naive-bayes, k-nearest-neighbors, support-vector-machine, decision-tree, random-forest, extra-trees, adaboost, gradient-boosting, hist-gradient-boosting, **neural-net-mlp** (feed-forward neural network) |
| classification (raw time-series) | **conv1d-classifier**, **lstm-classifier** — PyTorch deep tier, added when the package is `--type timeseries --ts-layout raw` and PyTorch is installed |
| regression | ridge, random-forest-reg, hist-gradient-boosting-reg |
| clustering | kmeans, gaussian-mixture |
| anomaly-detection | isolation-forest, lof-novelty |
| dimension-reduction | pca |

`neural-net-mlp` is the deep-learning classifier for **tabular** data — it
competes head-to-head with the others and wins only when it's genuinely more
accurate. For **ordered sequences** (raw time-series), the PyTorch tier adds
`conv1d-classifier` and `lstm-classifier`, which learn temporal patterns the
summary-feature classifiers can't; they train on the GPU when one is present and
still export to a portable ONNX graph (see §5). CNNs/LSTMs are *not* tabular
classifiers — they only join the search for the raw time-series modality; GANs
are generative models, not classifiers, so ATOM does not search them.

The **deep tier is optional**: `conv1d-classifier`/`lstm-classifier` register
only when `import torch` succeeds. On a machine without PyTorch they are simply
absent and the classical tier runs unchanged. `xgboost`/`lightgbm` register
automatically if installed (`pip install 'atom-ai[boosted]'`).

Restrict the search to specific methods with `atom run --methods A,B,C` — useful
to compare a shortlist or avoid diluting the budget across all 15 classifiers:

```bash
atom run mydata --methods neural-net-mlp,support-vector-machine,random-forest --yes
```

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
reproduce the trained model's outputs (the parity gate). This covers both the
classical (sklearn→ONNX) and deep-tier (PyTorch→ONNX) models — a single
`conv1d`/`lstm` model exports to a self-contained graph and passes the same gate.
If a model can't be faithfully exported, ATOM ships `deployable: false` and you
use `native/` — the run never fails for export reasons. (A deep model selected
*inside an ensemble* is not yet fused to ONNX and ships `deployable: false`; a
single deep model is fully deployable.)

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
# read manifest signature.outputs for the exact output list (see below)
```

- Feed features as `float32` in the order given by `signature.input.features`.
- **Always read `signature.outputs`** for the graph's actual outputs — they vary
  by model:
  - Classical (sklearn) classifiers output `["label", "probabilities"]`.
  - Deep-tier (`conv1d`/`lstm`) classifiers output `["probabilities"]` only —
    the graph is a self-contained raw→proba net; derive the label yourself with
    `label_map[probabilities.argmax(axis=1)]`.
  - Regression graphs output `["prediction"]`.
  - In every case the class order is `signature.label_map`.
- Deep-tier graphs are **self-preprocessing**: NaN-fill, standardization, and the
  sequence reshape are baked into the graph, so you feed the same raw feature
  vector — no external scaling needed.
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
| `ATOM_DEVICE` | force the PyTorch deep tier onto a device: `cpu`, `cuda`, or `mps`. Default is auto-detect (`cuda` → `mps` → `cpu`). No effect on the classical tier |

**Device auto-detection (deep tier).** When PyTorch is installed, the deep
models pick the best device automatically — CUDA GPU, then Apple-Silicon MPS,
then CPU — and the run log prints e.g. `device: mps (torch 2.13.0)`. The GPU only
accelerates training; the exported ONNX graph always serves on CPU onnxruntime.
Override with `ATOM_DEVICE=cpu` to force CPU training.

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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK
atom run "$DATA/pkgs/study" --time-budget 600 --max-rows 2000000 \
         --out "$RESULTS" --yes
```

**Many datasets on a cluster** → run them as a Slurm job array (one dataset per
array task). Full guide + ready-to-run scripts:
[`docs/slurm.md`](slurm.md) and [`scripts/slurm/`](../scripts/slurm/).

### 7.4 Reproducibility

Runs are deterministic given the same package, code, and `--seed`. Provenance
(`provenance/run.json`, `trials.jsonl`) records the exact task, budget, data
shape, and every trial, so any result is auditable.

### 7.5 Faster iteration vs. thorough search

- Quick look: `--time-budget 30` (or `--max-trials 50`).
- Thorough: raise `--time-budget`; ATOM keeps searching and finalizes the best
  models it found. The budget governs **search**; final full-fidelity refits and
  export happen after and may add time.

### 7.6 Classification: find & compare the most accurate method

```bash
atom pack data.csv --target species --name mydata
atom run mydata --time-budget 300 --yes
```

ATOM optimizes macro-F1 for multiclass (ROC-AUC for binary) and reports accuracy
too. See which method won and rank the finalists:

```bash
python3 -c "
import json, glob
m = json.load(open(sorted(glob.glob('runs/mydata-*/metrics.json'))[-1]))
print('selected:', m['final'], '| optimized:', m['primary_metric'])
print('test:', {k: round(v,4) for k,v in m['test'].items()})
for c in sorted(m['candidates'], key=lambda c: -c['val_score_oriented']):
    print(f\"  {c['val_score_oriented']:.4f}  {c['pipeline']['method']['name']}\")
"
```

### 7.7 Binary — one class vs. the rest

ATOM treats a 2-value target as binary. Collapse a multiclass label into
"target vs. rest" first (replace the column so the original can't leak):

```bash
python3 - <<'PY'
import csv
TARGET, POSITIVE = "species", "setosa"          # your column and the class of interest
rows = list(csv.DictReader(open("data.csv")))
with open("data_bin.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    for r in rows:
        r[TARGET] = "positive" if r[TARGET] == POSITIVE else "rest"
        w.writerow(r)
PY
atom pack data_bin.csv --target species --name mydata_bin
atom run mydata_bin --time-budget 300 --yes      # optimizes ROC-AUC, reports accuracy
```

### 7.8 Deep-learning classifiers

**Tabular** — the feed-forward network (`neural-net-mlp`) is searched
automatically, no special flag. To check whether it beat the classical methods,
use the ranking snippet in §7.6 and look for `neural-net-mlp`. Give it room with
a larger budget:

```bash
atom run mydata --time-budget 600 --yes
```

**Raw time-series** — pack with `--ts-layout raw` and the PyTorch deep tier
(`conv1d-classifier`, `lstm-classifier`) joins the search. They learn temporal
structure directly from the padded sequences and export to a portable ONNX graph
(`deployable=True`), so you can train on the GPU and serve on a no-torch box:

```bash
atom pack sensors.csv --target status --type timeseries \
     --time ts --group machine_id --ts-layout raw --name sensors_raw
atom run sensors_raw --methods conv1d-classifier,lstm-classifier --time-budget 300 --yes
# log prints e.g.  device: mps (torch 2.13.0)
#                  AMP: deployable=True (1 ONNX graph(s), parity ok, selected single)
```

Force CPU training with `ATOM_DEVICE=cpu` (see §6); the exported graph serves on
CPU either way. The deep tier needs PyTorch (`pip install 'atom-ai[torch]'`); if
it's absent these methods are simply not offered.

(For **image** classification, pack images with `atom pack-images`; CNN /
foundation-embedding models are the deferred adapters — see `docs/status.md`.)

### 7.9 Choosing the train/val/test split

```bash
atom pack data.csv --target y --split 0.7/0.15/0.15    # custom ratio (normalized)
atom pack data.csv --target y --split auto              # size-based (see §3.1)
```

A validation split is always kept (used to select the model), so all three
fractions must be > 0. Confirm what was written with `atom inspect <pkg>`.

---

## 8. Sample commands (try it on public datasets)

A self-contained walkthrough on five real, public datasets — each exercises a
different part of ATOM. Assumes `atom` is on your PATH (§1). Every command below
carries a comment describing what it does.

### 8.1 Download the datasets

```bash
mkdir -p ~/Downloads/sample && cd ~/Downloads/sample     # one folder for the samples
# five diverse, header-bearing, comma-separated CSVs (no login needed):
curl -fsSL -o titanic.csv  https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv   # binary
curl -fsSL -o iris.csv     https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv          # multiclass
curl -fsSL -o penguins.csv https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv      # multiclass, dirty
curl -fsSL -o diamonds.csv https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv      # multiclass, 54k rows
curl -fsSL -o diabetes.csv https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv           # binary
```

| File | Task | `--target` | What it exercises |
|---|---|---|---|
| `titanic.csv` (891) | binary | `Survived` | dirty real-world: free-text `Name`/`Ticket`/`Cabin`, missing `Age`, an id column — auto-cleaning |
| `iris.csv` (150) | multiclass (3) | `species` | fast smoke test |
| `penguins.csv` (344) | multiclass (3) | `species` | NaN rows + categorical `island`/`sex` |
| `diamonds.csv` (53,940) | multiclass (5) | `cut` | scale + quoted headers + mixed types |
| `diabetes.csv` (767) | binary | `Outcome` | all-numeric Pima diabetes |

### 8.2 Binary — Titanic (dirty real-world data)

```bash
# Convert titanic.csv into a package "titanic": auto-detect column types, drop the
# free-text/id columns (Name, Ticket, Cabin, PassengerId), fill missing Age,
# one-hot Sex/Embarked, and make train/val/test splits.
atom pack ~/Downloads/sample/titanic.csv --target Survived --name titanic

# Print the profile — modality, split sizes, roles, target balance, and which
# columns were coerced or dropped — before spending compute.
atom inspect titanic

# Search the classifiers under a 60s budget, evaluate the winner on the locked
# test split, export the model to runs/titanic-<timestamp>/. Binary → ROC-AUC.
atom run titanic --time-budget 60 --yes --out runs
```

### 8.3 Multiclass — Iris (fast)

```bash
atom pack ~/Downloads/sample/iris.csv --target species --name iris   # pack the 3-species set
atom run iris --time-budget 30 --yes --out runs                      # 30s search; 3 classes → macro-F1
```

### 8.4 Multiclass — Penguins (missing values + categoricals)

```bash
# Pack penguins: exercises the cleaning path — NaN rows plus categorical
# island/sex columns get imputed and encoded automatically.
atom pack ~/Downloads/sample/penguins.csv --target species --name penguins
atom run penguins --time-budget 45 --yes --out runs                  # predicts the 3 species
```

### 8.5 Multiclass at scale — Diamonds (54k rows)

```bash
# Pack the 53,940-row diamonds set (quoted headers, mixed numeric + categorical)
# to classify the "cut" grade — a scale/throughput test.
atom pack ~/Downloads/sample/diamonds.csv --target cut --name diamonds
atom run diamonds --time-budget 120 --yes --out runs                 # bigger budget for a bigger set
```

### 8.6 Binary — Pima diabetes

```bash
atom pack ~/Downloads/sample/diabetes.csv --target Outcome --name diabetes   # all-numeric, target 0/1
atom run diabetes --time-budget 60 --yes --out runs                          # binary → ROC-AUC
```

### 8.7 See which method won (and how every method did)

`atom run` already prints the full leaderboard at the end of the run (every method
with its best validation score, or why it was skipped). To reprint it from a
finished run — or pull it programmatically — read `metrics.json["leaderboard"]`:

```bash
# Reprint the full-field leaderboard for the newest run of a package (change the
# name): each method's validation metrics, trial count, winning config, and
# whether it fed the final model.
python3 -c "
import json, glob
m = json.load(open(sorted(glob.glob('runs/diabetes-*/metrics.json'))[-1]))   # change name
print('selected:', m['final'], '| optimized:', m['primary_metric'],
      '| test:', {k: round(v,4) for k,v in m['test'].items()})
for r in m['leaderboard']:
    star = '*' if r['in_final'] else ' '
    if r['status'] == 'scored':
        stats = '  '.join(f'{k}={v}' for k, v in r['metrics'].items())
        print(f\"  {star} {r['method']:<30s} {stats}  ({r['ok']} trials)  {r['config']}\")
    else:
        print(f\"    skipped  {r['method']:<30s} — {r['reason']}\")
"
```

### 8.8 Restrict the search to a shortlist

```bash
# Same run, but only search these 3 methods instead of all ~15 — focuses the whole
# budget on a shortlist (faster, or to compare specific models head-to-head).
atom run titanic --methods random-forest,hist-gradient-boosting,logistic-regression \
     --time-budget 30 --yes --out runs
```

### 8.9 Optional — the deep time-series tier (conv1d / lstm)

None of the five sets is a grouped time-series, so generate a small labeled sensor
dataset to exercise the PyTorch deep tier (needs `pip install 'atom-ai[torch]'`):

```bash
# Generate machines.csv: 400 machines × 24 timesteps, 2 channels (temperature,
# vibration), label failing/healthy. (Tidy grouped-TS CSVs aren't reliably
# downloadable, so we synthesize one.)
python3 - <<'PY'
import csv, random, os
p = os.path.expanduser("~/Downloads/sample/machines.csv"); rng = random.Random(42)
with open(p, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["machine_id","timestamp","temperature","vibration","status"])
    for mid in range(400):
        fail = rng.random() < 0.4
        for t in range(24):
            w.writerow([f"M{mid:03d}", t,
                        round(60+(t*0.9 if fail else 0)+rng.gauss(0,2),2),
                        round(0.5+(t*0.05 if fail else 0)+rng.gauss(0,0.15),3),
                        "failing" if fail else "healthy"])
print("wrote", p)
PY

# Pack as a RAW time-series: group rows per machine_id, order by timestamp, keep the
# padded raw sequences (channel-major) so the deep nets learn the temporal trend.
# Split is per-machine (no sequence leaks across train/test).
atom pack ~/Downloads/sample/machines.csv --target status --type timeseries \
     --time timestamp --group machine_id --ts-layout raw --name machines_raw

# Search only the two deep models. They train on the GPU (CUDA/MPS) when present and
# export a self-contained ONNX graph that serves on CPU anywhere (deployable=True).
atom run machines_raw --methods conv1d-classifier,lstm-classifier --time-budget 60 --yes --out runs
#   log shows:  device: mps (torch 2.13.0)
#               AMP: deployable=True (1 ONNX graph(s), parity ok, selected single)
```

Every `atom run` writes `runs/<name>-<timestamp>/` with `model/pipeline.onnx` (the
deployable model + `manifest.json` signature), `metrics.json` (all scores), and
`provenance/` (per-trial log).

---

## 9. Exit codes & troubleshooting

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

Health check any install: `atom modules verify` (expect `28/28 ... pass`).

---

## 10. Quick reference card

```
atom pack <csv> --target COL [--split R] -o DIR CSV  -> ADP   (R=0.7/0.15/0.15 | auto)
atom pack <csv> --target COL --type timeseries --time T --group G [--ts-layout raw]
atom fetch kaggle:<owner/ds> --target COL       Kaggle -> ADP   (needs [kaggle])
atom pack-images <folder>                       image folder -> ADP
atom inspect <pkg> [--json]                     profile a dataset
atom run <pkg> --time-budget S --yes            train -> ONNX model package
atom run <pkg> --methods A,B --yes              restrict search to methods A,B
atom run <pkg> --methods conv1d-classifier,lstm-classifier --yes   deep (needs [torch])
atom modules list | verify                      registry / health check

deploy: runs/<name>-<ts>/model/pipeline.onnx    (signature in manifest.json)
```
