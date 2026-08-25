# ATOM — AuTO ai Machine

**ATOM** is a modular AutoAI platform: a stable, frozen core wrapped by hot-swappable
module registries. Any algorithm — classical or newly published — is just a plug-in.

ATOM is not a single application and not only an LLM wrapper. It is a platform for
**multi-modal AI machines**, covering:

- Traditional ML — classification, regression/estimation, clustering
- Deep learning and vision recognition / vision models
- Image data processing
- Data abstraction and dimensionality reduction
- Dataset amplification and synthesis (augmentation, generative)
- Denoising and signal cleanup
- Prediction / forecasting
- Generative AI and LLM-based components

The unifying idea: **new research should never require touching the core.**
Implement the module contract, drop the module into a registry, and it is
auto-discovered and composed at runtime.

---

## Architecture

![ATOM architecture](docs/diagrams/atom-architecture.svg)

ATOM has three layers:

### ① Core Engine — frozen interfaces, rarely changes

A fixed pipeline that accepts labeled or unlabeled data of any modality:

| Stage | Responsibility |
|---|---|
| **Ingest & Profiler** | Load data, build a data **fingerprint** (modality, shape, statistics, quality signals) |
| **Task Inference** | Decide the objective / task-DAG from the fingerprint — with a ⚠ user **confirm gate** |
| **Search Orchestrator** | The hub. Multi-fidelity search with budget / bandit control over the inner loop: `preprocess × method × microcontrols` |
| **Nested Evaluation** | Leak-safe folds, locked test set — honest scores only |
| **Ensemble & Finalize** | Greedy ensemble selection over the candidate archive |
| **Model + Provenance** | Export the final model plus a full provenance record of how it was found |

### ② Pluggable Module Registries — add or revise modules without touching the core

Four independent, versioned registries. The orchestrator discovers and composes
modules from all of them at runtime:

| Registry | Examples |
|---|---|
| **Preprocessing** | Impute (MICE / KNN), Scale (robust / quantile), Resample (SMOTE / ADASYN), filtering, augmentation, image transforms |
| **Methods** (by task family) | Classification · Regression (incl. temporal) · Clustering · Dimension reduction · Anomaly detection · Generative · Structured prediction · Association mining · Preference learning |
| **Search Strategy** | BOHB / ASHA, TPE, SMAC, Evolutionary / TPOT |
| **Metrics / Evaluators** | F1 / AUC / RMSE, silhouette / ARI, TSTR + privacy risk |

A **diversity constraint** keeps candidate portfolios from collapsing to
near-duplicates. Task Inference reads the Methods and Metrics registries to
attach an evaluable objective to every inferred task.

### ③ Module Contract — the single interface that makes ATOM pluggable

Every module, regardless of registry, implements four methods:

```python
class Module:
    def declares(self) -> Declaration:
        """Task families + data modalities this module supports."""

    def space(self) -> SearchSpace:
        """Hyperparameter / 'microcontrol' ranges (conditional OK)."""

    def run(self, ctx: RunContext) -> RunResult:
        """fit · transform · generate · score."""

    def hints(self) -> ResourceHints:
        """cpu | gpu:N · fidelity levels for multi-fidelity search."""
```

New research → ① implement the contract → ② drop into a registry →
③ auto-discovered. **Core untouched.**

### Meta-Knowledge Base — the flywheel

Every run stores `fingerprint → winning config, score & cost`. New runs are
**warm-started** from the nearest fingerprints, so ATOM gets faster and smarter
with every dataset it sees.

---

## Repository layout

```
atom/
└── docs/
    ├── architecture.md          # full architecture reference
    ├── design/                  # ADRs (decision records) + open questions
    └── diagrams/                # architecture diagrams
```

## Installation

```bash
bash scripts/setup.sh     # fresh machine: OS packages + ATOM + health check
                          #   macOS (brew), RHEL 10/Rocky/Alma/Fedora (dnf), Debian 13/Ubuntu (apt)
bash scripts/install.sh   # Python 3.10+ already present: per-user ATOM only (no root on Linux)
bash scripts/sample_run.sh  # sample end-to-end test after install
```

Self-contained under `~/atom` — sources, venv, config, data, runs, meta-KB.
Details: [`docs/install.md`](docs/install.md).

## Using the `atom` command

Full command manual (Linux + macOS): [`docs/manual.md`](docs/manual.md) —
every subcommand, flag, output, serving the ONNX model package, and recipes.
Running many datasets on a cluster: [`docs/slurm.md`](docs/slurm.md) (Slurm job
arrays + ready-to-run scripts in [`scripts/slurm/`](scripts/slurm/)).

```bash
atom pack mydata.csv --target label --out pkgs   # CSV -> ATOM Dataset Package
atom inspect pkgs/mydata                          # profile it
atom run pkgs/mydata --time-budget 120 --yes      # train -> deployable ONNX package
atom modules verify                               # health check
```

## Example: an end-to-end run

Point ATOM at a CSV; it profiles the data, infers the task, searches every
applicable method under a time budget, ensembles the best, evaluates on a locked
test split, and exports a deployable ONNX model — one command. Grab a public
dataset and go (full walkthrough on five datasets: [`sample_command.md`](sample_command.md)):

```bash
curl -fsSL -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
atom pack titanic.csv --target Survived --name titanic     # CSV -> ATOM Dataset Package
atom run  titanic --time-budget 600 --yes --out runs        # search -> deployable ONNX package
```

The run streams a live search log — each rung names the winning algorithm and its
hyperparameters — then a full leaderboard of **every** method with its validation
metrics, and the final locked-test result:

```text
  ... (live search rungs above) ...
  finalizing: up to 5 candidates at full fidelity…
  decision threshold tuned on val: 0.545 — test balanced_accuracy 0.747 -> 0.774 (report-only; the AMP outputs probabilities)
  phases: load_s=0.0s  search_s=510.0s  finalize_s=1.1s  export_s=2.1s
  AMP: deployable=True (2 ONNX graph(s), parity ok, selected ensemble)
  === all 17 methods searched — validation metrics at each method's best trial (sorted by roc_auc; * = in the final model) ===
    * roc_auc=0.8802  acc=0.8203  bal_acc=0.7959  f1_macro=0.8003   extra-trees                      (267 trial(s))   -> final val 0.8782
      roc_auc=0.8799  acc=0.8422  bal_acc=0.8329  f1_macro=0.8293   gradient-boosting                (155 trial(s))   -> final val 0.8482
    * roc_auc=0.8788  acc=0.8189  bal_acc=0.8044  f1_macro=0.8033   random-forest                    (214 trial(s))   -> final val 0.8773
      roc_auc=0.8637  acc=0.8272  bal_acc=0.7953  f1_macro=0.8044   neural-net-mlp                   (155 trial(s))
      roc_auc=0.8626  acc=0.7819  bal_acc=0.7638  f1_macro=0.7619   support-vector-machine           (224 trial(s))
      roc_auc=0.8621  acc=0.8148  bal_acc=0.7951  f1_macro=0.7966   quadratic-discriminant-analysis  (204 trial(s))
      roc_auc=0.8620  acc=0.8066  bal_acc=0.7827  f1_macro=0.7857   linear-discriminant-analysis     (288 trial(s))
      roc_auc=0.8607  acc=0.8066  bal_acc=0.7717  f1_macro=0.7806   k-nearest-neighbors              (207 trial(s))
      roc_auc=0.8606  acc=0.7860  bal_acc=0.7859  f1_macro=0.7742   sgd-classifier                   (177 trial(s))
      roc_auc=0.8598  acc=0.8093  bal_acc=0.7834  f1_macro=0.7881   logistic-regression              (266 trial(s))
      roc_auc=0.8559  acc=0.7641  bal_acc=0.7626  f1_macro=0.7518   adaboost                         (175 trial(s))
      roc_auc=0.8467  acc=0.7970  bal_acc=0.7771  f1_macro=0.7781   conv1d-classifier                (144 trial(s) @f0.33)
      roc_auc=0.8331  acc=0.7942  bal_acc=0.7775  f1_macro=0.7764   gaussian-naive-bayes             (189 trial(s))
      roc_auc=0.8221  acc=0.7641  bal_acc=0.7511  f1_macro=0.7438   lstm-classifier                  (145 trial(s) @f0.33)
      roc_auc=0.8212  acc=0.8052  bal_acc=0.7629  f1_macro=0.7754   decision-tree                    (127 trial(s) @f0.33)
      roc_auc=0.8121  acc=0.7819  bal_acc=0.7541  f1_macro=0.7579   hist-gradient-boosting           (137 trial(s) @f0.33)
                      acc=0.7627  bal_acc=0.7229  f1_macro=0.7275   perceptron                       (141 trial(s) @f0.33)
  (sorted by cross-validated search score; '-> final val' = the finalist re-scored on the held-out val split after a full-data refit — the number selection used.)
  (* = chosen into the final model (greedy blend) by that held-out score, so it can rank differently from the leaderboard; the locked test set is scored once, on the final model only — see 'test' above.)
=== result ===
  final    : ensemble   trials: 3219   elapsed: 513s
  val      : roc_auc=0.8855
  test     : accuracy=0.7500  balanced_accuracy=0.7465  f1_macro=0.7473  roc_auc=0.8090  decision_threshold=0.5454  balanced_accuracy_tuned=0.7743  f1_macro_tuned=0.7755
  artifacts: runs/titanic-20260825-145909
```

**Reading it:**
- The **leaderboard** lists every searched method with its validation metrics
  (`roc_auc`, accuracy, balanced-accuracy, f1) at that method's best trial. A
  `@f0.33` tag means the method never reached full data, so its numbers rest on a
  subsample.
- `-> final val` is the finalist re-scored on the held-out val split after a
  full-data refit — the number model selection actually used. Note
  `gradient-boosting` tops the leaderboard on cross-validated search score
  (`0.8799`) but re-scores to `0.8482` at full fidelity, so it isn't picked.
- `*` marks the methods in the final model (a **greedy ensemble**); it follows the
  held-out score, so it can differ from the leaderboard order.
- Per-method numbers are **validation** scores; the locked **test** set is scored
  once, on the final model only (`=== result ===`). Outputs land in
  `runs/<name>-<timestamp>/` (`model/pipeline.onnx` + `manifest.json`, `metrics.json`,
  `provenance/`).

## Status

**Implemented and running (tabular).** The end-to-end pipeline works today:
pack → budgeted multi-fidelity search across 15 classical + 2 optional PyTorch
deep classifiers → greedy ensemble → locked-test evaluation → parity-gated,
deployable ONNX model package. Runs on Linux and macOS, CPU-only or GPU
(CUDA / Apple MPS), with **PyTorch optional** — the classical tier is the
always-available floor. Time-series packing (feature + raw-sequence layouts) is
in; image / text / generative adapters are in progress (ADR-0008).

The architecture is specified in [`docs/architecture.md`](docs/architecture.md)
and settled in accepted decision records ADR-0001..0008 under
[`docs/design/`](docs/design/):

| ADR | Decision |
|---|---|
| 0001 | Module contract (`declares/space/run/hints`) |
| 0002 | Registries & drop-in discovery |
| 0003 | ATOM Dataset Package (ADP) — zip/folder, loose-input conversion, DMS-aligned |
| 0004 | ATOM Model Package (AMP) on ONNX — pipeline export, parity gate |
| 0005 | v1 scope: 9 task families, foundation (zero/few-shot, PEFT), lab-server operating model |
| 0006 | Budget: wall-clock primary, optional trial count, estimated end time |
| 0007 | Module promotion: benchmark gate → AI-assisted dossier → minimal human approval |
| 0008 | Multi-modal + optional PyTorch deep tier (device auto-detect, torch→ONNX export) |

The method taxonomy lives in
[`docs/design/method-taxonomy.md`](docs/design/method-taxonomy.md);
the implementation plan in [`docs/roadmap.md`](docs/roadmap.md) (M1 skeleton
→ M2 tabular MVP on CIC-IDS-2017 → M3 ONNX model plane → M4 meta-KB
flywheel → M5 breadth/governance → M6 multi-modal & foundation).
