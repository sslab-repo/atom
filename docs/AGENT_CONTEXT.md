# ATOM — Agent Context & Continuity Document

**Read this first when resuming work on ATOM.** It consolidates purpose,
architecture, current state, conventions, environment gotchas, and what's next,
so any session can pick up without re-deriving. Keep it updated as things land.

Last updated: 2026-08-21 · Active branch: **`dev`** · Phase 2c (torch→ONNX) landed

---

## 1. What ATOM is (purpose)

ATOM (**AuTO ai Machine**) automatically finds the **best-accuracy
classification** — which method(s), hyperparameters, or ensemble — for a dataset,
then emits a **deployable model**. Clustering, regression, and anomaly detection
are supported too; **classification (mostly binary) is the primary focus**.

Hard invariants (never break):
- **One command, one line.** `atom <cmd> …`, scriptable, runs under Slurm/cron
  with `--yes`.
- **No LLM / AI agent in the runtime loop.** All task inference, search,
  selection, routing, ensembling are deterministic algorithms. (LLM-assisted
  *authoring* of modules is a maintainer convenience only — never in a run.)
- **Runs everywhere, degrades gracefully.** Linux workstation, macOS, Slurm
  supercomputer; sometimes GPU, sometimes CPU-only; sometimes no PyTorch. The
  CPU tier always works.

## 2. Operating constraints (from the user, Dr. Cho)

- Data: mostly **binary** classification tabular CSV; sometimes multiclass,
  **text CSV**, **image**, **time-series**.
- Environments: server has PyTorch (+CUDA); the MacBook has an Apple GPU (MPS);
  **some lab workstations have no PyTorch**.
- Therefore PyTorch and GPU are **optional**; ATOM must run and produce a model
  without them.

## 3. Architecture (see docs/design/ADR-0001..0008)

- **Module contract (ADR-0001, frozen):** every algorithm implements
  `declares() / space() / run(ctx) / hints()`. Kinds: `method`, `preprocessing`,
  `metric`, `search`. Task families: classification, regression, clustering,
  anomaly-detection, dimension-reduction. Modalities: tabular, text, timeseries,
  image. New algorithms drop into a registry and are auto-discovered — core
  untouched.
- **ADP — ATOM Dataset Package (ADR-0003):** a folder/zip with `manifest.json`,
  typed Parquet `processed/{train,val,test}.parquet`, `splits/`, roles
  (target/id/group/time), `dataset.source` provenance. Built by
  `pack_csv` / `pack_timeseries_csv` / `pack_images`.
- **AMP — ATOM Model Package (ADR-0004):** a run's deployable artifact — a fused
  **ONNX** graph (`model/pipeline.onnx`) verified to reproduce the trained model
  (parity gate), plus `manifest.json` (signature, parity, lineage) and a native
  `native/model.pkl` fallback. **ONNX serves on CPU onnxruntime anywhere** →
  train on GPU, deploy on a no-torch machine.
- **Run flow (`core/run.py`):** fingerprint → confirm gate → budgeted search
  (`core/orchestrator/search.py`, random+SHA multi-fidelity) → finalize top-K at
  full fidelity → greedy ensemble → **deployability-aware selection** (ship the
  best-scoring candidate whose ONNX also passes parity, within a 2% margin) →
  locked-test eval → provenance + AMP.
- **Two method tiers (ADR-0008):**
  - **CPU tier (always available):** sklearn — the floor, no GPU/torch.
  - **Deep tier (optional):** PyTorch modules; auto-register only when
    `import torch` works (like xgboost/lightgbm); device via
    `core/device.py` (cuda → mps → cpu, `$ATOM_DEVICE` override).
- **Meta-KB flywheel (ADR-0005, `core/metakb`):** append-only `records.jsonl`;
  nearest-fingerprint warm-start. Concurrent writers race → per-task KB + merge
  (see docs/slurm.md).

## 4. Current state — what exists

### Registered methods (as of HEAD)
- **Classification (15):** logistic-regression, sgd-classifier, perceptron,
  linear-discriminant-analysis, quadratic-discriminant-analysis,
  gaussian-naive-bayes, k-nearest-neighbors, support-vector-machine,
  decision-tree, random-forest, extra-trees, adaboost, gradient-boosting,
  hist-gradient-boosting, neural-net-mlp. **Deep (torch-only):**
  conv1d-classifier, lstm-classifier. **Optional libs:** xgboost, lightgbm
  (register when installed; `pip install 'atom-ai[boosted]'`).
- **Regression:** ridge, random-forest-reg, hist-gradient-boosting-reg.
- **Clustering:** kmeans, gaussian-mixture. **Anomaly:** isolation-forest,
  lof-novelty. **Dim-reduction:** pca. **Preprocessing:** impute-simple, scale.
- Files: `src/atom/registries/methods/sklearn_supervised.py`,
  `sklearn_unsupervised.py`, `torch_deep.py`.

### CLI (src/atom/cli.py)
- `atom inspect <pkg> [--json]` — profile.
- `atom pack <csv> --target COL [--name] [--out] [--split R] [--type T]
  [--time COL --group COL --ts-layout {features|raw}]`
  - `--split`: `0.7/0.15/0.15` (normalized) or `auto` (size-based:
    <1k→70/15/15, 1k–100k→80/10/10, ≥100k→90/5/5).
  - `--type {tabular|text|timeseries}`; text is gated (Phase 1 pending);
    timeseries needs `--time`/`--group`.
- `atom fetch kaggle:<slug> …` (needs `[kaggle]`), `atom pack-images <folder>`.
- `atom run <pkg> [--target] [--task] [--time-budget S] [--max-trials]
  [--max-rows] [--out] [--kb] [--methods A,B,C] [--seed] [--yes]
  [--include-experimental]`
  - `--methods` restricts the search (unknown name errors with the list).
- `atom modules list|verify`.

### Class balancing & imbalance
Classifiers whose `fit` accepts `sample_weight` carry a `class_balance`
search dim (`none|balanced`); the run shell applies balanced sample_weight
(`_Supervised.run`). Binary threshold tuning is report-only in `metrics.json`
(`decision_threshold`). Relative-imbalance advisory fires for minority <20%.

### Time-series (ADR-0008 Phase 2)
`pack_timeseries_csv` groups by `--group`, orders by `--time`, one row per
sequence, split per-sequence (no group leak). Two `--ts-layout`:
- `features` (default, torch-free): summary stats (mean/std/min/max/last/slope)
  per channel → the 15 classifiers.
- `raw`: padded channel-major sequences; `manifest.dataset_source` records
  `(n_channels, seq_len)`; `TabularMatrix.seq_shape` carries it
  (run_package → fit_pipeline FIT context) so conv1d/lstm reshape
  X (n, C*L) → (n, C, L).

### Testing / quality
`tests/` (12 files), run `.venv/bin/python -m pytest -q` (currently 91 pass),
`ruff check src tests`, `atom modules verify` (28 without torch, 30 with).
CI must pass WITHOUT torch (deep tests skip when torch absent).

## 5. Development plan (ADR-0008) — phase status

Leading with the torch tier per the user.
- **Phase 0 — foundation** ✅ (`d9b9f1e`): device.py, `--type`, gates.
- **Phase 2a — time-series features (torch-free)** ✅ (`8dc43fa`).
- **Phase 2b — conv1d/lstm on raw sequences (torch)** ✅ — trains on mps/cpu.
- **Phase 2c — torch→ONNX export for the deep tier** ✅ **GAP CLOSED**:
  conv1d/lstm now ship `deployable=True`. `_SeqNet` self-preprocesses (NaN-fill +
  standardize + reshape all IN the graph), `SELF_PREPROCESSING=True` makes
  `fit_pipeline` skip the sklearn impute/scale chain, and `core/provenance/amp.py`
  has a torch branch: `_torch_net` → `_export_torch` (`torch.onnx.export(...,
  opset_version=17, dynamo=False)`) → `_parity_torch` (onnx proba vs native
  predict). Single-member only; a torch model inside an ensemble → `deployable=False`
  (`skipped: torch-in-ensemble`), not yet fused. The torch graph outputs
  **`probabilities` only** (label = argmax via `label_map`), so the manifest
  signature `outputs` is `["probabilities"]` for torch AMPs (sklearn stays
  `["label","probabilities"]`). Regression test:
  `tests/test_device_and_modality.py::test_deep_tier_exports_onnx_and_is_deployable`
  (torch-gated; asserts deployable + parity + CPU-onnxruntime serving).
- **Phase 3 — images (CNN / foundation embeddings, torch).**
- **Phase 4 — GAN (GANomaly in the anomaly-detection family; binary
  normal-vs-rare, torch).**
- **Phase 1 — text CSV (TF-IDF/hashing, torch-free; optional transformer
  embeddings, torch).**

## 6. Environment & gotchas (this dev machine — the MacBook)

- **The venv is `uv`-managed** (`.venv`, no `pip` binary). Install with
  `VIRTUAL_ENV=$PWD/.venv uv pip install <pkg>` — NOT `python -m pip`.
- **torch 2.13 IS installed** here; `device.resolve_device()` → **mps**.
- **torch.onnx needs `dynamo=False`** (the default dynamo path needs
  `onnxscript`, not installed).
- **Torch nn.Module bodies must be MODULE-LEVEL classes** (local classes break
  the native pickle in `writer.write_model`).
- CLI binary that has torch/deep tier = `~/Dev/sslab-git/atom/.venv/bin/atom`.
  The `scripts/install.sh` install (`~/atom/bin/atom`) is torch-FREE (classical
  tier only) unless `[torch]` is installed there.
- macOS ships **bash 3.2**: empty-array `"${arr[@]}"` under `set -u` errors —
  use `${arr[@]+"${arr[@]}"}`. Shell scripts already fixed.
- **Repo-integrity note:** `.gitignore` was fixed to `/data/` (was `data/`,
  which had silently untracked `src/atom/data/`). Don't reintroduce a bare
  `data/` rule.

## 7. Working conventions (the user's expectations)

- **Autonomy + approval gate:** proceed without asking for routine work; get
  approval (with detail) only for changes >50 lines / >5% / architectural.
  Currently in a **full-auto-approval** multi-modal build on `dev`.
- **Compare-or-rollback:** improvements must beat baseline on measured evidence
  or be reverted (this killed several fixed-budget search tweaks — adding
  methods/CV/seeds dilutes a fixed `--time-budget`; use `--methods` to focus).
- **Generalize fixes** to all plausible cases, not the one failing dataset.
- **Validation rounds:** fetch N fresh Kaggle datasets, run full protocol,
  report bugs/observations. History in `docs/qa/`.
- Commit messages end with `Co-Authored-By: Claude …`. Develop on **`dev`**;
  `main` holds pre-multimodal work. Push when asked.

## 8. Key files & docs

- Design: `docs/design/ADR-0001..0008`. Command manual: `docs/manual.md`.
  Install: `docs/install.md` + `scripts/{setup,install,sample_run}.sh`.
  Slurm: `docs/slurm.md` + `scripts/slurm/`. Sample commands:
  `sample_command.md`. QA history: `docs/qa/`.
- Core: `src/atom/core/{run,dataset,device}.py`,
  `core/orchestrator/{search,pipeline,budget}.py`,
  `core/provenance/{amp,writer}.py`, `core/ingest/profiler.py`,
  `core/task_inference/infer.py`, `core/evaluation/evaluator.py`.
- Data plane: `src/atom/data/{packager,packager_images,package,manifest,source}.py`.
- Agent memory (outside repo, ~/.claude/…/memory/): `atom-purpose-and-runtime`,
  `atom-multimodal-dev-plan`, `atom-dirty-input-parser`, `atom-datetime-and-scope`,
  `atom-bugfix-workflow-approval`, `atom-validation-protocol`,
  `check-past-sessions-first`.

## 9. How to verify quickly

```bash
cd ~/Dev/sslab-git/atom
.venv/bin/python -m pytest -q          # 90 pass
.venv/bin/python -m ruff check src tests
.venv/bin/atom modules verify          # 30/30 (with torch) / 28 (without)
```
End-to-end examples: `sample_command.md` (datasets in `~/Download/Sample`).
