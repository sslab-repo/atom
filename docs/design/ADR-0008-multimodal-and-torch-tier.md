# ADR-0008: Multi-modal support & the optional PyTorch (deep) tier

- Status: **accepted** (2026-08-21)
- Supersedes parts of: ADR-0005 (v1 scope), the M6 "deferred" note in status.md
- Related: ADR-0001 (module contract), ADR-0002 (registry discovery),
  ADR-0004 (ONNX model package), ADR-0006 (run budget)

## Context

ATOM's purpose: **automatically find the best-accuracy classification** (later
clustering) — which method(s), hyperparameters, or ensemble — with **no human
and no LLM in the loop at runtime**. It must run as **one command, one line**,
across very different machines:

- a plain Linux workstation (often **no PyTorch, no GPU**),
- a macOS laptop (Apple-Silicon **GPU via MPS**),
- a Slurm supercomputer (**PyTorch + one or more CUDA GPUs**, or none).

Data is mostly **binary** tabular CSV, but sometimes **multiclass**, **text
CSV**, **image**, or **time-series/sequence** data. The same ATOM install and
the same command must do the right thing on all of these, degrading gracefully
where a dependency or accelerator is missing.

## Decision

### 1. Two method tiers, one registry

Methods split into two tiers, both discovered through the existing module
contract/registry (ADR-0001/0002):

- **CPU tier (always available):** the sklearn stack — 15 classifiers +
  regression/clustering/anomaly + preprocessing. Pure-Python + portable wheels,
  no GPU. **Guarantees a usable, good result on every machine.** This is the
  floor: ATOM never depends on the deep tier to produce an answer.
- **Deep tier (optional):** PyTorch-based modules — 1D-CNN/LSTM (sequences),
  CNN/foundation embeddings (images), transformer/embedding vectorizers (text),
  GANomaly (anomaly-style binary). Each module is wrapped in
  `try: import torch … except ImportError: pass` (the same pattern as
  xgboost/lightgbm), so **it registers only where torch is importable and is
  simply absent otherwise.** A run on a no-torch workstation sees the CPU tier
  only; the identical command on the server also sees the deep tier.

### 2. Device detection (cuda / mps / cpu)

A single `atom.core.device` helper resolves the compute device at runtime:

1. `cuda` if `torch.cuda.is_available()`,
2. else `mps` if `torch.backends.mps.is_available()` (Apple Silicon),
3. else `cpu`.

Deep modules move tensors/models to this device; on `cpu` they still run
(slower) unless admission control (ADR-0006) skips them for the budget. GPU
accelerates **search/training only** — never a requirement to get a result.
Overridable via `ATOM_DEVICE={auto|cpu|cuda|mps}` for reproducibility/testing.

### 3. Modality routing via `--type`

`atom pack`/`atom run` gain `--type {tabular|text|timeseries|image}` (default
inferred; `tabular` for CSV, `image` for `pack-images`). The declared modality
is recorded in the manifest (`dataset.modality`, already a field) and drives
**which methods and which feature handling** the run uses:

| `--type` | Feature handling | Methods searched |
|---|---|---|
| `tabular` | numeric coerce / one-hot / datetime (today) | CPU tier (15 classifiers), + deep MLP |
| `text` | text columns → TF-IDF/hashing (CPU) **or** transformer embeddings (deep) → then tabular features | CPU tier on vectors; deep text encoders when torch |
| `timeseries` | `--time`/`--group` → group, order, window into sequences | 1D-CNN, LSTM (deep); CPU tier on extracted per-window features |
| `image` | `pack-images` listing → pixels/embeddings | CNN / foundation embeddings (deep) |

Routing is deterministic and declared — **no inference by an LLM**. If a
modality's deep methods are unavailable (no torch), ATOM falls back to the
torch-free handling for that modality where one exists (text → TF-IDF;
timeseries → window features + tabular classifiers) and says so; for image it
gates gracefully (as today) until the deep tier is present.

### 4. Deployment stays CPU-portable

Every method — CPU or deep — exports to the same **ONNX model package**
(ADR-0004). Torch models export via `torch.onnx`; the fused graph serves through
CPU `onnxruntime` anywhere. **Train on a GPU Slurm node, deploy on a no-torch
workstation.** GPU is never needed to *serve*. Where a torch model can't be
faithfully exported, the run ships `deployable:false` + the native artifact
(existing rule), never failing.

### 5. No LLM, one command — invariant

ATOM's control flow (task inference, search, selection, ensembling, routing) is
**algorithmic and deterministic**. No module or orchestration step calls an LLM
or external AI service at runtime. The user interface remains a single
`atom <cmd> …` line, scriptable and runnable under Slurm/cron with `--yes`.
(LLM-assisted *authoring* of modules/dossiers, ADR-0007, is a maintainer-time
convenience, never part of a run.)

## Module additions (per phase)

Leading with the deep tier (server has PyTorch), each phase adds modules under
the existing contract; the CPU fallback for each modality ships alongside so
no-torch machines still function.

- **Phase 0 — foundation (torch-free):** `atom.core.device`; `--type` on
  pack/run + manifest modality; optional-torch registration group; verified
  graceful absence.
- **Phase 2 — time-series:** `--time/--group` packing; sequence assembly
  (group→order→window) + time/group-aware split; `conv1d-classifier`,
  `lstm-classifier` (deep); `ts-features` (torch-free window stats → CPU tier).
- **Phase 3 — images:** `cnn-classifier` and/or foundation-embedding +
  linear head (deep); image ADP already exists.
- **Phase 4 — GAN:** `ganomaly` detector in the anomaly-detection family
  (binary normal-vs-rare, deep).
- **Phase 1 — text (torch-free, any machine):** `tfidf`/`hashing` vectorizer →
  CPU tier; optional transformer-embedding vectorizer (deep).

Optional deps go in `pyproject` extras: `torch = ["torch>=2.2"]`,
`text = ["scikit-learn"]` (already core), etc. `pip install 'atom-ai[torch]'`.

## Consequences

- **Portability preserved:** identical command everywhere; the deep tier is
  additive, never required. No-torch workstations are first-class.
- **Determinism preserved:** device + seed fixed → reproducible; no LLM.
- **Budget dilution:** more methods dilute a fixed `--time-budget`; mitigated by
  the `--methods` filter and modality routing (a `text` run doesn't search
  image CNNs). Deep methods on CPU are usually skipped by admission control.
- **Testing:** CPU tier + routing + graceful-absence are unit-tested on any
  machine (incl. no-torch CI); deep-model training/export is validated on a
  torch machine (the server / a torch venv). CI must pass without torch.
- **Contract unchanged:** ADR-0001 stays frozen; deep modules implement the
  same `declares/space/run/hints`. New task families/modalities are additive.

## Status of development

Phase 0 in progress (device + `--type` + optional-torch scaffolding, all
torch-free and tested). Deep phases follow, validated on a PyTorch machine.
