# ATOM — Initial Development Status

Updated: 2026-07-16 · Suite: 67 tests green · `atom modules verify`: 17/17
· Reference dataset (CIC-IDS-2017, 623 MB zip) exercised at every milestone.
· Validation: 25 Kaggle datasets across rounds 3–6; bugfix workflow iter-5
  converged at 2 iterations (docs/qa/bugfix-iter5-report.md); iter-6 unified
  and generalized dirty-input parsing (docs/qa/bugfix-iter6-dirty-input.md).

## Milestones

| Milestone | Status | Exit evidence |
|---|---|---|
| M1 Skeleton & data plane | ✅ done | `atom inspect` profiles the zip in ~4 s; CSV→ADP round-trip tested |
| M2 Tabular MVP | ✅ done | one command → locked-test f1_macro ≈0.89 over 13 classes in a stated budget |
| M3 Model plane (AMP) | ✅ done | deployable ONNX AMP, parity-verified, serves standalone via onnxruntime |
| M4 Flywheel | ✅ done | meta-KB records + nearest-fingerprint warm-start (stored winner = trial #1 of next run) |
| M5 Breadth & governance | ✅ core done | unlabeled CIC-IDS auto-routes to outlier detection (17 s run); drop-ins enter experimental and are search-gated; smoke gate CLI |
| M6 Multi-modal & foundation | 🟡 data plane only | image ADP pack/inspect/fingerprint/task-inference work; `atom run` gates gracefully |

## What works today (CLI)

```
atom inspect <pkg.zip|dir>          # fingerprint + quality flags
atom pack <csv> --target <col>      # loose CSV -> typed ADP
atom pack-images <folder>           # class-per-subfolder -> image ADP
atom run <pkg> [--target C] [--task F] [--time-budget S] [--max-trials N]
                                    # confirm gate -> search -> ensemble ->
                                    # locked test -> provenance + ONNX AMP
atom modules list|verify            # registry + contract smoke gate
```

Families runnable: classification, regression (tabular), clustering
(silhouette-driven), anomaly-detection (outlier/novelty, descriptive).
Modules: 17 stable built-ins; external drop-ins via `atom.modules` entry
points enter as experimental (`--include-experimental` to search them).

## Deliberately deferred (with reasons)

1. **Foundation adapters (M6)** — CLIP/DINOv2 embeddings and zero-/few-shot
   modules require torch (+~2 GB) and model weights (hundreds of MB per
   backbone). Deferred until you green-light the dependency footprint on
   the lab server. Everything upstream (image ADP, fingerprint, task
   inference, the `adaptation` contract axis) is ready to receive them.
2. **Per-module venv isolation (ADR-0005)** — requires the serializable
   subprocess run boundary; current artifacts hold live sklearn objects.
   Design constraint recorded; no v1 module needs isolation yet.
3. **Promotion dossier automation (ADR-0007 stage 2)** — the automatic gate
   exists (`atom modules verify` + lifecycle); the AI-assisted dossier
   needs LLM plumbing. Promotion is currently: pass gate → maintainer
   flips lifecycle in code review.
4. **Temporal regression modules & group/time-aware split emission** —
   split *policy* is recorded in TaskSpec; ARIMA-class modules and
   `group_hash`/`time_holdout` packager output are the next registry drop.
5. **SMOTE/resampling modules** — imbalanced-learn optional dep, not yet
   wrapped; imbalance handled via metric choice today.
6. **Anomaly ONNX parity tolerance** — IsolationForest converts but hard
   label-agreement (99.5%) fails at 96.9% due to float32 threshold flips;
   needs a per-setting tolerance (score correlation) before anomaly AMPs
   mark deployable.

## Known limitations

Numeric features only in tabular runs (string columns dropped with recorded
reason) · low-fidelity subsamples not stratified (rare classes may vanish
from early rungs) · search strategy fixed to random+SHA pending Search
registry modules · budget clock includes package IO.

## Repo map

`src/atom/contract` (frozen interface) · `src/atom/registries` (4 registries
+ lifecycle + discovery) · `src/atom/core` (ingest, task_inference,
orchestrator, evaluation, ensemble, provenance incl. AMP export, run driver)
· `src/atom/data` (ADP reader zip/folder, packagers) · `src/atom/metakb`
(flywheel) · QA: `docs/qa/M1-M2-QA.md`.
