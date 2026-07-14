# ATOM Architecture Reference

> Status: draft · living document. Decisions that are settled move into
> [`design/`](design/) as ADRs; this file describes the current overall shape.

![ATOM architecture](diagrams/atom-architecture.svg)

## Design principles

1. **Frozen core, hot-swappable edges.** The core engine defines a small set of
   interfaces that rarely change. All algorithmic knowledge — preprocessors,
   methods, search strategies, metrics — lives in pluggable registries.
2. **Everything is a module.** A gradient-boosted classifier, a diffusion-based
   image amplifier, a SMOTE balancer, and an F1 evaluator all implement the same
   four-method contract. There is no privileged algorithm.
3. **Multi-modal by construction.** Data modality (tabular, image, text, signal,
   mixed) is part of the data fingerprint and every module's declaration; the
   orchestrator only composes modules whose declarations match the fingerprint.
4. **Honest evaluation.** Nested, leak-safe folds and a locked test set are core
   responsibilities, never delegated to plugins.
5. **Compounding knowledge.** The meta-knowledge base turns every completed run
   into warm-start priors for future runs.

## ① Core engine

```
data in ─▶ ingest ─▶ task_inference ─▶ orchestrator ─▶ evaluation ─▶ ensemble ─▶ provenance ─▶ model out
                          ⚠ confirm gate      ▲                                       │
                                              │ modules composed at runtime          │
                                     ② registries                          meta-KB ◀─┘ (store)
                                              ▲                                │
                                              └──────── warm-start ◀───────────┘
```

| Stage | Responsibility |
|---|---|
| Ingest & Profiler | Accepts labeled or unlabeled data, any modality. Produces the **Fingerprint**: modality, shapes, dtypes, missingness, class balance, statistical summaries, quality flags. |
| Task Inference | Maps fingerprint → objective / **task-DAG** (a run may chain steps, e.g. filter/repair → generative augmentation → classification). Reads Methods + Metrics registries so every inferred task has an evaluable objective. Emits a **confirm gate** for the user before spending budget. |
| Search Orchestrator | The hub. Runs the inner loop `preprocess × method × microcontrols` under multi-fidelity, budget/bandit control. Delegates the *strategy* to the Search registry; owns the *budget* itself. Enforces the diversity constraint on the candidate archive. |
| Nested Evaluation | Leak-safe fold construction, locked (never-touched) test set, per-fidelity scoring via the Metrics registry. |
| Ensemble & Finalize | Greedy ensemble selection over the candidate archive. |
| Model + Provenance | Exports the final artifact plus the full search history: every trial, config, score, cost, and data lineage. Feeds the meta-KB. |

## ② Module registries

Four registries, each an independent, versioned namespace:

| Registry | Task families served |
|---|---|
| Preprocessing | impute, scale, encode, tokenize, signal/image filtering, resampling (SMOTE), classical augmentation, data repair |
| Methods | classification, regression (incl. temporal/forecasting), clustering, dimension-reduction, anomaly-detection, generative, structured-prediction, association-mining, preference-learning — full taxonomy in [design/method-taxonomy.md](design/method-taxonomy.md) |
| Search Strategy | BOHB/ASHA, TPE, SMAC, evolutionary, random/grid baselines |
| Metrics / Evaluators | F1/AUC/RMSE, silhouette/ARI, TSTR + privacy risk, perceptual/vision metrics |

Discovery is automatic (see [ADR-0002](design/ADR-0002-registry-discovery.md)):
a module is registered by decorator inside the tree, or exposed as an entry
point from an external package — third-party plug-ins never require a change
in this repository.

## ③ Module contract

The single interface making ATOM pluggable — see
[ADR-0001](design/ADR-0001-module-contract.md) for the normative spec.

| Method | Returns | Purpose |
|---|---|---|
| `declares()` | `Declaration` | Task families + data modalities supported. Used by task inference and the orchestrator's compatibility filter. |
| `space()` | `SearchSpace` | Hyperparameter / "microcontrol" ranges; conditional parameters allowed. Consumed by the Search registry. |
| `run(ctx)` | `RunResult` | The work: fit · transform · generate · score, depending on module kind. |
| `hints()` | `ResourceHints` | `cpu | gpu:N`, memory, and **fidelity levels** (e.g. subsample fractions, epochs) enabling multi-fidelity search. |

## Meta-knowledge base

Append-only store of `fingerprint → (winning config, score, cost)` records.
On a new run, task inference and the orchestrator query the KB for the
nearest-fingerprint records and warm-start the search from those configs.
This is the flywheel: ATOM improves with every dataset it processes.

## Non-goals (for now)

- Serving / deployment infrastructure — ATOM produces models + provenance;
  serving is downstream.
- A UI. The confirm gate is an API/CLI concern first.
- Distributed execution — `hints()` is designed so a scheduler can be added
  later without contract changes.
