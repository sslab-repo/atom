# ADR-0001: The Module Contract

- Status: **accepted** (2026-07-14)
- Date: 2026-07-10

## Context

ATOM must absorb any algorithm — traditional classifiers, clustering, deep
vision models, dataset amplifiers, denoisers, generative synthesizers, LLM
components — without core changes. That requires one interface expressive
enough for all of them, yet small enough to stay frozen.

## Decision

Every module implements exactly four methods (sketch — normative shape, not
implementation):

```python
class Module(ABC):
    def declares(self) -> Declaration: ...
    def space(self) -> SearchSpace: ...
    def run(self, ctx: RunContext) -> RunResult: ...
    def hints(self) -> ResourceHints: ...
```

### `declares() -> Declaration`

Static capability statement, queried **before** any data is touched:

- `kind` — which registry the module belongs to
  (`preprocessing | method | search | metric`).
- `task_families` — set of `TaskFamily` values
  (classification, regression, clustering, dimension-reduction,
  anomaly-detection, generative, structured-prediction, association-mining,
  preference-learning — see [method-taxonomy.md](method-taxonomy.md);
  adding a family requires an ADR, adding a category or algorithm does not.
  Vision/language are modalities, not families; forecasting is
  `regression` + `temporal`).
- `modalities` — set of `Modality` values
  (tabular, image, text, timeseries, audio, mixed).
- `name`, `version` — stable identity for provenance and the meta-KB.

### `space() -> SearchSpace`

Hyperparameter / "microcontrol" ranges. Conditional parameters are allowed
(a parameter may be active only when another takes a given value). The core
never interprets the semantics of a parameter — only search strategies do.

### `run(ctx: RunContext) -> RunResult`

The single execution entry point. `RunContext` carries the data split, the
sampled config, the requested fidelity, and an operation selector
(`fit | transform | generate | score`). Not every module supports every
operation; unsupported operations raise `UnsupportedOperation`.

### `hints() -> ResourceHints`

- Resource needs: `cpu`, `gpu` count, memory estimate.
- **Fidelity levels**: ordered list of cheap→expensive fidelities
  (e.g. `[0.1, 0.33, 1.0]` subsample fractions, or epoch budgets) that the
  orchestrator exploits for multi-fidelity search (ASHA/BOHB).

## Consequences

- New research lands as: implement the contract → drop into a registry →
  auto-discovered. Core untouched.
- The contract is **frozen**: additions must be optional (default-implemented
  on the ABC) and never break existing modules.
- Uniformity has a cost: exotic algorithms must adapt to `run(ctx)` rather
  than exposing bespoke APIs. We accept this deliberately — it is what keeps
  the orchestrator generic.
