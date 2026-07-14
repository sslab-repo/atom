# ADR-0006: Run Budget Model

- Status: **accepted** (2026-07-14)
- Date: 2026-07-14

## Context

The Search Orchestrator owns the budget (ADR-0001 area); a run needs a
user-facing way to bound and predict cost.

## Decision

- **Primary budget: wall-clock time.** Every run has a wall-clock limit
  (explicit, or a default the confirm gate displays). The orchestrator
  plans fidelity schedules against remaining time and always reserves a
  tail slice for ensembling + finalization + export, so a run never ends
  budget-exhausted without a usable AMP.
- **Optional secondary budget: trial count.** A user may set a max-trials
  limit, a min-trials request, or both. When both time and trials are set,
  **whichever bound is reached first stops the search**.
- **Estimated end time is a first-class output.** The confirm gate shows
  the initial estimate (from meta-KB warm-start data when available); the
  orchestrator revises it continuously from observed trial costs and
  reports progress as `elapsed / estimated-total` plus trials completed.
- GPU-hours, monetary cost, and energy are **recorded** per trial in
  provenance (they feed the meta-KB cost model) but are not budget
  controls in v1.

## Consequences

- Multi-fidelity strategies (ASHA/BOHB) get an honest time axis to
  schedule against; `hints().fidelity_levels` + observed costs give the
  estimator its model.
- Early estimates on cold caches (no similar fingerprint in the meta-KB)
  will be poor; the confirm gate labels them as low-confidence.
- Adding a cost/energy budget later is additive — the budget interface
  takes a set of bounds, v1 implements two.
