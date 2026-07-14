# ADR-0005: v1 Scope and Operating Model

- Status: **accepted** (2026-07-14)
- Date: 2026-07-14

## Context

The architecture (ADR-0001..0004, method-taxonomy v2.1) left three
platform-shaping choices to the project owner: task-family scope,
deployment shape, and dependency isolation.

## Decision

### Task families and foundation-model scope

The **nine task families are confirmed** as the frozen `TaskFamily` enum:
classification, regression, clustering, dimension-reduction,
anomaly-detection, generative, structured-prediction, association-mining,
preference-learning (categories per `method-taxonomy.md`, which is hereby
accepted in structure).

**Foundation-model methods are in v1 scope** with adaptation modes
`zero-shot`, `few-shot`, and `peft` (LoRA/adapters). `distill` and
`full-finetune` are **deferred** — the enum values exist, but no v1 module
may declare them.

### Deployment shape

ATOM v1 runs **on a shared lab server inside a single user account** — a
per-user tool, not a service:

- No daemon, no multi-tenancy, no auth layer. One process per run.
- The meta-knowledge base is a **file store under the user's account**
  (append-only records); its schema must permit later merging/synchronizing
  across users, but no sync mechanism is built in v1.
- Fingerprints must remain privacy-safe to share (summary statistics only,
  never raw values), because the intended end state is lab-wide sharing.
- Once stabilized, the code is shared lab-wide (open-source style). This
  raises the bar on: reproducibility (ADP/AMP as the exchange units),
  documentation, and zero machine-specific assumptions (paths, GPUs
  resolved via `hints()` at runtime).

### Dependency isolation

- **Stable built-in modules** run in the single shared environment; their
  dependencies are pinned in ATOM's own dependency set.
- **Experimental wrapper modules** MAY declare an isolated per-module venv
  (declared requirements, materialized on first use, cached). Promotion to
  `stable` requires either folding the dependencies into the shared
  environment or an explicit decision to keep the module isolated.

## Consequences

- v1 needs no scheduler, queue, or user management — the orchestrator is a
  library + CLI.
- The meta-KB file layout is a compatibility surface from day one (others
  will eventually read it); version it like the package manifests.
- Foundation modules bring a weights cache (`~/.atom/models/` or similar,
  configurable) — licensing metadata is recorded per cached asset.
- Per-module venvs imply subprocess execution for isolated modules; the
  `run()` boundary must therefore be serializable (this constrains
  RunContext/RunResult design — no live Python object handoff across the
  isolation boundary).
