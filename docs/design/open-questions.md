# Open Design Questions

Design phase closed 2026-07-14: all platform-shaping questions are answered
and recorded in ADR-0001..0007. What remains here are **implementation
defaults** — decided in principle, to be detailed in the component ADR or
spec when that component is built. Changing one of these later needs a
superseding note, not a redesign.

## Locked defaults (detail during implementation)

| Area | Default locked |
|---|---|
| Anomaly-task routing | labels present → rare-class classification; normal-only trustworthy → novelty; unlabeled → outlier (+ drift if `time` role); confirm gate always shows the inferred setting (taxonomy v2.1 routing rule) |
| Exploration budget | experimental modules get a bandit-allocated slice, capped ~10% of run budget |
| Structured-prediction enablement | blocked until Metrics registry has mAP/IoU/task-specific evaluators |
| ADP zip layout | `processed/` members stored uncompressed for random access; lazy checksum verify |
| Multi-modal ADP | one manifest, per-file-group schema blocks |
| ONNX parity | per-family tolerance table, fixed sample-batch protocol at export |
| Contract progress channel | optional `report()` callback in RunContext for iterative trainers (checkpoint/resume, ASHA early report) |
| DAG data handoff | typed dataset handles per modality; serializable across the venv isolation boundary (ADR-0005) |
| Task-DAG validation | explicit legal-edge table (e.g. generative→classification OK; classification→generative not) |
| Confirm gate | everything overridable: objective, metric, budget, DAG shape; API/CLI first |
| Multi-objective | Pareto archive over (score, cost); scalarization only as user opt-in |
| Generated-data leak rule | synthetic samples visible to training folds only, never validation/test |
| Small datasets | repeated nested CV instead of fixed locked split under a size threshold |
| Fingerprint schema | versioned summary statistics only (privacy-safe, shareable — ADR-0005) |
| Repo layout | monorepo with registry namespaces; external modules via entry points |
| Runtime | Python ≥ 3.10, single machine (ADR-0005) |

## Genuinely open (future triggers)

- [ ] DMS API access — revisit fetch-from-DMS packager source when DMS
      exposes an API (ADR-0003).
- [ ] Meta-KB sharing/merge mechanism across lab users — design when the
      code is shared lab-wide (ADR-0005).
- [ ] `distill` / `full-finetune` adaptation modes — deferred from v1
      (ADR-0005).
- [ ] Distributed execution — out of scope for v1; `hints()` reserves the
      interface.
