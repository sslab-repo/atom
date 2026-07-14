# ADR-0007: Module Promotion Governance

- Status: **accepted** (2026-07-14)
- Date: 2026-07-14

## Context

Modules enter the registry as `experimental` (method-taxonomy lifecycle).
Promotion to `stable` needs a policy that keeps quality high while
minimizing human workload — new algorithms arrive too often for manual
review to scale.

## Decision

Promotion `experimental → stable` is a three-stage pipeline where the
human only acts last, on a prepared dossier:

1. **Automatic benchmark gate (machine).** The candidate must pass:
   contract conformance; the smoke suite (one small standard ADP per
   declared family × modality, end-to-end); a benchmark run where it is
   competitive with — or complementary to (improves ensembles /
   Pareto-nondominated on score-vs-cost) — the current stable set; and
   reproducibility (two seeded runs within tolerance). Plus accumulated
   real-run evidence from the meta-KB (participated in ≥N searches without
   crashes/timeouts above threshold).
2. **AI-assisted review (dynamic methods).** An LLM/agent reviewer drafts
   the promotion dossier: summarizes benchmark evidence, diffs the
   module's declared behavior against its paper/README, checks license
   compatibility, flags dependency risks (per-module venv vs. shared env
   — ADR-0005), and scans `space()` for suspicious defaults. Output: a
   recommendation with cited evidence, not a decision.
3. **Human approval (minimized).** A maintainer approves or rejects the
   dossier — one read, one click; no hand-testing expected. Rejections
   feed criteria back into stage 1/2.

Demotion (`stable → deprecated`) uses the same pipeline triggered by
sustained negative evidence (failures, superseding versions).

## Consequences

- Human effort scales with the number of *promotions*, not submissions —
  the gate filters first, the agent reads the papers.
- The benchmark suite becomes critical infrastructure: it needs its own
  versioned set of standard ADPs (small, fast, per family × modality).
- AI-reviewer conclusions are advisory by construction; provenance records
  which pipeline version promoted each module.
