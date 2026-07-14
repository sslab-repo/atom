# ATOM Roadmap

> Design phase closed 2026-07-14 (ADR-0001..0007 accepted). This is the
> implementation plan. Each milestone has an exit criterion — a thing that
> demonstrably works — and CIC-IDS-2017 (the reference ADP) is the thread
> through all of them.
>
> **Status (2026-07-14): M1–M5 complete; M6 data plane complete, foundation
> adapters pending dependency approval — see [status.md](status.md).**

## Development workflow

1. Every component starts from its ADR; deviations found while building go
   back into the ADR (superseding note), never silently into code.
2. The module contract and package manifests are compatibility surfaces —
   changes there require an ADR from day one.
3. New algorithms follow the taxonomy workflow: wrap → declare → drop in →
   smoke gate → `experimental` → promotion pipeline (ADR-0007).

## Milestones

### M1 — Skeleton & data plane
Contract types (`Module`, `Declaration`, `SearchSpace`, `RunContext/Result`,
`ResourceHints` + adaptation/setting tags) · registry with decorator +
entry-point discovery · **ADP reader (zip + folder, identical semantics)** ·
packager v0 (bare CSV → ADP) · tabular profiler → Fingerprint v1.
**Exit:** `atom inspect cic-ids-2017-ml-package.zip` prints a fingerprint;
a bare CSV round-trips into a valid ADP.

### M2 — Tabular MVP (end-to-end)
Task inference for tabular (incl. anomaly routing rule + confirm gate,
CLI) · orchestrator v1 (random + ASHA over wall-clock/trial budget,
ADR-0006) · nested evaluation with hash/group/time-aware splits · stable
seed modules: preprocessing (impute/scale/encode/SMOTE) + classification
& regression (linear, tree, RF, XGBoost/LightGBM) + core metrics ·
greedy ensemble · provenance record.
**Exit:** one command takes the CIC-IDS ADP to a trained, honestly-evaluated
model with full provenance, inside a stated time budget.

### M3 — Model plane (AMP)
ONNX pipeline export (fused preprocess+model) · parity gate · AMP writer
(manifest, signature, label map, metrics, provenance, lineage to ADP) ·
`native/` fallback with `deployable: false`.
**Exit:** the M2 winner ships as an AMP that serves via onnxruntime with
parity-verified outputs — shareable with lab members as-is.

### M4 — Flywheel
Meta-KB file store (versioned, privacy-safe fingerprints) · warm-start in
task inference + orchestrator · cost model feeding estimated-end-time.
**Exit:** a repeat run on a similar dataset is measurably faster/better
than cold start; confirm gate shows a meta-KB-informed estimate.

### M5 — Breadth & governance
Anomaly-detection (outlier/novelty settings) · temporal regression ·
clustering + dimension-reduction · experimental-module venv isolation
(subprocess runner) · benchmark suite + promotion pipeline with
AI-assisted dossier (ADR-0007).
**Exit:** an unlabeled variant of CIC-IDS routes to outlier detection
automatically; a third-party module goes drop-in → experimental →
gate-promoted without touching core.

### M6 — Multi-modal & foundation
Image modality (ADP file groups, image profiler) · pretrained-embedding
modules (CLIP/DINOv2) · foundation adapters: zero-shot / few-shot / peft
(ADR-0005 scope) with weights cache + licensing metadata ·
structured-prediction enablement behind mAP/IoU evaluators.
**Exit:** an image-classification ADP runs end-to-end, with a zero-shot
foundation baseline competing against fine-tuned classical/deep modules
in one search.

### Post-v1 (triggers in open-questions.md)
DMS API fetch · meta-KB lab-wide sharing · `distill`/`full-finetune` ·
distributed execution.
