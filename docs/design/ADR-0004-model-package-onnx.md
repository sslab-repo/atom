# ADR-0004: Standard Model Structure — ATOM Model Package (AMP) on ONNX

- Status: **accepted** (2026-07-14)
- Date: 2026-07-10

## Context

ATOM's core pipeline ends at "Model + Provenance out". That artifact must be
runnable outside ATOM (serving, edge, other languages) without dragging the
training framework along. ONNX is the natural interchange: framework-neutral,
versioned operator sets, mature runtime (onnxruntime, CPU/GPU), converters
from sklearn / PyTorch / XGBoost / LightGBM.

## Decision

Finalized models are exported as an **ATOM Model Package v1** — a folder or
zip, mirroring the dataset-package philosophy (manifest as contract,
checksums, provenance as data):

```
<model-slug>/
├── README.md            # human model card
├── manifest.json        # signature, label map, lineage, checksums
├── model/
│   └── pipeline.onnx    # THE artifact: preprocess + model, one graph
├── provenance/
│   ├── trials.jsonl     # every trial: config, fidelity, score, cost
│   ├── search.json      # orchestrator/search-strategy settings, budget
│   └── environment.json # module names+versions, opset, runtime versions
├── metrics.json         # final scores on the locked test set, per metric
└── native/              # OPTIONAL fallback: framework checkpoint
```

### Rules

1. **The unit of export is the inference pipeline, not the bare model.**
   Fitted preprocessing (impute/scale/encode) is composed into
   `pipeline.onnx` itself (single fused graph) so there is no train/serve
   skew. If fusion is impossible, the manifest declares an ordered chain
   of ONNX graphs; a bare model with out-of-band preprocessing is invalid.
2. **Signature in the manifest**: input/output tensor names, dtypes,
   shapes, and the label map (class index → class name), plus the task
   family and modality — enough to serve the model without reading code.
3. **Pinned opset + runtime.** Target ONNX opset is pinned per AMP major
   version (start: opset ≥ 17); `onnxruntime` is the reference runtime and
   CI gate: an AMP is valid only if `pipeline.onnx` loads and produces
   parity outputs (within tolerance) vs. the native model on a sample batch.
4. **Lineage.** The manifest records the source **ADP dataset id +
   `processed/` checksums + split version** and the winning config, tying
   every model to exactly the data and search that produced it. This is
   the record the meta-KB ingests.
5. **Ensembles** are exported as one composed ONNX graph where feasible
   (voting/averaging over member subgraphs); otherwise as declared chains
   with combination weights in the manifest.
6. **Every finalized model ships as an AMP — no exceptions.** The package
   is the sharing unit within the lab; a raw checkpoint outside an AMP is
   not a valid ATOM output. Search is NOT restricted to ONNX-exportable
   modules by default (accuracy first); a non-exportable winner ships
   `native/` only with `deployable: false` — see rule 7.
7. **ONNX-exportability is a Methods-module capability, not a hard gate.**
   Modules declare `exportable: onnx` in `hints()`/declaration. The
   orchestrator can (per run policy) restrict search to exportable
   modules, or allow non-exportable winners which then ship `native/`
   only, clearly flagged `deployable: false` in the manifest. Metric,
   search, and preprocessing-only modules are not models and are exempt.

## Consequences

- Any AMP is servable by generic ONNX infrastructure — no ATOM install
  needed at inference time.
- The parity check (rule 3) catches converter bugs before a model ships.
- Some algorithm families (exotic sklearn estimators, custom research
  code, large LLMs) convert poorly; rule 7 keeps them usable inside ATOM
  while making deployability an explicit, queryable property.
- Provenance-as-data inside the package means the meta-KB can be rebuilt
  from a shelf of AMPs — the flywheel survives KB loss.
