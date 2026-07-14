"""ATOM Model Package export (ADR-0004).

The unit of export is the inference PIPELINE (fitted preprocessing fused
with the model into one ONNX graph). An AMP is valid only if the ONNX
graph reproduces native outputs on a sample batch (parity gate). Ensembles
export as a declared chain of member graphs with the combination rule in
the manifest (rule 5). Export failure never kills a run — the package
ships native/ only with deployable:false (rules 6-7).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# Parity tolerances (locked default: per-family table)
PROBA_ATOL = 5e-3
LABEL_AGREEMENT_MIN = 0.995
REGRESSION_RTOL = 1e-3
# Pin both the core opset and ai.onnx.ml (tree ensembles): skl2onnx's tree
# converter targets ml opset 3 attribute layout.
TARGET_OPSET = {"": 17, "ai.onnx.ml": 3}


def _onnx_bool_shim() -> None:
    """skl2onnx 1.20 emits Python/numpy bools inside int-list node attributes
    (tree ensembles); onnx.helper with numpy>=2 rejects them. Coerce to int
    at attribute creation. Remove when skl2onnx ships a fix upstream."""
    import onnx.helper as helper

    if getattr(helper, "_atom_bool_shim", False):
        return
    orig = helper.make_attribute

    def patched(key, value, *args, **kwargs):
        if isinstance(value, (list, tuple, np.ndarray)) and len(value):
            vals = list(value)
            if any(isinstance(v, (bool, np.bool_)) for v in vals) and all(
                isinstance(v, (bool, np.bool_, int, np.integer)) for v in vals
            ):
                value = [int(v) for v in vals]
        return orig(key, value, *args, **kwargs)

    helper.make_attribute = patched
    helper._atom_bool_shim = True


def _sklearn_pipeline(fitted) -> "object":
    """Rebuild a fitted sklearn Pipeline from a FittedPipeline (fused graph)."""
    from sklearn.pipeline import Pipeline

    steps = []
    for i, (module, artifacts) in enumerate(fitted.pre_stages):
        transformer = artifacts.get("transformer")
        if transformer is not None:
            steps.append((f"pre{i}_{module.declares().name}", transformer))
    steps.append(("model", fitted.method_artifacts["model"]))
    return Pipeline(steps)


def _convert(pipeline, n_features: int) -> bytes:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    _onnx_bool_shim()

    onx = to_onnx(
        pipeline,
        initial_types=[("X", FloatTensorType([None, n_features]))],
        options={id(pipeline.steps[-1][1]): {"zipmap": False}}
        if hasattr(pipeline.steps[-1][1], "predict_proba") else None,
        target_opset=TARGET_OPSET,
    )
    return onx.SerializeToString()


def _parity(onnx_bytes: bytes, pipeline, X: np.ndarray, is_classifier: bool) -> dict[str, Any]:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    X32 = X.astype(np.float32)
    outs = sess.run(None, {"X": X32})
    if is_classifier:
        native_pred = pipeline.predict(X32).astype(str)
        onnx_pred = np.asarray(outs[0]).astype(str)
        agreement = float(np.mean(native_pred == onnx_pred))
        report = {"label_agreement": agreement}
        if hasattr(pipeline, "predict_proba") and len(outs) > 1:
            diff = float(np.max(np.abs(pipeline.predict_proba(X32) - outs[1])))
            report["proba_max_abs_diff"] = diff
            report["pass"] = agreement >= LABEL_AGREEMENT_MIN and diff <= PROBA_ATOL
        else:
            report["pass"] = agreement >= LABEL_AGREEMENT_MIN
        return report
    native = pipeline.predict(X32).ravel()
    onnx_out = np.asarray(outs[0]).ravel()
    rel = float(np.max(np.abs(native - onnx_out) / np.maximum(np.abs(native), 1e-6)))
    return {"max_rel_diff": rel, "pass": rel <= REGRESSION_RTOL}


def export_amp(
    run_dir: Path,
    task: dict[str, Any],
    candidates: list,  # FittedPipeline, index-aligned with ensemble members
    ensemble_members: list[int] | None,
    classes: list[str] | None,
    features: list[str],
    sample_X: np.ndarray,
    lineage: dict[str, Any],
    is_classifier: bool,
) -> dict[str, Any]:
    """Write model/ + manifest.json into run_dir. Returns the manifest."""
    model_dir = run_dir / "model"
    model_dir.mkdir(exist_ok=True)
    member_ids = sorted(set(ensemble_members)) if ensemble_members else [0]

    graphs, parities, deployable = [], [], True
    for idx in member_ids:
        name = "pipeline.onnx" if len(member_ids) == 1 else f"member_{idx}.onnx"
        try:
            pipeline = _sklearn_pipeline(candidates[idx])
            onnx_bytes = _convert(pipeline, len(features))
            parity = _parity(onnx_bytes, pipeline, sample_X, is_classifier)
            parities.append({"graph": name, **parity})
            if not parity["pass"]:
                deployable = False
            (model_dir / name).write_bytes(onnx_bytes)
            graphs.append({
                "file": f"model/{name}", "member": idx,
                "sha256": "sha256:" + hashlib.sha256(onnx_bytes).hexdigest(),
            })
        except Exception as exc:  # rule 6-7: non-exportable -> native only
            deployable = False
            parities.append({"graph": name, "pass": False, "error": str(exc)[:300]})

    manifest = {
        "manifest_version": "atom-model-v1",
        "task": task,
        "signature": {
            "input": {"name": "X", "dtype": "float32", "shape": [None, len(features)],
                      "features": features},
            "outputs": (["label", "probabilities"] if is_classifier else ["prediction"]),
            "label_map": classes,
        },
        "combination": (
            {"type": "mean_proba" if is_classifier else "mean",
             "members": ensemble_members} if ensemble_members else None
        ),
        "graphs": graphs,
        "parity": parities,
        "deployable": deployable and len(graphs) == len(member_ids),
        "opset": TARGET_OPSET,
        "lineage": lineage,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (run_dir / "README.md").write_text(
        f"# ATOM Model Package — {lineage.get('dataset_name', '')}\n\n"
        f"Deployable: {manifest['deployable']} · graphs: {len(graphs)} · "
        f"see manifest.json for signature, parity report, and lineage.\n"
    )
    return manifest
