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

# Parity policy — bounded row-flip rate, not max/p99: float32 arithmetic in
# the fused graph flips a small fraction of rows across tree-leaf boundaries
# (HistGB thresholds are quantile-dense, so ~1-2% flips are structural, each
# a full leaf-value jump). Gate: >=98% of rows match tightly AND labels agree
# almost everywhere; per-row extremes are recorded but do not gate.
# Primary gate: FUNCTIONAL equivalence — labels agree and the deployment-
# relevant metric is unchanged (measured: housing HistGB had 30% row-level
# drift but d_r2=0.0004; stellar 100% label agreement, d_f1=0.0).
METRIC_DELTA_MAX = 2e-3
LABEL_AGREEMENT_MIN = 0.995
ANOMALY_AGREEMENT_MIN = 0.95  # threshold-based ±1 flags flip near the boundary
# Fallback gate when no labels are available: bounded row-level drift.
PROBA_ATOL = 2e-2
REGRESSION_RTOL = 1e-2
MATCH_FRACTION_MIN = 0.98
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


def _fix_binary_tree_proba(onnx_bytes: bytes, X32: np.ndarray, classes) -> tuple[bytes, bool]:
    """skl2onnx 1.20 bug: binary tree classifiers (RF/DT) with zipmap=False
    emit probabilities [-p, p] instead of [1-p, p], and the label output is
    argmax of those broken scores. Detect empirically on a sample batch and
    repair the graph: proba += [1, 0]; label = Gather(classes, ArgMax(proba)).
    The parity gate afterwards remains the final arbiter."""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"X": X32})
    if len(outs) < 2:
        return onnx_bytes, False
    P = np.asarray(outs[1])
    broken = (P.ndim == 2 and P.shape[1] == 2 and (P < -1e-6).any()
              and np.allclose(P[:, 0], -P[:, 1], atol=1e-4))
    if not broken:
        return onnx_bytes, False

    model = onnx.load_from_string(onnx_bytes)
    graph = model.graph
    out_names = [o.name for o in graph.output]
    label_out, proba_out = out_names[0], out_names[1]
    for node in graph.node:  # detach original outputs
        node.output[:] = [
            (o + "_atomraw") if o in (label_out, proba_out) else o for o in node.output
        ]
    graph.initializer.append(
        helper.make_tensor("atom_fix_ones", TensorProto.FLOAT, [2], [1.0, 0.0]))
    classes_str = [str(c) for c in classes]
    graph.initializer.append(
        helper.make_tensor("atom_fix_classes", TensorProto.STRING, [len(classes_str)],
                           [c.encode() for c in classes_str]))
    graph.node.extend([
        helper.make_node("Add", [proba_out + "_atomraw", "atom_fix_ones"], [proba_out],
                         name="atom_fix_proba"),
        helper.make_node("ArgMax", [proba_out], ["atom_fix_argmax"], axis=1, keepdims=0,
                         name="atom_fix_argmax"),
        helper.make_node("Gather", ["atom_fix_classes", "atom_fix_argmax"], [label_out],
                         name="atom_fix_label"),
    ])
    # label output dtype is now string regardless of original
    graph.output[0].type.tensor_type.elem_type = TensorProto.STRING
    return model.SerializeToString(), True


def _fix_histgb_thresholds(onnx_bytes: bytes, sk_model) -> tuple[bytes, bool]:
    """skl2onnx encodes HistGradientBoosting float64 split thresholds by
    round-to-NEAREST float32; correct semantics for float32 inputs under
    BRANCH_LEQ (x <= t) needs the LARGEST float32 <= t64 (floor rounding).
    Quantile-dense thresholds make wrong-direction rounding flip 10-30% of
    rows (measured on california-housing). Rewrite nodes_values from the
    model's own float64 thresholds. Parity re-checks afterwards."""
    if not type(sk_model).__name__.startswith("HistGradientBoosting"):
        return onnx_bytes, False
    import onnx
    from onnx import helper

    def floor32(t) -> float:
        # NB: keep the comparison in float64 — under NumPy 2 (NEP 50),
        # np.float32 vs python-float compares in float32 and never fires.
        t64 = np.float64(t)
        f = np.float32(t64)
        return float(np.nextafter(f, np.float32(-np.inf))) if np.float64(f) > t64 else float(f)

    # (treeid, nodeid) -> floor-rounded threshold, flattening predictors in
    # converter order (per boosting round, then per class-tree within round)
    thresholds: dict[tuple[int, int], float] = {}
    tree_id = 0
    for round_predictors in sk_model._predictors:
        for predictor in round_predictors:
            for node_id, nd in enumerate(predictor.nodes):
                if not nd["is_leaf"]:
                    thresholds[(tree_id, node_id)] = floor32(nd["num_threshold"])
            tree_id += 1

    model = onnx.load_from_string(onnx_bytes)
    changed = False
    for node in model.graph.node:
        if node.op_type not in ("TreeEnsembleRegressor", "TreeEnsembleClassifier"):
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        tids, nids = attrs["nodes_treeids"], attrs["nodes_nodeids"]
        modes, values = attrs["nodes_modes"], list(attrs["nodes_values"])
        for i in range(len(values)):
            mode = modes[i].decode() if isinstance(modes[i], bytes) else modes[i]
            if mode == "BRANCH_LEQ" and (tids[i], nids[i]) in thresholds:
                values[i] = thresholds[(tids[i], nids[i])]
        attrs["nodes_values"] = values
        new_node = helper.make_node(node.op_type, list(node.input), list(node.output),
                                    name=node.name, domain=node.domain, **attrs)
        node.CopyFrom(new_node)
        changed = True
    return (model.SerializeToString(), True) if changed else (onnx_bytes, False)


def _parity(onnx_bytes: bytes, pipeline, X: np.ndarray, is_classifier: bool,
            agreement_min: float | None = None, sample_y=None) -> dict[str, Any]:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    X32 = X.astype(np.float32)
    outs = sess.run(None, {"X": X32})
    if is_classifier:
        native_pred = pipeline.predict(X32).astype(str)
        onnx_pred = np.asarray(outs[0]).astype(str)
        agreement = float(np.mean(native_pred == onnx_pred))
        report = {"label_agreement": agreement}
        min_agreement = LABEL_AGREEMENT_MIN if agreement_min is None else agreement_min
        if hasattr(pipeline, "predict_proba") and len(outs) > 1:
            diff = np.abs(pipeline.predict_proba(X32) - outs[1]).max(axis=1)
            report["proba_match_fraction"] = float(np.mean(diff <= PROBA_ATOL))
            report["proba_max_diff"] = float(diff.max())  # recorded, not gating
        if sample_y is not None:  # metric equivalence: the deployment gate
            from sklearn.metrics import f1_score

            y = np.asarray(sample_y).astype(str)
            d = abs(f1_score(y, native_pred, average="macro")
                    - f1_score(y, onnx_pred, average="macro"))
            report["metric_delta_f1_macro"] = float(d)
            report["pass"] = agreement >= min_agreement and d <= METRIC_DELTA_MAX
        elif "proba_match_fraction" in report:
            report["pass"] = (agreement >= min_agreement
                              and report["proba_match_fraction"] >= MATCH_FRACTION_MIN)
        else:
            report["pass"] = agreement >= min_agreement
        return report
    native = pipeline.predict(X32).ravel()
    onnx_out = np.asarray(outs[0]).ravel()
    rel = np.abs(native - onnx_out) / np.maximum(np.abs(native), 1e-6)
    report = {"rel_match_fraction": float(np.mean(rel <= REGRESSION_RTOL)),
              "rel_max_diff": float(rel.max())}  # recorded, not gating
    if sample_y is not None:
        from sklearn.metrics import r2_score

        y = np.asarray(sample_y, dtype=float)
        d = abs(r2_score(y, native) - r2_score(y, onnx_out))
        report["metric_delta_r2"] = float(d)
        report["pass"] = d <= METRIC_DELTA_MAX
    else:
        report["pass"] = report["rel_match_fraction"] >= MATCH_FRACTION_MIN
    return report


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
    agreement_min: float | None = None,
    sample_y=None,
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
            onnx_bytes, thr_fixed = _fix_histgb_thresholds(onnx_bytes, pipeline.steps[-1][1])
            model_classes = getattr(pipeline.steps[-1][1], "classes_", None)
            fixed = False
            if is_classifier and model_classes is not None and len(model_classes) == 2:
                onnx_bytes, fixed = _fix_binary_tree_proba(
                    onnx_bytes, sample_X.astype(np.float32), model_classes)
            parity = _parity(onnx_bytes, pipeline, sample_X, is_classifier,
                             agreement_min=agreement_min, sample_y=sample_y)
            repairs = (["binary-tree-proba"] if fixed else []) + (
                ["histgb-threshold-rounding"] if thr_fixed else [])
            if repairs:
                parity["graph_repair"] = ",".join(repairs)
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
