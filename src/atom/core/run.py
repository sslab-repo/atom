"""End-to-end run driver: ADP -> fingerprint -> task spec (confirm gate) ->
budgeted search -> refit + greedy ensemble -> LOCKED test evaluation ->
provenance. The CLI is a thin wrapper around run_package()."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from atom.contract import TaskFamily
from atom.core.dataset import load_matrix
from atom.core.ensemble import greedy_ensemble
from atom.core.evaluation import Evaluator
from atom.core.ingest import Fingerprint, fingerprint
from atom.core.orchestrator import Budget, Orchestrator, fit_pipeline
from atom.core.provenance import RunWriter
from atom.core.task_inference import TaskSpec, infer
from atom.data import DatasetPackage
from atom.registries.builtins import load_builtins

RUNNABLE_FAMILIES = {
    TaskFamily.CLASSIFICATION, TaskFamily.REGRESSION,
    TaskFamily.CLUSTERING, TaskFamily.ANOMALY_DETECTION,
}
SUPERVISED = {TaskFamily.CLASSIFICATION, TaskFamily.REGRESSION}
TOP_K = 5


@dataclass
class RunOutcome:
    run_dir: str
    task: TaskSpec
    final_kind: str  # "single" | "ensemble"
    val_score: float
    test_metrics: dict[str, float]
    n_trials: int
    elapsed_s: float


def run_package(
    package_path: str,
    target: str | None = None,
    wall_clock_s: float = 120.0,
    max_trials: int | None = None,
    min_trials: int | None = None,
    max_rows: int = 100_000,
    out_root: str = "runs",
    kb_root: str | None = None,  # default: $ATOM_HOME/metakb or ~/.atom/metakb
    force_task: str | None = None,  # task-family override (confirm gate)
    include_experimental: bool = False,
    seed: int = 0,
    confirm: Callable[[TaskSpec, Fingerprint], bool] = lambda spec, fp: True,
    progress: Callable[[str], None] = lambda s: None,
) -> RunOutcome:
    load_builtins()
    started = time.monotonic()

    with DatasetPackage.open(package_path) as pkg:
        fp = fingerprint(pkg)
        task = infer(fp, target_override=target, task_override=force_task)

        if task.family not in RUNNABLE_FAMILIES:
            raise SystemExit(
                f"task inferred as {task.family.value}"
                + (f"/{task.setting.value}" if task.setting else "")
                + " — no stable modules for this family yet. "
                "For supervised data pass --target <column>."
            )
        if not confirm(task, fp):  # the confirm gate (ADR: before spending budget)
            raise SystemExit("aborted at confirm gate")

        if pkg.manifest.mode != "tabular":
            raise SystemExit(
                f"modality '{fp.modality}' (mode={pkg.manifest.mode}): data plane is "
                "ready (pack/inspect work) but method modules for this modality land "
                "with the foundation adapters — deferred, see docs/status.md."
            )

        if task.family is TaskFamily.ANOMALY_DETECTION:
            return _run_anomaly(pkg, fp, task, wall_clock_s, max_rows, out_root,
                                include_experimental, seed, progress, started)

        budget = Budget(wall_clock_s=wall_clock_s, max_trials=max_trials, min_trials=min_trials)
        phases: dict[str, float] = {}
        progress(f"loading train/val (max {max_rows:,} train rows)…")
        train = load_matrix(pkg, fp, "train", task.target, max_rows=max_rows, seed=seed)
        val = load_matrix(pkg, fp, "val", task.target, max_rows=max(max_rows // 2, 10_000),
                          seed=seed + 1)
        phases["load_s"] = round(budget.elapsed, 1)
        if train.dropped:
            progress(f"dropped {len(train.dropped)} non-feature columns "
                     f"({', '.join(list(train.dropped)[:6])}…)")
        if not train.features:
            raise SystemExit(
                "no usable features — every column was dropped "
                f"({', '.join(f'{k}: {v}' for k, v in list(train.dropped.items())[:5])}…). "
                "Text/high-cardinality data needs the M6 foundation modules.")

        leaks = _leak_screen(train, task)
        for warning in leaks:
            task.notes.append(warning)
            progress(f"WARNING {warning}")

        # Target came via override (not in manifest roles): resolve class
        # count / imbalance / metric from the loaded training labels.
        if (task.family is TaskFamily.CLASSIFICATION and task.n_classes is None
                and train.y is not None):
            import numpy as np

            _, counts = np.unique(train.y.astype(str), return_counts=True)
            task.n_classes = int(len(counts))
            task.imbalanced = bool(counts.min() / counts.max() < 0.01)
            task.primary_metric = "roc_auc" if task.n_classes == 2 else "f1_macro"
            if task.imbalanced:
                task.notes.append("severe class imbalance (observed at load)")
            progress(f"resolved target: {task.n_classes} classes, "
                     f"metric={task.primary_metric}"
                     + (", imbalanced" if task.imbalanced else ""))

        # Meta-KB flywheel (M4): nearest-fingerprint winners warm-start the search.
        from atom.core.orchestrator.pipeline import PipelineSpec
        from atom.metakb import MetaKB, summarize_for_kb

        kb = MetaKB(kb_root)
        summary = summarize_for_kb(fp, task)
        neighbors = kb.nearest(summary)
        warm_specs = [PipelineSpec(**n["best_pipeline"]) for n in neighbors]
        if neighbors:
            prior_metric = neighbors[0].get("metric") or "score"
            progress(f"meta-KB: warm-starting from {len(neighbors)} similar run(s), "
                     f"best prior {prior_metric}≈{abs(neighbors[0]['val_score']):.3f}, "
                     f"cost≈{neighbors[0]['cost_s']:.0f}s")

        cv_folds = 3 if val.n < 1000 else 0  # small-data mode (locked default)
        if cv_folds:
            progress(f"small-data mode: {cv_folds}-fold CV scoring "
                     f"(val split has only {val.n} rows)")
            task.notes.append(f"small-data: {cv_folds}-fold CV used for trial scoring")
        evaluator = Evaluator(task, val, cv_folds=cv_folds)
        orch = Orchestrator(task, train, evaluator, budget, seed=seed, warm_specs=warm_specs,
                    include_experimental=include_experimental)
        progress(f"searching: {len(orch.methods)} methods × preprocessing, "
                 f"budget {wall_clock_s:.0f}s" + (f" / {max_trials} trials" if max_trials else ""))
        orch.run(progress=progress)
        phases["search_s"] = round(budget.elapsed - phases["load_s"], 1)

        # Finalize inside the reserved tail: refit top-K at full fidelity.
        # BUG-2: breadth adapts to what is left of the budget — load/search
        # already consumed their share.
        top_k = TOP_K if budget.elapsed < wall_clock_s * 0.9 else 2
        top = orch.best_trials(top_k)
        if not top:
            raise SystemExit("no successful trials within budget — increase --time-budget")
        progress(f"finalizing: up to {len(top)} candidates at full fidelity…")
        candidates, outputs, kept = [], [], []
        for t in top:
            # Finalize honors the budget too: always produce >=1 candidate
            # (never end without a usable artifact), stop adding more once
            # the wall clock is spent.
            est = orch.trial_cost_estimate(t.spec.method["name"], 1.0)
            over = (budget.elapsed >= wall_clock_s
                    or (est is not None and orch.get_fitted(t.spec.key()) is None
                        and budget.elapsed + est > wall_clock_s * 1.05))
            if candidates and over:
                progress(f"budget reached — finalizing with {len(candidates)} candidate(s)")
                break
            try:
                fitted = orch.get_fitted(t.spec.key())  # reuse search-time fit
                if fitted is None:
                    fitted = fit_pipeline(t.spec, orch.modules, train, 1.0, seed=t.seed)
                candidates.append(fitted)
                outputs.append(fitted.predict(val.X))
                kept.append(t)  # stays index-aligned with candidates/outputs
            except Exception as exc:  # a candidate failing must not kill the run
                progress(f"candidate failed at full fidelity, skipping: {exc}")
        if not candidates:
            raise SystemExit("all finalize candidates failed — see trials.jsonl")
        singles = [evaluator.oriented(evaluator.score_predictions(val.y, o, X=val.X))
                   for o in outputs]
        best_single_idx = max(range(len(singles)), key=singles.__getitem__)

        if task.family in SUPERVISED and budget.elapsed < wall_clock_s:
            ensemble, ens_score = greedy_ensemble(evaluator, outputs, val.y)
            use_ensemble = (ens_score > singles[best_single_idx]
                            and len(set(ensemble.members)) > 1)
        else:  # clustering (labels not comparable) or budget spent: best single
            ensemble, use_ensemble = None, False
        phases["finalize_s"] = round(budget.elapsed - phases["load_s"] - phases["search_s"], 1)

        # LOCKED test set: read once, here, for the final report only.
        test = load_matrix(pkg, fp, "test", task.target, max_rows=max(max_rows // 2, 10_000),
                           seed=seed + 2)
        test_outputs = [c.predict(test.X) for c in candidates]
        if use_ensemble:
            final_kind = "ensemble"
            for o in test_outputs:
                o["_proba_global"] = None  # recomputed inside combine via classes
            from atom.core.ensemble.greedy import _global_proba

            if ensemble.classes is not None:
                for o in test_outputs:
                    o["_proba_global"] = _global_proba(o, ensemble.classes)
            final_test = ensemble.combine(test_outputs)
            val_score = ens_score
        else:
            final_kind = "single"
            final_test = test_outputs[best_single_idx]
            val_score = singles[best_single_idx]
        test_metrics = evaluator.score_predictions(test.y, final_test, X=test.X)

        writer = RunWriter(out_root, pkg.source.name)
        writer.write_run({
            "package": {"id": pkg.manifest.content_id, "name": pkg.manifest.name,
                        "path": str(package_path)},
            "task": task.to_dict(),
            "budget": {"wall_clock_s": wall_clock_s, "max_trials": max_trials,
                       "min_trials": min_trials},
            "data": {"train_rows": train.n, "val_rows": val.n, "test_rows": test.n,
                     "features": len(train.features), "dropped": train.dropped},
            "phases": phases,
            "leak_warnings": leaks,
            "seed": seed,
        })
        for t in orch.archive:
            writer.append_trial(t.to_dict())
        writer.write_metrics({
            "primary_metric": task.primary_metric,
            "final": final_kind,
            "val_score_oriented": val_score,
            "test": test_metrics,
            "candidates": [
                {"pipeline": t.spec.to_dict(), "val_score_oriented": s}
                for t, s in zip(kept, singles)
            ],
            "ensemble_members": ensemble.members if use_ensemble else None,
        })
        writer.write_model({
            "task": task.to_dict(),
            "kind": final_kind,
            "pipelines": [c for c in (candidates if use_ensemble else
                                      [candidates[best_single_idx]])],
            "ensemble": ensemble if use_ensemble else None,
            "features": train.features,
        })
        # AMP export (ADR-0004): fused ONNX graph(s) + parity gate. Failure
        # never kills the run — deployable:false with native/ fallback.
        from atom.core.provenance.amp import export_amp

        parity_X, parity_y = _parity_sample(val, task)
        amp_candidates = candidates if use_ensemble else [candidates[best_single_idx]]
        amp = export_amp(
            run_dir=writer.dir,
            task=task.to_dict(),
            candidates=amp_candidates,
            ensemble_members=ensemble.members if use_ensemble else None,
            classes=(ensemble.classes if use_ensemble else
                     [str(c) for c in (test_outputs[best_single_idx].get("classes") or [])]
                     or None),
            features=train.features,
            sample_X=parity_X,
            sample_y=parity_y,
            lineage={
                "dataset_id": pkg.manifest.content_id,
                "dataset_name": pkg.manifest.name,
                "split": pkg.manifest.split.get("file"),
                "atom_run": writer.dir.name,
            },
            is_classifier=task.family is not TaskFamily.REGRESSION,
            should_stop=lambda: budget.elapsed > wall_clock_s * 1.2,
        )
        phases["export_s"] = round(
            budget.elapsed - sum(phases.values()), 1)
        progress("phases: " + "  ".join(f"{k}={v}s" for k, v in phases.items()))
        progress(f"AMP: deployable={amp['deployable']} "
                 f"({len(amp['graphs'])} ONNX graph(s), parity "
                 f"{'ok' if all(p.get('pass') for p in amp['parity']) else 'FAILED'})")
        writer.close()

        # Store the flywheel record: fingerprint summary -> winning config.
        best_spec = (kept[best_single_idx].spec if not use_ensemble
                     else kept[max(set(ensemble.members), key=ensemble.members.count)].spec)
        kb.append(summary, pkg.manifest.content_id, best_spec.to_dict(),
                  val_score, test_metrics, budget.elapsed, metric=task.primary_metric)

        return RunOutcome(
            run_dir=str(writer.dir), task=task, final_kind=final_kind, val_score=val_score,
            test_metrics=test_metrics, n_trials=len(orch.archive),
            elapsed_s=time.monotonic() - started,
        )


def _parity_sample(val, task, cap: int = 1024, per_class_min: int = 50):
    """Parity sample for AMP export. Classification samples are STRATIFIED
    with a minority floor — on a 3%-positive dataset a flat 512-row sample
    held ~15 positives, making f1-delta pure noise."""
    import numpy as np

    if task.family not in SUPERVISED or val.y is None:
        return val.X[:cap], (val.y[:cap] if task.family in SUPERVISED else None)
    if task.family is TaskFamily.REGRESSION:
        return val.X[:cap], val.y[:cap]
    y = val.y.astype(str)
    idx_parts = []
    classes = np.unique(y)
    per_class = max(per_class_min, cap // max(len(classes), 1))
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        idx_parts.append(cls_idx[:per_class])
    idx = np.sort(np.concatenate(idx_parts))[:cap]
    return val.X[idx], val.y[idx]


def _leak_screen(train, task, max_sample: int = 20_000, r_threshold: float = 0.98) -> list[str]:
    """Cheap target-leakage screen: flag features almost perfectly correlated
    with the target (measured need: szeged-weather 'Apparent Temperature',
    bank-churn precomputed Naive_Bayes columns both trained to 1.000 silently)."""
    import numpy as np

    if train.y is None or train.n < 50:
        return []
    n = min(train.n, max_sample)
    if task.family is TaskFamily.REGRESSION:
        try:
            yv = train.y[:n].astype(float)
        except (TypeError, ValueError):
            return []
    else:
        classes = np.unique(train.y[:n].astype(str))
        if len(classes) != 2:
            return []
        yv = (train.y[:n].astype(str) == classes[-1]).astype(float)
    flags = []
    for j, name in enumerate(train.features):
        xj = train.X[:n, j]
        mask = np.isfinite(xj)
        if mask.sum() < 50 or np.std(xj[mask]) == 0 or np.std(yv[mask]) == 0:
            continue
        r = abs(float(np.corrcoef(xj[mask], yv[mask])[0, 1]))
        if r >= r_threshold:
            flags.append(f"possible-target-leakage: '{name}' (|r|={r:.3f})")
        elif r >= 0.90:  # second tier: worth a human look, not a near-certain leak
            flags.append(f"suspicious-correlation: '{name}' (|r|={r:.3f})")
    return flags


def _run_anomaly(pkg, fp, task, wall_clock_s, max_rows, out_root,
                 include_experimental, seed, progress, started) -> RunOutcome:
    """Anomaly path (unlabeled): no honest model selection is possible
    without labels, so this is a DESCRIPTIVE run — fit the default config of
    the first matching stable detector, report score distribution and
    flagged fractions per split, ship the fitted detector + provenance."""
    import numpy as np

    from atom.contract import Modality, ModuleKind
    from atom.core.orchestrator.pipeline import PipelineSpec
    from atom.registries import find

    detectors = [
        m for m in find(ModuleKind.METHOD, task.family, Modality(task.modality),
                        include_experimental=include_experimental)
        if m.declares().setting == task.setting
    ]
    if not detectors:
        raise SystemExit(f"no {task.setting.value} detector modules registered")
    detector = detectors[0]
    decl = detector.declares()
    progress(f"anomaly ({task.setting.value}): {decl.name} with default config")

    train = load_matrix(pkg, fp, "train", None, max_rows=max_rows, seed=seed)
    modules = {decl.name: detector}
    pre = []
    for name in ("impute-simple", "scale"):
        for m in find(ModuleKind.PREPROCESSING, task.family, Modality(task.modality)):
            if m.declares().name == name:
                modules[name] = m
                pre.append({"name": name, "version": m.declares().version, "config": {}})
    defaults = {p.name: p.default for p in detector.space().parameters
                if p.default is not None}
    spec = PipelineSpec(preprocessing=pre,
                        method={"name": decl.name, "version": decl.version,
                                "config": defaults})
    fitted = fit_pipeline(spec, modules, train, 1.0, seed=seed)

    report: dict[str, dict] = {}
    for split in ("val", "test"):
        m = load_matrix(pkg, fp, split, None, max_rows=max(max_rows // 2, 10_000),
                        seed=seed + 1)
        out = fitted.predict(m.X)
        pred = np.asarray(out["pred"])
        entry = {"rows": int(len(pred)),
                 "flagged_fraction": float(np.mean(pred == -1))}
        if out.get("score") is not None:
            s = np.asarray(out["score"], dtype=float)
            entry["score_percentiles"] = {
                q: float(np.percentile(s, int(q))) for q in ("1", "5", "50", "95", "99")}
        report[split] = entry
        progress(f"{split}: flagged {entry['flagged_fraction']:.2%} of {entry['rows']:,} rows")

    writer = RunWriter(out_root, pkg.source.name)
    writer.write_run({
        "package": {"id": pkg.manifest.content_id, "name": pkg.manifest.name},
        "task": task.to_dict(),
        "pipeline": spec.to_dict(),
        "data": {"train_rows": train.n, "features": len(train.features),
                 "dropped": train.dropped},
        "seed": seed,
    })
    test_metrics = {"flagged_fraction": report["test"]["flagged_fraction"]}
    writer.write_metrics({"primary_metric": "descriptive", "final": "anomaly-detector",
                          "report": report})
    writer.write_model({"task": task.to_dict(), "kind": "anomaly-detector",
                        "pipelines": [fitted], "ensemble": None,
                        "features": train.features})

    from atom.core.provenance.amp import ANOMALY_AGREEMENT_MIN, export_amp

    amp = export_amp(
        run_dir=writer.dir, task=task.to_dict(), candidates=[fitted],
        ensemble_members=None, classes=None, features=train.features,
        sample_X=train.X[:256],
        lineage={"dataset_id": pkg.manifest.content_id,
                 "dataset_name": pkg.manifest.name,
                 "split": pkg.manifest.split.get("file"),
                 "atom_run": writer.dir.name},
        is_classifier=True,  # discrete -1/1 labels: parity by agreement
        agreement_min=ANOMALY_AGREEMENT_MIN,
    )
    progress(f"AMP: deployable={amp['deployable']}")
    writer.close()
    import time as _t

    return RunOutcome(run_dir=str(writer.dir), task=task, final_kind="anomaly-detector",
                      val_score=float("nan"), test_metrics=test_metrics,
                      n_trials=1, elapsed_s=_t.monotonic() - started)
