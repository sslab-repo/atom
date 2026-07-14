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

RUNNABLE_FAMILIES = {TaskFamily.CLASSIFICATION, TaskFamily.REGRESSION}
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
    seed: int = 0,
    confirm: Callable[[TaskSpec, Fingerprint], bool] = lambda spec, fp: True,
    progress: Callable[[str], None] = lambda s: None,
) -> RunOutcome:
    load_builtins()
    started = time.monotonic()

    with DatasetPackage.open(package_path) as pkg:
        fp = fingerprint(pkg)
        task = infer(fp, target_override=target)

        if task.family not in RUNNABLE_FAMILIES:
            raise SystemExit(
                f"task inferred as {task.family.value}"
                + (f"/{task.setting.value}" if task.setting else "")
                + " — no stable modules for this family yet (M5). "
                "For supervised data pass --target <column>."
            )
        if not confirm(task, fp):  # the confirm gate (ADR: before spending budget)
            raise SystemExit("aborted at confirm gate")

        budget = Budget(wall_clock_s=wall_clock_s, max_trials=max_trials, min_trials=min_trials)
        progress(f"loading train/val (max {max_rows:,} train rows)…")
        train = load_matrix(pkg, fp, "train", task.target, max_rows=max_rows, seed=seed)
        val = load_matrix(pkg, fp, "val", task.target, max_rows=max(max_rows // 2, 10_000),
                          seed=seed + 1)
        if train.dropped:
            progress(f"dropped {len(train.dropped)} non-feature columns "
                     f"({', '.join(list(train.dropped)[:6])}…)")

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

        evaluator = Evaluator(task, val)
        orch = Orchestrator(task, train, evaluator, budget, seed=seed)
        progress(f"searching: {len(orch.methods)} methods × preprocessing, "
                 f"budget {wall_clock_s:.0f}s" + (f" / {max_trials} trials" if max_trials else ""))
        orch.run(progress=progress)

        # Finalize inside the reserved tail: refit top-K at full fidelity.
        top = orch.best_trials(TOP_K)
        if not top:
            raise SystemExit("no successful trials within budget — increase --time-budget")
        progress(f"finalizing: up to {len(top)} candidates at full fidelity…")
        candidates, outputs = [], []
        for t in top:
            # Finalize honors the budget too: always produce >=1 candidate
            # (never end without a usable artifact), stop adding more once
            # the wall clock is spent.
            if candidates and budget.elapsed >= wall_clock_s:
                progress(f"budget reached — finalizing with {len(candidates)} candidate(s)")
                break
            try:
                fitted = orch.get_fitted(t.spec.key())  # reuse search-time fit
                if fitted is None:
                    fitted = fit_pipeline(t.spec, orch.modules, train, 1.0, seed=t.seed)
                candidates.append(fitted)
                outputs.append(fitted.predict(val.X))
            except Exception as exc:  # a candidate failing must not kill the run
                progress(f"candidate failed at full fidelity, skipping: {exc}")
        if not candidates:
            raise SystemExit("all finalize candidates failed — see trials.jsonl")
        singles = [evaluator.oriented(evaluator.score_predictions(val.y, o)) for o in outputs]
        best_single_idx = max(range(len(singles)), key=singles.__getitem__)

        ensemble, ens_score = greedy_ensemble(evaluator, outputs, val.y)
        use_ensemble = ens_score > singles[best_single_idx] and len(set(ensemble.members)) > 1

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
        test_metrics = evaluator.score_predictions(test.y, final_test)

        writer = RunWriter(out_root, pkg.source.name)
        writer.write_run({
            "package": {"id": pkg.manifest.content_id, "name": pkg.manifest.name,
                        "path": str(package_path)},
            "task": task.to_dict(),
            "budget": {"wall_clock_s": wall_clock_s, "max_trials": max_trials,
                       "min_trials": min_trials},
            "data": {"train_rows": train.n, "val_rows": val.n, "test_rows": test.n,
                     "features": len(train.features), "dropped": train.dropped},
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
                for t, s in zip(top, singles)
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
            sample_X=val.X[:256],
            lineage={
                "dataset_id": pkg.manifest.content_id,
                "dataset_name": pkg.manifest.name,
                "split": pkg.manifest.split.get("file"),
                "atom_run": writer.dir.name,
            },
            is_classifier=task.family is TaskFamily.CLASSIFICATION,
        )
        progress(f"AMP: deployable={amp['deployable']} "
                 f"({len(amp['graphs'])} ONNX graph(s), parity "
                 f"{'ok' if all(p.get('pass') for p in amp['parity']) else 'FAILED'})")
        writer.close()

        return RunOutcome(
            run_dir=str(writer.dir), task=task, final_kind=final_kind, val_score=val_score,
            test_metrics=test_metrics, n_trials=len(orch.archive),
            elapsed_s=time.monotonic() - started,
        )
