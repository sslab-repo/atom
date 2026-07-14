"""End-to-end run driver: ADP -> fingerprint -> task spec (confirm gate) ->
budgeted search -> refit + greedy ensemble -> LOCKED test evaluation ->
provenance. The CLI is a thin wrapper around run_package()."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from atom.contract import TaskFamily
from atom.core.dataset import TabularMatrix, load_matrix
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

        evaluator = Evaluator(task, val)
        orch = Orchestrator(task, train, evaluator, budget, seed=seed)
        progress(f"searching: {len(orch.methods)} methods × preprocessing, "
                 f"budget {wall_clock_s:.0f}s" + (f" / {max_trials} trials" if max_trials else ""))
        orch.run(progress=progress)

        # Finalize inside the reserved tail: refit top-K at full fidelity.
        top = orch.best_trials(TOP_K)
        if not top:
            raise SystemExit("no successful trials within budget — increase --time-budget")
        progress(f"finalizing: refitting top {len(top)} at full fidelity…")
        candidates, outputs = [], []
        for t in top:
            fitted = fit_pipeline(t.spec, orch.modules, train, 1.0, seed=t.seed)
            candidates.append(fitted)
            outputs.append(fitted.predict(val.X))
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
        writer.close()

        return RunOutcome(
            run_dir=str(writer.dir), task=task, final_kind=final_kind, val_score=val_score,
            test_metrics=test_metrics, n_trials=len(orch.archive),
            elapsed_s=time.monotonic() - started,
        )
