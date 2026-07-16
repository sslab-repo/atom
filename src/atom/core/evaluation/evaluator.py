"""Nested evaluation over ADP splits (ADR-0003/design):

- search fits on `train` (fidelity-subsampled), scores on `val`;
- `test` is LOCKED: read once, at the very end, for the final report only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from atom.contract import Module, ModuleKind, Operation, RunContext, TaskFamily
from atom.core.dataset import TabularMatrix
from atom.core.task_inference import TaskSpec

if TYPE_CHECKING:
    from atom.core.orchestrator.pipeline import FittedPipeline
from atom.registries import find
from atom.registries.metrics.basic import HIGHER_IS_BETTER


@dataclass
class EvalResult:
    score: float  # primary metric, oriented so HIGHER is always better
    metrics: dict[str, float]
    cost_s: float


class Evaluator:
    def __init__(self, spec: TaskSpec, val: TabularMatrix, cv_folds: int = 0):
        self.spec = spec
        self.val = val
        self.cv_folds = cv_folds  # >0: small-data mode, k-fold CV over train
        evaluators = find(ModuleKind.METRIC, spec.family, spec.modality)
        if not evaluators:
            raise RuntimeError(f"no metric module for {spec.family.value}")
        self.metric_module: Module = evaluators[0]

    def score_predictions(
        self, y_true, outputs: dict[str, Any], X=None
    ) -> dict[str, float]:
        data = {"y_true": y_true, **outputs}
        if X is not None:  # unsupervised metrics (silhouette) need the features
            data["X"] = X
        return self.metric_module.run(RunContext(Operation.SCORE, data)).metrics

    def metric_features(self, pipeline: "FittedPipeline | None", X):
        """Unsupervised metrics (silhouette) score in the model's feature
        space: raw X may hold NaNs that the pipeline's imputer owns."""
        if self.spec.family is TaskFamily.CLUSTERING and pipeline is not None:
            return pipeline.transform(X)
        return X

    def oriented(self, metrics: dict[str, float]) -> float:
        primary = self.spec.primary_metric
        value = metrics.get(primary)
        if value is None:  # fall back to any available metric
            primary, value = next(iter(metrics.items()))
        return value if HIGHER_IS_BETTER.get(primary, True) else -value

    def evaluate(self, pipeline: "FittedPipeline", started: float) -> EvalResult:
        outputs = pipeline.predict(self.val.X)
        metrics = self.score_predictions(self.val.y, outputs,
                                         X=self.metric_features(pipeline, self.val.X))
        return EvalResult(
            score=self.oriented(metrics), metrics=metrics, cost_s=time.monotonic() - started
        )

    def evaluate_spec(self, spec, modules, train: TabularMatrix, fidelity: float,
                      seed: int) -> tuple[EvalResult, Any]:
        """Fit + evaluate one candidate. Returns (result, fitted-or-None).

        Small-data mode (cv_folds > 0): k-fold CV over TRAIN — selection on a
        tiny fixed val split overfits it (measured: 485 trials on 77 val rows,
        val 0.933 vs test 0.853); fold-mean scores restore honest ranking."""
        from atom.core.orchestrator.pipeline import fit_pipeline  # runtime: avoids import cycle

        started = time.monotonic()
        if not self.cv_folds:
            fitted = fit_pipeline(spec, modules, train, fidelity, seed)
            return self.evaluate(fitted, started), fitted

        import numpy as np
        from sklearn.model_selection import KFold, StratifiedKFold

        y_str = train.y.astype(str) if train.y is not None else None
        try:
            splitter = (StratifiedKFold(self.cv_folds, shuffle=True, random_state=seed)
                        if y_str is not None else
                        KFold(self.cv_folds, shuffle=True, random_state=seed))
            folds = list(splitter.split(train.X, y_str))
        except ValueError:  # e.g. a class with < k members
            splitter = KFold(self.cv_folds, shuffle=True, random_state=seed)
            folds = list(splitter.split(train.X))
        all_metrics: dict[str, list[float]] = {}
        for tr_idx, ho_idx in folds:
            sub = TabularMatrix(X=train.X[tr_idx],
                                y=train.y[tr_idx] if train.y is not None else None,
                                features=train.features)
            fitted = fit_pipeline(spec, modules, sub, fidelity, seed)
            outputs = fitted.predict(train.X[ho_idx])
            fold_metrics = self.score_predictions(
                train.y[ho_idx] if train.y is not None else None,
                outputs, X=self.metric_features(fitted, train.X[ho_idx]))
            for k, v in fold_metrics.items():
                all_metrics.setdefault(k, []).append(v)
        metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        result = EvalResult(score=self.oriented(metrics), metrics=metrics,
                            cost_s=time.monotonic() - started)
        return result, None  # no single fitted pipeline in CV mode
