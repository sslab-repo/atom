"""Stable metric evaluators: classification + regression bundles.

Contract mapping: SCORE with data={"y_true", "pred", "proba"?, "classes"?}
returns RunResult.metrics (all metrics in the bundle; the TaskSpec names
the primary one).
"""

from __future__ import annotations

import numpy as np

from atom.contract import (
    Declaration,
    Modality,
    Module,
    ModuleKind,
    Operation,
    RunContext,
    RunResult,
    SearchSpace,
    TaskFamily,
    UnsupportedOperation,
)
from atom.registries import register

_ALL_MODALITIES = frozenset({Modality.MIXED})


class _MetricBundle(Module):
    def space(self) -> SearchSpace:
        return SearchSpace()

    def run(self, ctx: RunContext) -> RunResult:
        if ctx.operation is not Operation.SCORE:
            raise UnsupportedOperation(ctx.operation)
        return RunResult(metrics=self._compute(ctx.data))

    def _compute(self, data) -> dict[str, float]:
        raise NotImplementedError


#: Metrics where larger is better; used to orient search.
HIGHER_IS_BETTER = {
    "accuracy": True, "balanced_accuracy": True, "f1_macro": True, "roc_auc": True,
    "r2": True, "rmse": False, "mae": False,
}


@register
class ClassificationMetrics(_MetricBundle):
    def declares(self) -> Declaration:
        return Declaration(
            name="classification-basic", version="1.0", kind=ModuleKind.METRIC,
            task_families=frozenset({TaskFamily.CLASSIFICATION}),
            modalities=_ALL_MODALITIES, category="classification",
        )

    def _compute(self, data) -> dict[str, float]:
        from sklearn import metrics as sk

        y, pred = data["y_true"], data["pred"]
        out = {
            "accuracy": float(sk.accuracy_score(y, pred)),
            "balanced_accuracy": float(sk.balanced_accuracy_score(y, pred)),
            "f1_macro": float(sk.f1_score(y, pred, average="macro")),
        }
        proba, classes = data.get("proba"), data.get("classes")
        if proba is not None and classes is not None and len(classes) == 2:
            pos_idx = 1
            out["roc_auc"] = float(sk.roc_auc_score((np.asarray(y) == classes[pos_idx]).astype(int),
                                                    proba[:, pos_idx]))
        return out


@register
class RegressionMetrics(_MetricBundle):
    def declares(self) -> Declaration:
        return Declaration(
            name="regression-basic", version="1.0", kind=ModuleKind.METRIC,
            task_families=frozenset({TaskFamily.REGRESSION}),
            modalities=_ALL_MODALITIES, category="regression",
        )

    def _compute(self, data) -> dict[str, float]:
        from sklearn import metrics as sk

        y = np.asarray(data["y_true"], dtype=float)
        pred = np.asarray(data["pred"], dtype=float)
        return {
            "rmse": float(np.sqrt(sk.mean_squared_error(y, pred))),
            "mae": float(sk.mean_absolute_error(y, pred)),
            "r2": float(sk.r2_score(y, pred)),
        }
