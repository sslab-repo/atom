"""Nested evaluation over ADP splits (ADR-0003/design):

- search fits on `train` (fidelity-subsampled), scores on `val`;
- `test` is LOCKED: read once, at the very end, for the final report only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from atom.contract import Module, ModuleKind, Operation, RunContext
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
    def __init__(self, spec: TaskSpec, val: TabularMatrix):
        self.spec = spec
        self.val = val
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

    def oriented(self, metrics: dict[str, float]) -> float:
        primary = self.spec.primary_metric
        value = metrics.get(primary)
        if value is None:  # fall back to any available metric
            primary, value = next(iter(metrics.items()))
        return value if HIGHER_IS_BETTER.get(primary, True) else -value

    def evaluate(self, pipeline: "FittedPipeline", started: float) -> EvalResult:
        outputs = pipeline.predict(self.val.X)
        metrics = self.score_predictions(self.val.y, outputs, X=self.val.X)
        return EvalResult(
            score=self.oriented(metrics), metrics=metrics, cost_s=time.monotonic() - started
        )
