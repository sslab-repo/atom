"""Stable preprocessing modules: impute + scale (sklearn-backed)."""

from __future__ import annotations

from atom.contract import (
    Declaration,
    Modality,
    Module,
    ModuleKind,
    Operation,
    Parameter,
    RunContext,
    RunResult,
    SearchSpace,
    TaskFamily,
    UnsupportedOperation,
)
from atom.registries import register

_ALL_TABULAR_FAMILIES = frozenset(
    {TaskFamily.CLASSIFICATION, TaskFamily.REGRESSION, TaskFamily.CLUSTERING,
     TaskFamily.DIMENSION_REDUCTION, TaskFamily.ANOMALY_DETECTION}
)


class _FitTransform(Module):
    """Shared shell: FIT builds a sklearn transformer, TRANSFORM applies it."""

    def _build(self, config):  # -> sklearn transformer or None (passthrough)
        raise NotImplementedError

    def run(self, ctx: RunContext) -> RunResult:
        if ctx.operation is Operation.FIT:
            transformer = self._build(ctx.config)
            if transformer is not None:
                transformer.fit(ctx.data["X"])
            return RunResult(artifacts={"transformer": transformer})
        if ctx.operation is Operation.TRANSFORM:
            transformer = ctx.artifacts["transformer"]
            X = ctx.data["X"] if transformer is None else transformer.transform(ctx.data["X"])
            return RunResult(outputs={"X": X})
        raise UnsupportedOperation(ctx.operation)


@register
class Impute(_FitTransform):
    def declares(self) -> Declaration:
        return Declaration(
            name="impute-simple", version="1.0", kind=ModuleKind.PREPROCESSING,
            task_families=_ALL_TABULAR_FAMILIES,
            modalities=frozenset({Modality.TABULAR}), category="impute", exportable=True,
        )

    def space(self) -> SearchSpace:
        return SearchSpace((
            Parameter("strategy", "categorical", ("mean", "median", "most_frequent"), "median"),
        ))

    def _build(self, config):
        from sklearn.impute import SimpleImputer

        return SimpleImputer(strategy=config.get("strategy", "median"))


@register
class Scale(_FitTransform):
    def declares(self) -> Declaration:
        return Declaration(
            name="scale", version="1.0", kind=ModuleKind.PREPROCESSING,
            task_families=_ALL_TABULAR_FAMILIES,
            modalities=frozenset({Modality.TABULAR}), category="scale", exportable=True,
        )

    def space(self) -> SearchSpace:
        return SearchSpace((
            Parameter("kind", "categorical", ("standard", "robust", "minmax", "none"), "standard"),
        ))

    def _build(self, config):
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        kind = config.get("kind", "standard")
        return {"standard": StandardScaler(), "robust": RobustScaler(),
                "minmax": MinMaxScaler(), "none": None}[kind]
