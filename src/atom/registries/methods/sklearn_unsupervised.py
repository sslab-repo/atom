"""Stable unsupervised methods: anomaly detection, clustering, reduction.

Anomaly SCORE outputs: pred (1 normal / -1 anomaly), score (higher = more
normal, sklearn decision_function convention).
Clustering SCORE outputs: pred (cluster labels for X).
"""

from __future__ import annotations

from atom.contract import (
    Declaration,
    DetectionSetting,
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

_TAB = frozenset({Modality.TABULAR})


class _FitScore(Module):
    def _build(self, config, seed):
        raise NotImplementedError

    def run(self, ctx: RunContext) -> RunResult:
        if ctx.operation is Operation.FIT:
            model = self._build(ctx.config, int(ctx.config.get("_seed", 0)))
            model.fit(ctx.data["X"])
            return RunResult(artifacts={"model": model})
        if ctx.operation is Operation.SCORE:
            model = ctx.artifacts["model"]
            out = {"pred": model.predict(ctx.data["X"])}
            if hasattr(model, "decision_function"):
                out["score"] = model.decision_function(ctx.data["X"])
            return RunResult(outputs=out)
        if ctx.operation is Operation.TRANSFORM and hasattr(ctx.artifacts.get("model"), "transform"):
            return RunResult(outputs={"X": ctx.artifacts["model"].transform(ctx.data["X"])})
        raise UnsupportedOperation(ctx.operation)


@register
class IsolationForestM(_FitScore):
    def declares(self) -> Declaration:
        return Declaration(
            name="isolation-forest", version="1.0", kind=ModuleKind.METHOD,
            task_families=frozenset({TaskFamily.ANOMALY_DETECTION}), modalities=_TAB,
            category="isolation-subspace", setting=DetectionSetting.OUTLIER, exportable=True,
        )

    def space(self) -> SearchSpace:
        return SearchSpace((
            Parameter("n_estimators", "int", (50, 300), 100),
            Parameter("contamination", "float", (0.01, 0.2), 0.05),
        ))

    def _build(self, config, seed):
        from sklearn.ensemble import IsolationForest

        return IsolationForest(
            n_estimators=config.get("n_estimators", 100),
            contamination=config.get("contamination", 0.05), random_state=seed, n_jobs=-1)


@register
class LOFNoveltyM(_FitScore):
    def declares(self) -> Declaration:
        return Declaration(
            name="lof-novelty", version="1.0", kind=ModuleKind.METHOD,
            task_families=frozenset({TaskFamily.ANOMALY_DETECTION}), modalities=_TAB,
            category="proximity", setting=DetectionSetting.NOVELTY,
        )

    def space(self) -> SearchSpace:
        return SearchSpace((Parameter("n_neighbors", "int", (5, 50), 20),))

    def _build(self, config, seed):
        from sklearn.neighbors import LocalOutlierFactor

        return LocalOutlierFactor(n_neighbors=config.get("n_neighbors", 20), novelty=True)


@register
class KMeansM(_FitScore):
    def declares(self) -> Declaration:
        return Declaration(
            name="kmeans", version="1.0", kind=ModuleKind.METHOD,
            task_families=frozenset({TaskFamily.CLUSTERING}), modalities=_TAB,
            category="partitional", exportable=True,
        )

    def space(self) -> SearchSpace:
        return SearchSpace((Parameter("n_clusters", "int", (2, 12), 8),))

    def _build(self, config, seed):
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=config.get("n_clusters", 8), n_init=4, random_state=seed)


@register
class GaussianMixtureM(_FitScore):
    def declares(self) -> Declaration:
        return Declaration(
            name="gaussian-mixture", version="1.0", kind=ModuleKind.METHOD,
            task_families=frozenset({TaskFamily.CLUSTERING}), modalities=_TAB,
            category="model-based",
        )

    def space(self) -> SearchSpace:
        return SearchSpace((
            Parameter("n_components", "int", (2, 12), 5),
            Parameter("covariance_type", "categorical", ("full", "diag"), "diag"),
        ))

    def _build(self, config, seed):
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(
            n_components=config.get("n_components", 5),
            covariance_type=config.get("covariance_type", "diag"), random_state=seed)


@register
class PCAM(_FitScore):
    def declares(self) -> Declaration:
        return Declaration(
            name="pca", version="1.0", kind=ModuleKind.METHOD,
            task_families=frozenset({TaskFamily.DIMENSION_REDUCTION}), modalities=_TAB,
            category="linear-projection", exportable=True,
        )

    def space(self) -> SearchSpace:
        return SearchSpace((Parameter("n_components", "float", (0.5, 0.99), 0.95),))

    def _build(self, config, seed):
        from sklearn.decomposition import PCA

        return PCA(n_components=config.get("n_components", 0.95), random_state=seed)

    def run(self, ctx: RunContext) -> RunResult:  # PCA has no predict()
        if ctx.operation is Operation.SCORE:
            raise UnsupportedOperation(ctx.operation)
        return super().run(ctx)
