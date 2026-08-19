"""Stable seed methods: classification + regression (sklearn; xgboost/lightgbm
register themselves only when importable).

Contract mapping: FIT trains (data={"X","y"}, config, fidelity already applied
upstream) -> artifacts{"model"}; SCORE predicts (data={"X"}) ->
outputs{"pred", "proba"?, "classes"?}.
"""

from __future__ import annotations

from atom.contract import (
    Declaration,
    Modality,
    Module,
    ModuleKind,
    Operation,
    Parameter,
    ResourceHints,
    RunContext,
    RunResult,
    SearchSpace,
    TaskFamily,
    UnsupportedOperation,
)
from atom.registries import register

_TAB = frozenset({Modality.TABULAR})
_FIDELITY = (0.1, 0.33, 1.0)  # row-subsample ladder


class _Supervised(Module):
    FAMILY: TaskFamily
    NAME: str
    CATEGORY: str

    def declares(self) -> Declaration:
        return Declaration(
            name=self.NAME, version="1.0", kind=ModuleKind.METHOD,
            task_families=frozenset({self.FAMILY}), modalities=_TAB,
            category=self.CATEGORY, exportable=True,
        )

    def hints(self) -> ResourceHints:
        return ResourceHints(cpu=1, fidelity_levels=_FIDELITY)

    def _build(self, config, seed):
        raise NotImplementedError

    def run(self, ctx: RunContext) -> RunResult:
        if ctx.operation is Operation.FIT:
            model = self._build(ctx.config, seed=int(ctx.config.get("_seed", 0)))
            fit_kwargs = {}
            # Class balancing is a per-classifier search dimension: sample_weight
            # works uniformly across every sklearn classifier (incl. HistGB,
            # which has no class_weight ctor arg), and only reweights training —
            # the exported graph is unchanged. The search keeps it only when it
            # wins on the task metric (decisive for macro-F1 on imbalanced
            # multiclass; neutral otherwise).
            if (self.FAMILY is TaskFamily.CLASSIFICATION
                    and ctx.config.get("class_balance") == "balanced"):
                from sklearn.utils.class_weight import compute_sample_weight

                fit_kwargs["sample_weight"] = compute_sample_weight("balanced", ctx.data["y"])
            model.fit(ctx.data["X"], ctx.data["y"], **fit_kwargs)
            return RunResult(artifacts={"model": model})
        if ctx.operation is Operation.SCORE:
            model = ctx.artifacts["model"]
            out = {"pred": model.predict(ctx.data["X"])}
            if hasattr(model, "predict_proba"):
                out["proba"] = model.predict_proba(ctx.data["X"])
                out["classes"] = list(model.classes_)
            return RunResult(outputs=out)
        raise UnsupportedOperation(ctx.operation)


# --- classification -------------------------------------------------------

@register
class LogisticRegressionM(_Supervised):
    FAMILY, NAME, CATEGORY = TaskFamily.CLASSIFICATION, "logistic-regression", "linear"

    def space(self) -> SearchSpace:
        return SearchSpace((
            Parameter("C", "log_float", (1e-3, 1e2), 1.0),
            Parameter("class_balance", "categorical", ("none", "balanced"), "none"),
        ))

    def _build(self, config, seed):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=config.get("C", 1.0), max_iter=2000, random_state=seed)


@register
class DecisionTreeM(_Supervised):
    FAMILY, NAME, CATEGORY = TaskFamily.CLASSIFICATION, "decision-tree", "tree"

    def space(self) -> SearchSpace:
        return SearchSpace((
            Parameter("max_depth", "int", (3, 30), 12),
            Parameter("min_samples_leaf", "int", (1, 50), 5),
            Parameter("class_balance", "categorical", ("none", "balanced"), "none"),
        ))

    def _build(self, config, seed):
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(
            max_depth=config.get("max_depth"), min_samples_leaf=config.get("min_samples_leaf", 5),
            random_state=seed)


@register
class RandomForestM(_Supervised):
    FAMILY, NAME, CATEGORY = TaskFamily.CLASSIFICATION, "random-forest", "ensemble-bagging"

    def space(self) -> SearchSpace:
        return SearchSpace((
            Parameter("n_estimators", "int", (50, 300), 100),
            Parameter("max_depth", "int", (5, 40), 20),
            Parameter("min_samples_leaf", "int", (1, 20), 2),
            Parameter("class_balance", "categorical", ("none", "balanced"), "none"),
        ))

    def _build(self, config, seed):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100), max_depth=config.get("max_depth"),
            min_samples_leaf=config.get("min_samples_leaf", 2), n_jobs=-1, random_state=seed)


@register
class HistGBClassifierM(_Supervised):
    FAMILY, NAME, CATEGORY = TaskFamily.CLASSIFICATION, "hist-gradient-boosting", "ensemble-boosting"

    def space(self) -> SearchSpace:
        return SearchSpace((
            Parameter("learning_rate", "log_float", (0.01, 0.5), 0.1),
            Parameter("max_iter", "int", (50, 400), 150),
            Parameter("max_leaf_nodes", "int", (15, 127), 31),
            Parameter("class_balance", "categorical", ("none", "balanced"), "none"),
        ))

    def _build(self, config, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=config.get("learning_rate", 0.1), max_iter=config.get("max_iter", 150),
            max_leaf_nodes=config.get("max_leaf_nodes", 31), random_state=seed,
            early_stopping=False)  # auto-ES stratified-splits; breaks on 1-member classes


@register
class MLPClassifierM(_Supervised):
    """Feed-forward neural network (multi-layer perceptron) — the deep-learning
    classifier for TABULAR data. Searched and compared head-to-head with the
    classical methods by `atom run`; exports to ONNX like the rest. (No
    class_balance dim: MLPClassifier.fit takes no sample_weight.)"""

    FAMILY, NAME, CATEGORY = TaskFamily.CLASSIFICATION, "neural-net-mlp", "neural-network"

    def space(self) -> SearchSpace:
        return SearchSpace((
            Parameter("hidden_layer_sizes", "categorical",
                      ("64", "128,64", "256,128", "256,128,64"), "128,64"),
            Parameter("alpha", "log_float", (1e-6, 1e-1), 1e-4),
            Parameter("learning_rate_init", "log_float", (1e-4, 1e-2), 1e-3),
        ))

    def _build(self, config, seed):
        from sklearn.neural_network import MLPClassifier

        hls = tuple(int(x) for x in str(config.get("hidden_layer_sizes", "128,64")).split(","))
        return MLPClassifier(
            hidden_layer_sizes=hls, alpha=config.get("alpha", 1e-4),
            learning_rate_init=config.get("learning_rate_init", 1e-3),
            max_iter=300, early_stopping=True, random_state=seed)


# --- regression ------------------------------------------------------------

@register
class RidgeM(_Supervised):
    FAMILY, NAME, CATEGORY = TaskFamily.REGRESSION, "ridge", "linear"

    def space(self) -> SearchSpace:
        return SearchSpace((Parameter("alpha", "log_float", (1e-3, 1e3), 1.0),))

    def _build(self, config, seed):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=config.get("alpha", 1.0), random_state=seed)


@register
class RandomForestRegM(_Supervised):
    FAMILY, NAME, CATEGORY = TaskFamily.REGRESSION, "random-forest-reg", "tree-ensemble"

    def space(self) -> SearchSpace:
        return SearchSpace((
            Parameter("n_estimators", "int", (50, 300), 100),
            Parameter("max_depth", "int", (5, 40), 20),
        ))

    def _build(self, config, seed):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=config.get("n_estimators", 100), max_depth=config.get("max_depth"),
            n_jobs=-1, random_state=seed)


@register
class HistGBRegressorM(_Supervised):
    FAMILY, NAME, CATEGORY = TaskFamily.REGRESSION, "hist-gradient-boosting-reg", "tree-ensemble"

    def space(self) -> SearchSpace:
        return SearchSpace((
            Parameter("learning_rate", "log_float", (0.01, 0.5), 0.1),
            Parameter("max_iter", "int", (50, 400), 150),
        ))

    def _build(self, config, seed):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            learning_rate=config.get("learning_rate", 0.1), max_iter=config.get("max_iter", 150),
            random_state=seed)


# --- optional boosted libraries (drop-in if installed) ----------------------

try:  # pragma: no cover - depends on environment
    import xgboost  # noqa: F401

    @register
    class XGBoostM(_Supervised):
        FAMILY, NAME, CATEGORY = TaskFamily.CLASSIFICATION, "xgboost", "ensemble-boosting"

        def space(self) -> SearchSpace:
            return SearchSpace((
                Parameter("learning_rate", "log_float", (0.01, 0.5), 0.1),
                Parameter("n_estimators", "int", (50, 400), 200),
                Parameter("max_depth", "int", (3, 12), 6),
                Parameter("class_balance", "categorical", ("none", "balanced"), "none"),
            ))

        def _build(self, config, seed):
            from xgboost import XGBClassifier

            return XGBClassifier(
                learning_rate=config.get("learning_rate", 0.1),
                n_estimators=config.get("n_estimators", 200),
                max_depth=config.get("max_depth", 6), n_jobs=-1, random_state=seed)
except ImportError:
    pass
