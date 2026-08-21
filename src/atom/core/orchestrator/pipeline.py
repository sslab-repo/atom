"""Pipeline runner: (preprocessing chain + method) over the module contract.

Leak-safety by construction: every stage is FIT on the training portion
only, then TRANSFORM/SCORE applied to evaluation portions with the fitted
artifacts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from atom.contract import Module, Operation, Parameter, RunContext
from atom.core.dataset import TabularMatrix


def sample_config(space, rng: random.Random) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for p in space.parameters:
        config[p.name] = _sample_param(p, rng)
    for p in space.parameters:  # conditional params: drop when inactive
        if p.condition is not None:
            dep, value = p.condition
            active = config.get(dep) == value or (
                isinstance(value, (list, tuple)) and config.get(dep) in value
            )
            if not active:
                config.pop(p.name, None)
    return config


def _sample_param(p: Parameter, rng: random.Random):
    if p.kind == "categorical":
        return rng.choice(list(p.domain))
    lo, hi = p.domain
    if p.kind == "int":
        return rng.randint(int(lo), int(hi))
    if p.kind == "float":
        return rng.uniform(lo, hi)
    if p.kind == "log_float":
        import math

        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    raise ValueError(f"unknown parameter kind: {p.kind}")


@dataclass
class PipelineSpec:
    """One candidate: named modules + sampled configs. JSON-serializable."""

    preprocessing: list[dict[str, Any]]  # [{"name", "version", "config"}]
    method: dict[str, Any]  # {"name", "version", "config"}

    def key(self) -> str:
        import json

        return json.dumps({"pre": self.preprocessing, "m": self.method}, sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        return {"preprocessing": self.preprocessing, "method": self.method}


@dataclass
class FittedPipeline:
    spec: PipelineSpec
    pre_stages: list[tuple[Module, dict]] = field(default_factory=list)  # (module, artifacts)
    method: Module | None = None
    method_artifacts: dict = field(default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        for module, artifacts in self.pre_stages:
            out = module.run(RunContext(Operation.TRANSFORM, {"X": X}, artifacts=artifacts))
            X = out.outputs["X"]
        return X

    def predict(self, X: np.ndarray) -> dict[str, Any]:
        X = self.transform(X)
        result = self.method.run(
            RunContext(Operation.SCORE, {"X": X}, artifacts=self.method_artifacts)
        )
        return result.outputs


def fit_pipeline(
    spec: PipelineSpec,
    modules: dict[str, Module],
    train: TabularMatrix,
    fidelity: float,
    seed: int,
) -> FittedPipeline:
    """Fit on a seeded row-subsample of `train` at the given fidelity."""
    n = train.n
    if fidelity < 1.0:
        k = max(int(n * fidelity), 50)
        idx = np.random.default_rng(seed).choice(n, size=min(k, n), replace=False)
        X, y = train.X[idx], (train.y[idx] if train.y is not None else None)
    else:
        X, y = train.X, train.y

    fitted = FittedPipeline(spec=spec)
    for stage in spec.preprocessing:
        module = modules[stage["name"]]
        result = module.run(RunContext(Operation.FIT, {"X": X}, config=dict(stage["config"])))
        fitted.pre_stages.append((module, result.artifacts))
        X = module.run(
            RunContext(Operation.TRANSFORM, {"X": X}, artifacts=result.artifacts)
        ).outputs["X"]

    method = modules[spec.method["name"]]
    config = dict(spec.method["config"])
    config["_seed"] = seed
    fit_data = {"X": X, "y": y}
    if train.seq_shape is not None:  # raw sequences: hand the (C, L) shape to deep models
        fit_data["seq_shape"] = train.seq_shape
    result = method.run(RunContext(Operation.FIT, fit_data, config=config))
    fitted.method = method
    fitted.method_artifacts = result.artifacts
    return fitted
