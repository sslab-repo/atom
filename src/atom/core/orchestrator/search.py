"""Search Orchestrator v1: random sampling + successive-halving (SHA batches)
over the module fidelity ladder, under a wall-clock/trial budget (ADR-0006).

Strategy note: v1 hard-codes random+SHA; the Search registry takes over
strategy selection when strategy modules land (the loop below only touches
modules through the contract, so that swap is local to this file).
"""

from __future__ import annotations

import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from atom.contract import Modality, Module, ModuleKind
from atom.core.dataset import TabularMatrix
from atom.core.evaluation import Evaluator
from atom.core.orchestrator.budget import Budget
from atom.core.orchestrator.pipeline import PipelineSpec, fit_pipeline, sample_config
from atom.core.task_inference import TaskSpec
from atom.registries import find

PREPROCESSING_CHAIN = ("impute-simple", "scale")
DEFAULT_BATCH = 9
REDUCTION = 3


@dataclass
class Trial:
    id: int
    spec: PipelineSpec
    fidelity: float
    seed: int
    status: str = "ok"
    score: float | None = None  # oriented: higher is better
    metrics: dict[str, float] = field(default_factory=dict)
    cost_s: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "pipeline": self.spec.to_dict(), "fidelity": self.fidelity,
            "seed": self.seed, "status": self.status, "score": self.score,
            "metrics": self.metrics, "cost_s": round(self.cost_s, 3), "error": self.error,
        }


class Orchestrator:
    def __init__(
        self,
        task: TaskSpec,
        train: TabularMatrix,
        evaluator: Evaluator,
        budget: Budget,
        seed: int = 0,
        batch_size: int = DEFAULT_BATCH,
        warm_specs: list[PipelineSpec] | None = None,
    ):
        self.task = task
        self.train = train
        self.evaluator = evaluator
        self.budget = budget
        self.rng = random.Random(seed)
        self.batch_size = batch_size
        self.archive: list[Trial] = []
        self._next_id = 0
        self._cost: dict[tuple[str, float], tuple[float, int]] = {}  # (method,fid)->(mean,n)
        self._fitted_full: dict[str, tuple[float, Any]] = {}  # spec.key -> (score, fitted)
        self._warm = list(warm_specs or [])  # meta-KB winners: injected first (ADR flywheel)

        self.methods: dict[str, Module] = {
            m.declares().name: m
            for m in find(ModuleKind.METHOD, task.family, Modality(task.modality))
        }
        if not self.methods:
            raise RuntimeError(f"no method modules for {task.family.value}/{task.modality.value}")
        self.preprocessors: dict[str, Module] = {
            m.declares().name: m
            for m in find(ModuleKind.PREPROCESSING, task.family, Modality(task.modality))
            if m.declares().name in PREPROCESSING_CHAIN
        }
        self.modules = {**self.methods, **self.preprocessors}

    # -- candidate sampling ------------------------------------------------

    def _sample_spec(self, method_name: str) -> PipelineSpec:
        pre = []
        for name in PREPROCESSING_CHAIN:
            module = self.preprocessors.get(name)
            if module is None:
                continue
            pre.append({
                "name": name, "version": module.declares().version,
                "config": sample_config(module.space(), self.rng),
            })
        method = self.methods[method_name]
        return PipelineSpec(
            preprocessing=pre,
            method={
                "name": method_name, "version": method.declares().version,
                "config": sample_config(method.space(), self.rng),
            },
        )

    def _fidelity_ladder(self, method_name: str) -> tuple[float, ...]:
        levels = self.methods[method_name].hints().fidelity_levels
        return tuple(levels) or (1.0,)

    # -- trial execution ----------------------------------------------------

    def _run_trial(self, spec: PipelineSpec, fidelity: float) -> Trial:
        trial = Trial(id=self._next_id, spec=spec, fidelity=fidelity, seed=1000 + self._next_id)
        self._next_id += 1
        started = time.monotonic()
        try:
            fitted = fit_pipeline(spec, self.modules, self.train, fidelity, trial.seed)
            result = self.evaluator.evaluate(fitted, started)
            trial.score, trial.metrics, trial.cost_s = result.score, result.metrics, result.cost_s
            key = (spec.method["name"], fidelity)
            mean, n = self._cost.get(key, (0.0, 0))
            self._cost[key] = ((mean * n + trial.cost_s) / (n + 1), n + 1)
            if fidelity >= 1.0:  # keep fitted top candidates: finalize reuses, no refit
                self._fitted_full[spec.key()] = (trial.score, fitted)
                if len(self._fitted_full) > 5:
                    worst = min(self._fitted_full, key=lambda k: self._fitted_full[k][0])
                    del self._fitted_full[worst]
        except Exception:
            trial.status = "error"
            trial.error = traceback.format_exc(limit=3)
            trial.cost_s = time.monotonic() - started
        self.archive.append(trial)
        self.budget.trials_done += 1
        return trial

    # -- main loop -----------------------------------------------------------

    def run(self, progress: Callable[[str], None] = lambda s: None) -> list[Trial]:
        method_names = list(self.methods)
        rung_fidelities = self._fidelity_ladder(method_names[0])
        while not self.budget.search_exhausted():
            specs = [
                self._sample_spec(method_names[i % len(method_names)])
                for i in range(self.batch_size)
            ]
            if self._warm:  # warm-starts lead the first batch(es)
                known = {m for m in self.methods}
                usable = [s for s in self._warm if s.method["name"] in known]
                specs = (usable + specs)[: max(self.batch_size, len(usable))]
                self._warm = []
            self.rng.shuffle(method_names)
            survivors = specs
            for rung, fidelity in enumerate(rung_fidelities):
                results = []
                for spec in survivors:
                    if self.budget.search_exhausted():
                        break
                    if not self._affordable(spec.method["name"], fidelity):
                        continue  # admission control: don't start what can't finish
                    results.append(self._run_trial(spec, fidelity))
                ok = sorted(
                    (t for t in results if t.status == "ok"), key=lambda t: t.score, reverse=True
                )
                est = self.budget.estimate()
                progress(
                    f"rung f={fidelity:g}: {len(ok)}/{len(results)} ok"
                    + (f", best {self.task.primary_metric}={abs(ok[0].score):.4f}" if ok else "")
                    + f"  [{est['elapsed_s']:.0f}s elapsed, ~{est['estimated_end_in_s']:.0f}s left]"
                )
                if rung == len(rung_fidelities) - 1 or not ok:
                    break
                survivors = [t.spec for t in ok[: max(1, len(ok) // REDUCTION)]]
        return self.archive

    def _affordable(self, method_name: str, fidelity: float) -> bool:
        """Estimated trial cost must fit the remaining search window.
        Unknown costs are admitted (extrapolated from lower fidelity when
        available) — the estimate improves as the archive grows."""
        if fidelity >= 1.0 and not self._fitted_full:
            return True  # always invest in one full-fidelity fit: finalize reuses it
        remaining = self.budget.search_deadline_s - self.budget.elapsed
        est = self._cost.get((method_name, fidelity))
        if est is None:
            lower = [(f, m) for (n, f), (m, _) in self._cost.items()
                     if n == method_name and f < fidelity]
            if not lower:
                return True
            f, m = max(lower)
            est = (m * (fidelity / f), 1)
        return est[0] <= remaining

    def get_fitted(self, spec_key: str):
        entry = self._fitted_full.get(spec_key)
        return entry[1] if entry else None

    def best_trials(self, k: int) -> list[Trial]:
        """Top-k distinct pipelines by score, preferring higher-fidelity
        evidence; lower-fidelity survivors fill remaining slots (they are
        re-validated at full fidelity during finalize)."""
        seen: set[str] = set()
        out = []
        for t in sorted(
            (t for t in self.archive if t.status == "ok"),
            key=lambda t: (t.fidelity, t.score), reverse=True,
        ):
            if t.spec.key() in seen:
                continue
            seen.add(t.spec.key())
            out.append(t)
        out.sort(key=lambda t: t.score, reverse=True)
        return out[:k]
