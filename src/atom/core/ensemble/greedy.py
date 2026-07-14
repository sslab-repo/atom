"""Greedy ensemble selection (Caruana-style) over refit candidates.

Members are chosen with replacement on validation predictions; the final
model is the ensemble iff it beats the best single candidate on val.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from atom.contract import TaskFamily
from atom.core.evaluation import Evaluator

MAX_MEMBERS = 5


@dataclass
class EnsembleModel:
    members: list[int]  # indices into the candidate list (repeats = weight)
    task_family: TaskFamily
    classes: list[str] | None = None

    def combine(self, candidate_outputs: list[dict[str, Any]]) -> dict[str, Any]:
        picked = [candidate_outputs[i] for i in self.members]
        if self.task_family is TaskFamily.CLASSIFICATION:
            stack = np.mean([o["_proba_global"] for o in picked], axis=0)
            pred = np.array([self.classes[i] for i in np.argmax(stack, axis=1)], dtype=object)
            return {"pred": pred, "proba": stack, "classes": self.classes}
        pred = np.mean([np.asarray(o["pred"], dtype=float) for o in picked], axis=0)
        return {"pred": pred}


def _global_proba(outputs: dict[str, Any], classes: list[str]) -> np.ndarray:
    """Map a model's proba columns onto the global class list (zeros for
    classes the model never saw)."""
    proba = np.zeros((len(outputs["pred"]), len(classes)))
    if outputs.get("proba") is None:
        for i, c in enumerate(classes):  # hard votes fallback
            proba[np.asarray(outputs["pred"], dtype=object) == c, i] = 1.0
        return proba
    col = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(outputs["classes"]):
        proba[:, col[str(c)]] = outputs["proba"][:, j]
    return proba


def greedy_ensemble(
    evaluator: Evaluator, candidate_outputs: list[dict[str, Any]], y_val
) -> tuple[EnsembleModel, float]:
    """Returns (ensemble, oriented val score). Single-member ensemble =
    best single model, so the caller can compare fairly."""
    family = evaluator.spec.family
    classes = None
    if family is TaskFamily.CLASSIFICATION:
        classes = sorted({str(c) for o in candidate_outputs for c in (o.get("classes") or [])}
                         | {str(v) for v in y_val})
        for o in candidate_outputs:
            o["_proba_global"] = _global_proba(o, classes)

    members: list[int] = []
    best_score = -np.inf
    while len(members) < MAX_MEMBERS:
        step_best, step_idx = best_score, None
        for i in range(len(candidate_outputs)):
            trial = EnsembleModel(members + [i], family, classes)
            score = evaluator.oriented(
                evaluator.score_predictions(y_val, trial.combine(candidate_outputs))
            )
            if score > step_best:
                step_best, step_idx = score, i
        if step_idx is None:
            break
        members.append(step_idx)
        best_score = step_best
    return EnsembleModel(members, family, classes), float(best_score)
