"""The module contract (ADR-0001)."""

from atom.contract.module import Module
from atom.contract.types import (
    V1_BARRED_ADAPTATIONS,
    Adaptation,
    Declaration,
    DetectionSetting,
    Modality,
    ModuleKind,
    Operation,
    Paradigm,
    Parameter,
    ResourceHints,
    RunContext,
    RunResult,
    SearchSpace,
    TaskFamily,
    UnsupportedOperation,
)

__all__ = [
    "Module",
    "Adaptation",
    "Declaration",
    "DetectionSetting",
    "Modality",
    "ModuleKind",
    "Operation",
    "Paradigm",
    "Parameter",
    "ResourceHints",
    "RunContext",
    "RunResult",
    "SearchSpace",
    "TaskFamily",
    "UnsupportedOperation",
    "V1_BARRED_ADAPTATIONS",
]
