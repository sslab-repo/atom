"""Shared vocabulary of the module contract (ADR-0001, ADR-0005).

Frozen core interface: extend enums by adding values, never repurpose —
provenance and the meta-KB persist them by value. Everything crossing the
run() boundary must be JSON-serializable (isolated modules run as
subprocesses, ADR-0005).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModuleKind(str, Enum):
    PREPROCESSING = "preprocessing"
    METHOD = "method"
    SEARCH = "search"
    METRIC = "metric"


class TaskFamily(str, Enum):
    """The nine canonical families (method-taxonomy v2.1, ADR-0005)."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"  # incl. temporal/forecasting via 'temporal' tag
    CLUSTERING = "clustering"
    DIMENSION_REDUCTION = "dimension-reduction"
    ANOMALY_DETECTION = "anomaly-detection"
    GENERATIVE = "generative"
    STRUCTURED_PREDICTION = "structured-prediction"
    ASSOCIATION_MINING = "association-mining"
    PREFERENCE_LEARNING = "preference-learning"


class Modality(str, Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    TIMESERIES = "timeseries"
    AUDIO = "audio"
    VIDEO = "video"
    MIXED = "mixed"


class Paradigm(str, Enum):
    CLASSICAL = "classical"
    DEEP = "deep"
    FOUNDATION = "foundation"


class Adaptation(str, Enum):
    """Foundation-model adaptation modes (method-taxonomy v2.1)."""

    ZERO_SHOT = "zero-shot"
    FEW_SHOT = "few-shot"
    PROMPT_TUNING = "prompt-tuning"
    PEFT = "peft"
    FULL_FINETUNE = "full-finetune"  # barred in v1 (ADR-0005)
    DISTILL = "distill"  # barred in v1 (ADR-0005)


#: Adaptation modes no v1 module may declare (ADR-0005).
V1_BARRED_ADAPTATIONS = frozenset({Adaptation.FULL_FINETUNE, Adaptation.DISTILL})


class DetectionSetting(str, Enum):
    """Required for anomaly-detection modules (method-taxonomy v2.1)."""

    OUTLIER = "outlier"
    NOVELTY = "novelty"
    OOD = "ood"
    DRIFT = "drift"


class Operation(str, Enum):
    FIT = "fit"
    TRANSFORM = "transform"
    GENERATE = "generate"
    SCORE = "score"


@dataclass(frozen=True)
class Declaration:
    """Static capability statement returned by Module.declares().

    Cheap to construct: no data access, no weight loading.
    """

    name: str
    version: str
    kind: ModuleKind
    task_families: frozenset[TaskFamily]
    modalities: frozenset[Modality]
    category: str = ""  # open convention tag, e.g. "ensemble-boosting"
    paradigm: Paradigm = Paradigm.CLASSICAL
    adaptation: Adaptation | None = None  # required iff paradigm=FOUNDATION
    setting: DetectionSetting | None = None  # required iff ANOMALY_DETECTION
    tags: frozenset[str] = frozenset()  # e.g. {"temporal", "augment"}
    exportable: bool = False  # ONNX-exportable (ADR-0004 rules 6-7)

    def supports(self, task_family: TaskFamily, modality: Modality) -> bool:
        return task_family in self.task_families and (
            modality in self.modalities or Modality.MIXED in self.modalities
        )

    def validate(self) -> list[str]:
        """Return contract violations (empty = valid)."""
        problems = []
        if self.paradigm is Paradigm.FOUNDATION and self.adaptation is None:
            problems.append("foundation module must declare an adaptation mode")
        if self.adaptation in V1_BARRED_ADAPTATIONS:
            problems.append(f"adaptation '{self.adaptation.value}' is barred in v1 (ADR-0005)")
        if (
            self.kind is ModuleKind.METHOD
            and TaskFamily.ANOMALY_DETECTION in self.task_families
            and self.setting is None
        ):
            problems.append("anomaly-detection module must declare a setting")
        return problems


@dataclass(frozen=True)
class Parameter:
    """One hyperparameter / microcontrol dimension; `condition` names another
    parameter and the value(s) that activate this one."""

    name: str
    # ("categorical", choices) | ("int", (lo, hi)) | ("float", (lo, hi)) | ("log_float", (lo, hi))
    kind: str
    domain: Any
    default: Any = None
    condition: tuple[str, Any] | None = None


@dataclass(frozen=True)
class SearchSpace:
    parameters: tuple[Parameter, ...] = ()


@dataclass(frozen=True)
class ResourceHints:
    """`fidelity_levels` ordered cheap → expensive drives multi-fidelity
    search; for foundation modules the adaptation ladder is the natural
    fidelity axis."""

    cpu: int = 1
    gpu: int = 0
    memory_gb: float | None = None
    fidelity_levels: tuple[float, ...] = (1.0,)


@dataclass
class RunContext:
    """Input to Module.run(). `data` is a dataset handle (package locator +
    split + fidelity), never a live object — the boundary is serializable."""

    operation: Operation
    data: Any
    config: dict[str, Any] = field(default_factory=dict)
    fidelity: float = 1.0
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    outputs: Any = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    cost: float | None = None


class UnsupportedOperation(RuntimeError):
    """run() was asked for an Operation the module doesn't support."""
