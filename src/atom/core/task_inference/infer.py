"""Task Inference: Fingerprint -> TaskSpec, with the anomaly routing rule
(method-taxonomy v2.1) and an evaluable objective from the Metrics registry.

Routing: labeled target present -> classification/regression (rare-class
classification if severely imbalanced). No trustworthy labels -> anomaly
detection: outlier by default, drift when a time role exists.

The TaskSpec is what the confirm gate shows; every field is overridable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from atom.contract import DetectionSetting, Modality, ModuleKind, TaskFamily
from atom.core.ingest.profiler import Fingerprint
from atom.registries import find


@dataclass
class TaskSpec:
    family: TaskFamily
    modality: Modality
    target: str | None = None
    setting: DetectionSetting | None = None  # anomaly-detection only
    primary_metric: str = ""
    n_classes: int | None = None
    imbalanced: bool = False
    split_policy: str = "provided"  # ADP splits; group/time-aware when roles exist
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CLASSIFICATION_MAX_NUMERIC_CLASSES = 20


def infer(
    fp: Fingerprint,
    target_override: str | None = None,
    task_override: str | None = None,
) -> TaskSpec:
    modality = Modality(fp.modality) if fp.modality else Modality.TABULAR
    target = target_override or (fp.roles.get("target") or [None])[0]
    notes: list[str] = []

    if task_override:  # confirm-gate override: everything is overridable
        family = TaskFamily(task_override)
        notes.append(f"task family overridden by user: {family.value}")
        spec = TaskSpec(
            family=family, modality=modality,
            target=target if family in (TaskFamily.CLASSIFICATION, TaskFamily.REGRESSION) else None,
            setting=DetectionSetting.OUTLIER if family is TaskFamily.ANOMALY_DETECTION else None,
            notes=notes,
        )
        _attach_metric(spec)
        return spec
    if target_override and target_override != (fp.roles.get("target") or [None])[0]:
        notes.append(f"target overridden by user: {target_override}")

    if fp.roles.get("time"):
        notes.append("time role present: time-aware evaluation applies")
    if fp.roles.get("group"):
        notes.append("group role present: group-aware evaluation applies")

    if target is None:
        # Anomaly routing: nothing labeled -> outlier; streams -> drift too.
        setting = DetectionSetting.OUTLIER
        if fp.roles.get("time"):
            notes.append("time role: drift detection also applicable")
        spec = TaskSpec(
            family=TaskFamily.ANOMALY_DETECTION,
            modality=modality,
            setting=setting,
            primary_metric="",
            notes=notes,
        )
        _attach_metric(spec)
        return spec

    profile = next((c for c in fp.columns if c.name == target), None)
    classes = fp.target_classes if target == ((fp.roles.get("target") or [None])[0]) else {}
    if profile is None:
        notes.append(f"target '{target}' not profiled; assuming classification")
        family = TaskFamily.CLASSIFICATION
    elif profile.dtype == "string" or profile.distinct_sampled <= CLASSIFICATION_MAX_NUMERIC_CLASSES:
        family = TaskFamily.CLASSIFICATION
    else:
        family = TaskFamily.REGRESSION

    spec = TaskSpec(family=family, modality=modality, target=target, notes=notes)
    if (family is TaskFamily.CLASSIFICATION and profile is not None
            and profile.dtype in ("integer", "number")
            and 2 < profile.distinct_sampled <= CLASSIFICATION_MAX_NUMERIC_CLASSES):
        notes.append(f"numeric target with {profile.distinct_sampled} levels — "
                     "if ordinal (grades, ratings, stages), consider --task regression")
    if family is TaskFamily.CLASSIFICATION:
        spec.n_classes = len(classes) or None
        if classes:
            top, rare = max(classes.values()), min(classes.values())
            minority = rare / sum(classes.values())
            spec.imbalanced = rare / top < 0.01
            if spec.imbalanced:
                notes.append("severe class imbalance: rare-class classification (not anomaly detection)")
            elif minority < 0.20:
                notes.append(f"class imbalance: minority class is {minority:.1%} — "
                             "default-threshold metrics may understate minority recall")
    _attach_metric(spec)
    return spec


def _attach_metric(spec: TaskSpec) -> None:
    """Attach an evaluable objective from the Metrics registry (ADR-0001:
    task inference reads Methods + Metrics so every task is evaluable)."""
    evaluators = find(ModuleKind.METRIC, spec.family, spec.modality)
    if not evaluators:
        spec.notes.append(f"no metric module registered for {spec.family.value} — not evaluable")
        return
    decl = evaluators[0].declares()
    defaults = {
        # roc_auc only when the class count is KNOWN to be 2; unknown -> f1_macro
        TaskFamily.CLASSIFICATION: "roc_auc" if spec.n_classes == 2 else "f1_macro",
        TaskFamily.REGRESSION: "rmse",
    }
    spec.primary_metric = defaults.get(spec.family, decl.category or decl.name)
