"""Task inference: family choice, anomaly routing rule, evaluable objective."""

from atom.contract import DetectionSetting, TaskFamily
from atom.core.ingest.profiler import ColumnProfile, Fingerprint
from atom.core.task_inference import infer
from atom.registries.builtins import load_builtins

load_builtins()


def _fp(columns, roles=None, target_classes=None):
    return Fingerprint(
        version="fingerprint-v1", package_id="sha256:x", name="t", modality="tabular",
        dataset_type="supervised", n_columns=len(columns), counts={"train": 100},
        roles={"id": "sample_id", "target": [], "ignore": [], "group": None, "time": None,
               **(roles or {})},
        target_classes=target_classes or {}, columns=columns, sampled_rows=100,
    )


def test_string_target_routes_to_classification():
    fp = _fp([ColumnProfile("label", "string", distinct_sampled=3)],
             roles={"target": ["label"]}, target_classes={"a": 50, "b": 45, "c": 5})
    spec = infer(fp)
    assert spec.family is TaskFamily.CLASSIFICATION
    assert spec.primary_metric == "f1_macro"
    assert not spec.imbalanced


def test_rare_class_stays_classification_not_anomaly():
    fp = _fp([ColumnProfile("label", "string", distinct_sampled=2)],
             roles={"target": ["label"]}, target_classes={"BENIGN": 1000, "ATTACK": 5})
    spec = infer(fp)
    assert spec.family is TaskFamily.CLASSIFICATION and spec.imbalanced


def test_high_cardinality_numeric_target_routes_to_regression():
    fp = _fp([ColumnProfile("price", "number", distinct_sampled=90)],
             roles={"target": ["price"]})
    spec = infer(fp)
    assert spec.family is TaskFamily.REGRESSION and spec.primary_metric == "rmse"


def test_unlabeled_routes_to_outlier_detection():
    spec = infer(_fp([ColumnProfile("x", "number", distinct_sampled=90)]))
    assert spec.family is TaskFamily.ANOMALY_DETECTION
    assert spec.setting is DetectionSetting.OUTLIER


def test_target_override_is_the_confirm_gate_path():
    fp = _fp([ColumnProfile("Label", "string", distinct_sampled=5)])
    spec = infer(fp, target_override="Label")
    assert spec.family is TaskFamily.CLASSIFICATION
    assert any("overridden" in n for n in spec.notes)
