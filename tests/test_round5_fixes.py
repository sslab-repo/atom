"""Round-5 bugfix regressions:

- BUG-1: numeric columns polluted with missing-marker strings ("N/A", "?")
  must be typed numeric (coerced), not dropped as high-cardinality strings.
- BUG-2: rows with a missing target are dropped, never a phantom class.
- BUG-3: unsupervised metrics score in the pipeline's feature space, and an
  all-trials-failing search stops early with the error surfaced.
"""

import csv

import numpy as np
import pytest

from atom.core.dataset import load_matrix, select_features
from atom.core.ingest import fingerprint
from atom.data import DatasetPackage, pack_csv

ROWS = 300


@pytest.fixture(scope="module")
def dirty_pkg(tmp_path_factory):
    """bmi-style dirty numeric column + NaN-target rows in one package."""
    path = tmp_path_factory.mktemp("dirty") / "dirty.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["age", "bmi", "hp", "odd", "stage"])
        for i in range(ROWS):
            bmi = "N/A" if i % 25 == 0 else f"{18 + (i % 20)}.{i % 10}"
            hp = "?" if i % 40 == 0 else str(60 + i % 120)
            odd = "unknown" if i % 50 == 0 else f"{i}.25"  # non-sentinel marker
            stage = "" if i % 60 == 0 else str(1 + i % 4)
            w.writerow([str(20 + i % 50), bmi, hp, odd, stage])
    out = tmp_path_factory.mktemp("pkg")
    return DatasetPackage.open(pack_csv(path, out, name="dirty-test", target="stage"))


def test_bug1_dirty_numeric_coerced_not_dropped(dirty_pkg):
    fp = fingerprint(dirty_pkg)
    by_name = {c.name: c for c in fp.columns}
    assert by_name["bmi"].dtype == "number"  # sentinel path: "N/A" is missing
    assert by_name["hp"].dtype in ("integer", "number")  # sentinel path: "?"
    assert by_name["odd"].dtype == "number"  # probe path: "unknown" coerced
    features, _, dropped = select_features(fp, "stage")
    assert {"bmi", "hp", "odd"} <= set(features)
    assert not {"bmi", "hp", "odd"} & set(dropped)
    assert "numeric-coerced:odd" in fp.quality_flags


def test_bug1_sentinels_load_as_nan(dirty_pkg):
    fp = fingerprint(dirty_pkg)
    m = load_matrix(dirty_pkg, fp, "train", "stage")
    j = m.features.index("bmi")
    col = m.X[:, j]
    assert np.isnan(col).any()  # "N/A" cells became NaN for the imputer
    assert np.nanmax(col) < 50  # and the numeric values came through


def test_bug2_unlabeled_rows_dropped(dirty_pkg):
    fp = fingerprint(dirty_pkg)
    # profiler: missing target is not a class
    assert set(fp.target_classes) == {"1", "2", "3", "4"}
    assert any(f.startswith("unlabeled-rows:stage:") for f in fp.quality_flags)
    m = load_matrix(dirty_pkg, fp, "train", "stage")
    assert m.unlabeled_dropped > 0
    assert m.X.shape[0] == len(m.y)
    assert "" not in set(m.y)


def test_bug3_metric_features_transforms_for_clustering():
    from atom.core.evaluation import Evaluator
    from atom.core.orchestrator.pipeline import fit_pipeline, PipelineSpec
    from atom.core.task_inference import infer
    from atom.registries.builtins import load_builtins
    from atom.registries import find
    from atom.contract import Modality, ModuleKind, TaskFamily
    from atom.core.dataset import TabularMatrix

    load_builtins()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    X[::17, 2] = np.nan  # raw features hold NaNs the imputer owns
    train = TabularMatrix(X=X, y=None, features=[f"f{i}" for i in range(4)])

    class FP:  # minimal fingerprint stand-in for infer()
        modality, roles, target_classes, columns, quality_flags = (
            "tabular", {"target": []}, {}, [], [])

    task = infer(FP, task_override="clustering")
    modules = {m.declares().name: m
               for m in find(ModuleKind.METHOD, TaskFamily.CLUSTERING, Modality.TABULAR)}
    modules.update({m.declares().name: m
                    for m in find(ModuleKind.PREPROCESSING, TaskFamily.CLUSTERING,
                                  Modality.TABULAR)})
    spec = PipelineSpec(
        preprocessing=[{"name": "impute-simple", "version": "1.0", "config": {}},
                       {"name": "scale", "version": "1.0", "config": {}}],
        method={"name": "kmeans", "version": next(
            m.declares().version for n, m in modules.items() if n == "kmeans"),
            "config": {"n_clusters": 3}},
    )
    fitted = fit_pipeline(spec, modules, train, 1.0, seed=0)
    ev = Evaluator(task, val=train)
    Xm = ev.metric_features(fitted, X)
    assert not np.isnan(Xm).any()  # silhouette input is imputed
    result, _ = ev.evaluate_spec(spec, modules, train, 1.0, seed=0)
    assert "silhouette" in result.metrics  # scores instead of raising


def test_bug3_all_error_batches_stop_early():
    from atom.core.evaluation import Evaluator
    from atom.core.orchestrator.budget import Budget
    from atom.core.orchestrator.search import MAX_ALL_ERROR_BATCHES, Orchestrator
    from atom.core.task_inference import infer
    from atom.core.dataset import TabularMatrix
    from atom.registries.builtins import load_builtins

    load_builtins()
    X = np.zeros((60, 3))
    y = np.array(["a", "b"] * 30, dtype=object)
    train = TabularMatrix(X=X, y=y, features=["f0", "f1", "f2"])

    class FP:
        modality, roles, columns, quality_flags = "tabular", {"target": ["y"]}, [], []
        target_classes = {"a": 30, "b": 30}

    task = infer(FP, target_override="y")
    ev = Evaluator(task, val=train)
    budget = Budget(wall_clock_s=3600)  # only the breaker can stop this quickly
    orch = Orchestrator(task, train, ev, budget, seed=0)

    def broken(*a, **k):
        raise ValueError("boom")

    ev.evaluate_spec = broken  # force every trial to fail
    lines = []
    orch.run(progress=lines.append)
    batches = sum("stopping search" in ln for ln in lines)
    assert batches == 1  # breaker fired once, loop ended
    sig, count, total = orch.error_summary()
    assert "boom" in sig and count == total > 0
    assert total <= MAX_ALL_ERROR_BATCHES * 9 * 3  # no budget-long spin
