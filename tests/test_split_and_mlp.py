"""New capabilities: --split (custom + auto ratios) and the neural-net (MLP)
tabular classifier."""

import csv

import numpy as np
import pytest

from atom.data import DatasetPackage, pack_csv
from atom.data.packager import resolve_split


def test_resolve_split_default_custom_auto():
    assert resolve_split(None, 5000)[0] == {"train": 0.8, "val": 0.1, "test": 0.1}
    r, mode = resolve_split("0.7/0.15/0.15", 5000)
    assert mode == "custom" and abs(r["train"] - 0.7) < 1e-9
    r2, _ = resolve_split("70/15/15", 5000)          # normalized, same result
    assert abs(r2["train"] - 0.7) < 1e-9
    assert resolve_split("auto", 500)[0]["train"] == 0.70      # small data
    assert resolve_split("auto", 50_000)[0]["train"] == 0.80   # medium
    assert resolve_split("auto", 500_000)[0]["train"] == 0.90  # large


@pytest.mark.parametrize("bad", ["0.7/0.3", "a/b/c", "1/2/3/4", "0/0.5/0.5"])
def test_resolve_split_rejects_bad(bad):
    with pytest.raises(ValueError, match="--split"):
        resolve_split(bad, 1000)


def test_pack_respects_custom_split(tmp_path):
    csvp = tmp_path / "d.csv"
    with csvp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for i in range(1000):
            w.writerow([i, i % 2])
    root = pack_csv(csvp, tmp_path, name="d", target="y", split="0.6/0.2/0.2")
    with DatasetPackage.open(root) as pkg:
        c = pkg.manifest.counts
        assert 0.55 < c["train"] / 1000 < 0.65      # ~60% train
        assert c["train"] + c["val"] + c["test"] == 1000
        assert pkg.manifest.split["mode"] == "custom"


def test_mlp_registered_and_exports_onnx():
    from atom.contract import Modality, ModuleKind, Operation, RunContext, TaskFamily
    from atom.registries import find
    from atom.registries.builtins import load_builtins
    load_builtins()
    names = {m.declares().name for m in find(ModuleKind.METHOD, TaskFamily.CLASSIFICATION,
                                             Modality.TABULAR)}
    assert "neural-net-mlp" in names
    mlp = next(m for m in find(ModuleKind.METHOD, TaskFamily.CLASSIFICATION, Modality.TABULAR)
               if m.declares().name == "neural-net-mlp")
    X = np.random.RandomState(0).normal(size=(120, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(str)
    art = mlp.run(RunContext(Operation.FIT, {"X": X, "y": y},
                  config={"hidden_layer_sizes": "32", "_seed": 0})).artifacts
    out = mlp.run(RunContext(Operation.SCORE, {"X": X}, artifacts=art)).outputs
    assert "proba" in out and len(out["pred"]) == 120


def test_expanded_classifier_set_registered():
    from atom.contract import Modality, ModuleKind, TaskFamily
    from atom.registries import find
    from atom.registries.builtins import load_builtins
    load_builtins()
    names = {m.declares().name for m in
             find(ModuleKind.METHOD, TaskFamily.CLASSIFICATION, Modality.TABULAR)}
    expected = {"logistic-regression", "decision-tree", "random-forest",
                "hist-gradient-boosting", "neural-net-mlp", "k-nearest-neighbors",
                "support-vector-machine", "gaussian-naive-bayes", "extra-trees",
                "gradient-boosting", "adaboost", "sgd-classifier",
                "linear-discriminant-analysis", "quadratic-discriminant-analysis",
                "perceptron"}
    assert expected <= names


def test_methods_filter_restricts_and_rejects_unknown(tmp_path):
    import csv
    from atom.core.run import run_package
    csvp = tmp_path / "d.csv"
    rng = __import__("random").Random(0)
    with csvp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "y"])
        for _ in range(400):
            k = rng.random() < 0.5
            w.writerow([rng.gauss(2 if k else 0, 1), rng.gauss(0, 1), "P" if k else "N"])
    from atom.data import pack_csv
    pkg = pack_csv(csvp, tmp_path, name="d", target="y")

    out = run_package(str(pkg), wall_clock_s=8, out_root=str(tmp_path / "r1"),
                      kb_root=str(tmp_path / "kb"), only_methods={"gaussian-naive-bayes"})
    assert out.n_trials > 0   # ran with just the one method

    with pytest.raises(SystemExit, match="unknown"):
        run_package(str(pkg), wall_clock_s=5, out_root=str(tmp_path / "r2"),
                    kb_root=str(tmp_path / "kb"), only_methods={"nope"})
