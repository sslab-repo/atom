"""Phase 0 (ADR-0008): device detection is torch-free & CPU-safe, the deep tier
is gracefully absent without torch, and --type records the declared modality."""

import csv

import pytest

from atom.core import device


def test_device_torch_free_and_cpu_safe(monkeypatch):
    # importing atom.core.device must never require torch
    assert device.resolve_device("cpu") == "cpu"
    if not device.torch_available():
        # no-torch machine (a lab workstation / this CI): always CPU, never raises
        assert device.resolve_device() == "cpu"
        assert device.resolve_device("cuda") == "cpu"
        assert device.resolve_device("mps") == "cpu"
        assert "no PyTorch" in device.describe()
    monkeypatch.setenv("ATOM_DEVICE", "cpu")
    assert device.resolve_device() == "cpu"


def test_deep_tier_absent_without_torch():
    # the CPU tier must stand alone: no torch => no torch-only method modules,
    # and the smoke gate still passes.
    from atom.registries.builtins import load_builtins
    from atom.registries import all_modules
    load_builtins()
    names = {m.declares().name for m in all_modules()}
    if not device.torch_available():
        assert not (names & {"lstm-classifier", "conv1d-classifier", "cnn-classifier",
                             "ganomaly"})
    # the 15-classifier CPU tier is present regardless
    assert {"logistic-regression", "random-forest", "neural-net-mlp"} <= names


@pytest.mark.parametrize("modality", ["tabular", "text", "timeseries"])
def test_pack_records_declared_modality(tmp_path, modality):
    from atom.data import DatasetPackage, pack_csv
    csvp = tmp_path / "d.csv"
    with csvp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "y"])
        for i in range(60):
            w.writerow([i, i % 3, i % 2])
    root = pack_csv(csvp, tmp_path, name=f"d_{modality}", target="y", modality=modality)
    with DatasetPackage.open(root) as pkg:
        assert pkg.manifest.modality == modality


def test_pack_rejects_bad_modality(tmp_path):
    from atom.data import pack_csv
    csvp = tmp_path / "d.csv"
    csvp.write_text("a,y\n1,0\n2,1\n")
    with pytest.raises(ValueError, match="--type"):
        pack_csv(csvp, tmp_path, name="d", target="y", modality="image")


def test_timeseries_feature_extraction(tmp_path):
    """Phase 2 (torch-free): a time-series CSV packs into a tabular ADP of
    per-sequence summary features, one row per group, split per-sequence."""
    import csv as _csv
    import random
    from atom.data import DatasetPackage, pack_timeseries_csv

    rng = random.Random(0)
    csvp = tmp_path / "ts.csv"
    with csvp.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["entity", "t", "s1", "s2", "label"])
        for e in range(120):
            rising = rng.random() < 0.5
            for t in range(15):
                w.writerow([f"e{e}", t, round((t * 0.4 if rising else rng.gauss(0, 1)), 3),
                            round(rng.gauss(5, 1), 3), "up" if rising else "flat"])
    root = pack_timeseries_csv(csvp, tmp_path, name="ts", target="label",
                               time_col="t", group_col="entity")
    with DatasetPackage.open(root) as pkg:
        m = pkg.manifest
        # one row per sequence (120 groups), not per raw row (1800)
        assert sum(m.counts[s] for s in ("train", "val", "test")) == 120
        # 2 numeric channels x 6 summary stats = 12 feature columns (+ id + target)
        feat = [c.name for c in m.columns if c.name.endswith(
            ("__mean", "__std", "__min", "__max", "__last", "__slope"))]
        assert len(feat) == 12 and "s1__slope" in feat
        assert m.modality == "tabular"          # runnable by the tabular classifiers
        assert m.dataset_source["modality"] == "timeseries"  # provenance preserved


def test_timeseries_requires_time_and_group(tmp_path):
    from atom.data import pack_timeseries_csv
    csvp = tmp_path / "x.csv"
    csvp.write_text("g,t,v,y\na,1,2,up\na,2,3,up\n")
    import pytest as _pytest
    with _pytest.raises(ValueError, match="needs --time"):
        pack_timeseries_csv(csvp, tmp_path, name="x", target="y", time_col=None, group_col="g")


def test_timeseries_raw_layout(tmp_path):
    """--ts-layout raw packs padded sequences (channel-major) + records shape."""
    import csv as _csv
    import random
    from atom.data import DatasetPackage, pack_timeseries_csv
    rng = random.Random(0)
    csvp = tmp_path / "ts.csv"
    with csvp.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["e", "t", "s1", "s2", "y"])
        for e in range(50):
            up = rng.random() < 0.5
            for t in range(12):
                w.writerow([f"e{e}", t, round(t * 0.3 if up else rng.gauss(0, 1), 3),
                            round(rng.gauss(5, 1), 3), "up" if up else "flat"])
    root = pack_timeseries_csv(csvp, tmp_path, name="tsr", target="y",
                               time_col="t", group_col="e", layout="raw")
    with DatasetPackage.open(root) as pkg:
        src = pkg.manifest.dataset_source
        assert src["layout"] == "raw" and src["n_channels"] == 2 and src["seq_len"] == 12
        # 2 channels x 12 steps = 24 sequence columns
        seq_cols = [c.name for c in pkg.manifest.columns if "__t" in c.name]
        assert len(seq_cols) == 24
        assert sum(pkg.manifest.counts[s] for s in ("train", "val", "test")) == 50


def test_conv1d_lstm_train_predict_when_torch():
    """Deep sequence classifiers register with torch and fit/predict via the
    seq_shape context; skipped where torch is absent (CPU-only machines)."""
    if not device.torch_available():
        import pytest as _pytest
        _pytest.skip("torch not installed (CPU-only machine)")
    import numpy as np
    from atom.contract import Modality, ModuleKind, Operation, RunContext, TaskFamily
    from atom.registries import find
    from atom.registries.builtins import load_builtins
    load_builtins()
    methods = {m.declares().name: m for m in
               find(ModuleKind.METHOD, TaskFamily.CLASSIFICATION, Modality.TABULAR)}
    assert {"conv1d-classifier", "lstm-classifier"} <= set(methods)
    C, L = 2, 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, C * L)).astype(np.float32)
    y = np.where(X[:, :L].mean(1) > 0, "a", "b").astype(object)
    for name in ("conv1d-classifier", "lstm-classifier"):
        m = methods[name]
        art = m.run(RunContext(Operation.FIT, {"X": X, "y": y, "seq_shape": (C, L)},
                    config={"epochs": 5, "_seed": 0})).artifacts
        out = m.run(RunContext(Operation.SCORE, {"X": X}, artifacts=art)).outputs
        assert out["proba"].shape == (60, 2) and len(out["pred"]) == 60
