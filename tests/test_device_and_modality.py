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
