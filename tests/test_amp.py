"""M3: AMP export — ONNX serves with parity vs native (ADR-0004)."""

import json
from pathlib import Path

import numpy as np
import pytest

from atom.core.run import run_package
from atom.data import pack_csv


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    import csv
    import random

    rng = random.Random(11)
    src = tmp_path_factory.mktemp("ampsrc") / "waves.csv"
    with src.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "c", "label"])
        for _ in range(900):
            cls = rng.random() < 0.5
            w.writerow([f"{rng.gauss(2 if cls else -2, 1):.3f}",
                        f"{rng.gauss(0, 1):.3f}", f"{rng.gauss(1, 2):.3f}",
                        "x" if cls else "y"])
    adp = pack_csv(src, tmp_path_factory.mktemp("amppkg"), name="waves", target="label")
    outcome = run_package(str(adp), wall_clock_s=25, max_trials=8, max_rows=1500,
                          out_root=str(tmp_path_factory.mktemp("ampruns")), seed=3)
    return Path(outcome.run_dir)


def test_amp_manifest_and_parity(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["manifest_version"] == "atom-model-v1"
    assert manifest["deployable"] is True
    assert all(p["pass"] for p in manifest["parity"])
    assert manifest["signature"]["input"]["features"]
    assert manifest["lineage"]["dataset_id"].startswith("sha256:")


def test_onnx_graph_serves(run_dir):
    import onnxruntime as ort

    manifest = json.loads((run_dir / "manifest.json").read_text())
    graph = run_dir / manifest["graphs"][0]["file"]
    sess = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    d = len(manifest["signature"]["input"]["features"])
    X = np.random.default_rng(0).normal(size=(16, d)).astype(np.float32)
    outs = sess.run(None, {"X": X})
    assert len(outs[0]) == 16  # labels for 16 rows
    labels = {str(v) for v in np.asarray(outs[0]).ravel()}
    assert labels <= set(manifest["signature"]["label_map"])
