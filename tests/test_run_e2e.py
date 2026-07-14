"""End-to-end M2: packed CSV -> atom run -> model + provenance, small budget."""

import csv
import json
import math
import random
from pathlib import Path

import pytest

from atom.core.run import run_package
from atom.data import pack_csv


@pytest.fixture(scope="module")
def adp(tmp_path_factory):
    rng = random.Random(7)
    path = tmp_path_factory.mktemp("e2e") / "blobs.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x1", "x2", "x3", "label"])
        for _ in range(1200):
            cls = rng.random() < 0.4
            cx = 3.0 if cls else 0.0
            w.writerow([
                f"{rng.gauss(cx, 1):.4f}", f"{rng.gauss(-cx, 1):.4f}",
                f"{rng.gauss(0, 1):.4f}", "pos" if cls else "neg",
            ])
    out = tmp_path_factory.mktemp("pkg")
    return pack_csv(path, out, name="blobs", target="label")


def test_run_end_to_end(adp, tmp_path):
    outcome = run_package(
        str(adp), wall_clock_s=30, max_trials=12, max_rows=2000,
        out_root=str(tmp_path / "runs"), seed=1,
    )
    # separable blobs: any sane model should be near-perfect on roc_auc
    assert outcome.test_metrics["accuracy"] > 0.9
    assert outcome.n_trials >= 1

    run_dir = Path(outcome.run_dir)
    run_doc = json.loads((run_dir / "provenance" / "run.json").read_text())
    assert run_doc["package"]["id"].startswith("sha256:")
    assert run_doc["task"]["family"] == "classification"
    trials = [json.loads(l) for l in (run_dir / "provenance" / "trials.jsonl").read_text().splitlines()]
    assert len(trials) == outcome.n_trials
    assert all("pipeline" in t and "cost_s" in t for t in trials)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["primary_metric"] == "roc_auc"
    assert (run_dir / "native" / "model.pkl").exists()


def test_budget_trial_bound_respected(adp, tmp_path):
    outcome = run_package(
        str(adp), wall_clock_s=60, max_trials=4, max_rows=1500,
        out_root=str(tmp_path / "runs"), seed=2,
    )
    assert outcome.n_trials <= 4 + 0  # first bound reached stops the search
    assert not math.isnan(outcome.val_score)
