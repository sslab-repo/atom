"""M4: meta-KB — records appended, nearest lookup, warm-start injection."""

import csv
import json
import random
from pathlib import Path

from atom.core.run import run_package
from atom.data import pack_csv
from atom.metakb import MetaKB


def _make_adp(tmp, name, seed):
    rng = random.Random(seed)
    src = tmp / f"{name}.csv"
    with src.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x1", "x2", "label"])
        for _ in range(700):
            cls = rng.random() < 0.5
            w.writerow([f"{rng.gauss(2 if cls else -2, 1):.3f}",
                        f"{rng.gauss(-1 if cls else 1, 1):.3f}", "a" if cls else "b"])
    return pack_csv(src, tmp, name=name, target="label")


def test_flywheel_warm_start(tmp_path):
    kb_root = tmp_path / "kb"
    adp1 = _make_adp(tmp_path, "fly1", 5)
    adp2 = _make_adp(tmp_path, "fly2", 6)  # similar dataset, different package

    o1 = run_package(str(adp1), wall_clock_s=20, max_trials=6, max_rows=1000,
                     out_root=str(tmp_path / "r1"), kb_root=str(kb_root), seed=4)
    kb = MetaKB(kb_root)
    recs = kb.records()
    assert len(recs) == 1
    stored = recs[0]["best_pipeline"]
    assert recs[0]["summary"]["family"] == "classification"

    o2 = run_package(str(adp2), wall_clock_s=20, max_trials=6, max_rows=1000,
                     out_root=str(tmp_path / "r2"), kb_root=str(kb_root), seed=9)
    assert len(kb.records()) == 2

    # warm-start: the stored winner must be the FIRST trial of run 2
    trials_path = Path(o2.run_dir) / "provenance" / "trials.jsonl"
    first = json.loads(trials_path.read_text().splitlines()[0])
    assert first["pipeline"] == stored
    assert o1.test_metrics and o2.test_metrics


def test_nearest_filters_family(tmp_path):
    kb = MetaKB(tmp_path / "kb2")
    kb.append({"family": "regression", "modality": "tabular", "n_rows": 100,
               "n_features": 3, "n_classes": 0, "missing_mean": 0.0, "imbalanced": False},
              "sha256:a", {"preprocessing": [], "method": {}}, 0.5, {}, 1.0)
    query = {"family": "classification", "modality": "tabular", "n_rows": 100,
             "n_features": 3, "n_classes": 2, "missing_mean": 0.0, "imbalanced": False}
    assert kb.nearest(query) == []
