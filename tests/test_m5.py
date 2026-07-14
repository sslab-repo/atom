"""M5: lifecycle gating + drop-in discovery, clustering & anomaly paths."""

import csv
import random

from atom.contract import (
    Declaration, Modality, Module, ModuleKind, RunContext, RunResult,
    SearchSpace, TaskFamily, UnsupportedOperation,
)
from atom.core.run import run_package
from atom.data import pack_csv
from atom.registries import find, lifecycle_of
from atom.registries.builtins import load_builtins

load_builtins()


class ThirdPartyClassifier(Module):
    """Simulated external drop-in (registered via entry point)."""

    def declares(self):
        return Declaration(
            name="thirdparty-clf", version="0.1", kind=ModuleKind.METHOD,
            task_families=frozenset({TaskFamily.CLASSIFICATION}),
            modalities=frozenset({Modality.TABULAR}), category="linear",
        )

    def space(self):
        return SearchSpace()

    def run(self, ctx: RunContext) -> RunResult:
        raise UnsupportedOperation(ctx.operation)


def test_dropin_enters_experimental_and_is_gated(monkeypatch):
    import atom.registries.registry as reg

    class FakeEP:
        def load(self):
            return ThirdPartyClassifier

    monkeypatch.setattr(reg, "entry_points", lambda group: [FakeEP()])
    assert reg.discover() == 1

    module = next(m for m in reg.all_modules()
                  if m.declares().name == "thirdparty-clf")
    assert lifecycle_of(module) == "experimental"
    # excluded from the default (stable-only) search set...
    names = {m.declares().name
             for m in find(ModuleKind.METHOD, TaskFamily.CLASSIFICATION, Modality.TABULAR)}
    assert "thirdparty-clf" not in names
    # ...included on explicit opt-in
    names_x = {m.declares().name
               for m in find(ModuleKind.METHOD, TaskFamily.CLASSIFICATION, Modality.TABULAR,
                             include_experimental=True)}
    assert "thirdparty-clf" in names_x


def _unlabeled_blobs_adp(tmp_path, name="blobsu"):
    rng = random.Random(3)
    src = tmp_path / f"{name}.csv"
    with src.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x1", "x2"])
        for _ in range(900):
            c = rng.choice([(0, 0), (6, 6), (-6, 6)])
            w.writerow([f"{rng.gauss(c[0], 0.7):.3f}", f"{rng.gauss(c[1], 0.7):.3f}"])
    return pack_csv(src, tmp_path, name=name)  # no target -> unlabeled


def test_clustering_path(tmp_path):
    adp = _unlabeled_blobs_adp(tmp_path)
    outcome = run_package(str(adp), force_task="clustering", wall_clock_s=25, max_trials=8,
                          max_rows=900, out_root=str(tmp_path / "runs"), seed=2)
    assert outcome.task.family is TaskFamily.CLUSTERING
    # 3 well-separated blobs: silhouette should be high
    assert outcome.test_metrics["silhouette"] > 0.6
    assert outcome.final_kind == "single"


def test_anomaly_path_default_routing(tmp_path):
    adp = _unlabeled_blobs_adp(tmp_path, name="blobsa")
    outcome = run_package(str(adp), wall_clock_s=20, max_rows=900,
                          out_root=str(tmp_path / "runs"), seed=2)
    assert outcome.task.family is TaskFamily.ANOMALY_DETECTION
    assert outcome.final_kind == "anomaly-detector"
    assert 0.0 <= outcome.test_metrics["flagged_fraction"] <= 0.3
