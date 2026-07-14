"""Provenance out: every run leaves a complete, self-describing record
(AMP export proper lands in M3; this is the provenance/ + native/ portion)."""

from __future__ import annotations

import json
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any


class RunWriter:
    def __init__(self, out_root: str | Path, run_name: str):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(out_root) / f"{run_name}-{stamp}"
        (self.dir / "provenance").mkdir(parents=True, exist_ok=True)
        (self.dir / "native").mkdir(exist_ok=True)
        self._trials = (self.dir / "provenance" / "trials.jsonl").open("w")

    def write_run(self, doc: dict[str, Any]) -> None:
        doc = {**doc, "environment": _environment()}
        (self.dir / "provenance" / "run.json").write_text(json.dumps(doc, indent=2, default=str))

    def append_trial(self, trial_doc: dict[str, Any]) -> None:
        self._trials.write(json.dumps(trial_doc, default=str) + "\n")

    def write_metrics(self, doc: dict[str, Any]) -> None:
        (self.dir / "metrics.json").write_text(json.dumps(doc, indent=2, default=str))

    def write_model(self, payload: Any) -> Path:
        path = self.dir / "native" / "model.pkl"
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        return path

    def close(self) -> None:
        self._trials.close()


def _environment() -> dict[str, str]:
    import numpy
    import pyarrow
    import sklearn

    import atom

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "atom": atom.__version__,
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit-learn": sklearn.__version__,
    }
