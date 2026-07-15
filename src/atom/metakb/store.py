"""Meta-Knowledge Base (ADR-0005): append-only file store under the user's
account (`$ATOM_HOME` or `~/.atom`), privacy-safe by construction — records
hold fingerprint SUMMARY statistics and winning configs, never data.

Record schema (metakb-v1) is a compatibility surface: other lab members'
stores must merge cleanly later, so version it and never repurpose fields.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

RECORD_VERSION = "metakb-v1"


def default_root() -> Path:
    return Path(os.environ.get("ATOM_HOME", Path.home() / ".atom")) / "metakb"


def summarize_for_kb(fp, task) -> dict[str, Any]:
    """The privacy-safe fingerprint summary used for similarity lookup."""
    numeric = [c for c in fp.columns if c.dtype in ("integer", "number")]
    return {
        "family": task.family.value,
        "modality": task.modality.value,
        "n_rows": sum(fp.counts.values()) or fp.sampled_rows,
        "n_features": len(numeric),
        "n_classes": task.n_classes or 0,
        "missing_mean": round(
            sum(c.missing_rate for c in numeric) / max(len(numeric), 1), 4),
        "imbalanced": bool(task.imbalanced),
    }


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    d = abs(math.log10(max(a["n_rows"], 1)) - math.log10(max(b["n_rows"], 1)))
    d += abs(math.log10(max(a["n_features"], 1)) - math.log10(max(b["n_features"], 1)))
    d += abs(a["n_classes"] - b["n_classes"]) / 10.0
    d += abs(a["missing_mean"] - b["missing_mean"])
    d += 0.5 * (a["imbalanced"] != b["imbalanced"])
    return d


class MetaKB:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else default_root()
        self.path = self.root / "records.jsonl"

    def append(self, summary: dict[str, Any], package_id: str,
               best_pipeline: dict[str, Any], val_score: float,
               test_metrics: dict[str, float], cost_s: float,
               metric: str = "") -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "version": RECORD_VERSION,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": summary,
            "package_id": package_id,
            "best_pipeline": best_pipeline,
            "metric": metric,
            "val_score": val_score,
            "test": test_metrics,
            "cost_s": round(cost_s, 1),
        }
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get("version") == RECORD_VERSION:
                    out.append(rec)
            except json.JSONDecodeError:
                continue  # tolerate a torn append
        return out

    def nearest(self, summary: dict[str, Any], k: int = 3,
                max_distance: float = 2.0) -> list[dict[str, Any]]:
        """Closest same-family/modality records within max_distance, best
        first. The cutoff keeps wildly dissimilar datasets from diluting
        warm-starts (a 768-row clinic table is no prior for 3M flows)."""
        pool = [r for r in self.records()
                if r["summary"]["family"] == summary["family"]
                and r["summary"]["modality"] == summary["modality"]
                and _distance(r["summary"], summary) <= max_distance]
        pool.sort(key=lambda r: (_distance(r["summary"], summary), -r["val_score"]))
        return pool[:k]
