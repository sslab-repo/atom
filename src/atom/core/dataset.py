"""Tabular dataset loading: DatasetPackage split -> numeric matrix.

M2 scope: numeric features only. String-typed columns (identifiers, IPs,
timestamps, free text) are dropped with a recorded reason; categorical
encoding arrives with the encode preprocessing modules (M5). Missing values
and infinities become NaN — the impute module owns them downstream.

Row bounding spreads reads across the whole parquet file (every batch
contributes proportionally) so ordered-by-source-file packages, like
CIC-IDS-2017, don't yield a single-day sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atom.core.ingest.profiler import (INF_SENTINELS, MISSING_SENTINELS, Fingerprint,
    column_to_pylist)
from atom.data.package import DatasetPackage


@dataclass
class TabularMatrix:
    X: np.ndarray  # (n, d) float64, NaN = missing
    y: np.ndarray | None  # object (classification) or float64 (regression)
    features: list[str]
    dropped: dict[str, str] = field(default_factory=dict)  # column -> reason

    @property
    def n(self) -> int:
        return self.X.shape[0]


def _to_float(v) -> float:
    if v is None:
        return math.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in MISSING_SENTINELS or s in INF_SENTINELS:
        return math.nan
    try:
        f = float(s)
        return f if math.isfinite(f) else math.nan
    except ValueError:
        return math.nan


def select_features(fp: Fingerprint, target: str | None) -> tuple[list[str], dict[str, str]]:
    """Numeric feature columns, with drop reasons for the rest."""
    roles = fp.roles
    ignore = set(roles.get("ignore") or [])
    id_col = roles.get("id")
    features, dropped = [], {}
    for col in fp.columns:
        if col.name == target:
            continue
        if col.name == id_col:
            dropped[col.name] = "id"
        elif col.name in ignore:
            dropped[col.name] = "role:ignore"
        elif col.dtype == "string":
            dropped[col.name] = "non-numeric (M2 numeric-only)"
        else:
            features.append(col.name)
    return features, dropped


def load_matrix(
    pkg: DatasetPackage,
    fp: Fingerprint,
    split: str,
    target: str | None,
    max_rows: int | None = None,
    seed: int = 0,
) -> TabularMatrix:
    import pyarrow.parquet as pq

    features, dropped = select_features(fp, target)
    columns = features + ([target] if target else [])
    member = pkg.processed_member(split)

    with pkg.source.open(member) as fh:
        pf = pq.ParquetFile(fh)
        total = pf.metadata.num_rows
        take_frac = 1.0 if not max_rows else min(1.0, max_rows / max(total, 1))
        rng = np.random.default_rng(seed)
        chunks = []
        for batch in pf.iter_batches(batch_size=65536, columns=columns):
            if take_frac >= 1.0:
                chunks.append(batch)
                continue
            k = int(round(batch.num_rows * take_frac))
            if k <= 0:
                continue
            idx = np.sort(rng.choice(batch.num_rows, size=k, replace=False))
            chunks.append(batch.take(idx))

    n = sum(c.num_rows for c in chunks)
    X = np.empty((n, len(features)), dtype=np.float64)
    for j, name in enumerate(features):
        pos = 0
        for chunk in chunks:
            col = chunk.column(chunk.schema.get_field_index(name))
            if str(col.type) in ("double", "float", "int64", "int32"):
                arr = col.to_numpy(zero_copy_only=False).astype(np.float64)
                arr[~np.isfinite(arr)] = np.nan
            else:
                arr = np.array([_to_float(v) for v in column_to_pylist(col)], dtype=np.float64)
            X[pos : pos + len(arr), j] = arr
            pos += len(arr)

    y = None
    if target:
        parts = []
        for chunk in chunks:
            col = chunk.column(chunk.schema.get_field_index(target))
            parts.extend("" if v is None else str(v).strip() for v in column_to_pylist(col))
        y = np.array(parts, dtype=object)

    return TabularMatrix(X=X, y=y, features=features, dropped=dropped)
