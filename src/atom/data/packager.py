"""Packager v0: convert a loose CSV into a valid ATOM Dataset Package
(ADR-0003 — loose inputs are converted, not rejected).

Produces: manifest.json (atom-dataset-v1), raw/ (byte-exact copy),
processed/{train,val,test}.parquet (typed), splits/split_v1.json, README.md.
Split rule matches the reference package: seeded sha256 hash, 80/10/10.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}  # default
SPLIT_SEED = 42


def resolve_split(spec: str | None, n_rows: int) -> tuple[dict[str, float], str]:
    """Turn a --split spec into (ratios, mode).

    spec: None -> default 80/10/10; "auto" -> size-based heuristic;
    "0.7/0.15/0.15" (or "70/15/15") -> custom train/val/test, normalized.
    A validation split is always kept — ATOM selects the model on it."""
    if spec is None:
        return dict(SPLIT_RATIOS), "default"
    if spec.strip().lower() == "auto":
        # more data -> a smaller test fraction still gives a reliable estimate;
        # less data -> keep val/test large enough to be stable.
        if n_rows < 1_000:
            r = (0.70, 0.15, 0.15)
        elif n_rows < 100_000:
            r = (0.80, 0.10, 0.10)
        else:
            r = (0.90, 0.05, 0.05)
        return {"train": r[0], "val": r[1], "test": r[2]}, f"auto(n={n_rows})"
    parts = spec.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"--split must be TRAIN/VAL/TEST (e.g. 0.7/0.15/0.15) or 'auto', got {spec!r}")
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"--split ratios must be numbers, got {spec!r}") from exc
    if any(v <= 0 for v in vals):
        raise ValueError("--split ratios must all be > 0 (a validation split is required)")
    total = sum(vals)
    return {"train": vals[0] / total, "val": vals[1] / total, "test": vals[2] / total}, "custom"


def _hash_split(sample_id: str, ratios: dict[str, float], seed: int = SPLIT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    frac = int.from_bytes(digest[:8], "big") / 2**64
    if frac < ratios["train"]:
        return "train"
    if frac < ratios["train"] + ratios["val"]:
        return "val"
    return "test"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def open_text_tolerant(path: Path):
    """Open a text file as UTF-8, falling back to cp1252 (BUG-1: real-world
    CSVs like uciml/sms-spam are Latin-1/cp1252; never crash on encoding)."""
    raw = path.read_bytes()
    probe = raw if len(raw) < (64 << 20) else raw[: (8 << 20) - 4]
    try:
        probe.decode("utf-8")
        enc = "utf-8"
    except UnicodeDecodeError:
        enc = "cp1252"
    return path.open("r", encoding=enc, errors="replace", newline="")


def _dedupe(names: list[str]) -> list[str]:
    used, out = set(), []
    for i, raw in enumerate(names):
        name = raw.strip() or f"column_{i + 1}"
        candidate, n = name, 2
        while candidate in used:
            candidate = f"{name}.{n}"
            n += 1
        used.add(candidate)
        out.append(candidate)
    return out


def _infer_arrow_type(values: list[str]):
    """int64 -> float64 -> string, from non-missing sampled values."""
    import pyarrow as pa

    non_missing = [v for v in values if v.strip() not in ("", "NaN", "NA", "null")]
    if not non_missing:
        return pa.string()
    try:
        for v in non_missing:
            int(v)
        return pa.int64()
    except ValueError:
        pass
    try:
        for v in non_missing:
            float(v)
        return pa.float64()
    except ValueError:
        return pa.string()


TABULAR_MODALITIES = ("tabular", "text", "timeseries")


def pack_csv(
    csv_path: str | Path,
    out_dir: str | Path,
    name: str | None = None,
    target: str | None = None,
    id_column: str = "sample_id",
    split: str | None = None,
    modality: str = "tabular",
    source: dict | None = None,
) -> Path:
    """Build an ADP folder from one CSV. Returns the package root path.

    split: train/val/test ratios — None (default 80/10/10), 'auto' (size-based),
    or 'TRAIN/VAL/TEST' e.g. '0.7/0.15/0.15'.
    modality: declared input type (ADR-0008) — 'tabular' (default), 'text', or
    'timeseries'. Recorded in the manifest; drives which methods a run uses.
    source: optional provenance dict recorded under dataset.source (e.g. how a
    timeseries CSV was feature-extracted into this tabular package)."""
    if modality not in TABULAR_MODALITIES:
        raise ValueError(
            f"--type must be one of {', '.join(TABULAR_MODALITIES)} for a CSV "
            f"(images use 'atom pack-images'), got {modality!r}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    csv_path = Path(csv_path)
    name = name or csv_path.stem
    root = Path(out_dir) / name
    for sub in ("raw", "processed", "splits"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    raw_dest = root / "raw" / csv_path.name
    shutil.copyfile(csv_path, raw_dest)

    with open_text_tolerant(csv_path) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        columns = _dedupe(header)
        if target is not None and target not in columns:
            raise ValueError(f"target column {target!r} not in CSV header")
        all_columns = columns + [id_column]
        records = list(reader)   # materialize so 'auto' can size the split
        ratios, split_mode = resolve_split(split, len(records))
        buckets = {s: {c: [] for c in all_columns} for s in ratios}
        counts = {s: 0 for s in ratios}
        label_counts: dict[str, int] = {}
        tgt_idx = columns.index(target) if target is not None else -1
        for idx, record in enumerate(records):
            sid = f"{csv_path.name}#{idx}"
            sp = _hash_split(sid, ratios)
            counts[sp] += 1
            bucket = buckets[sp]
            for i, col in enumerate(columns):
                bucket[col].append(record[i] if i < len(record) else "")
            bucket[id_column].append(sid)
            if target is not None:
                v = record[tgt_idx] if tgt_idx < len(record) else ""
                label_counts[v] = label_counts.get(v, 0) + 1

    # typed schema from a sample of train values (ADR-0003: processed is typed)
    types = {}
    for col in columns:
        sample = buckets["train"][col][:1000] or buckets["val"][col][:1000]
        types[col] = _infer_arrow_type(sample) if col != target else pa.string()
    types[id_column] = pa.string()

    def _array(col: str, values: list[str]):
        t = types[col]
        if t == pa.string():
            return pa.array(values, type=t)
        cast = []
        for v in values:
            v = v.strip()
            try:
                cast.append(int(v) if t == pa.int64() else float(v))
            except ValueError:
                cast.append(None)
        return pa.array(cast, type=t)

    schema = pa.schema([(c, types[c]) for c in all_columns])
    checksums = {}
    for split, data in buckets.items():
        table = pa.table({c: _array(c, data[c]) for c in all_columns}, schema=schema)
        dest = root / "processed" / f"{split}.parquet"
        pq.write_table(table, dest, compression="snappy")
        checksums[f"processed/{split}.parquet"] = _sha256_file(dest)
    checksums[f"raw/{csv_path.name}"] = _sha256_file(raw_dest)

    split_doc = {
        "version": "split_v1",
        "method": "hash",
        "mode": split_mode,
        "seed": SPLIT_SEED,
        "ratios": {k: round(v, 6) for k, v in ratios.items()},
        "id_scheme": "<raw file name>#<0-based row index>",
        "counts": {**counts, "total": sum(counts.values())},
        "ids_included": False,
    }
    (root / "splits" / "split_v1.json").write_text(json.dumps(split_doc, indent=2))

    manifest = {
        "manifest_version": "atom-dataset-v1",
        "dataset": {
            "name": name,
            "modality": modality,
            **({"source": source} if source else {}),
            "dataset_type": "supervised" if target else "unlabeled",
        },
        "counts": {"files": 1, "samples": split_doc["counts"]},
        "schema": {
            "mode": "tabular",
            "id_column": id_column,
            "columns": [{"name": c, "stored_type": str(types[c])} for c in all_columns],
        },
        "roles": {
            "id": id_column,
            "target": [target] if target else [],
            "ignore": [],
            "group": None,
            "time": None,
        },
        "labels": (
            [{"column": target, "classes": dict(sorted(label_counts.items()))}] if target else []
        ),
        "split": {"method": "hash", "mode": split_mode, "seed": SPLIT_SEED,
                  "ratios": {k: round(v, 6) for k, v in ratios.items()},
                  "file": "splits/split_v1.json"},
        "files": [
            {
                "name": csv_path.name,
                "path": f"raw/{csv_path.name}",
                "size_bytes": raw_dest.stat().st_size,
                "sha256": checksums[f"raw/{csv_path.name}"],
                "detected_type": "csv",
                "role": "data",
            }
        ],
        "checksums": checksums,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (root / "README.md").write_text(
        f"# {name} — ATOM Dataset Package\n\nGenerated by `atom pack` from "
        f"`{csv_path.name}`. See manifest.json for schema, roles, and checksums.\n"
    )
    return root


_TS_STATS = ("mean", "std", "min", "max", "last", "slope")


def _summarize(values: list[float]) -> list[float]:
    """Per-sequence summary stats over one ordered numeric channel."""
    import math

    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return [float("nan")] * len(_TS_STATS)
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
    if n > 1:  # linear-trend slope over the ordered index
        mx = (n - 1) / 2
        den = sum((i - mx) ** 2 for i in range(n))
        slope = sum((i - mx) * (v - mean) for i, v in enumerate(vals)) / den if den else 0.0
    else:
        slope = 0.0
    return [mean, std, min(vals), max(vals), vals[-1], slope]


def pack_timeseries_csv(
    csv_path: str | Path,
    out_dir: str | Path,
    name: str | None = None,
    target: str | None = None,
    time_col: str | None = None,
    group_col: str | None = None,
    split: str | None = None,
) -> Path:
    """Time-series CSV -> tabular ADP by per-sequence feature extraction (ADR-0008
    Phase 2, torch-free). Rows are grouped by `group_col` (one sequence per
    group), ordered by `time_col`, and each numeric feature channel is summarized
    (mean/std/min/max/last/slope). The result is a tabular package (one row per
    sequence) the existing classifiers run on — on any machine, no PyTorch. The
    split is per-sequence, so no group leaks across train/val/test."""
    from collections import OrderedDict

    from atom.core.ingest.profiler import parse_numeric

    csv_path = Path(csv_path)
    name = name or csv_path.stem
    if not (time_col and group_col and target):
        raise ValueError("--type timeseries needs --time, --group and --target")

    with open_text_tolerant(csv_path) as fh:
        reader = csv.reader(fh)
        header = _dedupe(next(reader))
        for col in (time_col, group_col, target):
            if col not in header:
                raise ValueError(f"column {col!r} not in CSV header")
        idx = {c: i for i, c in enumerate(header)}
        rows = [rec for rec in reader]

    feat_cols = [c for c in header if c not in (time_col, group_col, target)]
    probe = rows[:500] or rows
    numeric_feats = [
        c for c in feat_cols
        if probe and sum(parse_numeric(r[idx[c]] if idx[c] < len(r) else "") is not None
                         for r in probe) >= 0.8 * len(probe)
    ]
    if not numeric_feats:
        raise ValueError("no numeric feature columns to summarize for the time series")

    groups: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        groups.setdefault(r[idx[group_col]] if idx[group_col] < len(r) else "", []).append(r)

    def _tkey(rec):
        v = parse_numeric(rec[idx[time_col]] if idx[time_col] < len(rec) else "")
        return (0, v) if v is not None else (1, rec[idx[time_col]] if idx[time_col] < len(rec) else "")

    out_header = [f"{c}__{s}" for c in numeric_feats for s in _TS_STATS] + [target]
    out_rows = []
    for grp in groups.values():
        grp_sorted = sorted(grp, key=_tkey)
        row: list = []
        for c in numeric_feats:
            vals = [parse_numeric(r[idx[c]] if idx[c] < len(r) else "") for r in grp_sorted]
            row += _summarize(vals)
        row.append(grp_sorted[-1][idx[target]] if idx[target] < len(grp_sorted[-1]) else "")
        out_rows.append(row)

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        feat_csv = Path(td) / f"{name}_tsfeat.csv"
        with feat_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(out_header)
            w.writerows(out_rows)
        return pack_csv(
            feat_csv, out_dir, name=name, target=target, split=split, modality="tabular",
            source={"modality": "timeseries", "time": time_col, "group": group_col,
                    "aggregation": "summary-stats", "stats": list(_TS_STATS),
                    "n_sequences": len(out_rows)})
