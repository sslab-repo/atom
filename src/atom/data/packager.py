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

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 42


def _hash_split(sample_id: str, seed: int = SPLIT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    frac = int.from_bytes(digest[:8], "big") / 2**64
    if frac < SPLIT_RATIOS["train"]:
        return "train"
    if frac < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
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


def pack_csv(
    csv_path: str | Path,
    out_dir: str | Path,
    name: str | None = None,
    target: str | None = None,
    id_column: str = "sample_id",
) -> Path:
    """Build an ADP folder from one CSV. Returns the package root path."""
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
        buckets = {s: {c: [] for c in all_columns} for s in SPLIT_RATIOS}
        counts = {s: 0 for s in SPLIT_RATIOS}
        label_counts: dict[str, int] = {}
        for idx, record in enumerate(reader):
            sid = f"{csv_path.name}#{idx}"
            split = _hash_split(sid)
            counts[split] += 1
            bucket = buckets[split]
            for i, col in enumerate(columns):
                bucket[col].append(record[i] if i < len(record) else "")
            bucket[id_column].append(sid)
            if target is not None:
                v = record[columns.index(target)] if columns.index(target) < len(record) else ""
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
        "seed": SPLIT_SEED,
        "ratios": SPLIT_RATIOS,
        "id_scheme": "<raw file name>#<0-based row index>",
        "counts": {**counts, "total": sum(counts.values())},
        "ids_included": False,
    }
    (root / "splits" / "split_v1.json").write_text(json.dumps(split_doc, indent=2))

    manifest = {
        "manifest_version": "atom-dataset-v1",
        "dataset": {
            "name": name,
            "modality": "tabular",
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
        "split": {"method": "hash", "seed": SPLIT_SEED, "ratios": SPLIT_RATIOS,
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
