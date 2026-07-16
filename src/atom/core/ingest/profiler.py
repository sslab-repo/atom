"""Ingest & Profiler: DatasetPackage -> Fingerprint (v1, tabular).

The fingerprint is privacy-safe by construction (ADR-0005): summary
statistics only, never raw values. Profiling is sample-bounded — it reads
at most `sample_rows` from the train split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from atom.data.package import DatasetPackage

MISSING_SENTINELS = {"", "NaN", "nan", "NA", "N/A", "n/a", "null", "NULL", "None", "?"}
INF_SENTINELS = {"Infinity", "-Infinity", "inf", "-inf", "Inf"}

FINGERPRINT_VERSION = "fingerprint-v1"


def column_to_pylist(col) -> list:
    """Arrow column -> python list, tolerating invalid UTF-8 in string data
    (seen in the wild: Windows-1252 bytes inside DMS parquet). Bad bytes are
    replaced, never fatal."""
    try:
        return col.to_pylist()
    except UnicodeDecodeError:
        import pyarrow as pa

        def _decode(v):
            return v.decode("utf-8", "replace") if isinstance(v, bytes) else v

        return [_decode(v) for v in col.cast(pa.binary()).to_pylist()]


@dataclass
class ColumnProfile:
    name: str
    dtype: str  # observed: "integer" | "number" | "string"
    missing_rate: float = 0.0
    inf_rate: float = 0.0
    distinct_sampled: int = 0
    categories: list[str] = field(default_factory=list)  # small string vocabularies


@dataclass
class Fingerprint:
    version: str
    package_id: str
    name: str
    modality: str
    dataset_type: str
    n_columns: int
    counts: dict[str, int]
    roles: dict[str, Any]
    target_classes: dict[str, int]  # class -> count in sample (train)
    columns: list[ColumnProfile] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    sampled_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify_value(v: Any) -> str:
    """Observed type of one non-missing cell value."""
    if isinstance(v, bool):
        return "string"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    s = str(v)
    try:
        int(s)
        return "integer"
    except ValueError:
        pass
    try:
        float(s)
        return "number"
    except ValueError:
        return "string"


def fingerprint(pkg: DatasetPackage, sample_rows: int = 50_000) -> Fingerprint:
    m = pkg.manifest
    fp = Fingerprint(
        version=FINGERPRINT_VERSION,
        package_id=m.content_id,
        name=m.name,
        modality=m.modality,
        dataset_type=m.dataset_type,
        n_columns=len(m.columns),
        counts={s: c for s, c in m.counts.items()},
        roles={
            "id": m.roles.id,
            "target": list(m.roles.target),
            "ignore": list(m.roles.ignore),
            "group": m.roles.group,
            "time": m.roles.time,
        },
        target_classes={},
    )

    if not m.roles.target:
        fp.quality_flags.append("no-target-declared")
    if m.roles.target and not m.labels:
        fp.quality_flags.append("target-declared-but-labels-not-enumerated")
    dupish = [c.name for c in m.columns if "." in c.name and c.name.rsplit(".", 1)[-1].isdigit()]
    if dupish:
        fp.quality_flags.append(f"deduplicated-column-names:{','.join(sorted(dupish)[:5])}")

    if m.mode == "files":  # per-sample files (image etc.): listing-based profile
        import json as _json

        member = "processed/train.jsonl"
        if not pkg.source.exists(member):
            fp.quality_flags.append("no-processed-train")
            return fp
        rows = [_json.loads(line)
                for line in pkg.source.read_text(member).splitlines()[:sample_rows]]
        fp.sampled_rows = len(rows)
        counts: dict[str, int] = {}
        for r in rows:
            if r.get("label"):
                counts[str(r["label"])] = counts.get(str(r["label"]), 0) + 1
        fp.target_classes = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
        if counts and min(counts.values()) / max(counts.values()) < 0.01:
            fp.quality_flags.append("severe-class-imbalance")
        return fp

    try:
        table = pkg.read_split("train", max_rows=sample_rows)
    except FileNotFoundError:
        fp.quality_flags.append("no-processed-train")
        return fp

    fp.sampled_rows = table.num_rows
    id_col = m.roles.id
    for col_name in table.column_names:
        values = column_to_pylist(table.column(col_name))
        n = len(values)
        missing = inf = 0
        type_votes: dict[str, int] = {"integer": 0, "number": 0, "string": 0}
        distinct = set()
        for v in values:
            if v is None or (isinstance(v, str) and v.strip() in MISSING_SENTINELS):
                missing += 1
                continue
            if isinstance(v, str) and v.strip() in INF_SENTINELS:
                inf += 1
                continue
            type_votes[_classify_value(v)] += 1
            if len(distinct) <= 10_000:
                distinct.add(str(v))
        numeric_votes = type_votes["number"] + type_votes["integer"]
        if type_votes["string"] and numeric_votes >= 19 * type_votes["string"]:
            # ≥95% of non-missing values parse as numbers: a numeric column
            # polluted by unrecognized missing markers, not a real string
            # column (stroke bmi "N/A", auto-mpg horsepower "?"). Loading
            # coerces the stragglers to NaN for the imputer.
            dtype = "number" if type_votes["number"] else "integer"
            fp.quality_flags.append(f"numeric-coerced:{col_name}")
        elif type_votes["string"]:
            dtype = "string"
        elif type_votes["number"]:
            dtype = "number"
        elif type_votes["integer"]:
            dtype = "integer"
        else:
            dtype = "string"
        fp.columns.append(
            ColumnProfile(
                name=col_name,
                dtype=dtype,
                missing_rate=round(missing / n, 6) if n else 0.0,
                inf_rate=round(inf / n, 6) if n else 0.0,
                distinct_sampled=len(distinct),
                categories=(sorted(distinct) if dtype == "string"
                            and 0 < len(distinct) <= 64 and col_name != id_col else []),
            )
        )
        if inf and col_name != id_col:
            fp.quality_flags.append(f"infinity-values:{col_name}")

    for target in m.roles.target:
        if target in table.column_names:
            counts: dict[str, int] = {}
            unlabeled = 0
            for v in column_to_pylist(table.column(target)):
                if v is None or str(v).strip() in MISSING_SENTINELS:
                    unlabeled += 1  # missing target is absent data, not a class
                    continue
                key = str(v)
                counts[key] = counts.get(key, 0) + 1
            if unlabeled:
                fp.quality_flags.append(f"unlabeled-rows:{target}:{unlabeled}")
            fp.target_classes = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
            if len(counts) > 1:
                top = max(counts.values())
                rare = min(counts.values())
                if rare / top < 0.01:
                    fp.quality_flags.append("severe-class-imbalance")
    return fp
