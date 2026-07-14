"""Ingest & Profiler: DatasetPackage -> Fingerprint (v1, tabular).

The fingerprint is privacy-safe by construction (ADR-0005): summary
statistics only, never raw values. Profiling is sample-bounded — it reads
at most `sample_rows` from the train split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from atom.data.package import DatasetPackage

MISSING_SENTINELS = {"", "NaN", "nan", "NA", "null", "None"}
INF_SENTINELS = {"Infinity", "-Infinity", "inf", "-inf", "Inf"}

FINGERPRINT_VERSION = "fingerprint-v1"


@dataclass
class ColumnProfile:
    name: str
    dtype: str  # observed: "integer" | "number" | "string"
    missing_rate: float = 0.0
    inf_rate: float = 0.0
    distinct_sampled: int = 0


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

    try:
        table = pkg.read_split("train", max_rows=sample_rows)
    except FileNotFoundError:
        fp.quality_flags.append("no-processed-train")
        return fp

    fp.sampled_rows = table.num_rows
    id_col = m.roles.id
    for col_name in table.column_names:
        values = table.column(col_name).to_pylist()
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
        present = max(n - missing, 1)
        if type_votes["string"]:
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
            )
        )
        if inf and col_name != id_col:
            fp.quality_flags.append(f"infinity-values:{col_name}")

    for target in m.roles.target:
        if target in table.column_names:
            counts: dict[str, int] = {}
            for v in table.column(target).to_pylist():
                key = "" if v is None else str(v)
                counts[key] = counts.get(key, 0) + 1
            fp.target_classes = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
            if len(counts) > 1:
                top = max(counts.values())
                rare = min(counts.values())
                if rare / top < 0.01:
                    fp.quality_flags.append("severe-class-imbalance")
    return fp
