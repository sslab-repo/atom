"""Ingest & Profiler: DatasetPackage -> Fingerprint (v1, tabular).

The fingerprint is privacy-safe by construction (ADR-0005): summary
statistics only, never raw values. Profiling is sample-bounded — it reads
at most `sample_rows` from the train split.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from atom.data.package import DatasetPackage

# matched case-insensitively via is_missing(); covers the common junk seen
# across CSV exporters (SAS ".", accounting "-", spreadsheet "#N/A", …)
MISSING_SENTINELS = {"", "na", "n/a", "n.a.", "n.a", "#n/a", "#na", "nan",
                     "null", "none", "nil", "missing", "unknown", "?", "-",
                     "--", "---", ".", ".."}
INF_SENTINELS = {"infinity", "-infinity", "inf", "-inf", "+inf"}

FINGERPRINT_VERSION = "fingerprint-v1"


def is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in MISSING_SENTINELS)


def _is_inf_token(v: Any) -> bool:
    return isinstance(v, str) and v.strip().lower() in INF_SENTINELS


_ACCOUNTING = re.compile(r"^\((.*)\)$")           # "(50)" -> negative 50
_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")  # "1,234,567" / "1,234.56"
_SINGLE_COMMA = re.compile(r"^-?\d+,\d+$")         # "27,3" — decimal vs thousands: caller decides
_CURRENCY_CHARS = "$€£¥₩¢"


def parse_numeric(value: Any, decimal_comma: bool = False) -> float | None:
    """Best-effort numeric parse under full exception control: returns a
    float, or None when the value is missing/non-numeric. Generalizes the
    real-world dirtiness of CSV numeric columns — currency symbols, percent
    signs, thousands separators, accounting-parenthesis negatives, stray
    whitespace, and (when the column is flagged) locale decimal commas.
    Unit-suffixed values ('5 kg') are intentionally NOT parsed — stripping
    arbitrary units guesses at semantics — so they fall through to None and
    the ≥95%-numeric column gate drops the column as categorical."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if value is None:
        return None
    s = str(value).strip()
    if is_missing(s):
        return None
    negative = False
    acct = _ACCOUNTING.match(s)
    if acct:
        s, negative = acct.group(1).strip(), True
    for ch in _CURRENCY_CHARS:
        s = s.replace(ch, "")
    s = s.strip().lstrip("+")
    percent = s.endswith("%")
    if percent:
        s = s[:-1].strip()
    if decimal_comma:                 # locale "1.234,56" -> "1234.56"
        s = s.replace(".", "").replace(",", ".")
    elif _THOUSANDS.match(s):         # grouped separators "1,234,567"
        s = s.replace(",", "")
    try:
        f = float(s)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(f):
        return None
    if percent:
        f /= 100.0
    return -f if negative else f


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
    decimal_comma: bool = False  # numeric column written with ',' as the radix point
    datetime_format: str | None = None  # strptime format if the column parses as dates


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


# Ordered so the radix-comma / day-first conventions are tried before their
# ambiguous US counterparts; whichever parses the most sample values wins.
_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
    "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M",
)
_EPOCH = _dt.date(1970, 1, 1)
DATETIME_PARTS = ("year", "month", "day", "dayofweek", "epoch_days")


def detect_datetime(values: list) -> str | None:
    """strptime format that parses >=90% of a column's non-missing sample, or
    None. Cheap pre-filter (needs a digit + '-'/'/'/'':') avoids scanning
    every string column against every format."""
    sample = [str(v).strip() for v in values if not is_missing(v)][:300]
    if len(sample) < 10:
        return None
    if not any(any(c.isdigit() for c in s) and ("-" in s or "/" in s or ":" in s)
               for s in sample[:25]):
        return None
    best_fmt, best_ok = None, 0
    for fmt in _DATE_FORMATS:
        ok = 0
        for s in sample:
            try:
                _dt.datetime.strptime(s, fmt)
                ok += 1
            except (ValueError, TypeError):
                pass
        if ok > best_ok:
            best_ok, best_fmt = ok, fmt
        if best_ok == len(sample):
            break
    return best_fmt if best_ok >= 0.9 * len(sample) else None


def datetime_parts(fmt: str) -> tuple[str, ...]:
    return DATETIME_PARTS + (("hour",) if "%H" in fmt else ())


def datetime_features(values: list, fmt: str) -> dict:
    """Expand a datetime column into numeric parts (NaN where unparseable).
    Ordinal epoch_days carries trend; year/month/day/dow carry seasonality."""
    parts = datetime_parts(fmt)
    cols = {p: np.full(len(values), np.nan) for p in parts}
    for i, v in enumerate(values):
        if is_missing(v):
            continue
        try:
            d = _dt.datetime.strptime(str(v).strip(), fmt)
        except (ValueError, TypeError):
            continue
        cols["year"][i] = d.year
        cols["month"][i] = d.month
        cols["day"][i] = d.day
        cols["dayofweek"][i] = d.weekday()
        cols["epoch_days"][i] = (d.date() - _EPOCH).days
        if "hour" in cols:
            cols["hour"][i] = d.hour
    return cols


def _classify_value(v: Any) -> str:
    """Observed type of one non-missing cell value: 'integer', 'number',
    'comma' (single-comma value, ambiguous decimal-vs-thousands — the column
    decides), or 'string'. All numeric dirtiness routes through
    parse_numeric so the profiler and the loader agree on what is numeric."""
    if isinstance(v, bool):
        return "string"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    s = str(v).strip()
    try:
        int(s)
        return "integer"
    except ValueError:
        pass
    if _SINGLE_COMMA.match(s):  # "27,3" (decimal) or "1,234" (thousands): defer
        return "comma"
    return "number" if parse_numeric(v) is not None else "string"


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
        type_votes: dict[str, int] = {"integer": 0, "number": 0, "string": 0,
                                      "comma": 0}
        comma_frac_lengths: set[int] = set()
        distinct = set()
        for v in values:
            if is_missing(v):
                missing += 1
                continue
            if _is_inf_token(v):
                inf += 1
                continue
            vote = _classify_value(v)
            type_votes[vote] += 1
            if vote == "comma":
                comma_frac_lengths.add(len(str(v).strip().split(",")[1]))
            if len(distinct) <= 10_000:
                distinct.add(str(v))
        # single-comma values are numeric either way; the fractions decide HOW
        # to read them — a non-3-digit fraction means the comma is the radix
        # point ("27,3"), all-3-digit means a thousands separator ("1,234").
        comma = type_votes["comma"]
        decimal_comma = comma > 0 and bool(comma_frac_lengths - {3})
        numeric_votes = type_votes["number"] + type_votes["integer"] + comma
        string_votes = type_votes["string"]
        if numeric_votes and numeric_votes >= 19 * string_votes:
            # ≥95% of non-missing values parse once currency/percent/thousands/
            # radix-comma dirtiness and missing markers are handled: a numeric
            # column, not a category (stroke bmi "N/A", auto-mpg hp "?", beer
            # "27,3", a "$1,234" price). Load coerces the stragglers to NaN.
            dtype = "integer" if not (type_votes["number"] or comma) else "number"
            if decimal_comma:
                fp.quality_flags.append(f"decimal-comma:{col_name}")
            elif string_votes or comma:
                fp.quality_flags.append(f"numeric-coerced:{col_name}")
        else:
            dtype = "string"
        # a string column that parses as dates becomes datetime (expanded to
        # numeric parts at load) instead of being one-hot'd or dropped
        datetime_format = None
        if dtype == "string" and col_name != id_col:
            datetime_format = detect_datetime(values)
            if datetime_format:
                dtype = "datetime"
                fp.quality_flags.append(f"datetime:{col_name}")
        fp.columns.append(
            ColumnProfile(
                name=col_name,
                dtype=dtype,
                missing_rate=round(missing / n, 6) if n else 0.0,
                inf_rate=round(inf / n, 6) if n else 0.0,
                distinct_sampled=len(distinct),
                decimal_comma=decimal_comma,
                datetime_format=datetime_format,
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
                if is_missing(v):
                    unlabeled += 1  # missing target is absent data, not a class
                    continue
                key = str(v)
                counts[key] = counts.get(key, 0) + 1
            if unlabeled:
                fp.quality_flags.append(f"unlabeled-rows:{target}:{unlabeled}")
            fp.target_classes = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
            if len(counts) > 1:
                total = sum(counts.values())
                minority = min(counts.values()) / total  # smallest-class fraction
                if min(counts.values()) / max(counts.values()) < 0.01:
                    fp.quality_flags.append("severe-class-imbalance")
                elif minority < 0.20:  # notable but not rare-class (pokemon 8%)
                    fp.quality_flags.append(f"class-imbalance:{minority:.3f}")
    return fp
