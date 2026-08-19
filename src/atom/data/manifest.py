"""Manifest model for ATOM Dataset Packages (ADR-0003).

Reads both `atom-dataset-v1` and its baseline `dms-ml-package-v1`.
Package identity is the sha256 of the manifest bytes as stored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SUPPORTED_VERSIONS = ("atom-dataset-v1", "dms-ml-package-v1")


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    stored_type: str = "string"
    inferred_type: str | None = None


@dataclass(frozen=True)
class Roles:
    """ADR-0003 roles block. dms-ml-package-v1 has only id_column; the rest
    default to empty and the profiler flags the gap."""

    id: str | None = None
    target: tuple[str, ...] = ()
    ignore: tuple[str, ...] = ()
    group: str | None = None
    time: str | None = None


@dataclass(frozen=True)
class FileEntry:
    name: str
    path: str
    size_bytes: int
    sha256: str
    detected_type: str = ""
    role: str = "data"


@dataclass(frozen=True)
class Manifest:
    manifest_version: str
    name: str
    modality: str
    mode: str  # "tabular" (parquet) | "files" (per-sample files, e.g. images)
    dataset_type: str
    columns: tuple[ColumnSpec, ...]
    roles: Roles
    labels: tuple[dict[str, Any], ...]
    counts: dict[str, int]  # per-split sample counts
    split: dict[str, Any]
    files: tuple[FileEntry, ...]
    checksums: dict[str, str]  # member path -> "sha256:<hex>"
    content_id: str  # sha256 of the manifest bytes
    dms_dataset_id: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_manifest(data: bytes) -> Manifest:
    doc = json.loads(data)
    version = doc.get("manifest_version", "")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported manifest_version: {version!r}")

    ds = doc.get("dataset", {})
    schema = doc.get("schema", {})
    columns = tuple(
        ColumnSpec(c["name"], c.get("stored_type", "string"), c.get("inferred_type"))
        for c in schema.get("columns", [])
    )

    roles_doc = doc.get("roles", {})
    roles = Roles(
        id=roles_doc.get("id") or schema.get("id_column"),
        target=tuple(roles_doc.get("target", [])),
        ignore=tuple(roles_doc.get("ignore", [])),
        group=roles_doc.get("group"),
        time=roles_doc.get("time"),
    )

    files = tuple(
        FileEntry(
            name=f["name"],
            path=f["path"],
            size_bytes=f.get("size_bytes", 0),
            sha256=f.get("sha256", ""),
            detected_type=f.get("detected_type", ""),
            role=f.get("role", "data"),
        )
        for f in doc.get("files", [])
    )

    return Manifest(
        manifest_version=version,
        name=ds.get("name", ""),
        modality=ds.get("modality", ""),
        mode=schema.get("mode", "tabular"),
        dataset_type=ds.get("dataset_type", ""),
        columns=columns,
        roles=roles,
        labels=tuple(doc.get("labels", []) or []),
        counts=dict(doc.get("counts", {}).get("samples", {})),
        split=dict(doc.get("split", {})),
        files=files,
        checksums=dict(doc.get("checksums", {})),
        content_id="sha256:" + hashlib.sha256(data).hexdigest(),
        dms_dataset_id=ds.get("dms_dataset_id"),
        raw=doc,
    )
