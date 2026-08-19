"""DatasetPackage — open, verify, and read an ATOM Dataset Package (ADR-0003).

Works identically on a directory or a .zip. Checksums verify lazily,
per member, on request (never a full-package hash at open time).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import IO

from atom.data.manifest import Manifest, parse_manifest
from atom.data.source import PackageSource

SPLITS = ("train", "val", "test")


class ChecksumMismatch(RuntimeError):
    pass


class DatasetPackage:
    def __init__(self, source: PackageSource, manifest: Manifest):
        self.source = source
        self.manifest = manifest

    @classmethod
    def open(cls, path: str | Path) -> "DatasetPackage":
        source = PackageSource(path)
        if not source.exists("manifest.json"):
            raise FileNotFoundError(f"no manifest.json in package: {path}")
        manifest = parse_manifest(source.read_bytes("manifest.json"))
        return cls(source, manifest)

    def close(self) -> None:
        self.source.close()

    def __enter__(self) -> "DatasetPackage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- members ---------------------------------------------------------

    def processed_member(self, split: str) -> str:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        return f"processed/{split}.parquet"

    def open_member(self, member: str, verify: bool = False) -> IO[bytes]:
        if verify:
            self.verify(member)
        return self.source.open(member)

    def verify(self, member: str) -> None:
        """Hash one member against the manifest checksum (lazy integrity)."""
        expected = self.manifest.checksums.get(member)
        if expected is None:
            return  # not covered by the manifest; nothing to check
        h = hashlib.sha256()
        with self.source.open(member) as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        actual = "sha256:" + h.hexdigest()
        if actual != expected:
            raise ChecksumMismatch(f"{member}: expected {expected}, got {actual}")

    # -- parquet access --------------------------------------------------

    def read_split(self, split: str, columns: list[str] | None = None, max_rows: int | None = None):
        """Read a processed split as a pyarrow Table (optionally bounded)."""
        import pyarrow.parquet as pq

        member = self.processed_member(split)
        if not self.source.exists(member):
            raise FileNotFoundError(f"package has no {member}")
        with self.source.open(member) as fh:
            pf = pq.ParquetFile(fh)
            if max_rows is None:
                return pf.read(columns=columns)
            batches = []
            remaining = max_rows
            for batch in pf.iter_batches(batch_size=min(remaining, 65536), columns=columns):
                batches.append(batch.slice(0, remaining))
                remaining -= min(batch.num_rows, remaining)
                if remaining <= 0:
                    break
            import pyarrow as pa

            return pa.Table.from_batches(batches) if batches else pf.schema_arrow.empty_table()

    def split_row_count(self, split: str) -> int | None:
        """Row count from parquet metadata (cheap), else manifest counts."""
        import pyarrow.parquet as pq

        member = self.processed_member(split)
        if self.source.exists(member):
            with self.source.open(member) as fh:
                return pq.ParquetFile(fh).metadata.num_rows
        return self.manifest.counts.get(split)
