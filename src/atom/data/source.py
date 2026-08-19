"""Uniform access to a package root that is a directory OR a .zip (ADR-0003).

Semantics are identical in both forms; all member paths are relative to the
package root (the single top-level directory inside a zip, when present).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import IO


class PackageSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.is_dir():
            self._zip: zipfile.ZipFile | None = None
            self._prefix = ""
            self.name = self.path.name
        elif zipfile.is_zipfile(self.path):
            self._zip = zipfile.ZipFile(self.path)
            names = [n for n in self._zip.namelist() if not n.endswith("/")]
            tops = {n.split("/", 1)[0] for n in names}
            if len(tops) == 1 and all("/" in n for n in names):
                self._prefix = next(iter(tops)) + "/"
                self.name = next(iter(tops))
            else:
                self._prefix = ""
                self.name = self.path.stem
        else:
            raise FileNotFoundError(f"not a package directory or zip: {path}")

    @property
    def is_zip(self) -> bool:
        return self._zip is not None

    def exists(self, member: str) -> bool:
        if self._zip is None:
            return (self.path / member).is_file()
        try:
            self._zip.getinfo(self._prefix + member)
            return True
        except KeyError:
            return False

    def open(self, member: str) -> IO[bytes]:
        """Binary stream for a member. Zip streams are seekable (ZipExtFile),
        which pyarrow needs; stored (uncompressed) members seek cheaply."""
        if self._zip is None:
            return (self.path / member).open("rb")
        return self._zip.open(self._prefix + member)

    def read_bytes(self, member: str) -> bytes:
        with self.open(member) as fh:
            return fh.read()

    def read_text(self, member: str) -> str:
        return self.read_bytes(member).decode("utf-8")

    def size(self, member: str) -> int:
        if self._zip is None:
            return (self.path / member).stat().st_size
        return self._zip.getinfo(self._prefix + member).file_size

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
