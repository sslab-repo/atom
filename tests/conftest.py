"""Test isolation: never touch the user's real ~/.atom (ADR-0005 home)."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_atom_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOM_HOME", str(tmp_path / "atom-home"))
