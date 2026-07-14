"""Pluggable module registries (ADR-0002): preprocessing, methods, search, metrics."""

from atom.registries.registry import (
    ENTRY_POINT_GROUP,
    DuplicateModuleError,
    InvalidDeclarationError,
    discover,
    find,
    register,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "DuplicateModuleError",
    "InvalidDeclarationError",
    "discover",
    "find",
    "register",
]
