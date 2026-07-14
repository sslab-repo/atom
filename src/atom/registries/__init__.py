"""Pluggable module registries (ADR-0002): preprocessing, methods, search, metrics."""

from atom.registries.registry import (
    ENTRY_POINT_GROUP,
    DuplicateModuleError,
    InvalidDeclarationError,
    all_modules,
    discover,
    find,
    lifecycle_of,
    register,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "DuplicateModuleError",
    "InvalidDeclarationError",
    "all_modules",
    "discover",
    "find",
    "lifecycle_of",
    "register",
]
