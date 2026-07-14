"""Registry mechanics: registration, entry-point discovery, lookup (ADR-0002).

The core never imports a concrete module — everything goes through
register / discover / find. Declarations are validated at registration
time (contract + ADR-0005 v1 bars).
"""

from __future__ import annotations

from importlib.metadata import entry_points

from atom.contract import Modality, Module, ModuleKind, TaskFamily

ENTRY_POINT_GROUP = "atom.modules"

_registries: dict[ModuleKind, dict[tuple[str, str], Module]] = {k: {} for k in ModuleKind}
# Lifecycle per ADR-0007: in-tree modules are stable; external drop-ins ENTER
# as experimental and are promoted via the gate, never by self-declaration.
_lifecycle: dict[tuple[ModuleKind, str, str], str] = {}


class DuplicateModuleError(RuntimeError):
    """Two modules declared the same (kind, name, version)."""


class InvalidDeclarationError(RuntimeError):
    """A module's declaration violates the contract (see Declaration.validate)."""


def register(cls: type[Module], lifecycle: str = "stable") -> type[Module]:
    """Class decorator: file a Module under the registry named by declares().kind."""
    module = cls()
    decl = module.declares()
    problems = decl.validate()
    if problems:
        raise InvalidDeclarationError(f"{decl.name}@{decl.version}: " + "; ".join(problems))
    key = (decl.name, decl.version)
    registry = _registries[decl.kind]
    if key in registry:
        raise DuplicateModuleError(f"{decl.kind.value}:{decl.name}@{decl.version}")
    registry[key] = module
    _lifecycle[(decl.kind, *key)] = lifecycle
    return cls


def lifecycle_of(module: Module) -> str:
    decl = module.declares()
    return _lifecycle.get((decl.kind, decl.name, decl.version), "experimental")


def discover() -> int:
    """Load external modules from the `atom.modules` entry-point group.
    Drop-ins always enter as EXPERIMENTAL (ADR-0007). Returns the count."""
    count = 0
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        register(ep.load(), lifecycle="experimental")
        count += 1
    return count


def all_modules() -> list[Module]:
    return [m for reg in _registries.values() for m in reg.values()]


def find(
    kind: ModuleKind,
    task_family: TaskFamily | None = None,
    modality: Modality | None = None,
    include_experimental: bool = False,
) -> list[Module]:
    """Registered modules of `kind` compatible with the given task/modality.
    Experimental modules are excluded unless explicitly requested."""
    modules = [
        m for m in _registries[kind].values()
        if include_experimental or lifecycle_of(m) == "stable"
    ]
    if task_family is not None and modality is not None:
        return [m for m in modules if m.declares().supports(task_family, modality)]
    if task_family is not None:
        return [m for m in modules if task_family in m.declares().task_families]
    if modality is not None:
        return [m for m in modules if modality in m.declares().modalities]
    return modules
