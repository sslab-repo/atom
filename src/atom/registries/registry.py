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


class DuplicateModuleError(RuntimeError):
    """Two modules declared the same (kind, name, version)."""


class InvalidDeclarationError(RuntimeError):
    """A module's declaration violates the contract (see Declaration.validate)."""


def register(cls: type[Module]) -> type[Module]:
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
    return cls


def discover() -> int:
    """Load external modules from the `atom.modules` entry-point group.

    Returns the number of newly registered modules.
    """
    count = 0
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        register(ep.load())
        count += 1
    return count


def find(
    kind: ModuleKind,
    task_family: TaskFamily | None = None,
    modality: Modality | None = None,
) -> list[Module]:
    """Registered modules of `kind` compatible with the given task/modality."""
    modules = list(_registries[kind].values())
    if task_family is not None and modality is not None:
        return [m for m in modules if m.declares().supports(task_family, modality)]
    if task_family is not None:
        return [m for m in modules if task_family in m.declares().task_families]
    if modality is not None:
        return [m for m in modules if modality in m.declares().modalities]
    return modules
