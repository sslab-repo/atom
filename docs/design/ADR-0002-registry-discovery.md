# ADR-0002: Registry Structure and Module Discovery

- Status: **accepted** (2026-07-14)
- Date: 2026-07-10

## Context

Modules must be addable — including from packages outside this repository —
without editing core code or maintaining a central list by hand.

## Decision

### Four registries

One registry per module kind: `preprocessing`, `methods`, `search`,
`metrics`. Each is an independent, versioned namespace; a registry release
never forces a core release.

### Registration — two paths

1. **In-tree decorator**:

   ```python
   from atom.registries import register

   @register
   class RobustScaler(Module):
       ...
   ```

   The decorator files the class under the registry named by
   `declares().kind`.

2. **External entry points** — third-party packages expose modules via the
   `atom.modules` entry-point group in their own `pyproject.toml`:

   ```toml
   [project.entry-points."atom.modules"]
   my_new_optimizer = "my_pkg.opt:MyOptimizer"
   ```

   `discover()` loads all entry points at startup. Drop-in, no ATOM change.

### Lookup

The orchestrator and task inference query registries only through
`Registry.find(kind, task_family, modality)`, which filters on each module's
`declares()`. Nothing in the core ever imports a concrete module.

### Diversity constraint

Registries answer *what exists*; the orchestrator enforces a diversity
constraint over the *candidate archive* so portfolios don't collapse to
near-duplicate configs. This lives in the core, not the registries, because
it is a property of the search, not of any single module.

## Consequences

- Adding an algorithm is a one-file change (in-tree) or a zero-file change
  (external package).
- Duplicate names within a registry are an error at discovery time —
  identity is `(kind, name, version)`.
- Discovery is import-time cheap: `declares()` must not touch data or load
  weights; heavy initialization belongs in `run()`.
