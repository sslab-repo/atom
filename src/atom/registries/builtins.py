"""Load ATOM's built-in stable module set (idempotent — import side effects)."""


def load_builtins() -> None:
    from atom.registries.metrics import basic  # noqa: F401
    from atom.registries.methods import sklearn_supervised  # noqa: F401
    from atom.registries.preprocessing import simple  # noqa: F401
