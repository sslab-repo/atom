"""Compute-device resolution for the optional PyTorch (deep) tier (ADR-0008).

Torch-free by construction: torch is imported lazily *inside* the functions, so
importing this module never requires torch. On a machine without torch every
call reports CPU / unavailable and the deep tier simply does not register —
the CPU (sklearn) tier still produces a result. GPU accelerates training/search
only; it is never required to produce or to serve a model.

Resolution order (ADR-0008): cuda -> mps (Apple Silicon) -> cpu.
Override with ATOM_DEVICE={auto|cpu|cuda|mps} for reproducibility/testing.
"""

from __future__ import annotations

import os


def torch_available() -> bool:
    """True iff PyTorch can be imported here."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _mps_ok(torch) -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def resolve_device(override: str | None = None) -> str:
    """Return 'cuda' | 'mps' | 'cpu'.

    Honors `override` then $ATOM_DEVICE (both {auto|cpu|cuda|mps}); falls back to
    'cpu' whenever torch or the requested accelerator is absent — so a run never
    fails for lack of a GPU."""
    choice = (override or os.environ.get("ATOM_DEVICE") or "auto").strip().lower()
    if choice == "cpu" or not torch_available():
        return "cpu"
    import torch

    if choice in ("cuda", "gpu"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "mps":
        return "mps" if _mps_ok(torch) else "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if _mps_ok(torch):
        return "mps"
    return "cpu"


def describe() -> str:
    """One-line summary for run logs, e.g. 'device: cuda (torch 2.3.0)'."""
    if not torch_available():
        return "device: cpu (no PyTorch — deep tier disabled)"
    import torch

    dev = resolve_device()
    return f"device: {dev} (torch {torch.__version__})"
