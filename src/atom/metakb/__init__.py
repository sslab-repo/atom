"""Meta-Knowledge Base — the flywheel (fingerprint -> winning config, score, cost)."""

from atom.metakb.store import MetaKB, default_root, summarize_for_kb

__all__ = ["MetaKB", "default_root", "summarize_for_kb"]
