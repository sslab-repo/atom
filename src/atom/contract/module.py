"""The Module contract — the single interface that makes ATOM pluggable.

Normative spec: ADR-0001. FROZEN: additions must be optional
(default-implemented here) and never break existing modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from atom.contract.types import Declaration, ResourceHints, RunContext, RunResult, SearchSpace


class Module(ABC):
    """Base class for every ATOM module, regardless of registry."""

    @abstractmethod
    def declares(self) -> Declaration:
        """Static capabilities. Cheap: no data access, no weight loading."""

    @abstractmethod
    def space(self) -> SearchSpace:
        """Hyperparameter / microcontrol ranges (conditional parameters OK)."""

    @abstractmethod
    def run(self, ctx: RunContext) -> RunResult:
        """Execute per ctx.operation (fit/transform/generate/score).

        Raise UnsupportedOperation for operations this module doesn't do.
        Heavy initialization belongs here, not in __init__ or declares().
        """

    def hints(self) -> ResourceHints:
        """Resources + fidelity levels. Default: 1 CPU, full fidelity."""
        return ResourceHints()
