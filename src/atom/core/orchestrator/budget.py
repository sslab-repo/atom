"""Run budget (ADR-0006): wall-clock primary, optional trial count.

A tail fraction is reserved so a run always finishes with ensembling +
finalization, never budget-exhausted without a usable artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

TAIL_FRACTION = 0.15


@dataclass
class Budget:
    wall_clock_s: float
    max_trials: int | None = None
    min_trials: int | None = None
    started_at: float = field(default_factory=time.monotonic)
    trials_done: int = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def search_deadline_s(self) -> float:
        return self.wall_clock_s * (1 - TAIL_FRACTION)

    def search_exhausted(self) -> bool:
        """First bound reached stops the search (ADR-0006) — unless the
        user's min-trials request is still unmet and time remains."""
        if self.min_trials and self.trials_done < self.min_trials and self.elapsed < self.wall_clock_s:
            return False
        if self.elapsed >= self.search_deadline_s:
            return True
        if self.max_trials is not None and self.trials_done >= self.max_trials:
            return True
        return False

    def estimate(self) -> dict[str, float]:
        """Elapsed / total for progress + estimated end (ADR-0006)."""
        return {
            "elapsed_s": round(self.elapsed, 1),
            "budget_s": self.wall_clock_s,
            "estimated_end_in_s": round(max(self.wall_clock_s - self.elapsed, 0.0), 1),
            "trials": self.trials_done,
        }
