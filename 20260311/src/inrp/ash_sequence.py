"""ASH (Oliveira et al., 2021) sequence policy, independent of geometry.

Algorithms 1–2, DOI 10.1007/978-3-030-85476-8_8. A caller using another
initializer/decoder is transplanting this policy, not reproducing full ASH.
"""

from __future__ import annotations

import random
from typing import Callable, Optional, Sequence, Tuple


class ASHSequencePolicy:
    """Adapt perturbation probability using strict board-count improvements."""

    def __init__(self, sequence: Sequence[int], panels: int, seed: int) -> None:
        base = tuple(sequence)
        if not base or len(set(base)) != len(base):
            raise ValueError("initial sequence must contain distinct identities")
        self._validate_panels(panels)
        self.base_sequence = base
        self.best_panels = panels
        self.alpha = 0.1
        self.rng = random.Random(seed)
        self.random_draws = 0
        self.pending_sequence: Optional[Tuple[int, ...]] = None

    @staticmethod
    def _validate_panels(panels: int) -> None:
        if type(panels) is not int or panels < 1:
            raise ValueError("panel count must be a positive integer")

    def propose(self, check: Optional[Callable[[], None]] = None) -> Tuple[int, ...]:
        if self.pending_sequence is not None:
            raise RuntimeError("observe the previous proposal before generating another")
        remaining = list(self.base_sequence)
        output = []
        while remaining:
            index = 0
            while True:
                if check is not None:
                    check()
                self.random_draws += 1
                if self.rng.random() >= self.alpha:
                    output.append(remaining.pop(index))
                    break
                index = (index + 1) % len(remaining)
        self.pending_sequence = tuple(output)
        return self.pending_sequence

    def observe(self, panels: int) -> None:
        if self.pending_sequence is None:
            raise RuntimeError("no generated proposal to observe")
        self._validate_panels(panels)
        if panels < self.best_panels:
            self.base_sequence = self.pending_sequence
            self.best_panels = panels
            self.alpha = 0.1
        else:
            self.alpha = min(0.65, self.alpha + 0.002)
        self.pending_sequence = None
