"""Lifecycle-neutral wall-clock budget primitives for formal experiments."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


Clock = Callable[[], float]


class BudgetExpired(RuntimeError):
    """Raised when a formal pipeline cannot finish a required phase in budget."""

    def __init__(self, phase: str) -> None:
        self.phase = str(phase)
        super().__init__(f"total budget expired during phase: {self.phase}")


@dataclass(frozen=True)
class BudgetStamp:
    phase: str
    elapsed_s: float
    remaining_s: float


class TotalBudget:
    """One absolute deadline shared by every budgeted pipeline phase.

    Only primitive timing data are retained. In particular, this object never
    stores solver, candidate, layout, or callback objects that could alter their
    lifetimes during a formal timing run.
    """

    def __init__(
        self,
        total_s: float,
        closing_reserve_s: float,
        *,
        clock: Clock = time.perf_counter,
        started_at: Optional[float] = None,
    ) -> None:
        total = float(total_s)
        reserve = float(closing_reserve_s)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("total_s must be positive and finite")
        if not math.isfinite(reserve) or reserve < 0.0 or reserve >= total:
            raise ValueError("closing_reserve_s must be finite and in [0, total_s)")
        self._clock = clock
        self.total_s = total
        self.closing_reserve_s = reserve
        self.started_at = float(clock() if started_at is None else started_at)
        self.deadline = self.started_at + total
        self.search_deadline = self.deadline - reserve
        self._stamps: List[BudgetStamp] = []

    def now(self) -> float:
        return float(self._clock())

    def elapsed_s(self) -> float:
        return max(0.0, self.now() - self.started_at)

    def remaining_s(self) -> float:
        return max(0.0, self.deadline - self.now())

    def search_remaining_s(self) -> float:
        return max(0.0, self.search_deadline - self.now())

    def expired(self, *, phase: str = "total") -> bool:
        boundary = self.search_deadline if phase == "search" else self.deadline
        return self.now() >= boundary

    def require_open(self, phase: str) -> None:
        boundary_phase = "search" if str(phase) == "search" else "total"
        if self.expired(phase=boundary_phase):
            raise BudgetExpired(str(phase))

    def allowance_s(self, cap_s: Optional[float] = None, *, phase: str = "search") -> float:
        remaining = self.search_remaining_s() if phase == "search" else self.remaining_s()
        if cap_s is None:
            return remaining
        cap = float(cap_s)
        if not math.isfinite(cap) or cap < 0.0:
            raise ValueError("cap_s must be nonnegative and finite")
        return min(remaining, cap)

    def mark(self, phase: str) -> BudgetStamp:
        now = self.now()
        stamp = BudgetStamp(
            phase=str(phase),
            elapsed_s=max(0.0, now - self.started_at),
            remaining_s=max(0.0, self.deadline - now),
        )
        self._stamps.append(stamp)
        return stamp

    @property
    def stamps(self) -> Tuple[BudgetStamp, ...]:
        return tuple(self._stamps)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_s": self.total_s,
            "closing_reserve_s": self.closing_reserve_s,
            "started_at": self.started_at,
            "deadline": self.deadline,
            "search_deadline": self.search_deadline,
            "elapsed_s": self.elapsed_s(),
            "remaining_s": self.remaining_s(),
            "stamps": [asdict(stamp) for stamp in self._stamps],
        }


def atomic_write_json_with_budget(
    path: Path,
    payload: Mapping[str, Any],
    budget: TotalBudget,
    *,
    phase: str = "minimal_result_write",
) -> None:
    """Atomically replace ``path`` only if serialization finishes in budget."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    budget.require_open(phase)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(dict(payload), fh, ensure_ascii=False, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        budget.require_open(phase)
        os.replace(tmp_path, target)
        budget.mark(f"{phase}_complete")
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
