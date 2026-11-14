from __future__ import annotations

import json
from pathlib import Path

import pytest

from inrp.budget import BudgetExpired, TotalBudget, atomic_write_json_with_budget
from inrp.mcts import RecedingHorizonMCTS


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


def test_total_budget_uses_one_absolute_deadline_and_closing_reserve() -> None:
    clock = FakeClock(100.0)
    budget = TotalBudget(10.0, 2.0, clock=clock)

    assert budget.deadline == 110.0
    assert budget.search_deadline == 108.0
    assert budget.allowance_s(20.0, phase="search") == 8.0
    clock.advance(7.5)
    assert budget.allowance_s(20.0, phase="search") == 0.5
    assert budget.allowance_s(20.0, phase="closing") == 2.5
    clock.advance(0.5)
    assert budget.expired(phase="search")
    assert not budget.expired(phase="total")
    assert budget.allowance_s(20.0, phase="search") == 0.0


def test_budget_does_not_force_a_minimum_allowance_after_expiry() -> None:
    clock = FakeClock()
    budget = TotalBudget(1.0, 0.1, clock=clock)
    clock.advance(1.0)

    assert budget.allowance_s(0.2, phase="search") == 0.0
    assert budget.allowance_s(0.2, phase="closing") == 0.0
    with pytest.raises(BudgetExpired, match="validation"):
        budget.require_open("validation")


def test_stamps_and_summary_contain_only_primitive_timing_data() -> None:
    clock = FakeClock(5.0)
    budget = TotalBudget(3.0, 0.5, clock=clock)
    clock.advance(0.25)
    budget.mark("input_loaded")

    summary = budget.summary()
    assert summary["stamps"] == [
        {"phase": "input_loaded", "elapsed_s": 0.25, "remaining_s": 2.75}
    ]
    json.dumps(summary)


def test_atomic_write_replaces_target_on_success(tmp_path: Path) -> None:
    clock = FakeClock()
    budget = TotalBudget(2.0, 0.5, clock=clock)
    target = tmp_path / "minimal_result.json"

    atomic_write_json_with_budget(target, {"status": "usable", "panel_count": 2}, budget)

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "usable", "panel_count": 2}
    assert budget.stamps[-1].phase == "minimal_result_write_complete"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_failure_preserves_existing_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = FakeClock()
    budget = TotalBudget(2.0, 0.5, clock=clock)
    target = tmp_path / "minimal_result.json"
    target.write_text('{"status":"old"}', encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr("inrp.budget.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        atomic_write_json_with_budget(target, {"status": "new"}, budget)

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "old"}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_refuses_to_start_after_deadline(tmp_path: Path) -> None:
    clock = FakeClock()
    budget = TotalBudget(1.0, 0.1, clock=clock)
    clock.advance(1.0)
    target = tmp_path / "minimal_result.json"

    with pytest.raises(BudgetExpired, match="minimal_result_write"):
        atomic_write_json_with_budget(target, {"status": "late"}, budget)

    assert not target.exists()


def test_mcts_uses_propagated_absolute_deadline_and_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = RecedingHorizonMCTS.__new__(RecedingHorizonMCTS)
    solver.formal_total_deadline = 50.0
    solver.formal_closing_reserve_s = 4.0
    solver.solver_time_limit_s = 999.0
    solver.meta = {}
    monkeypatch.setattr("inrp.mcts.time.perf_counter", lambda: 10.0)

    solver._configure_solve_deadlines()

    assert solver.solve_deadline == 50.0
    assert solver.search_deadline == 46.0
    assert solver.meta["post_mcts_reserve_s"] == 4.0
    assert solver.meta["formal_total_budget"] == 1.0


def test_constructive_warmstart_stops_when_search_reserve_begins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = RecedingHorizonMCTS.__new__(RecedingHorizonMCTS)
    solver.search_deadline = 5.0
    solver.meta = {"solver_timeout_hit": 0.0}
    monkeypatch.setattr("inrp.mcts.time.perf_counter", lambda: 5.0)

    solver._prime_constructive_warmstarts()

    assert solver.meta["solver_timeout_hit"] == 1.0
