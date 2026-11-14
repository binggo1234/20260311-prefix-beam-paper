from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import inrp.handoff as handoff_module
from inrp.handoff import (
    PACKING_HANDOFF_SCHEMA_VERSION,
    PackingPart,
    PackingRequest,
    solve_packing_handoff,
    solve_packing_handoff_with_budget,
)
from inrp.budget import BudgetExpired, TotalBudget
from inrp.mcts import SearchResult
from inrp.packer import Board, PlacedPart, Rect


def _request() -> PackingRequest:
    return PackingRequest(
        case_id="handoff-smoke",
        parts=(
            PackingPart("A", "source-A", 40.0, 20.0),
            PackingPart("B", "source-B", 30.0, 10.0),
        ),
        board_width_mm=200.0,
        board_height_mm=100.0,
        trim_mm=5.0,
        tool_diameter_mm=6.0,
        seed=17,
        time_limit_s=0.2,
    )


def test_handoff_preserves_nominal_geometry_and_identity(monkeypatch) -> None:
    def fake_solver(parts, cfg, seed):
        assert seed == 17
        assert cfg.BINARY_SPACING is False
        assert cfg.GAP == 6.0
        assert [(part.w, part.h) for part in parts] == [(46.0, 26.0), (36.0, 16.0)]
        board = Board(1, 200.0, 100.0, trim=5.0)
        board.placed = [
            PlacedPart(1, "A", Rect(5.0, 5.0, 46.0, 26.0), 40.0, 20.0, False),
            PlacedPart(2, "B", Rect(51.0, 5.0, 36.0, 16.0), 30.0, 10.0, True),
        ]
        return SearchResult([board], (2, 1), (1.0, 2.0), -1.0, {"fake": True})

    monkeypatch.setattr(handoff_module, "solve_with_receding_horizon_mcts", fake_solver)
    result = solve_packing_handoff(_request())

    assert result.schema_version == PACKING_HANDOFF_SCHEMA_VERSION
    assert result.feasible
    assert result.panel_count == 1
    assert result.sequence == ("B", "A")
    assert result.placements[0].x_mm == 8.0
    assert result.placements[0].nominal_width_mm == 40.0
    assert result.placements[1].x_mm == 54.0
    assert result.placements[1].nominal_width_mm == 10.0
    assert result.placements[1].nominal_height_mm == 30.0
    assert result.provenance["geometry_semantics"] == "nominal_with_tool_clearance"


def test_handoff_rejects_duplicate_ids_and_unsupported_geometry() -> None:
    request = _request()
    duplicate = PackingRequest(
        **{
            **request.__dict__,
            "parts": (
                PackingPart("A", "one", 10.0, 10.0),
                PackingPart("A", "two", 10.0, 10.0),
            ),
        }
    )
    with pytest.raises(ValueError, match="part_id must be unique"):
        solve_packing_handoff(duplicate)

    unsupported = PackingRequest(**{**request.__dict__, "geometry_semantics": "legacy"})
    with pytest.raises(ValueError, match="unsupported geometry semantics"):
        solve_packing_handoff(unsupported)


def test_budgeted_handoff_propagates_absolute_deadline_before_solver(monkeypatch) -> None:
    now = [100.0]
    budget = TotalBudget(10.0, 2.0, clock=lambda: now[0])

    def fake_solver(parts, cfg, seed):
        assert cfg.FORMAL_TOTAL_DEADLINE == 110.0
        assert cfg.FORMAL_CLOSING_RESERVE_S == 2.0
        assert seed == 17
        board = Board(1, 200.0, 100.0, trim=5.0)
        board.placed = [
            PlacedPart(1, "A", Rect(5.0, 5.0, 46.0, 26.0), 40.0, 20.0, False),
            PlacedPart(2, "B", Rect(51.0, 5.0, 36.0, 16.0), 30.0, 10.0, True),
        ]
        now[0] += 1.0
        return SearchResult([board], (1, 2), (1.0, 2.0), -1.0, {"fake": True})

    monkeypatch.setattr(handoff_module, "solve_with_receding_horizon_mcts", fake_solver)
    result = solve_packing_handoff_with_budget(_request(), budget)

    assert result.feasible
    assert result.runtime_s == 1.0
    assert [stamp.phase for stamp in budget.stamps] == [
        "request_validated",
        "solver_input_built",
        "solver_returned",
        "usable_handoff_ready",
    ]


def test_budgeted_handoff_rejects_solver_result_returned_after_deadline(monkeypatch) -> None:
    now = [0.0]
    budget = TotalBudget(1.0, 0.2, clock=lambda: now[0])

    def late_solver(parts, cfg, seed):
        del parts, cfg, seed
        now[0] = 1.0
        return SearchResult([], tuple(), tuple(), 0.0, {})

    monkeypatch.setattr(handoff_module, "solve_with_receding_horizon_mcts", late_solver)
    with pytest.raises(BudgetExpired, match="result_validation"):
        solve_packing_handoff_with_budget(_request(), budget)


def test_budgeted_handoff_supports_greedy_calibration_variant() -> None:
    request = PackingRequest(**{**_request().__dict__, "solver_variant": "baseline_greedy"})
    budget = TotalBudget(2.0, 0.5)

    result = solve_packing_handoff_with_budget(request, budget)

    assert result.feasible
    assert result.panel_count == 1
    assert result.provenance["solver_meta"]["solver_kind"] == "baseline_greedy"


def test_budgeted_prefix_handoff_closes_from_deadline_incumbent() -> None:
    budget = TotalBudget(2.0, 0.5)

    result = solve_packing_handoff_with_budget(_request(), budget)

    assert result.feasible
    solver_meta = result.provenance["solver_meta"]
    assert solver_meta["formal_total_budget"] == 1.0
    assert solver_meta["result_source"] == "formal_deadline_incumbent"
    assert solver_meta["formal_optional_postprocess_skipped"] == 1.0
