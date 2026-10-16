from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import inrp.handoff as handoff_module
from inrp.budget import BudgetExpired
from inrp.dataio import Part
from inrp.handoff import PackingPart, PackingRequest, solve_packing_handoff
from inrp.mcts import SearchResult, _orientations
from inrp.packer import Board, PlacedPart, Rect, _part_dims, pack_multi_board, pack_one_board
from inrp.preprocess import preprocess_parts
from inrp.cfg import CFG


def _part(*, allowed_rotations_deg=(0, 90), w=126.0, h=66.0) -> Part:
    return Part(1, "p", w, h, w, h, allowed_rotations_deg=allowed_rotations_deg)


def _request(*, part: PackingPart, allow_rotation=True) -> PackingRequest:
    return PackingRequest(
        case_id="allowed-rotations",
        parts=(part,),
        board_width_mm=200.0,
        board_height_mm=100.0,
        trim_mm=0.0,
        tool_diameter_mm=0.0,
        allow_rotation=allow_rotation,
        time_limit_s=0.1,
    )


def test_default_rotation_contract_preserves_legacy_behavior() -> None:
    part = Part(1, "legacy", 126.0, 66.0, 126.0, 66.0)
    source = PackingPart("legacy", "source", 126.0, 66.0)

    assert part.allowed_rotations_deg == (0, 90)
    assert source.allowed_rotations_deg == (0, 90)
    assert _part_dims(part, True) == [(126.0, 66.0, False), (66.0, 126.0, True)]


def test_part_rotation_permissions_control_packer_and_mcts_orientations() -> None:
    locked = _part(allowed_rotations_deg=(0,))
    rotated_only = _part(allowed_rotations_deg=(90,))
    both = _part(allowed_rotations_deg=(0, 90))

    assert _part_dims(locked, True) == [(126.0, 66.0, False)]
    assert _part_dims(rotated_only, True) == [(66.0, 126.0, True)]
    assert _part_dims(both, True) == [(126.0, 66.0, False), (66.0, 126.0, True)]
    assert _orientations(rotated_only, True) == ((66.0, 126.0),)


def test_locked_part_does_not_fit_only_by_rotation() -> None:
    locked = _part(allowed_rotations_deg=(0,), w=126.0, h=66.0)

    with pytest.raises(RuntimeError, match="Part too large"):
        pack_one_board([locked], 100.0, 150.0, True, bid=1)


def test_constructive_packer_rejects_work_after_deadline() -> None:
    with pytest.raises(BudgetExpired, match="constructive_warmstart"):
        pack_multi_board(
            [_part(w=40.0, h=20.0)],
            100.0,
            100.0,
            True,
            deadline=0.0,
            deadline_phase="constructive_warmstart",
        )


def test_preprocess_does_not_merge_rotated_duplicates_with_different_permissions() -> None:
    cfg = CFG()
    cfg.ALLOW_ROT = True
    parts = [
        Part(1, "a", 10.0, 5.0, 10.0, 5.0, allowed_rotations_deg=(0,)),
        Part(2, "b", 5.0, 10.0, 5.0, 10.0, allowed_rotations_deg=(90,)),
    ]

    processed, summary = preprocess_parts(parts, cfg)

    assert summary.n_distinct_shapes == 2
    assert [part.dup_count for part in processed] == [1, 1]


@pytest.mark.parametrize("angles", [(), (0, 0), (45,), (0, 45), (0.0,), (True,)])
def test_handoff_rejects_invalid_allowed_rotation_fields(angles) -> None:
    request = _request(part=PackingPart("p", "source", 40.0, 20.0, angles))

    with pytest.raises(ValueError, match="allowed_rotations_deg"):
        solve_packing_handoff(request)


def test_handoff_rejects_part_with_no_global_rotation_intersection() -> None:
    request = _request(
        part=PackingPart("p", "source", 40.0, 20.0, (90,)),
        allow_rotation=False,
    )

    with pytest.raises(ValueError, match="allowed_rotations_deg"):
        solve_packing_handoff(request)


def test_handoff_keeps_ninety_only_square_rotation_flag(monkeypatch) -> None:
    request = _request(part=PackingPart("square", "source", 40.0, 40.0, (90,)))

    def fake_solver(parts, cfg, seed):
        board = Board(1, 200.0, 100.0)
        board.placed = [PlacedPart(1, "square", Rect(0.0, 0.0, 40.0, 40.0), 40.0, 40.0, True)]
        return SearchResult([board], (1,), (1.0,), 1.0, {})

    monkeypatch.setattr(handoff_module, "solve_with_receding_horizon_mcts", fake_solver)
    result = solve_packing_handoff(request)

    assert result.placements[0].rotated is True


def test_handoff_rejects_returned_rotation_outside_requested_permission(monkeypatch) -> None:
    request = _request(part=PackingPart("locked", "source", 40.0, 20.0, (0,)))

    def fake_solver(parts, cfg, seed):
        board = Board(1, 200.0, 100.0)
        board.placed = [PlacedPart(1, "locked", Rect(0.0, 0.0, 20.0, 40.0), 40.0, 20.0, True)]
        return SearchResult([board], (1,), (1.0,), 1.0, {})

    monkeypatch.setattr(handoff_module, "solve_with_receding_horizon_mcts", fake_solver)

    with pytest.raises(RuntimeError, match="rotation"):
        solve_packing_handoff(request)
