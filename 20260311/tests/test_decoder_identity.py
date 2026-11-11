from __future__ import annotations

from functools import reduce
import operator
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inrp.budget import BudgetExpired, TotalBudget
from inrp.cfg import CFG
from inrp.dataio import Part
from inrp.handoff import PackingPart, PackingRequest, solve_packing_handoff_with_budget
import inrp.mcts as mcts_module
from inrp.mcts import OmnipotentDecoder, _board_state_hash
import inrp.ordering as ordering
from inrp.packer import board_signature, make_empty_board
from inrp.runner import _variant_cfg, run_one_seed


def _cfg():
    return CFG(BOARD_W=100, BOARD_H=100, TRIM=0, TOOL_D=0,
               BINARY_SPACING=False, SAFE_GAP=0, SOLVER_TIME_LIMIT_S=0.05)


def _parts(rotations=(0, 90)):
    return [Part(1, "a", 40, 10, 40, 10, allowed_rotations_deg=rotations),
            Part(2, "b", 40, 10, 40, 10, allowed_rotations_deg=rotations),
            Part(3, "c", 30, 20, 30, 20)]


def _assert_real_hashes(decoder, snapshot):
    hashes = [_board_state_hash(board, decoder.sig_nd) for board in snapshot.boards]
    mask = (1 << 64) - 1
    assert snapshot.layout_hash == (sum(hashes) & mask, reduce(operator.xor, hashes, 0))
    assert snapshot.ordered_hash == sum(h * decoder.pos_hash[i] for i, h in enumerate(hashes)) & mask


@pytest.mark.parametrize("rotations", [(0, 90), (90,)])
def test_shared_decoder_duplicate_shapes_match_cold_replay_identities_and_hashes(rotations):
    by_id = {part.uid: part for part in _parts(rotations)}
    shared = OmnipotentDecoder(by_id, _cfg())
    for order in ((1, 3, 2), (2, 3, 1), (3, 2, 1), (1, 3, 2)):
        snapshot, states = shared.replay_order(order)
        cold, _ = OmnipotentDecoder(by_id, _cfg()).replay_order(order)
        assert sorted(p.uid for board in snapshot.boards for p in board.placed) == [1, 2, 3]
        assert [board_signature(b) for b in snapshot.boards] == [board_signature(b) for b in cold.boards]
        for state in states:
            _assert_real_hashes(shared, state)


def test_same_identity_and_state_still_reuse_micro_and_complete_caches():
    parts = _parts()
    decoder = OmnipotentDecoder({p.uid: p for p in parts}, _cfg())
    board = make_empty_board(1, 100, 100, trim=0, safe_gap=0)
    first = decoder._micro_decode(board, parts[0])
    assert first is not None
    assert decoder._micro_decode(board, parts[0]) is first
    assert decoder.micro_hits == 1
    snapshot, _ = decoder.replay_order((1, 3, 2))
    hits_before = decoder.decode_hits
    repeated, _ = decoder.replay_order((1, 3, 2))
    assert repeated is snapshot
    assert decoder.decode_hits - hits_before == 3


def test_interrupted_replay_does_not_contaminate_later_identity_or_saved_incumbent(monkeypatch):
    decoder = OmnipotentDecoder({p.uid: p for p in _parts()}, _cfg())
    saved, _ = decoder.replay_order((1, 3, 2))
    now = [0.0]
    monkeypatch.setattr(mcts_module.time, "perf_counter", lambda: now[0])
    real_decode = decoder.decode

    def interrupt_after_first(snapshot, action, **kwargs):
        out = real_decode(snapshot, action, **kwargs)
        if kwargs.get("deadline_phase") == "forced_interrupt":
            now[0] = 8.0
        return out

    monkeypatch.setattr(decoder, "decode", interrupt_after_first)
    with pytest.raises(BudgetExpired):
        decoder.replay_order((2, 3, 1), deadline=8.0, deadline_phase="forced_interrupt")
    now[0] = 0.0
    resumed, _ = decoder.replay_order((2, 3, 1), deadline=8.0)
    assert sorted(p.uid for board in resumed.boards for p in board.placed) == [1, 2, 3]
    assert sorted(p.uid for board in saved.boards for p in board.placed) == [1, 2, 3]
    _assert_real_hashes(decoder, resumed)


def test_short_edge_order_uses_area_then_uid_as_stable_ties():
    assert hasattr(ordering, "short_edge_sequence"), "missing deterministic short-edge reference"
    parts = [Part(uid, str(uid), w, h, w, h) for uid, w, h in (
        (10, 80, 5), (8, 25, 20), (2, 20, 20), (3, 25, 20))]
    assert ordering.short_edge_sequence(parts) == (3, 8, 2, 10)


@pytest.mark.parametrize("count", [20, 178, 739, 2416])
def test_short_edge_reference_uses_staged_decoder_settings_at_each_tier(count):
    cfg = _variant_cfg(_cfg(), "baseline_short_edge", {"n_parts": count})
    staged = _variant_cfg(_cfg(), "rh_mcts_prefix_beam_staged", {"n_parts": count})
    assert cfg.SOLVER_KIND == "baseline_short_edge"
    for key in ("MAIN_PROFILE_TIER", "BOARD_W", "BOARD_H", "TRIM", "GAP", "TOOL_D",
                "BINARY_SPACING", "SAFE_GAP", "ALLOW_ROT", "BASELINE_PLACE_MODE", "UTIL_GAMMA", "STATE_SIG_ND"):
        assert getattr(cfg, key) == getattr(staged, key)


def test_short_edge_budgeted_handoff_has_the_expected_complete_order():
    request = PackingRequest(
        case_id="short-edge", parts=(PackingPart("a", "a", 40, 10), PackingPart("b", "b", 20, 20),
                                      PackingPart("c", "c", 30, 30)),
        board_width_mm=100, board_height_mm=100, trim_mm=2, tool_diameter_mm=2,
        time_limit_s=0.1, solver_variant="baseline_short_edge")
    result = solve_packing_handoff_with_budget(request, TotalBudget(1.0, 0.1))
    assert result.feasible and result.panel_count == 1
    assert result.sequence == ("c", "b", "a")
    assert result.provenance["solver_meta"]["solver_kind"] == "baseline_short_edge"


def test_short_edge_runner_dispatches_deterministic_solver(tmp_path):
    cfg = _variant_cfg(_cfg(), "baseline_short_edge", {"n_parts": 3})
    totals = run_one_seed(_parts(), cfg, 17, tmp_path, "baseline_short_edge")
    assert totals["solver_kind"] == "baseline_short_edge"


def test_short_edge_solver_stops_at_the_search_boundary(monkeypatch):
    import inrp.runner as runner
    cfg = _variant_cfg(_cfg(), "baseline_short_edge", {"n_parts": 3})
    cfg.FORMAL_TOTAL_DEADLINE = 10.0
    cfg.FORMAL_CLOSING_RESERVE_S = 2.0
    monkeypatch.setattr(mcts_module.time, "perf_counter", lambda: 8.0)
    with pytest.raises(BudgetExpired):
        runner._solve_baseline_greedy(_parts(), cfg, 17)
