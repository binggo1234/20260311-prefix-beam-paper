"""Behavior contracts for the first restricted-prefix development iteration."""

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import inrp.mcts as mcts_module
from inrp.budget import BudgetExpired
from inrp.cfg import CFG
from inrp.dataio import Part
from inrp.mcts import LayoutSnapshot, PersistentBoards, RecedingHorizonMCTS, SearchNode
from inrp.runner import _baseline_sequence
from inrp.runner import _case_stats, _variant_cfg
from inrp.budget import TotalBudget
from inrp.handoff import PackingPart, PackingRequest, solve_packing_handoff_with_budget
from inrp.validate import validate_solution


def _cfg():
    cfg = CFG()
    cfg.BOARD_W = cfg.BOARD_H = 10.0
    cfg.TRIM = cfg.GAP = cfg.TOOL_D = cfg.SAFE_GAP = 0.0
    cfg.BINARY_SPACING = False
    cfg.PREFIX_BEAM_ENABLE = True
    cfg.MACRO_ACTION_ENABLE = True
    cfg.MACRO_EARLY_SINGLE_ONLY_RATIO = 0.0
    cfg.BASELINE_RESTARTS = 0
    cfg.GLOBAL_PATTERN_INCLUDE_BASELINE = False
    cfg.SOLVER_TIME_LIMIT_S = 1.0
    return cfg


def _parts():
    parts = [
        Part(i, str(i), float(w), float(h), float(w), float(h))
        for i, (w, h) in enumerate(((6, 6), (4, 4), (2, 8)), 1)
    ]
    parts[2].is_strip = True
    return parts


def _empty():
    return LayoutSnapshot(PersistentBoards.empty(), 0.0, (0, 0), 0)


def test_prefix_injects_exact_greedy_even_without_other_warmstarts():
    solver = RecedingHorizonMCTS(_parts(), _cfg(), 1)
    assert _baseline_sequence(_parts()) == (1, 3, 2)
    assert solver._area_aspect_sequence() != (1, 3, 2)

    solver._prime_constructive_warmstarts()

    assert solver.global_best_sequence == (1, 3, 2)
    assert solver.meta["greedy_floor_panels"] == len(solver.global_best_snapshot.boards)
    assert solver.meta["global_best_events"][0]["source"] == "greedy_floor"
    assert solver.meta["global_best_events"][0]["phase"] == "warmstart"
    assert solver.meta["phase_time_s"]["warmstart"] >= 0.0
    assert solver.meta["warmstart_best_panels"] == len(solver.global_best_snapshot.boards)


def test_formal_prefix_keeps_greedy_when_later_replays_expire(monkeypatch):
    cfg = _cfg()
    cfg.BASELINE_RESTARTS = 2
    cfg.GLOBAL_PATTERN_INCLUDE_BASELINE = True
    cfg.FORMAL_TOTAL_DEADLINE = time.perf_counter() + 10.0
    cfg.FORMAL_CLOSING_RESERVE_S = 1.0
    solver = RecedingHorizonMCTS(_parts(), cfg, 1)
    original = solver.decoder.replay_order
    calls = []

    def expire_after_greedy(sequence, **kwargs):
        calls.append(tuple(sequence))
        if len(calls) > 1:
            raise BudgetExpired(kwargs.get("deadline_phase", "completion_replay"))
        return original(sequence, **kwargs)

    monkeypatch.setattr(solver.decoder, "replay_order", expire_after_greedy)
    result = solver.solve()

    assert calls[0] == (1, 3, 2)
    assert result.sequence == (1, 3, 2)
    assert result.meta["solver_timeout_hit"] == 1.0
    assert result.meta["phase_time_s"]["warmstart"] >= 0.0
    assert result.meta["phase_time_s"]["prefix_search"] >= 0.0
    validate_solution(_parts(), result.boards, 10.0, 10.0)


def test_incumbent_events_use_solver_start_and_explicit_context(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(mcts_module.time, "perf_counter", lambda: now[0])
    cfg = _cfg()
    cfg.FORMAL_TOTAL_DEADLINE = 1000.0
    solver = RecedingHorizonMCTS(_parts(), cfg, 1)
    snapshot, _ = solver.decoder.replay_order((1, 3, 2))
    now[0] = 101.25

    solver._update_global_best(
        (1, 3, 2), snapshot,
        source="prefix_completion", completion_mode="long_edge", prefix_depth=2,
    )
    event = solver.meta["global_best_events"][0]
    assert event["elapsed_s"] == 1.25
    assert event["source"] == "prefix_completion"
    assert event["completion_mode"] == "long_edge"
    assert event["prefix_depth"] == 2


def test_completion_timings_include_interrupted_mode_without_caching_group(monkeypatch):
    solver = RecedingHorizonMCTS(_parts(), _cfg(), 1)
    original = solver.decoder.replay_order
    calls = []

    def expire_second_mode(sequence, **kwargs):
        calls.append(tuple(sequence))
        if len(calls) == 2:
            raise BudgetExpired("completion_replay")
        return original(sequence, **kwargs)

    monkeypatch.setattr(solver.decoder, "replay_order", expire_second_mode)
    sequence, snapshot, _ = solver._complete_with_baseline(
        _empty(), solver.full_remaining_mask, prefix_depth=0,
    )

    assert sorted(sequence) == [1, 2, 3]
    assert len(snapshot.boards) >= 1
    assert solver.meta["oracle_completion_mode_completed"] == 1.0
    assert solver.meta["oracle_completion_mode_timeouts"] == 1.0
    assert set(solver.meta["completion_mode_time_s"]) == {"baseline", "long_edge"}
    assert all(value >= 0.0 for value in solver.meta["completion_mode_time_s"].values())
    assert solver.meta["phase_time_s"]["completion"] >= 0.0
    assert not solver.oracle_completion_cache


def test_prefix_reports_real_puct_selections():
    solver = RecedingHorizonMCTS(_parts(), _cfg(), 1)
    root = SearchNode(snapshot=_empty(), remaining_mask=solver.full_remaining_mask)
    solver._run_simulations(root, 16)
    assert solver.meta["puct_selections"] > 0
    assert solver.meta["puct_selections"] <= solver.meta["simulations"] * len(_parts())


def _locked_parts(dims):
    return [
        Part(i, str(i), float(w), float(h), float(w), float(h), allowed_rotations_deg=(0,))
        for i, (w, h) in enumerate(dims, 1)
    ]


def test_complement_pairs_choose_compact_fit_and_decode_both_orders():
    cfg = _cfg()
    cfg.MACRO_COMPLEMENT_ENABLE = True
    parts = _locked_parts(((6, 10), (6, 9), (4, 10)))
    solver = RecedingHorizonMCTS(parts, cfg, 1)
    candidates = solver._complement_macro_actions(_empty(), solver.full_remaining_mask, 1)

    assert {tuple(item[1]) for item in candidates} == {(1, 3), (3, 1)}
    for _, sequence, _ in candidates:
        snapshot, _ = solver.decoder.replay_order(sequence)
        assert len(snapshot.boards) == 1
        validate_solution([parts[i - 1] for i in sequence], list(snapshot.boards), 10.0, 10.0)


def test_complement_pairs_respect_rotation_permissions_and_global_switch():
    cfg = _cfg()
    cfg.BOARD_H = 6.0
    cfg.MACRO_COMPLEMENT_ENABLE = True
    parts = _locked_parts(((6, 6), (6, 4)))
    locked = RecedingHorizonMCTS(parts, cfg, 1)
    assert locked._complement_macro_actions(_empty(), locked.full_remaining_mask, 1) == []
    parts[1].allowed_rotations_deg = (90,)
    allowed = RecedingHorizonMCTS(parts, cfg, 1)
    assert allowed._complement_macro_actions(_empty(), allowed.full_remaining_mask, 1)
    cfg.ALLOW_ROT = False
    forbidden = RecedingHorizonMCTS(parts, cfg, 1)
    assert forbidden._complement_macro_actions(_empty(), forbidden.full_remaining_mask, 1) == []


def test_complement_fit_uses_inflated_dimensions_and_trim():
    cfg = _cfg()
    cfg.MACRO_COMPLEMENT_ENABLE = True
    parts = [Part(i, str(i), 6.0, 6.0, 4.0, 4.0) for i in (1, 2)]
    solver = RecedingHorizonMCTS(parts, cfg, 1)
    assert solver._complement_macro_actions(_empty(), solver.full_remaining_mask, 1) == []
    cfg.TRIM = 1.0
    trimmed = RecedingHorizonMCTS(_locked_parts(((6, 8), (3, 8))), cfg, 1)
    assert trimmed._complement_macro_actions(_empty(), trimmed.full_remaining_mask, 1) == []


@pytest.mark.parametrize("cap", [1, 2, 3, 8])
def test_complement_selection_preserves_cap_metadata_and_single_fallback(cap):
    cfg = _cfg()
    cfg.MACRO_COMPLEMENT_ENABLE = True
    cfg.MACRO_ACTION_TOPK = cap
    solver = RecedingHorizonMCTS(_locked_parts(((6, 10), (6, 9), (4, 10))), cfg, 1)
    actions, metadata, decoded = solver._generate_macro_actions(_empty(), solver.full_remaining_mask, cap)

    assert 1 <= len(actions) <= cap
    assert set(actions) == set(metadata) == set(decoded)
    assert any(metadata[action].kind == "single" for action in actions)
    pairs = [metadata[action] for action in actions if metadata[action].source == "complement"]
    if cap > 1:
        assert pairs
    assert len({item.part_ids for item in pairs}) == len(pairs)
    assert all(len(set(item.sequence)) == len(item.sequence) for item in metadata.values())


def test_complement_generation_is_opt_in_and_stops_at_deadline():
    solver = RecedingHorizonMCTS(_parts(), _cfg(), 1)
    assert solver._complement_macro_actions(_empty(), solver.full_remaining_mask, 2) == []
    cfg = _cfg()
    cfg.MACRO_COMPLEMENT_ENABLE = True
    expired = RecedingHorizonMCTS(_parts(), cfg, 1)
    expired.search_deadline = 0.0
    assert expired._complement_macro_actions(_empty(), expired.full_remaining_mask, 2) == []
    assert expired.meta["complement_pairs_considered"] == 0


def test_zero_complement_quota_retains_original_candidate_policy():
    cfg = _cfg()
    cfg.MACRO_ACTION_TOPK = 2
    parts = _locked_parts(((6, 10), (6, 9), (4, 10)))
    original = RecedingHorizonMCTS(parts, cfg, 1)
    before = original._generate_macro_actions(_empty(), original.full_remaining_mask, 2)
    cfg.MACRO_COMPLEMENT_ENABLE = True
    cfg.MACRO_COMPLEMENT_TOPK = 0
    disabled = RecedingHorizonMCTS(parts, cfg, 1)
    after = disabled._generate_macro_actions(_empty(), disabled.full_remaining_mask, 2)
    assert before[0] == after[0]
    assert before[1] == after[1]


def test_complement_variant_runs_real_budgeted_handoff():
    cfg = _cfg()
    original = _variant_cfg(cfg, "rh_mcts_prefix_beam", _case_stats(_parts(), cfg))
    assert not original.MACRO_COMPLEMENT_ENABLE
    request = PackingRequest(
        case_id="synthetic-complement",
        parts=(PackingPart("a", "a", 54, 94, (0,)), PackingPart("b", "b", 34, 94, (0,))),
        board_width_mm=110.0, board_height_mm=110.0,
        trim_mm=5.0, tool_diameter_mm=6.0,
        solver_variant="rh_mcts_prefix_beam_complement", time_limit_s=1.0,
    )
    result = solve_packing_handoff_with_budget(request, TotalBudget(2.0, 0.5))
    meta = result.provenance["solver_meta"]
    assert result.panel_count == 1
    assert {p.part_id for p in result.placements} == {"a", "b"}
    assert meta["complement_enabled"] is True
    assert meta["macro_complement_actions"] > 0
    assert meta["candidate_unique_parts_max"] == 2
    assert meta["main_profile_tier"] == "small"
    assert result.panel_count <= meta["greedy_floor_panels"]
