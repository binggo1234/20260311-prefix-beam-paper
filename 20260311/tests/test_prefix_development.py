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
