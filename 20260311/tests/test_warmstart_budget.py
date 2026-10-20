"""A local initialization deadline must leave the search deadline intact."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import inrp.mcts as mcts_module
from inrp.cfg import CFG
from inrp.dataio import Part
from inrp.mcts import LayoutSnapshot, PersistentBoards, RecedingHorizonMCTS, SearchNode
from inrp.validate import validate_solution


def _solver(monkeypatch, *, available=27.0, enabled=True, prefix=True):
    now = [100.0]
    monkeypatch.setattr(mcts_module.time, "perf_counter", lambda: now[0])
    cfg = CFG()
    cfg.BOARD_W = cfg.BOARD_H = 10.0
    cfg.TRIM = cfg.GAP = cfg.TOOL_D = cfg.SAFE_GAP = 0.0
    cfg.BINARY_SPACING = False
    cfg.PREFIX_BEAM_ENABLE = prefix
    cfg.MACRO_ACTION_ENABLE = True
    cfg.MACRO_EARLY_SINGLE_ONLY_RATIO = 0.0
    cfg.PREFIX_WARMSTART_BUDGET_ENABLE = enabled
    cfg.BASELINE_RESTARTS = 2
    cfg.GLOBAL_PATTERN_INCLUDE_BASELINE = True
    parts = [
        Part(i, str(i), float(w), float(h), float(w), float(h))
        for i, (w, h) in enumerate(((6, 6), (4, 4), (2, 8)), 1)
    ]
    parts[2].is_strip = True
    solver = RecedingHorizonMCTS(parts, cfg, 1)
    solver.solve_deadline = 100.0 + available + 3.0
    solver.search_deadline = 100.0 + available
    return solver, now, parts


@pytest.mark.parametrize("available,after_greedy,limit", [(27.0, 106.0, 5.0), (10.0, 103.0, 2.0)])
def test_greedy_may_exceed_local_cap_but_optional_work_yields_to_search(
    monkeypatch, available, after_greedy, limit,
):
    solver, now, parts = _solver(monkeypatch, available=available)
    original = solver.decoder.replay_order

    def complete_greedy(sequence, **kwargs):
        result = original(sequence, **kwargs)
        now[0] = after_greedy
        return result

    monkeypatch.setattr(solver.decoder, "replay_order", complete_greedy)
    solver._prime_constructive_warmstarts()

    assert solver.global_best_sequence == (1, 3, 2)
    assert solver.meta["warmstart_budget_s"] == limit
    assert solver.meta["warmstart_budget_hit"] is True
    assert solver.meta["warmstart_stop_reason"] == "warmstart_budget"
    assert solver.meta["solver_timeout_hit"] == 0.0
    assert solver.meta["search_budget_remaining_s"] > 0.0
    assert solver.search_deadline == 100.0 + available
    assert not solver.seed_suffix_cache
    validate_solution(parts, list(solver.global_best_snapshot.boards), 10.0, 10.0)

    root = SearchNode(
        snapshot=LayoutSnapshot(PersistentBoards.empty(), 0.0, (0, 0), 0),
        remaining_mask=solver.full_remaining_mask,
    )
    solver._run_simulations(root, 1)
    assert solver.meta["simulations"] == 1.0
    solver.meta["formal_total_budget"] = 1.0
    result = solver._solve_prefix_beam(root)
    assert solver.meta["prefix_search_budget_at_entry_s"] == 100.0 + available - after_greedy
    validate_solution(parts, result.boards, 10.0, 10.0)


def test_extra_replay_obeys_local_deadline_without_claiming_total_timeout(monkeypatch):
    solver, now, _ = _solver(monkeypatch, available=20.0)
    original = solver.decoder.decode

    def timed_decode(snapshot, pid, **kwargs):
        result = original(snapshot, pid, **kwargs)
        now[0] += 1.0
        return result

    monkeypatch.setattr(solver.decoder, "decode", timed_decode)
    solver._prime_constructive_warmstarts()

    # Greedy completes at 103; the extra replay reaches the 104 local limit.
    assert now[0] == 104.0
    assert solver.meta["warmstart_budget_hit"] is True
    assert solver.meta["solver_timeout_hit"] == 0.0
    assert solver.meta["search_budget_remaining_s"] == 16.0
    assert solver.global_best_sequence == (1, 3, 2)
    assert len(solver.meta["global_best_events"]) == 1


def test_cache_registration_also_yields_at_local_deadline(monkeypatch):
    solver, now, _ = _solver(monkeypatch, available=10.0)
    original = solver._state_key

    def timed_key(snapshot, remaining_mask):
        key = original(snapshot, remaining_mask)
        now[0] += 1.0
        return key

    monkeypatch.setattr(solver, "_state_key", timed_key)
    solver._prime_constructive_warmstarts()

    assert now[0] == 102.0
    assert solver.meta["warmstart_budget_hit"] is True
    assert 0 < len(solver.seed_suffix_cache) < 4
    assert solver.meta["warmstart_candidates_kept"] == len(solver.constructive_warmstarts)
    assert solver.meta["warmstart_cache_states"] == len(solver.seed_suffix_cache)
    assert solver.meta["solver_timeout_hit"] == 0.0
    assert solver.global_best_snapshot is not None


def test_constructive_packer_receives_local_deadline(monkeypatch):
    solver, now, _ = _solver(monkeypatch)
    original = mcts_module.pack_multi_board
    deadlines = []

    def reach_local_deadline(*args, **kwargs):
        deadlines.append(kwargs["deadline"])
        now[0] = 105.0
        return original(*args, **kwargs)

    monkeypatch.setattr(mcts_module, "pack_multi_board", reach_local_deadline)
    solver._prime_constructive_warmstarts()

    assert deadlines == [105.0]
    assert solver.global_best_sequence == (1, 3, 2)
    assert solver.meta["warmstart_budget_hit"] is True
    assert solver.meta["solver_timeout_hit"] == 0.0
    assert solver.meta["search_budget_remaining_s"] == 22.0


def test_disabled_cap_reports_expiry_in_last_constructive_pack(monkeypatch):
    solver, now, _ = _solver(monkeypatch, enabled=False)
    original = mcts_module.pack_multi_board
    attempts = []

    def expire_last_pack(*args, **kwargs):
        attempts.append(kwargs["deadline"])
        if len(attempts) == 3:
            now[0] = 127.0
        return original(*args, **kwargs)

    monkeypatch.setattr(mcts_module, "pack_multi_board", expire_last_pack)
    solver._prime_constructive_warmstarts()

    assert len(attempts) == 3
    assert solver.meta["solver_timeout_hit"] == 1.0
    assert solver.meta["warmstart_stop_reason"] == "search_deadline"
    assert solver.meta["search_budget_remaining_s"] == 0.0


def test_pattern_archive_registration_stops_between_boards(monkeypatch):
    base, now, _ = _solver(monkeypatch, available=10.0)
    base.cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    parts = [Part(i, str(i), 6.0, 10.0, 6.0, 10.0, allowed_rotations_deg=(0,)) for i in (1, 2, 3)]
    solver = RecedingHorizonMCTS(parts, base.cfg, 1)
    solver.search_deadline = 110.0
    original = solver._record_pattern_candidate

    def timed_pattern(pattern):
        original(pattern)
        now[0] += 1.0

    monkeypatch.setattr(solver, "_record_pattern_candidate", timed_pattern)
    solver._prime_constructive_warmstarts()

    assert now[0] == 102.0
    assert len(solver.global_best_snapshot.boards) == 3
    assert len(solver.global_pattern_memory) == 2
    assert solver.meta["warmstart_budget_hit"] is True
    assert solver.meta["solver_timeout_hit"] == 0.0


def test_completed_better_warmstart_survives_registration_cutoff(monkeypatch):
    base, now, _ = _solver(monkeypatch, available=10.0)
    dims = ((6, 3), (4, 4), (2, 4), (4, 2), (4, 7), (2, 8))
    parts = [Part(i, str(i), float(w), float(h), float(w), float(h)) for i, (w, h) in enumerate(dims, 1)]
    parts[-1].is_strip = True
    solver = RecedingHorizonMCTS(parts, base.cfg, 1)
    solver.search_deadline = 110.0
    original = solver.decoder.replay_order
    replays = []

    def finish_at_registration_boundary(sequence, **kwargs):
        result = original(sequence, **kwargs)
        replays.append(tuple(sequence))
        if len(replays) == 2:
            now[0] = 102.0
        return result

    monkeypatch.setattr(solver.decoder, "replay_order", finish_at_registration_boundary)
    solver._prime_constructive_warmstarts()

    assert solver.meta["greedy_floor_panels"] == 2
    assert len(solver.global_best_snapshot.boards) == 1
    assert solver.global_best_sequence == (6, 5, 1, 2, 3, 4)
    assert solver.meta["warmstart_best_panels"] == 1
    assert solver.meta["warmstart_budget_hit"] is True
    assert solver.meta["solver_timeout_hit"] == 0.0
    validate_solution(parts, list(solver.global_best_snapshot.boards), 10.0, 10.0)


def test_unbounded_search_still_caps_optional_initialization(monkeypatch):
    solver, now, _ = _solver(monkeypatch, available=float("inf"))
    original = solver.decoder.replay_order

    def slow_greedy(sequence, **kwargs):
        result = original(sequence, **kwargs)
        now[0] = 106.0
        return result

    monkeypatch.setattr(solver.decoder, "replay_order", slow_greedy)
    solver._prime_constructive_warmstarts()

    assert solver.global_best_snapshot is not None
    assert solver.meta["warmstart_budget_s"] == 5.0
    assert solver.meta["warmstart_budget_hit"] is True
    assert solver.meta["search_budget_remaining_s"] is None


@pytest.mark.parametrize("setting,value", [
    ("PREFIX_WARMSTART_BUDGET_FRAC", float("nan")),
    ("PREFIX_WARMSTART_BUDGET_FRAC", 1.1),
    ("PREFIX_WARMSTART_BUDGET_MAX_S", -1.0),
    ("PREFIX_WARMSTART_BUDGET_MAX_S", float("inf")),
])
def test_invalid_budget_configuration_is_rejected(monkeypatch, setting, value):
    solver, _, _ = _solver(monkeypatch)
    setattr(solver.cfg, setting, value)

    with pytest.raises(ValueError, match=setting):
        solver._prime_constructive_warmstarts()


@pytest.mark.parametrize("enabled,prefix", [(False, True), (True, False)])
def test_disabled_or_nonprefix_initialization_retains_legacy_budget(monkeypatch, enabled, prefix):
    solver, now, _ = _solver(monkeypatch, available=27.0, enabled=enabled, prefix=prefix)
    original = solver.decoder.replay_order

    def slow_first_replay(sequence, **kwargs):
        result = original(sequence, **kwargs)
        now[0] = 106.0
        return result

    monkeypatch.setattr(solver.decoder, "replay_order", slow_first_replay)
    solver._prime_constructive_warmstarts()

    assert solver.meta["warmstart_budget_enabled"] is False
    assert solver.meta["warmstart_budget_hit"] is False
    assert solver.meta["warmstart_stop_reason"] == "completed"
    assert solver.constructive_warmstarts
    assert len(solver.seed_suffix_cache) >= 4
    assert solver.meta["solver_timeout_hit"] == 0.0


def test_zero_optional_budget_keeps_only_completed_greedy(monkeypatch):
    solver, _, _ = _solver(monkeypatch)
    solver.cfg.PREFIX_WARMSTART_BUDGET_MAX_S = 0.0
    solver.global_pattern_memory_enable = True
    solver.global_pattern_memory_min_util = 0.0
    solver._prime_constructive_warmstarts()

    assert solver.global_best_sequence == (1, 3, 2)
    assert solver.meta["warmstart_budget_hit"] is True
    assert solver.meta["warmstart_budget_s"] == 0.0
    assert solver.meta["solver_timeout_hit"] == 0.0
    assert not solver.constructive_warmstarts
    assert not solver.elite_archive
    assert not solver.harvested_layouts
    assert not solver.global_pattern_memory


def test_mandatory_greedy_still_obeys_search_deadline(monkeypatch):
    solver, now, _ = _solver(monkeypatch, available=1.0)
    original = solver.decoder.decode

    def timed_decode(snapshot, pid, **kwargs):
        result = original(snapshot, pid, **kwargs)
        now[0] += 1.0
        return result

    monkeypatch.setattr(solver.decoder, "decode", timed_decode)
    solver._prime_constructive_warmstarts()

    assert solver.global_best_snapshot is None
    assert solver.meta["solver_timeout_hit"] == 1.0
    assert solver.meta["warmstart_stop_reason"] == "search_deadline"
    assert solver.meta["warmstart_budget_hit"] is False
