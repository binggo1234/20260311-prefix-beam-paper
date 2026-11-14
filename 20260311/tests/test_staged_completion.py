"""Completion fidelity is distinct from a complete, feasible layout."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inrp.budget import BudgetExpired, TotalBudget
from inrp.cfg import CFG
from inrp.dataio import Part
from inrp.handoff import PackingPart, PackingRequest, solve_packing_handoff_with_budget
from inrp.mcts import LayoutSnapshot, PersistentBoards, RecedingHorizonMCTS
from inrp.runner import _case_stats, _variant_cfg
from inrp.validate import validate_solution


def _cfg():
    cfg = CFG()
    cfg.BOARD_W = cfg.BOARD_H = 10.0
    cfg.TRIM = cfg.GAP = cfg.TOOL_D = cfg.SAFE_GAP = 0.0
    cfg.BINARY_SPACING = False
    cfg.PREFIX_BEAM_ENABLE = True
    cfg.PREFIX_BEAM_STAGED_ENABLE = True
    cfg.BASELINE_RESTARTS = 0
    return cfg


def _parts():
    # Hand-checked real decoder: baseline uses 3 boards, long_edge uses 2.
    dims = ((4, 7), (6, 8), (3, 6), (2, 7), (8, 3), (5, 5))
    return [Part(i, str(i), float(w), float(h), float(w), float(h)) for i, (w, h) in enumerate(dims, 1)]


def _empty():
    return LayoutSnapshot(PersistentBoards.empty(), 0.0, (0, 0), 0)


def _complete(solver, depth=0):
    return solver._complete_with_baseline(_empty(), solver.full_remaining_mask, prefix_depth=depth)


def test_primary_is_feasible_but_not_cached_as_a_full_group():
    solver = RecedingHorizonMCTS(_parts(), _cfg(), 1)
    suffix, snapshot, _ = _complete(solver)

    assert suffix == (2, 1, 6, 5, 3, 4)
    assert len(snapshot.boards) == 3
    validate_solution(_parts(), list(snapshot.boards), 10.0, 10.0)
    assert solver.meta["oracle_completion_mode_evals"] == 1
    assert solver.meta["oracle_completion_last_tier"] == "primary_only"
    assert not solver.oracle_completion_cache
    _complete(solver)
    assert solver.meta["oracle_completion_mode_evals"] == 1
    assert solver.meta["staged_primary_completed"] == 1
    assert solver.meta["staged_completion_cache_hits"] == 1


def test_four_screens_grant_one_promotion_and_full_cache():
    solver = RecedingHorizonMCTS(_parts(), _cfg(), 1)
    for depth in range(3):
        assert len(_complete(solver, depth)[1].boards) == 3
    suffix, snapshot, _ = _complete(solver, 3)

    assert len(snapshot.boards) == 2
    assert suffix == (2, 5, 1, 4, 3, 6)
    assert solver.meta["staged_primary_completed"] == 4
    assert solver.meta["staged_promotions"] == 1
    assert solver.meta["staged_promotions_quality"] == 1
    assert solver.meta["oracle_completion_mode_evals"] == 6
    assert len(solver.oracle_completion_cache) == 1
    assert solver.meta["oracle_completion_last_tier"] == "full"
    assert _complete(solver, 3)[0] == suffix
    assert solver.meta["oracle_completion_cache_hits"] == 1
    assert solver.meta["oracle_completion_mode_evals"] == 6


def test_primary_cache_can_be_promoted_later_without_replaying_primary():
    solver = RecedingHorizonMCTS(_parts(), _cfg(), 1)
    _complete(solver, 0)
    # A one-board incumbent makes three-board primaries miss the quality gate.
    solver.global_best_snapshot = LayoutSnapshot(PersistentBoards.empty().append(object()), 0.0, (0, 0), 0)
    for depth in (1, 2, 3):
        _complete(solver, depth)
    assert solver.meta["staged_promotions"] == 0
    solver.global_best_snapshot = None
    suffix, snapshot, _ = _complete(solver, 0)

    assert suffix == (2, 5, 1, 4, 3, 6)
    assert len(snapshot.boards) == 2
    assert solver.meta["staged_primary_completed"] == 4
    assert solver.meta["oracle_completion_mode_evals"] == 6
    assert solver.meta["staged_promotions"] == 1


def test_exploration_credits_do_not_discard_poor_primary_forever():
    solver = RecedingHorizonMCTS(_parts(), _cfg(), 1)
    solver.global_best_snapshot = LayoutSnapshot(PersistentBoards.empty().append(object()), 0.0, (0, 0), 0)
    for depth in range(8):
        _complete(solver, depth)

    assert solver.meta["staged_primary_completed"] == 8
    assert solver.meta["staged_promotions"] == 1
    assert solver.meta["staged_promotions_exploration"] == 1
    assert solver.meta["oracle_completion_mode_evals"] == 10


def test_partial_promotion_keeps_completed_modes_and_resumes_only_missing(monkeypatch):
    cfg = _cfg()
    cfg.PREFIX_BEAM_STAGED_PROMOTE_EVERY = 1
    solver = RecedingHorizonMCTS(_parts(), cfg, 1)
    replay = solver.decoder.replay_order
    attempts = []

    def expire_last(sequence, **kwargs):
        attempts.append(tuple(sequence))
        if len(attempts) == 3:
            raise BudgetExpired("completion_replay")
        return replay(sequence, **kwargs)

    monkeypatch.setattr(solver.decoder, "replay_order", expire_last)
    suffix, snapshot, _ = _complete(solver)

    assert suffix == (2, 5, 1, 4, 3, 6)
    assert len(snapshot.boards) == 2
    assert solver.meta["oracle_completion_last_tier"] == "promoted_partial"
    assert solver.meta["oracle_completion_mode_completed"] == 2
    assert solver.meta["oracle_completion_mode_timeouts"] == 1
    assert solver.meta["oracle_completion_last_best_mode"] == "long_edge"
    assert not solver.oracle_completion_cache
    _complete(solver)
    assert len(attempts) == 4
    assert solver.meta["oracle_completion_mode_completed"] == 3
    assert solver.meta["oracle_completion_last_tier"] == "full"
    assert solver.meta["staged_promotions"] == 1
    assert solver.meta["staged_primary_completed"] == 1


def test_failed_primary_never_creates_usable_cache(monkeypatch):
    solver = RecedingHorizonMCTS(_parts(), _cfg(), 1)
    solver.search_deadline = time.perf_counter() - 1.0
    with pytest.raises(BudgetExpired, match="completion_replay"):
        _complete(solver)
    assert not solver.oracle_completion_cache
    assert solver.meta["staged_primary_completed"] == 0
    assert not solver.staged_completion_cache


def test_cache_disabled_replays_primary_and_never_stores_either_tier():
    cfg = _cfg()
    cfg.PREFIX_BEAM_ORACLE_CACHE_ENABLE = False
    solver = RecedingHorizonMCTS(_parts(), cfg, 1)
    _complete(solver)
    _complete(solver)

    assert solver.meta["oracle_completion_mode_evals"] == 2
    assert not solver.oracle_completion_cache
    assert not solver.staged_completion_cache


def test_single_mode_is_already_full_without_spending_promotion_credit():
    cfg = _cfg()
    cfg.PREFIX_BEAM_ORACLE_MODES = ("long_edge",)
    solver = RecedingHorizonMCTS(_parts(), cfg, 1)
    assert len(_complete(solver)[1].boards) == 2
    assert solver.meta["oracle_completion_last_tier"] == "full"
    assert solver.meta["staged_promotions"] == 0
    assert len(solver.oracle_completion_cache) == 1


@pytest.mark.parametrize("n_parts", [6, 739, 2416])
def test_staged_variant_changes_only_completion_policy_and_identity(n_parts):
    cfg = CFG()
    stats = _case_stats(_parts(), cfg)
    stats["n_parts"] = n_parts
    control = _variant_cfg(cfg, "rh_mcts_prefix_beam", stats)
    staged = _variant_cfg(cfg, "rh_mcts_prefix_beam_staged", stats)
    ignored = {"PREFIX_BEAM_STAGED_ENABLE", "SOLVER_ROLE", "SOLVER_BUNDLE", "VARIANTS"}
    assert {k: v for k, v in vars(control).items() if k not in ignored} == {
        k: v for k, v in vars(staged).items() if k not in ignored
    }
    assert not control.PREFIX_BEAM_STAGED_ENABLE
    assert staged.PREFIX_BEAM_STAGED_ENABLE
    assert not staged.MACRO_COMPLEMENT_ENABLE
    control_solver = RecedingHorizonMCTS(_parts(), control, 41)
    staged_solver = RecedingHorizonMCTS(_parts(), staged, 41)
    before = control_solver._generate_macro_actions(_empty(), control_solver.full_remaining_mask, 8)
    after = staged_solver._generate_macro_actions(_empty(), staged_solver.full_remaining_mask, 8)
    assert before[0] == after[0]
    assert before[1] == after[1]
    assert {key: value.snapshot.layout_hash for key, value in before[2].items()} == {
        key: value.snapshot.layout_hash for key, value in after[2].items()
    }


def test_disabling_staging_retains_exhaustive_mode_evaluation():
    cfg = _cfg()
    cfg.PREFIX_BEAM_STAGED_ENABLE = False
    solver = RecedingHorizonMCTS(_parts(), cfg, 1)
    assert len(_complete(solver)[1].boards) == 2
    assert solver.meta["oracle_completion_mode_evals"] == 3
    assert solver.meta["staged_primary_completed"] == 0
    assert not solver.staged_completion_cache
    assert len(solver.oracle_completion_cache) == 1


def test_staged_deadline_preserves_first_complete_result_and_floor(monkeypatch):
    cfg = _cfg()
    cfg.PREFIX_BEAM_STAGED_PROMOTE_EVERY = 1
    solver = RecedingHorizonMCTS(_parts(), cfg, 1)
    original = solver.decoder.replay_order

    def expire_after_complete_primary(sequence, **kwargs):
        result = original(sequence, **kwargs)
        solver.search_deadline = 0.0
        return result

    monkeypatch.setattr(solver.decoder, "replay_order", expire_after_complete_primary)
    suffix, snapshot, _ = _complete(solver)
    solver._update_global_best(
        suffix, snapshot, source="prefix_completion",
        completion_mode=solver.meta["oracle_completion_last_best_mode"], prefix_depth=0,
    )
    assert len(snapshot.boards) == 3
    assert solver.meta["solver_timeout_hit"] == 1
    assert solver.meta["oracle_completion_mode_completed"] == 1
    assert solver.meta["oracle_completion_mode_timeouts"] == 1
    assert solver.meta["oracle_completion_last_tier"] == "promoted_partial"
    assert solver.meta["global_best_events"][0]["completion_tier"] == "promoted_partial"
    validate_solution(_parts(), list(solver.global_best_snapshot.boards), 10.0, 10.0)


def test_promotion_during_root_ranking_immediately_retains_better_complete_layout():
    solver = RecedingHorizonMCTS(_parts(), _cfg(), 1)
    for depth in range(3):
        solver._evaluate_prefix_with_baseline((), _empty(), solver.full_remaining_mask, prefix_depth=depth)
    assert len(solver.global_best_snapshot.boards) == 3
    # The beam ranking path evaluates completions too, outside _run_simulations.
    sequence, snapshot, _ = solver._evaluate_prefix_with_baseline(
        (), _empty(), solver.full_remaining_mask, prefix_depth=3,
    )
    assert len(snapshot.boards) == 2
    assert solver.global_best_sequence == sequence
    assert solver.global_best_snapshot is snapshot
    event = solver.meta["global_best_events"][-1]
    assert event["source"] == "prefix_completion"
    assert event["completion_mode"] == "long_edge"
    assert event["completion_tier"] == "full"


@pytest.mark.parametrize("field,value", [
    ("PREFIX_BEAM_STAGED_PROMOTE_EVERY", 0),
    ("PREFIX_BEAM_STAGED_EXPLORE_CREDITS", 0),
    ("PREFIX_BEAM_STAGED_PANEL_MARGIN", -1),
])
def test_invalid_staged_settings_rejected(field, value):
    cfg = _cfg()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        RecedingHorizonMCTS(_parts(), cfg, 1)


def test_staged_policy_does_not_silently_change_random_ablation():
    cfg = _cfg()
    cfg.PREFIX_BEAM_COMPLETION_POLICY = "random"
    with pytest.raises(ValueError, match="deterministic"):
        RecedingHorizonMCTS(_parts(), cfg, 1)


def test_staged_variant_runs_real_budgeted_handoff():
    request = PackingRequest(
        case_id="staged-synthetic", parts=(
            PackingPart("a", "a", 54, 94, (0,)), PackingPart("b", "b", 34, 94, (0,)),
        ), board_width_mm=110, board_height_mm=110,
        solver_variant="rh_mcts_prefix_beam_staged", time_limit_s=1.0,
    )
    result = solve_packing_handoff_with_budget(request, TotalBudget(2.0, 0.5))
    assert result.panel_count == 1
    assert {item.part_id for item in result.placements} == {"a", "b"}
    meta = result.provenance["solver_meta"]
    assert meta["staged_completion_enabled"] is True
    assert result.panel_count <= meta["greedy_floor_panels"]
    assert meta["staged_primary_completed"] > 0
