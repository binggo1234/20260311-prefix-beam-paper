"""A complete board-saving witness may advance, never relax prefix limits."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inrp.budget import BudgetExpired, TotalBudget
from inrp.cfg import CFG
from inrp.dataio import Part
from inrp.handoff import PackingPart, PackingRequest, solve_packing_handoff_with_budget
from inrp.mcts import RecedingHorizonMCTS, SearchNode, _empty_snapshot
from inrp.ordering import baseline_sequence
from inrp.runner import _case_stats, _variant_cfg
from inrp.validate import validate_solution


def parts():
    return [Part(i, str(i), float(w), float(h), float(w), float(h))
            for i, (w, h) in enumerate(((4, 7), (6, 8), (3, 6), (2, 7), (8, 3), (5, 5)), 1)]


def subject(enabled=True):
    cfg = CFG()
    cfg.BOARD_W = cfg.BOARD_H = 10.0
    cfg.TRIM = cfg.GAP = cfg.TOOL_D = cfg.SAFE_GAP = 0.0
    cfg.BINARY_SPACING = False
    cfg.PREFIX_BEAM_ENABLE = cfg.PREFIX_BEAM_STAGED_ENABLE = True
    cfg.PREFIX_BEAM_WITNESS_ADVANCE = enabled
    cfg.PREFIX_BEAM_STAGED_PROMOTE_EVERY = 1
    cfg.PREFIX_BEAM_USE_MACRO = cfg.MACRO_ACTION_ENABLE = False
    cfg.COMMIT_MAX_STEPS = 1
    cfg.BASELINE_RESTARTS = 0
    solver = RecedingHorizonMCTS(parts(), cfg, 1000)
    sequence = tuple(baseline_sequence(parts()))
    snapshot, _ = solver.decoder.replay_order(sequence)
    assert len(snapshot.boards) == 3
    solver._update_global_best(sequence, snapshot, source="greedy_floor")
    root = SearchNode(_empty_snapshot(), solver.full_remaining_mask)
    return solver, root


def select_path(solver, root, monkeypatch, sequence=(2,)):
    path = [root]
    for uid in sequence:
        path.append(solver._expand(path[-1], uid))
    monkeypatch.setattr(solver, "_select_leaf", lambda current: path)
    return path


def test_strict_complete_saving_interrupts_batch_after_backup(monkeypatch):
    solver, root = subject()
    path = select_path(solver, root, monkeypatch)
    witness = solver._run_simulations(root, 8)
    assert witness is not None
    chain, snapshot = witness
    assert chain == [(2, path[-1])]
    assert len(snapshot.boards) == len(solver.global_best_snapshot.boards) == 2
    validate_solution(parts(), list(snapshot.boards), 10.0, 10.0)
    assert solver.meta["simulations"] == root.visits == path[-1].visits == 1
    assert solver.meta["prefix_witness_triggers"] == 1


def test_disabled_policy_finishes_original_batch_despite_saving(monkeypatch):
    solver, root = subject(False)
    select_path(solver, root, monkeypatch)
    assert solver._run_simulations(root, 8) is None
    assert solver.meta["simulations"] == root.visits == 8
    assert len(solver.global_best_snapshot.boards) == 2
    assert solver.meta["prefix_witness_triggers"] == 0


def test_equal_panel_count_cannot_trigger_on_secondary_score(monkeypatch):
    solver, root = subject()
    path = select_path(solver, root, monkeypatch)
    solver._evaluate_prefix_with_baseline(path[-1].prefix_sequence, path[-1].snapshot, path[-1].remaining_mask, prefix_depth=1)
    assert len(solver.global_best_snapshot.boards) == 2
    assert solver._run_simulations(root, 2) is None
    assert solver.meta["simulations"] == 2
    assert solver.meta["prefix_witness_triggers"] == 0


@pytest.mark.parametrize("blocked", ["length", "depth", "admissibility", "terminal"])
def test_witness_never_bypasses_existing_commit_limits(monkeypatch, blocked):
    solver, root = subject()
    if blocked in ("length", "depth"):
        path = select_path(solver, root, monkeypatch, (2, 5))
        if blocked == "depth":
            solver.commit_max_steps = 2
            solver.prefix_beam_depth = 1
    elif blocked == "terminal":
        path = select_path(solver, root, monkeypatch, (2, 5, 1, 4, 3, 6))
        solver.commit_max_steps = solver.prefix_beam_depth = 6
    else:
        path = select_path(solver, root, monkeypatch)
        monkeypatch.setattr(solver, "_prefix_beam_child_allowed", lambda node, action, child: False)
    assert solver._run_simulations(root, 1) is None
    assert len(solver.global_best_snapshot.boards) == 2
    assert solver.meta["prefix_witness_triggers"] == 0
    assert path[-1].visits == 1


def test_expiry_after_complete_saving_keeps_incumbent_without_advancing(monkeypatch):
    solver, root = subject()
    select_path(solver, root, monkeypatch)
    evaluate = solver._evaluate_prefix_with_baseline

    def expire(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        solver.search_deadline = 0.0
        return result

    monkeypatch.setattr(solver, "_evaluate_prefix_with_baseline", expire)
    assert solver._run_simulations(root, 8) is None
    assert len(solver.global_best_snapshot.boards) == 2
    assert solver.meta["simulations"] == 1
    assert solver.meta["prefix_witness_triggers"] == 0


def test_failed_completion_does_not_advance_or_backup(monkeypatch):
    solver, root = subject()
    path = select_path(solver, root, monkeypatch)

    def expire(*args, **kwargs):
        raise BudgetExpired("completion_replay")

    monkeypatch.setattr(solver, "_evaluate_prefix_with_baseline", expire)
    assert solver._run_simulations(root, 8) is None
    assert solver.meta["simulations"] == root.visits == path[-1].visits == 0
    assert len(solver.global_best_snapshot.boards) == 3
    assert solver.meta["prefix_witness_triggers"] == 0


def test_incomplete_sequence_is_not_a_witness(monkeypatch):
    solver, root = subject()
    select_path(solver, root, monkeypatch)
    evaluate = solver._evaluate_prefix_with_baseline

    def incomplete(*args, **kwargs):
        sequence, snapshot, score = evaluate(*args, **kwargs)
        return sequence[:-1], snapshot, score

    monkeypatch.setattr(solver, "_evaluate_prefix_with_baseline", incomplete)
    assert solver._run_simulations(root, 1) is None
    assert solver.meta["prefix_witness_triggers"] == 0


def test_deeper_evidence_requires_a_complete_sequence(monkeypatch):
    solver, root = subject()
    select_path(solver, root, monkeypatch)
    solver.prefix_witness_advanced_prefixes.add(tuple(root.prefix_sequence))
    evaluate = solver._evaluate_prefix_with_baseline

    def incomplete(*args, **kwargs):
        sequence, snapshot, score = evaluate(*args, **kwargs)
        return sequence[:-1], snapshot, score

    monkeypatch.setattr(solver, "_evaluate_prefix_with_baseline", incomplete)
    solver._run_simulations(root, 1)
    assert solver.meta["prefix_witness_deeper_evals"] == 0


def test_partial_fidelity_complete_layout_can_advance_without_full_cache(monkeypatch):
    solver, root = subject()
    select_path(solver, root, monkeypatch, (3,))
    replay = solver.decoder.replay_order
    calls = []

    def stop_promotion(sequence, **kwargs):
        calls.append(tuple(sequence))
        if len(calls) == 2:
            raise BudgetExpired("completion_replay")
        return replay(sequence, **kwargs)

    monkeypatch.setattr(solver.decoder, "replay_order", stop_promotion)
    witness = solver._run_simulations(root, 8)
    assert witness is not None and len(witness[1].boards) == 2
    assert solver.meta["oracle_completion_last_tier"] == "promoted_partial"
    assert not solver.oracle_completion_cache
    assert len(solver.staged_completion_cache) == 1


def test_cached_deeper_return_is_not_fresh_decoder_work(monkeypatch):
    solver, root = subject()
    child = solver._expand(root, 2)
    grandchild = solver._expand(child, 5)
    solver._evaluate_prefix_with_baseline(grandchild.prefix_sequence, grandchild.snapshot,
                                         grandchild.remaining_mask, prefix_depth=2)
    solver.prefix_witness_advanced_prefixes.add(tuple(child.prefix_sequence))
    monkeypatch.setattr(solver, "_select_leaf", lambda current: [child, grandchild])
    solver._run_simulations(child, 1)
    assert solver.meta["prefix_witness_deeper_evals"] == 1
    assert solver.meta["prefix_witness_deeper_fresh_evals"] == 0


def test_commit_reuses_witness_and_actually_completes_deeper_evaluation(monkeypatch):
    solver, root = subject()
    solver.meta["formal_total_budget"] = 1.0
    solver.prefix_beam_depth = 2
    solver.prefix_beam_width = 1
    ranked = solver._prefix_beam_root_rank

    def select(current):
        uid = 2 if current.depth == 0 else 5
        if uid not in current.children:
            solver._expand(current, uid)
        return [current, current.children[uid]]

    def rank(current):
        assert tuple(current.prefix_sequence) != (2,), "witness ranking must not evaluate completion again"
        return ranked(current)

    monkeypatch.setattr(solver, "_select_leaf", select)
    monkeypatch.setattr(solver, "_prefix_beam_root_rank", rank)
    result = solver._solve_prefix_beam(root)
    assert len(result.boards) == 2
    assert solver.meta["prefix_witness_advances"] == 1
    assert solver.meta["prefix_witness_deeper_evals"] >= 1
    assert solver.meta["prefix_witness_deeper_fresh_evals"] >= 1
    assert solver.meta["prefix_witness_deeper_depth_max"] == 2
    assert solver.meta["prefix_witness_events"][0]["from_depth"] == 0
    assert solver.meta["prefix_witness_events"][0]["to_depth"] == 1


@pytest.mark.parametrize("n_parts", [6, 178, 739, 2416])
def test_witness_variant_isolates_one_policy_across_adaptive_profiles(n_parts):
    cfg = CFG()
    stats = _case_stats(parts(), cfg)
    stats["n_parts"] = n_parts
    control = _variant_cfg(cfg, "rh_mcts_prefix_beam_staged", stats)
    witness = _variant_cfg(cfg, "rh_mcts_prefix_beam_staged_witness", stats)
    ignored = {"PREFIX_BEAM_WITNESS_ADVANCE", "SOLVER_ROLE", "SOLVER_BUNDLE", "VARIANTS"}
    assert {key: value for key, value in vars(control).items() if key not in ignored} == {
        key: value for key, value in vars(witness).items() if key not in ignored}
    assert witness.PREFIX_BEAM_WITNESS_ADVANCE and not control.PREFIX_BEAM_WITNESS_ADVANCE
    assert witness.PREFIX_BEAM_STAGED_ENABLE and not witness.PREFIX_BEAM_TYPE_FAIR_EXPANSION


@pytest.mark.parametrize("gate", ["PREFIX_BEAM_ENABLE", "PREFIX_BEAM_STAGED_ENABLE"])
def test_inactive_prefix_or_staging_ignores_witness_flag(gate):
    cfg = CFG()
    cfg.PREFIX_BEAM_ENABLE = cfg.PREFIX_BEAM_STAGED_ENABLE = cfg.PREFIX_BEAM_WITNESS_ADVANCE = True
    setattr(cfg, gate, False)
    solver = RecedingHorizonMCTS(parts(), cfg, 1)
    assert not solver.meta["prefix_witness_enabled"]


def test_real_budgeted_handoff_preserves_feasible_floor():
    request = PackingRequest(case_id="witness", parts=(PackingPart("a", "a", 54, 94), PackingPart("b", "b", 34, 94)),
                             board_width_mm=110, board_height_mm=110, time_limit_s=1.0,
                             solver_variant="rh_mcts_prefix_beam_staged_witness")
    result = solve_packing_handoff_with_budget(request, TotalBudget(2.0, .5))
    meta = result.provenance["solver_meta"]
    assert result.panel_count == 1 and result.panel_count <= meta["greedy_floor_panels"]
    assert meta["prefix_witness_enabled"] and meta["prefix_witness_triggers"] == 0
    assert {part.part_id for part in result.placements} == {"a", "b"}
