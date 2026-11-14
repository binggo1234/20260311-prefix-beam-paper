from __future__ import annotations

from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inrp.budget import BudgetExpired, TotalBudget
from inrp.cfg import CFG
from inrp.dataio import Part
from inrp.handoff import PackingPart, PackingRequest, solve_packing_handoff_with_budget
import inrp.mcts as module
from inrp.mcts import RecedingHorizonMCTS
from inrp.ordering import short_edge_sequence
from inrp.packer import board_signature
from inrp.runner import _variant_cfg


ON = "rh_mcts_prefix_short_completion_on"
OFF = "rh_mcts_prefix_short_completion_off"
MODES = ("short_edge", "long_edge", "strip_bias")
SIZES = [(30, 50), (15, 35), (55, 50), (30, 35), (20, 70), (15, 75)]


def _solver(parts=None):
    cfg = CFG(BOARD_W=100, BOARD_H=100, TRIM=0, TOOL_D=0,
              BINARY_SPACING=False, SAFE_GAP=0, SOLVER_TIME_LIMIT_S=0.05)
    cfg = _variant_cfg(cfg, ON, {"n_parts": 6})
    cfg.FORMAL_TOTAL_DEADLINE = time.perf_counter() + 2
    cfg.FORMAL_CLOSING_RESERVE_S = 0.1
    if parts is None:
        parts = [Part(i, str(i), w, h, w, h) for i, (w, h) in enumerate(SIZES, 1)]
    return RecedingHorizonMCTS(parts, cfg, 17)


def test_short_edge_subset_order_matches_strong_seed_with_uid_ties_and_tiny_edges():
    sizes = [(10, 80, 5), (8, 25, 20), (2, 20, 20), (3, 25, 20),
             (11, 1, 1e-12), (12, 0.1, 2e-12)]
    parts = [Part(uid, str(uid), w, h, w, h) for uid, w, h in sizes]
    solver = _solver(parts)
    expected = (3, 8, 2, 10, 12, 11)
    assert short_edge_sequence(parts) == expected
    for subset in (expected, (8, 3, 11, 12), (10, 2, 12)):
        mask = sum(solver.part_bit[uid] for uid in subset)
        assert solver._baseline_completion_order(mask, mode="short_edge") == tuple(
            uid for uid in expected if uid in subset)


def test_active_modes_are_registered_and_recorded():
    solver = _solver()
    assert solver.prefix_beam_oracle_modes == MODES
    assert solver.meta["oracle_completion_modes"] == list(MODES)


@pytest.mark.parametrize("count", [20, 178, 739, 2416])
def test_profiles_change_only_primary_mode_and_retain_identical_adaptation(count):
    configs = [_variant_cfg(CFG(), name, {"n_parts": count}) for name in
               (ON, OFF, "rh_mcts_prefix_common_init_on", "rh_mcts_prefix_common_init_off")]
    on, off, old_on, old_off = configs
    assert on.PREFIX_BEAM_ORACLE_MODES == off.PREFIX_BEAM_ORACLE_MODES == MODES
    assert old_on.PREFIX_BEAM_ORACLE_MODES == old_off.PREFIX_BEAM_ORACLE_MODES == (
        "baseline", "long_edge", "strip_bias")
    for a, b, allowed in ((on, off, {"VARIANTS", "PREFIX_ABLATION_SEARCH_ENABLE"}),
                          (on, old_on, {"VARIANTS", "PREFIX_BEAM_ORACLE_MODES"}),
                          (off, old_off, {"VARIANTS", "PREFIX_BEAM_ORACLE_MODES"})):
        assert vars(a).keys() == vars(b).keys()
        assert {key for key in vars(a) if getattr(a, key) != getattr(b, key)} <= allowed


def test_real_primary_and_fourth_state_promotion_use_the_existing_three_mode_protocol():
    solver = _solver()
    solver._configure_solve_deadlines()
    solver._prime_common_initial_solution()
    order = (3, 1, 4, 5, 6, 2)
    for depth in range(4):
        prefix = order[:depth]
        snapshot, _ = solver.decoder.replay_order(prefix)
        remaining = solver.full_remaining_mask
        for uid in prefix:
            remaining &= ~solver.part_bit[uid]
        sequence, completed, _ = solver._evaluate_prefix_with_baseline(
            prefix, snapshot, remaining, prefix_depth=depth)
        assert len(completed.boards) == 1
        assert set(sequence) == set(order)
        if depth == 0:
            assert sequence == order
            assert solver.meta["oracle_completion_mode_evals"] == 1
    assert solver.meta["oracle_completion_mode_eval_short_edge"] == 4
    assert solver.meta["oracle_completion_mode_eval_long_edge"] == 1
    assert solver.meta["oracle_completion_mode_eval_strip_bias"] == 1
    assert solver.meta.get("oracle_completion_mode_eval_baseline", 0) == 0
    assert solver.meta["staged_promotions"] == 1


def test_interrupted_primary_replay_preserves_complete_common_floor(monkeypatch):
    monkeypatch.setattr(module.time, "perf_counter", lambda: 0.0)
    solver = _solver()
    solver.formal_total_deadline = 10.0
    solver.formal_closing_reserve_s = 2.0
    solver._configure_solve_deadlines()
    solver._prime_common_initial_solution()
    before = (solver.global_best_sequence, solver.global_best_score,
              [board_signature(b) for b in solver.global_best_snapshot.boards])
    replay = solver.decoder.replay_order

    def interrupted(order, **kwargs):
        assert kwargs["deadline"] == 8.0
        replay(order[:1], **kwargs)
        raise BudgetExpired("completion_replay")

    monkeypatch.setattr(solver.decoder, "replay_order", interrupted)
    snapshot, _ = replay(())
    with pytest.raises(BudgetExpired):
        solver._evaluate_prefix_with_baseline((), snapshot, solver.full_remaining_mask, prefix_depth=0)
    assert solver.meta["oracle_completion_mode_eval_short_edge"] == 1
    assert solver.meta["oracle_completion_mode_evals"] == 1
    assert (solver.global_best_sequence, solver.global_best_score,
            [board_signature(b) for b in solver.global_best_snapshot.boards]) == before
    assert not solver.staged_completion_cache


def test_budgeted_off_handoff_has_the_same_initial_and_final_receipts_as_old_profile():
    results = []
    for variant in (OFF, "rh_mcts_prefix_common_init_off"):
        request = PackingRequest(case_id="short-completion", parts=tuple(
            PackingPart(str(i), str(i), w, h) for i, (w, h) in enumerate(SIZES, 1)),
            board_width_mm=100, board_height_mm=100, trim_mm=0, tool_diameter_mm=0,
            solver_variant=variant, time_limit_s=0.05)
        results.append(solve_packing_handoff_with_budget(request, TotalBudget(0.3, 0.03)))
    new, old = results
    assert new.feasible and new.panel_count == old.panel_count == 1
    assert new.sequence == old.sequence
    for key in ("common_initial_panels", "common_initial_score", "common_initial_candidate_panels",
                "common_initial_sequence_sha256", "common_initial_layout_sha256",
                "ablation_final_sequence_sha256", "ablation_final_layout_sha256"):
        assert new.provenance["solver_meta"][key] == old.provenance["solver_meta"][key]
    assert new.provenance["solver_meta"]["oracle_completion_modes"] == list(MODES)
    assert new.provenance["solver_meta"]["simulations"] == 0
