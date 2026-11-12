from dataclasses import asdict
import time

import pytest

from inrp.budget import BudgetExpired, TotalBudget
from inrp.handoff import PackingPart, PackingRequest, _build_solver_input, _build_solver_config
from inrp.mcts import RecedingHorizonMCTS, OmnipotentDecoder, _complete_solution_fingerprints
from inrp.early_prefix_control import solve_two_completion_handoff


SIZES = [(15, 15), (40, 45), (20, 20), (50, 50), (25, 55), (50, 15), (30, 55)]
ARMS = ("single_reference", "current_search")


def request(variant="rh_mcts_prefix_short_completion_on"):
    return PackingRequest(case_id="early-control-toy", parts=tuple(
        PackingPart(str(i), str(i), w, h) for i, (w, h) in enumerate(SIZES, 1)),
        board_width_mm=100, board_height_mm=100, trim_mm=0, tool_diameter_mm=0,
        solver_variant=variant, seed=17, time_limit_s=3)


def original_solver(req):
    parts, _ = _build_solver_input(req)
    parts, cfg = _build_solver_config(req, parts)
    cfg.FORMAL_TOTAL_DEADLINE = time.perf_counter() + 3
    cfg.FORMAL_CLOSING_RESERVE_S = 0.3
    return RecedingHorizonMCTS(parts, cfg, req.seed)


@pytest.mark.parametrize("arm", ARMS)
def test_real_runs_have_two_initial_and_only_two_post_initial_replays(monkeypatch, arm):
    calls = []
    replay = OmnipotentDecoder.replay_order

    def observed(self, order, **kwargs):
        calls.append(kwargs["deadline_phase"])
        return replay(self, order, **kwargs)

    monkeypatch.setattr(OmnipotentDecoder, "replay_order", observed)
    result = solve_two_completion_handoff(request(), TotalBudget(3, 0.3), arm)
    evidence = result.provenance["solver_meta"]["two_completion_control"]
    assert calls == ["common_initial_greedy", "common_initial_short_edge", "completion_replay", "completion_replay"]
    assert evidence["initial_starts"] == evidence["initial_completed"] == 2
    assert evidence["completion_starts"] == evidence["completion_completed"] == 2
    assert evidence["stop_reason"] == "completion_quota"
    assert evidence["export_complete"] and len(evidence["evaluations"]) == 2
    assert not result.provenance["solver_meta"]["solver_timeout_hit"]
    assert result.provenance["solver_meta"]["oracle_completion_mode_evals"] == 2
    assert result.provenance["solver_meta"]["oracle_completion_mode_completed"] == 2
    assert [row["mode"] for row in evidence["evaluations"]] == ["short_edge", "short_edge"]
    assert result.panel_count <= evidence["initial_receipt"]["panels"]
    scores = [tuple(row["score"]) for row in [evidence["initial_receipt"], *evidence["evaluations"]]]
    assert result.score_global == max(scores)
    assert len(result.placements) == len(result.sequence) == 7


def test_reference_uses_exact_existing_two_entries_and_common_state_matches_search():
    req = request()
    original = original_solver(req)
    original._configure_solve_deadlines()
    original._prime_common_initial_solution()
    entries = [list(sequence) for _, sequence, _ in original._single_macro_actions(original.full_remaining_mask, 2)]
    left = solve_two_completion_handoff(req, TotalBudget(3, 0.3), ARMS[0])
    right = solve_two_completion_handoff(req, TotalBudget(3, 0.3), ARMS[1])
    a = left.provenance["solver_meta"]["two_completion_control"]
    b = right.provenance["solver_meta"]["two_completion_control"]
    assert [row["prefix_sequence"] for row in a["evaluations"]] == entries
    assert a["entry_state"] == b["entry_state"]
    assert a["initial_receipt"] == b["initial_receipt"]
    assert left.provenance["request"] == right.provenance["request"] == asdict(req)
    assert left.provenance["solver_meta"]["simulations"] == 0
    assert left.provenance["solver_meta"]["puct_selections"] == 0


def test_current_arm_preserves_original_search_first_two_complete_evaluations():
    req = request()
    original = original_solver(req)
    evaluate = original._evaluate_prefix_with_baseline
    observed = []

    class Enough(Exception):
        pass

    def record(*args, **kwargs):
        if len(observed) == 2:
            raise Enough()
        sequence, snapshot, score = evaluate(*args, **kwargs)
        observed.append((tuple(args[0]), _complete_solution_fingerprints(sequence, snapshot), tuple(score)))
        return sequence, snapshot, score

    original._evaluate_prefix_with_baseline = record
    with pytest.raises(Enough):
        original.solve()
    result = solve_two_completion_handoff(req, TotalBudget(3, 0.3), "current_search")
    receipts = result.provenance["solver_meta"]["two_completion_control"]["evaluations"]
    assert [(tuple(row["prefix_sequence"]), (row["sequence_sha256"], row["layout_sha256"]), tuple(row["score"]))
            for row in receipts] == observed


@pytest.mark.parametrize("arm", ARMS)
def test_second_completion_timeout_spends_credit_and_keeps_valid_incumbent(monkeypatch, arm):
    replay = OmnipotentDecoder.replay_order
    calls = []

    def fail_second(self, order, **kwargs):
        if kwargs["deadline_phase"] == "completion_replay":
            calls.append(tuple(order))
            if len(calls) == 2:
                raise BudgetExpired("completion_replay")
        return replay(self, order, **kwargs)

    monkeypatch.setattr(OmnipotentDecoder, "replay_order", fail_second)
    result = solve_two_completion_handoff(request(), TotalBudget(3, 0.3), arm)
    evidence = result.provenance["solver_meta"]["two_completion_control"]
    assert len(calls) == evidence["completion_starts"] == 2
    assert evidence["completion_completed"] == 1
    assert evidence["evaluations"][1]["status"] == "failed"
    assert evidence["stop_reason"] == "search_deadline"
    assert result.provenance["solver_meta"]["solver_timeout_hit"]
    assert len(result.placements) == 7 and result.panel_count <= evidence["initial_receipt"]["panels"]


def test_incomplete_common_initialization_is_not_a_valid_two_completion_comparison(monkeypatch):
    replay = OmnipotentDecoder.replay_order

    def fail_short(self, order, **kwargs):
        if kwargs["deadline_phase"] == "common_initial_short_edge":
            raise BudgetExpired("common_initial_short_edge")
        return replay(self, order, **kwargs)

    monkeypatch.setattr(OmnipotentDecoder, "replay_order", fail_short)
    result = solve_two_completion_handoff(request(), TotalBudget(3, 0.3), "current_search")
    evidence = result.provenance["solver_meta"]["two_completion_control"]
    assert evidence["initial_starts"] == 2 and evidence["initial_completed"] == 1
    assert evidence["completion_starts"] == evidence["completion_completed"] == 0
    assert evidence["stop_reason"] == "initial_incomplete"
    assert evidence["entry_state"] is None and evidence["initial_receipt"] is None
    assert len(result.placements) == 7


@pytest.mark.parametrize("arm,variant", [("bad", "rh_mcts_prefix_short_completion_on"),
                                         ("single_reference", "rh_mcts_prefix_short_receipt"),
                                         ("current_search", "baseline_greedy")])
def test_unsupported_experiments_reject_before_any_decoder_construction(monkeypatch, arm, variant):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported adapter request reached decoder")
    monkeypatch.setattr(OmnipotentDecoder, "__init__", forbidden)
    with pytest.raises(ValueError):
        solve_two_completion_handoff(request(variant), TotalBudget(3, 0.3), arm)


def test_absolute_close_expiry_is_not_extended_to_export_evidence(monkeypatch):
    clock = [0.0]
    replay = OmnipotentDecoder.replay_order
    count = []

    def expire_after_second(self, order, **kwargs):
        result = replay(self, order, **kwargs)
        if kwargs["deadline_phase"] == "completion_replay":
            count.append(1)
            if len(count) == 2:
                clock[0] = 10.0
        return result

    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(OmnipotentDecoder, "replay_order", expire_after_second)
    with pytest.raises(BudgetExpired):
        solve_two_completion_handoff(request(), TotalBudget(10, 1, clock=lambda: clock[0]), "single_reference")
    assert len(count) == 2


def test_cached_completion_reuse_does_not_spend_a_second_replay_credit():
    from inrp.early_prefix_control import _TwoCompletionSolver, _CompletionQuotaReached
    from inrp.mcts import _empty_snapshot

    req = request()
    parts, _ = _build_solver_input(req)
    parts, cfg = _build_solver_config(req, parts)
    cfg.FORMAL_TOTAL_DEADLINE = time.perf_counter() + 3
    cfg.FORMAL_CLOSING_RESERVE_S = 0.3
    solver = _TwoCompletionSolver(parts, cfg, req.seed, "single_reference")
    solver._configure_solve_deadlines()
    solver._prime_common_initial_solution()
    entries = solver._single_macro_actions(solver.full_remaining_mask, 2)
    for _, sequence, _ in [entries[0], entries[0], entries[1]]:
        snapshot = solver._apply_sequence(_empty_snapshot(), sequence, key=sequence[0]).snapshot
        remaining = solver.full_remaining_mask & ~solver.part_bit[sequence[0]]
        solver._evaluate_prefix_with_baseline(sequence, snapshot, remaining, prefix_depth=1)
    assert solver.completion_starts == solver.completion_completed == 2
    assert solver.meta["oracle_completion_calls"] == 3
    assert solver.meta["oracle_completion_mode_evals"] == 2
    with pytest.raises(_CompletionQuotaReached):
        solver._select_leaf(None)


def test_changed_promotion_configuration_is_rejected_before_decoding(monkeypatch):
    import inrp.early_prefix_control as adapter
    build = adapter._build_solver_config

    def unsupported(*args):
        parts, cfg = build(*args)
        cfg.PREFIX_BEAM_STAGED_PROMOTE_EVERY = 1
        return parts, cfg

    monkeypatch.setattr(adapter, "_build_solver_config", unsupported)
    monkeypatch.setattr(OmnipotentDecoder, "__init__", lambda *a, **k: pytest.fail("unsupported promotion reached decoder"))
    with pytest.raises(ValueError, match="configuration"):
        solve_two_completion_handoff(request(), TotalBudget(3, 0.3), "current_search")
