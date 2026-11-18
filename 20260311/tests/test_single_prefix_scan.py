from dataclasses import replace
import importlib
import importlib.util

import pytest

from inrp.budget import BudgetExpired, TotalBudget
from inrp.early_prefix_control import solve_two_completion_handoff
from inrp.handoff import PackingPart, PackingRequest
from inrp.mcts import OmnipotentDecoder


def module():
    assert importlib.util.find_spec("inrp.single_prefix_scan") is not None, "single-prefix scan is not implemented"
    return importlib.import_module("inrp.single_prefix_scan")


def request():
    sizes = [(15, 15), (40, 45), (20, 20), (50, 50), (25, 55), (50, 15), (30, 55)]
    return PackingRequest(case_id="scan-toy", parts=tuple(
        PackingPart(str(i), str(i), w, h) for i, (w, h) in enumerate(sizes, 1)),
        board_width_mm=100, board_height_mm=100, trim_mm=0, tool_diameter_mm=0,
        solver_variant="rh_mcts_prefix_short_completion_on", seed=17, time_limit_s=5)


def run(req=None, limit=None):
    return module().solve_single_prefix_scan_handoff(req or request(), TotalBudget(5, .5), max_evaluations=limit)


def evidence(result):
    return result.provenance["solver_meta"]["single_prefix_scan"]


def test_complete_scan_visits_every_uid_once_and_preserves_suffix():
    result = run()
    e = evidence(result)
    assert e["initial_starts"] == e["initial_completed"] == 2
    assert e["completion_starts"] == e["completion_completed"] == 7
    assert e["stop_reason"] == "candidate_exhausted"
    assert sorted(e["candidate_order"]) == list(range(1, 8))
    assert [r["prefix_sequence"][0] for r in e["evaluations"]] == e["candidate_order"]
    base = e["short_edge_order"]
    for row in e["evaluations"]:
        uid, = row["prefix_sequence"]
        assert row["sequence"] == [uid] + [x for x in base if x != uid]
        assert row["mode"] == "short_edge"
    scores = [tuple(r["score"]) for r in [e["initial_receipt"], *e["evaluations"]]]
    assert result.score_global == max(scores)
    assert result.panel_count <= e["initial_receipt"]["panels"]
    assert result.provenance["solver_meta"]["simulations"] == 0
    assert result.provenance["solver_meta"]["puct_selections"] == 0


def test_two_quota_matches_frozen_reference_initial_state_and_both_outputs():
    req = request()
    old = solve_two_completion_handoff(req, TotalBudget(5, .5), "single_reference")
    new = run(req, 2)
    a = old.provenance["solver_meta"]["two_completion_control"]
    b = evidence(new)
    assert a["entry_state"] == b["entry_state"]
    assert a["initial_receipt"] == b["initial_receipt"]
    assert a["final_receipt"] == b["final_receipt"]
    for x, y in zip(a["evaluations"], b["evaluations"]):
        for key in ("prefix_sequence", "sequence", "sequence_sha256", "layout_sha256", "score", "panels"):
            assert x[key] == y[key]
    assert b["completion_starts"] == 2 and b["stop_reason"] == "completion_quota"


def test_scout_cap_does_not_truncate_global_scan(monkeypatch):
    target = module()
    build = target._build_solver_config

    def bounded(*args, **kwargs):
        parts, cfg = build(*args, **kwargs)
        cfg.DYN_POOL_MAX = 2
        return parts, cfg

    monkeypatch.setattr(target, "_build_solver_config", bounded)
    assert evidence(run())["completion_completed"] == 7


def test_replay_counts_include_two_initial_and_all_full_candidates(monkeypatch):
    module()
    replay = OmnipotentDecoder.replay_order
    observed = []

    def record(self, order, **kwargs):
        observed.append((tuple(order), kwargs.get("deadline_phase")))
        return replay(self, order, **kwargs)

    monkeypatch.setattr(OmnipotentDecoder, "replay_order", record)
    run()
    assert len(observed) == 9
    assert [phase for _, phase in observed[:2]] == ["common_initial_greedy", "common_initial_short_edge"]
    assert all(len(order) == 7 for order, _ in observed)
    assert all(phase == "single_prefix_scan_replay" for _, phase in observed[2:])


def test_timeout_consumes_start_retains_best_and_does_not_retry(monkeypatch):
    module()
    replay = OmnipotentDecoder.replay_order
    count = 0

    def interrupted(self, order, **kwargs):
        nonlocal count
        if kwargs.get("deadline_phase") == "single_prefix_scan_replay":
            count += 1
            if count == 3:
                raise BudgetExpired("single_prefix_scan_replay")
        return replay(self, order, **kwargs)

    monkeypatch.setattr(OmnipotentDecoder, "replay_order", interrupted)
    result = run()
    e = evidence(result)
    assert e["completion_starts"] == 3 and e["completion_completed"] == 2
    assert e["stop_reason"] == "search_deadline"
    assert e["evaluations"][-1]["status"] == "timed_out"
    assert count == 3
    completed = [e["initial_receipt"], *e["evaluations"][:2]]
    assert result.score_global == max(tuple(r["score"]) for r in completed)


def test_partial_initialization_is_labeled_and_keeps_completed_floor(monkeypatch):
    module()
    replay = OmnipotentDecoder.replay_order

    def interrupted(self, order, **kwargs):
        if kwargs.get("deadline_phase") == "common_initial_short_edge":
            raise BudgetExpired("common_initial_short_edge")
        return replay(self, order, **kwargs)

    monkeypatch.setattr(OmnipotentDecoder, "replay_order", interrupted)
    e = evidence(run())
    assert e["initial_starts"] == 2 and e["initial_completed"] == 1
    assert e["initial_receipt"] is None
    assert e["final_receipt"] is not None and e["stop_reason"] == "initial_incomplete"
    assert e["completion_starts"] == 0


def test_no_initial_incumbent_propagates_timeout(monkeypatch):
    module()

    def interrupted(*args, **kwargs):
        raise BudgetExpired("common_initial_greedy")

    monkeypatch.setattr(OmnipotentDecoder, "replay_order", interrupted)
    with pytest.raises(BudgetExpired):
        run()


def test_non_timeout_error_is_not_disguised_as_valid_result(monkeypatch):
    module()
    replay = OmnipotentDecoder.replay_order

    def broken(self, order, **kwargs):
        if kwargs.get("deadline_phase") == "single_prefix_scan_replay":
            raise ValueError("invalid geometry")
        return replay(self, order, **kwargs)

    monkeypatch.setattr(OmnipotentDecoder, "replay_order", broken)
    with pytest.raises(ValueError, match="invalid geometry") as caught:
        run()
    assert caught.value.single_prefix_scan_failure["completion_starts"] == 1
    assert caught.value.single_prefix_scan_failure["completion_completed"] == 0


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_limits_are_rejected(limit):
    with pytest.raises(ValueError):
        run(limit=limit)


def test_zero_quota_returns_initial_without_candidate_replay():
    e = evidence(run(limit=0))
    assert e["completion_starts"] == e["completion_completed"] == 0
    assert e["stop_reason"] == "completion_quota"
    assert e["initial_receipt"] == e["final_receipt"]


def test_unsupported_profile_rejected():
    with pytest.raises(ValueError):
        run(replace(request(), solver_variant="baseline_greedy"))


def test_absolute_closed_budget_never_resets_to_request_limit():
    target = module()
    import time
    with pytest.raises(BudgetExpired):
        target.solve_single_prefix_scan_handoff(request(), TotalBudget(1, .1, started_at=time.perf_counter()-2))


def test_expired_export_does_not_claim_a_completed_handoff(monkeypatch):
    target = module()
    original = target._SinglePrefixScanSolver._capture

    def closed(self, capture):
        import time
        self.solve_deadline = time.perf_counter()-1
        return original(self, capture)

    monkeypatch.setattr(target._SinglePrefixScanSolver, "_capture", closed)
    with pytest.raises(BudgetExpired, match="single_prefix_scan_export") as caught:
        run(limit=2)
    assert caught.value.single_prefix_scan_failure["completion_completed"] == 2


def test_equal_shapes_retain_all_distinct_candidate_identities():
    req = replace(request(), parts=tuple(PackingPart(str(i), str(i), 15, 15) for i in range(1, 5)))
    e = evidence(run(req))
    assert e["completion_completed"] == 4
    assert sorted(r["prefix_sequence"][0] for r in e["evaluations"]) == [1, 2, 3, 4]
    for row in e["evaluations"]:
        assert sorted(row["sequence"]) == sorted(p["uid"] for p in row["placements"]) == [1, 2, 3, 4]
