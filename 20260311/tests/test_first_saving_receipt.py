from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inrp.budget import BudgetExpired, TotalBudget
from inrp.cfg import CFG
from inrp.dataio import Part
from inrp.handoff import PackingPart, PackingRequest, _convert_result, solve_packing_handoff_with_budget
import inrp.mcts as module
from inrp.mcts import MacroAction, RecedingHorizonMCTS, SearchNode, _empty_snapshot, _complete_solution_fingerprints
from inrp.runner import _variant_cfg


PARENT = "rh_mcts_prefix_short_completion_on"
DIAGNOSTIC = "rh_mcts_prefix_short_receipt"
SIZES = [(15, 15), (40, 45), (20, 20), (50, 50), (25, 55), (50, 15), (30, 55)]
LARGER = [(35, 20), (55, 35), (20, 40), (25, 65), (50, 70),
          (50, 20), (65, 20), (60, 30), (45, 25), (55, 60)]


def _solver(*, enabled=True, larger=False, promote_every=4, sizes=None):
    sizes = sizes if sizes is not None else (LARGER if larger else SIZES)
    cfg = CFG(BOARD_W=100, BOARD_H=100, TRIM=0, TOOL_D=0, GAP=0,
              BINARY_SPACING=False, SAFE_GAP=0, SOLVER_TIME_LIMIT_S=0.05)
    cfg = _variant_cfg(cfg, PARENT, {"n_parts": len(sizes)})
    cfg.PREFIX_FIRST_SAVING_RECEIPT_ENABLE = enabled
    cfg.PREFIX_BEAM_STAGED_PROMOTE_EVERY = promote_every
    cfg.FORMAL_TOTAL_DEADLINE = time.perf_counter() + 5
    cfg.FORMAL_CLOSING_RESERVE_S = 0.1
    parts = [Part(i, str(i), w, h, w, h) for i, (w, h) in enumerate(sizes, 1)]
    solver = RecedingHorizonMCTS(parts, cfg, 17)
    solver._configure_solve_deadlines()
    solver._prime_common_initial_solution()
    return solver


def _child(solver, uid, root=None):
    root = root or SearchNode(snapshot=_empty_snapshot(), remaining_mask=solver.full_remaining_mask)
    root.action_meta[uid] = MacroAction(uid, "single", (uid,), (uid,), 1.0, (), "single")
    return solver._expand(root, uid)


def _evaluate(solver, child):
    return solver._evaluate_prefix_with_baseline(
        child.prefix_sequence, child.snapshot, child.remaining_mask, prefix_depth=child.depth,
        receipt_macro_chain=child.receipt_macro_chain)


def _forbid(*args, **kwargs):
    raise AssertionError("receipt export must not replay or search")


def test_common_initializer_improvement_is_excluded_and_first_real_search_saving_is_exact():
    solver = _solver()
    assert solver.meta["common_initial_candidate_panels"] == {"greedy": 2, "short_edge": 2}
    assert solver.meta["first_saving_receipt"] is None
    child = _child(solver, 2)
    sequence, snapshot, _ = _evaluate(solver, child)
    assert sequence == (2, 4, 7, 5, 3, 6, 1) and len(snapshot.boards) == 1
    receipt = solver.meta["first_saving_receipt"]
    assert receipt["initial_panel_count"] == 2 and receipt["panel_count"] == 1
    assert receipt["prefix_sequence"] == [2] and receipt["prefix_piece_count"] == receipt["prefix_depth"] == 1
    assert receipt["macro_chain"] == [{"kind": "single", "source": "single", "sequence": [2]}]
    assert receipt["context_complete"] is True
    assert receipt["completion_mode"] == "short_edge" and receipt["completion_origin"] == "fresh_mode_evaluation"
    assert receipt["sequence"] == list(sequence)
    assert receipt["event"] == solver.meta["global_best_events"][receipt["event_index"]]
    assert receipt["event_index"] == next(i for i, e in enumerate(solver.meta["global_best_events"]) if e["panel_count"] < 2)


def test_default_off_has_no_receipt_payload_or_diagnostic_path_work():
    solver = _solver(enabled=False)
    child = _child(solver, 2)
    sequence, snapshot, _ = solver._evaluate_prefix_with_baseline(
        child.prefix_sequence, child.snapshot, child.remaining_mask, prefix_depth=child.depth)
    assert sequence == (2, 4, 7, 5, 3, 6, 1) and len(snapshot.boards) == 1
    assert "first_saving_receipt" not in solver.meta
    assert child.receipt_macro_chain == ()


def test_enabling_diagnostics_preserves_initial_rng_cache_and_search_settings():
    off, on = _solver(enabled=False), _solver()
    assert off.rng.getstate() == on.rng.getstate()
    assert off.decoder.micro_cache.keys() == on.decoder.micro_cache.keys()
    assert off.decoder.decode_cache.keys() == on.decoder.decode_cache.keys()
    assert off.global_best_sequence == on.global_best_sequence
    assert off.global_best_score == on.global_best_score
    for key in ("common_initial_sequence_sha256", "common_initial_layout_sha256", "common_initial_score"):
        assert off.meta[key] == on.meta[key]


def test_greedy_to_strong_initial_saving_is_not_recorded_as_search_saving():
    solver = _solver(sizes=[(30, 50), (15, 35), (55, 50), (30, 35), (20, 70), (15, 75)])
    assert solver.meta["common_initial_candidate_panels"] == {"greedy": 2, "short_edge": 1}
    assert solver.meta["first_saving_receipt"] is None


def test_diagnostic_work_does_not_change_rng_cache_or_completed_prefix_results():
    off, on = _solver(enabled=False, larger=True), _solver(larger=True)
    for uid in (4, 6):
        a, b = _child(off, uid), _child(on, uid)
        left = off._evaluate_prefix_with_baseline(a.prefix_sequence, a.snapshot, a.remaining_mask, prefix_depth=a.depth)
        right = _evaluate(on, b)
        assert left[0] == right[0] and left[2] == right[2]
    assert off.rng.getstate() == on.rng.getstate()
    assert off.decoder.micro_cache.keys() == on.decoder.micro_cache.keys()
    assert off.decoder.decode_cache.keys() == on.decoder.decode_cache.keys()
    assert off.oracle_completion_cache.keys() == on.oracle_completion_cache.keys()
    for key in ("oracle_completion_mode_evals", "oracle_completion_cache_hits", "staged_primary_completed"):
        assert off.meta[key] == on.meta[key]


@pytest.mark.parametrize("disabled", ["PREFIX_COMMON_INIT_ABLATION", "PREFIX_ABLATION_SEARCH_ENABLE",
                                      "PREFIX_BEAM_STAGED_ENABLE", "PREFIX_BEAM_ENABLE"])
def test_diagnostic_rejects_incompatible_pipelines(disabled):
    cfg = _variant_cfg(CFG(), PARENT, {"n_parts": 7})
    cfg.PREFIX_FIRST_SAVING_RECEIPT_ENABLE = True
    setattr(cfg, disabled, False)
    parts = [Part(i, str(i), w, h, w, h) for i, (w, h) in enumerate(SIZES, 1)]
    with pytest.raises(ValueError, match="first-saving"):
        RecedingHorizonMCTS(parts, cfg, 17)


def test_later_strictly_better_incumbent_does_not_overwrite_first_snapshot_or_receipt(monkeypatch):
    solver = _solver(larger=True)
    first_seq, first_snapshot, first_score = _evaluate(solver, _child(solver, 4))
    assert solver.meta["common_initial_panels"] == 3 and len(first_snapshot.boards) == 2
    assert first_seq == (4, 10, 5, 2, 8, 9, 7, 6, 3, 1)
    before = json.loads(json.dumps(solver.meta["first_saving_receipt"]))
    later_seq, later_snapshot, later_score = _evaluate(solver, _child(solver, 6))
    assert len(later_snapshot.boards) == 2 and later_score > first_score and later_seq != first_seq
    assert solver.global_best_sequence == later_seq
    assert solver.meta["first_saving_receipt"] == before
    monkeypatch.setattr(solver.decoder, "replay_order", _forbid)
    result = solver._formal_result_from_incumbent()
    receipt = json.loads(json.dumps(result.meta))["first_saving_receipt"]
    assert result.sequence == later_seq and receipt["sequence"] == list(first_seq)
    assert receipt["export_complete"] is True and len(receipt["placements"]) == 10
    assert (receipt["sequence_sha256"], receipt["layout_sha256"]) == _complete_solution_fingerprints(first_seq, first_snapshot)
    assert receipt["score"] == list(first_score)


def test_macro_ancestry_survives_root_detachment():
    solver = _solver()
    first = _child(solver, 4)
    solver._detach_root(first)
    assert first.parent is None
    second = _child(solver, 2, first)
    assert second.prefix_sequence == (4, 2) and second.depth == 2
    assert second.receipt_macro_chain == (("single", "single", (4,)), ("single", "single", (2,)))


def test_ranking_time_completion_supplies_actual_macro_context():
    solver = _solver()
    child = _child(solver, 2)
    solver._prefix_beam_root_rank(child)
    receipt = solver.meta["first_saving_receipt"]
    assert receipt["prefix_sequence"] == [2] and receipt["context_complete"] is True


@pytest.mark.parametrize("full_group", [False, True])
def test_cached_winning_mode_is_not_mislabeled_fresh(full_group):
    solver = _solver(promote_every=1 if full_group else 4)
    child = _child(solver, 2)
    solver._complete_with_baseline(child.snapshot, child.remaining_mask, prefix_depth=child.depth)
    assert solver.meta["first_saving_receipt"] is None
    _evaluate(solver, child)
    receipt = solver.meta["first_saving_receipt"]
    assert receipt["completion_origin"] == "completed_mode_cache"
    assert receipt["cache_evidence"]["new_mode_evaluations"] == 0
    assert receipt["cache_evidence"]["full_group_cache_hit"] is full_group
    assert receipt["cache_evidence"]["staged_entry_reused"] is not full_group


def test_cached_winner_stays_cached_when_other_modes_are_evaluated_on_promotion():
    solver = _solver()
    child = _child(solver, 2)
    solver._complete_with_baseline(child.snapshot, child.remaining_mask, prefix_depth=child.depth)
    solver.staged_promote_every = 1  # Exercise a cached primary with two fresh fallback evaluations.
    _evaluate(solver, child)
    receipt = solver.meta["first_saving_receipt"]
    assert receipt["completion_mode"] == "short_edge"
    assert receipt["completion_origin"] == "completed_mode_cache"
    assert receipt["cache_evidence"]["new_mode_evaluations"] == 2
    assert receipt["cache_evidence"]["winning_mode_evaluated_now"] is False


def test_missing_macro_context_is_reported_incomplete_instead_of_invented():
    solver = _solver()
    child = _child(solver, 2)
    sequence, snapshot, _ = solver._complete_with_baseline(child.snapshot, child.remaining_mask, prefix_depth=1)
    solver._update_global_best((2,) + sequence, snapshot, source="prefix_completion",
                               completion_mode="short_edge", prefix_depth=1, record_archive=False)
    receipt = solver.meta["first_saving_receipt"]
    assert receipt["context_complete"] is False
    assert receipt["prefix_sequence"] is None and receipt["macro_chain"] is None


def test_receipt_close_cannot_extend_the_absolute_total_deadline(monkeypatch):
    solver = _solver()
    _evaluate(solver, _child(solver, 2))
    solver.solve_deadline = 10.0
    ticks = iter((0.0, 0.0, 0.0, 11.0, 11.0))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks, 11.0))
    with pytest.raises(BudgetExpired, match="receipt_export"):
        solver._formal_result_from_incumbent()


@pytest.mark.parametrize("count", [20, 178, 739, 2416])
def test_opt_in_profile_changes_only_identity_and_receipt_flag(count):
    parent = _variant_cfg(CFG(), PARENT, {"n_parts": count})
    diagnostic = _variant_cfg(CFG(), DIAGNOSTIC, {"n_parts": count})
    assert diagnostic.PREFIX_FIRST_SAVING_RECEIPT_ENABLE is True
    assert parent.PREFIX_FIRST_SAVING_RECEIPT_ENABLE is False
    assert vars(parent).keys() == vars(diagnostic).keys()
    assert {k for k in vars(parent) if getattr(parent, k) != getattr(diagnostic, k)} == {
        "VARIANTS", "PREFIX_FIRST_SAVING_RECEIPT_ENABLE"}


def _receipt_payload():
    solver = _solver(larger=True)
    _evaluate(solver, _child(solver, 4))
    _evaluate(solver, _child(solver, 6))
    result = solver._formal_result_from_incumbent()
    request = PackingRequest(case_id="receipt-toy", parts=tuple(
        PackingPart(str(i), str(i), w, h) for i, (w, h) in enumerate(LARGER, 1)),
        board_width_mm=100, board_height_mm=100, trim_mm=0, tool_diameter_mm=0,
        seed=17, time_limit_s=5, solver_variant=DIAGNOSTIC)
    by_uid = {i: part for i, part in enumerate(request.parts, 1)}
    converted = _convert_result(request, result, by_uid, 0.1)
    return {"request": asdict(request), "handoff": asdict(converted)}


def test_raw_receipt_is_json_safe_through_existing_handoff_and_exports_all_parts():
    payload = json.loads(json.dumps(_receipt_payload()))
    receipt = payload["handoff"]["provenance"]["solver_meta"]["first_saving_receipt"]
    assert sorted(p["uid"] for p in receipt["placements"]) == list(range(1, 11))
    assert receipt["schema_version"] == "inrp-first-saving-receipt-v1"
    assert receipt["export_complete"] is True


def test_real_budgeted_diagnostic_handoff_delivers_floor_and_enabled_metadata():
    request = PackingRequest(case_id="receipt-budgeted", parts=tuple(
        PackingPart(str(i), str(i), w, h) for i, (w, h) in enumerate(SIZES, 1)),
        board_width_mm=100, board_height_mm=100, trim_mm=0, tool_diameter_mm=0,
        seed=17, time_limit_s=0.1, solver_variant=DIAGNOSTIC)
    result = solve_packing_handoff_with_budget(request, TotalBudget(0.8, 0.08))
    assert result.feasible and result.panel_count <= 2
    assert result.provenance["solver_meta"]["first_saving_receipt_enabled"] is True
