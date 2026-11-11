from __future__ import annotations

import hashlib
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
from inrp.packer import board_signature
from inrp.runner import _variant_cfg


ON = "rh_mcts_prefix_common_init_on"
OFF = "rh_mcts_prefix_common_init_off"
SIZES = [(30, 50), (15, 35), (55, 50), (30, 35), (20, 70), (15, 75)]


def _solver(enabled, *, formal=True):
    cfg = CFG(BOARD_W=100, BOARD_H=100, TRIM=0, TOOL_D=0, BINARY_SPACING=False,
              SAFE_GAP=0, SOLVER_TIME_LIMIT_S=0.05)
    cfg = _variant_cfg(cfg, "rh_mcts_prefix_beam_staged", {"n_parts": 6})
    cfg.PREFIX_COMMON_INIT_ABLATION = True
    cfg.PREFIX_ABLATION_SEARCH_ENABLE = enabled
    if formal:
        cfg.FORMAL_TOTAL_DEADLINE = time.perf_counter() + 1
        cfg.FORMAL_CLOSING_RESERVE_S = 0.1
    parts = [Part(i, str(i), w, h, w, h) for i, (w, h) in enumerate(SIZES, 1)]
    return RecedingHorizonMCTS(parts, cfg, 17)


def _forbid_optional(*args, **kwargs):
    raise AssertionError("ablation entered an optional or alternative pipeline")


def test_off_delivers_complete_strong_seed_without_any_search_or_other_warmstarts(monkeypatch):
    solver = _solver(False)
    for method in ("_prime_constructive_warmstarts", "_warm_start", "_solve_prefix_beam", "_solve_beam",
                   "_sequence_sa_lns", "_memetic_tail_search", "_global_pattern_master_repair"):
        monkeypatch.setattr(solver, method, _forbid_optional)
    result = solver.solve()
    assert len(result.boards) == 1
    assert result.sequence == (3, 1, 4, 5, 6, 2)
    assert sorted(p.uid for b in result.boards for p in b.placed) == [1, 2, 3, 4, 5, 6]
    assert result.meta["common_initial_candidate_panels"] == {"greedy": 2, "short_edge": 1}
    assert result.meta["common_initial_complete"] is True
    assert result.meta["common_initial_panels"] == 1
    assert result.meta["common_initial_sequence_sha256"] == hashlib.sha256(repr((3, 1, 4, 5, 6, 2)).encode()).hexdigest()
    assert result.meta["ablation_final_sequence_sha256"] == result.meta["common_initial_sequence_sha256"]
    assert result.meta["ablation_final_layout_sha256"] == result.meta["common_initial_layout_sha256"]
    assert result.meta["result_source"] == "common_initial_search_disabled"
    assert result.meta["simulations"] == result.meta["oracle_completion_calls"] == 0
    assert not result.meta["solver_timeout_hit"]


def test_on_and_off_have_identical_initial_geometry_score_rng_and_cache_state(monkeypatch):
    off, on = _solver(False), _solver(True)
    captured = {}

    def enter_prefix(root):
        captured.update({"sequence": on.global_best_sequence,
                         "boards": [board_signature(b) for b in on.global_best_snapshot.boards],
                         "score": on.global_best_score, "rng": on.rng.getstate(),
                         "cache_sizes": (len(on.decoder.micro_cache), len(on.decoder.decode_cache))})
        return on._formal_result_from_incumbent()

    monkeypatch.setattr(on, "_prime_constructive_warmstarts", _forbid_optional)
    monkeypatch.setattr(on, "_solve_prefix_beam", enter_prefix)
    before = off.solve()
    after = on.solve()
    for key in ("common_initial_sequence_sha256", "common_initial_layout_sha256", "common_initial_score", "common_initial_panels"):
        assert before.meta[key] == after.meta[key]
    assert captured == {"sequence": before.sequence, "boards": [board_signature(b) for b in before.boards],
                        "score": before.score_global, "rng": off.rng.getstate(),
                        "cache_sizes": (len(off.decoder.micro_cache), len(off.decoder.decode_cache))}


@pytest.mark.parametrize("enabled", [False, True])
def test_second_initial_order_timeout_retains_greedy_but_invalidates_ablation(monkeypatch, enabled):
    monkeypatch.setattr(module.time, "perf_counter", lambda: 0.0)
    solver = _solver(enabled)
    solver.formal_total_deadline = 10.0
    solver.formal_closing_reserve_s = 2.0
    replay = solver.decoder.replay_order
    calls = []

    def interrupt_second(order, **kwargs):
        assert kwargs["deadline"] == 8.0
        calls.append(tuple(order))
        if len(calls) == 2:
            replay(order[:2], **kwargs)
            raise BudgetExpired("common_initial_short_edge")
        return replay(order, **kwargs)

    monkeypatch.setattr(solver.decoder, "replay_order", interrupt_second)
    monkeypatch.setattr(solver, "_solve_prefix_beam", _forbid_optional)
    result = solver.solve()
    assert len(calls) == 2
    assert len(result.boards) == 2
    assert result.meta["common_initial_complete"] is False
    assert result.meta["common_initial_candidate_panels"] == {"greedy": 2}
    assert result.meta["common_initial_failure_phase"] == "common_initial_short_edge"
    assert result.meta["simulations"] == 0
    assert sorted(p.uid for b in result.boards for p in b.placed) == [1, 2, 3, 4, 5, 6]


def test_initial_expiry_without_a_complete_seed_propagates(monkeypatch):
    solver = _solver(False)
    monkeypatch.setattr(solver.decoder, "replay_order", lambda *a, **kw: (_ for _ in ()).throw(BudgetExpired("common_initial_greedy")))
    with pytest.raises(BudgetExpired):
        solver.solve()


def test_search_interruption_preserves_complete_initial_geometry_and_score(monkeypatch):
    solver = _solver(True)
    captured = {}

    def interrupted_search(root):
        captured.update({"boards": [board_signature(b) for b in solver.global_best_snapshot.boards],
                         "score": solver.global_best_score, "sequence": solver.global_best_sequence})
        solver.decoder.replay_order((2, 1))
        raise BudgetExpired("completion_replay")

    monkeypatch.setattr(solver, "_solve_prefix_beam", interrupted_search)
    result = solver.solve()
    assert len(result.boards) == 1
    assert result.sequence == captured["sequence"]
    assert result.score_global == captured["score"]
    assert [board_signature(b) for b in result.boards] == captured["boards"]
    assert result.meta["common_initial_complete"] is True


def test_ablation_requires_an_absolute_total_deadline():
    solver = _solver(False, formal=False)
    with pytest.raises(ValueError, match="absolute total deadline"):
        solver.solve()


def test_ablation_rejects_non_prefix_mode_before_decoding(monkeypatch):
    solver = _solver(False)
    solver.prefix_beam_enable = False
    monkeypatch.setattr(solver.decoder, "replay_order", _forbid_optional)
    with pytest.raises(ValueError, match="prefix"):
        solver.solve()


@pytest.mark.parametrize("count", [20, 178, 739, 2416])
def test_profiles_differ_only_in_identity_and_the_search_gate(count):
    on = _variant_cfg(CFG(), ON, {"n_parts": count})
    off = _variant_cfg(CFG(), OFF, {"n_parts": count})
    assert on.PREFIX_COMMON_INIT_ABLATION and off.PREFIX_COMMON_INIT_ABLATION
    assert on.PREFIX_ABLATION_SEARCH_ENABLE is True
    assert off.PREFIX_ABLATION_SEARCH_ENABLE is False
    assert on.PREFIX_BEAM_ENABLE and off.PREFIX_BEAM_ENABLE
    for key in vars(on):
        if key not in {"VARIANTS", "PREFIX_ABLATION_SEARCH_ENABLE"}:
            assert getattr(on, key) == getattr(off, key)


def test_existing_staged_variant_still_uses_its_original_initializer(monkeypatch):
    solver = _solver(True)
    solver.cfg.PREFIX_COMMON_INIT_ABLATION = False
    def original_initializer():
        raise RuntimeError("original initializer reached")
    monkeypatch.setattr(solver, "_prime_constructive_warmstarts", original_initializer)
    with pytest.raises(RuntimeError, match="original initializer reached"):
        solver.solve()


@pytest.mark.parametrize("variant", [ON, OFF])
def test_real_budgeted_handoff_retains_common_strong_floor(variant):
    request = PackingRequest(case_id="common-init", parts=tuple(
        PackingPart(str(i), str(i), w, h) for i, (w, h) in enumerate(SIZES, 1)),
        board_width_mm=100, board_height_mm=100, trim_mm=0, tool_diameter_mm=0,
        solver_variant=variant, time_limit_s=0.05)
    result = solve_packing_handoff_with_budget(request, TotalBudget(0.3, 0.03))
    assert result.feasible and result.panel_count == 1
    assert result.provenance["solver_meta"]["common_initial_complete"] is True
