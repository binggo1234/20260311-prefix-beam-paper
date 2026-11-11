from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inrp.budget import BudgetExpired, TotalBudget
from inrp.cfg import CFG
from inrp.handoff import (
    PackingPart, PackingRequest, _build_solver_input,
    solve_packing_handoff_with_budget,
)
from inrp.mcts import OmnipotentDecoder
from inrp.ordering import baseline_sequence
from inrp.runner import _variant_cfg


VARIANTS = ("baseline_order_multistart", "baseline_sequence_sa_budgeted")


def _module(name):
    assert importlib.util.find_spec(name) is not None, f"missing evaluation feature: {name}"
    return importlib.import_module(name)


def _request(**overrides):
    request = PackingRequest(
        case_id="yardstick", board_width_mm=100, board_height_mm=100,
        trim_mm=5, tool_diameter_mm=2, time_limit_s=0.05,
        parts=(PackingPart("a", "a", 44, 88), PackingPart("b", "b", 44, 88)),
    )
    return replace(request, **overrides)


def test_envelope_area_bound_accounts_for_trim_and_clearance():
    bounds = _module("inrp.evaluation_bounds").packing_area_bounds(_request())
    assert bounds["nominal_area_mm2"] == 7744
    assert bounds["envelope_area_mm2"] == 8280
    assert bounds["nominal_area_lower_bound"] == 1
    assert bounds["envelope_area_lower_bound"] == bounds["panel_lower_bound"] == 2
    assert bounds["net_utilization_ceiling"] == pytest.approx(0.3872)


def test_decimal_boundary_does_not_add_a_spurious_board():
    request = _request(board_width_mm=10, board_height_mm=10, trim_mm=0,
                       tool_diameter_mm=0.1, parts=(PackingPart("a", "a", 9.9, 9.9),))
    bounds = _module("inrp.evaluation_bounds").packing_area_bounds(request)
    assert bounds["envelope_area_mm2"] == 100
    assert bounds["panel_lower_bound"] == 1


@pytest.mark.parametrize("overrides", [
    {"board_width_mm": float("nan")}, {"trim_mm": 50},
    {"tool_diameter_mm": -1}, {"parts": ()},
    {"parts": (PackingPart("a", "a", float("inf"), 2),)},
    {"parts": (PackingPart("a", "a", 91, 10),)},
])
def test_bounds_reject_invalid_or_unfit_geometry(overrides):
    with pytest.raises(ValueError):
        _module("inrp.evaluation_bounds").packing_area_bounds(_request(**overrides))


def test_bounds_respect_rotation_only_part():
    bounds_module = _module("inrp.evaluation_bounds")
    request = _request(board_width_mm=60, board_height_mm=100,
                       parts=(PackingPart("a", "a", 80, 40, (90,)),))
    assert bounds_module.packing_area_bounds(request)["panel_lower_bound"] == 1
    with pytest.raises(ValueError):
        bounds_module.packing_area_bounds(replace(request, allow_rotation=False))


@pytest.mark.parametrize("variant", VARIANTS)
def test_budgeted_reference_handoff_returns_complete_greedy_or_better(variant):
    _module("inrp.evaluation_baselines")
    result = solve_packing_handoff_with_budget(_request(solver_variant=variant), TotalBudget(0.2, 0.02))
    assert result.feasible and result.panel_count == 2
    assert set(result.sequence) == {"a", "b"}
    assert len(result.placements) == 2
    meta = result.provenance["solver_meta"]
    assert meta["baseline_completed_evaluations"] >= 1
    assert meta["baseline_attempted_evaluations"] == (
        meta["baseline_completed_evaluations"] + meta["baseline_interrupted_evaluations"])
    assert meta["baseline_best_panels"] <= meta["greedy_floor_panels"]


@pytest.mark.parametrize("count", [178, 739, 2416])
def test_references_share_staged_decoder_configuration(count):
    _module("inrp.evaluation_baselines")
    staged = _variant_cfg(CFG(), "rh_mcts_prefix_beam_staged", {"n_parts": count})
    for variant in VARIANTS:
        cfg = _variant_cfg(CFG(), variant, {"n_parts": count})
        for key in ("BOARD_W", "BOARD_H", "TRIM", "GAP", "BINARY_SPACING", "SAFE_GAP",
                    "TOOL_D", "ALLOW_ROT", "BASELINE_PLACE_MODE", "UTIL_GAMMA", "STATE_SIG_ND"):
            assert getattr(cfg, key) == getattr(staged, key)
        assert cfg.MAIN_PROFILE_TIER == staged.MAIN_PROFILE_TIER
        assert cfg.SOLVER_KIND == variant


@pytest.mark.parametrize("variant", VARIANTS)
def test_optional_partial_replay_cannot_replace_completed_incumbent(monkeypatch, variant):
    module = _module("inrp.evaluation_baselines")
    request = _request(parts=tuple(PackingPart(str(i), str(i), 8 + i, 5) for i in range(1, 6)))
    parts, _ = _build_solver_input(request)
    cfg = _variant_cfg(CFG(), variant, {"n_parts": len(parts)})
    cfg.BOARD_W = cfg.BOARD_H = 100
    cfg.BINARY_SPACING = False
    cfg.SAFE_GAP = 0
    cfg.SOLVER_TIME_LIMIT_S = 100
    cfg.FORMAL_TOTAL_DEADLINE = 10
    cfg.FORMAL_CLOSING_RESERVE_S = 2
    now = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    real_decode = OmnipotentDecoder.decode
    count = [0]

    def interrupted_decode(self, snapshot, action, **kwargs):
        assert kwargs["deadline"] == 8
        out = real_decode(self, snapshot, action, **kwargs)
        count[0] += 1
        if count[0] == 7:
            now[0] = 8.0
        return out

    monkeypatch.setattr(OmnipotentDecoder, "decode", interrupted_decode)
    result = module.solve_evaluation_baseline(parts, cfg, 17)
    assert result.sequence == baseline_sequence(parts)
    assert sorted(p.uid for b in result.boards for p in b.placed) == [1, 2, 3, 4, 5]
    assert result.meta["baseline_completed_evaluations"] == 1
    assert result.meta["baseline_interrupted_evaluations"] == 1


def test_initial_replay_expiry_propagates_without_partial_result(monkeypatch):
    module = _module("inrp.evaluation_baselines")
    parts, _ = _build_solver_input(_request())
    cfg = _variant_cfg(CFG(), VARIANTS[0])
    cfg.FORMAL_TOTAL_DEADLINE = 10
    cfg.FORMAL_CLOSING_RESERVE_S = 2
    monkeypatch.setattr(module.time, "perf_counter", lambda: 8.0)
    with pytest.raises(BudgetExpired):
        module.solve_evaluation_baseline(parts, cfg, 1)


def test_global_sequence_moves_preserve_permutations_and_reach_the_tail():
    module = _module("inrp.evaluation_baselines")
    sequence = tuple(range(100))
    left, right = random.Random(42), random.Random(42)
    proposals = [module._sequence_proposal(sequence, left) for _ in range(100)]
    assert proposals == [module._sequence_proposal(sequence, right) for _ in range(100)]
    assert all(sorted(proposal) == list(sequence) for proposal in proposals)
    assert any(proposal[:80] == sequence[:80] and proposal[80:] != sequence[80:] for proposal in proposals)
