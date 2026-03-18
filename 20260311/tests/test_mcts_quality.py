import json
from pathlib import Path
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.cfg import CFG
from inrp.dataio import Part, read_sample_parts
from inrp.geometry import resolve_geometry_policy
from inrp.mcts import (
    AppliedActionResult,
    LayoutSnapshot,
    MacroAction,
    PatternCandidate,
    PersistentBoards,
    RecedingHorizonMCTS,
    RegionProposal,
    RepairWindow,
    SearchNode,
    solve_with_receding_horizon_mcts,
)
from inrp.packer import make_empty_board
from inrp.preprocess import preprocess_parts
from inrp.runner import (
    _adaptive_main_profile,
    _area_lower_bound,
    _case_stats,
    _effective_board_area,
    _load_cached_seed_totals,
    _solve_baseline_sa,
    _variant_cfg,
    aggregate_totals,
    run_one_seed,
    run_stepF,
)
from inrp.validate import validate_solution
from run_autonomous_beam_loop import _choose_next_bundle, _promotion_ready


def _cfg() -> CFG:
    cfg = CFG()
    cfg.BOARD_W = 10.0
    cfg.BOARD_H = 10.0
    cfg.TRIM = 0.0
    cfg.GAP = 0.0
    cfg.TOOL_D = 0.0
    cfg.SAFE_GAP = 0.0
    cfg.BINARY_SPACING = False
    cfg.MCTS_N_SIM = 1
    cfg.COMMIT_MIN_ROOT_VISITS = 2
    cfg.COMMIT_MIN_CHILD_VISITS = 1
    cfg.COMMIT_CONFIRM_PASSES = 1
    cfg.COMMIT_FORCE_ENABLE = False
    cfg.COMMIT_TIMEOUT_FORCE = False
    cfg.SOLVER_TIME_LIMIT_S = 5.0
    cfg.ELITE_ARCHIVE_K = 2
    cfg.ELITE_TAIL_LEN = 2
    cfg.REPAIR_ELITE_TOPK = 2
    cfg.REPAIR_TAIL_BOARDS = 2
    cfg.REPAIR_BLOCKER_WINDOW = 8
    cfg.REPAIR_BLOCKER_TOPK = 2
    cfg.REPAIR_BEAM_WIDTH = 4
    cfg.REPAIR_NODE_LIMIT = 32
    cfg.REPAIR_ACTIONS_PER_STATE = 6
    cfg.REPAIR_PASS_TIME_SLICE_S = 0.2
    cfg.LOCAL_MAX_ITERS = 4
    cfg.LOCAL_FAIL_LIMIT = 2
    return cfg


def _parts() -> List[Part]:
    dims = [(6, 4), (4, 6), (4, 4), (3, 3), (2, 5), (5, 2)]
    return [
        Part(uid=i + 1, pid_raw=str(i + 1), w=float(w), h=float(h), w0=float(w), h0=float(h))
        for i, (w, h) in enumerate(dims)
    ]


def _dummy_snapshot(board_count: int, sum_u_gamma: float) -> LayoutSnapshot:
    boards = PersistentBoards.empty()
    for bid in range(board_count):
        boards = boards.append(make_empty_board(bid + 1, 10.0, 10.0, trim=0.0, safe_gap=0.0, touch_tol=1e-6))
    return LayoutSnapshot(
        boards=boards,
        sum_u_gamma=float(sum_u_gamma),
        layout_hash=(int(board_count), int(sum_u_gamma * 1000)),
        ordered_hash=int(board_count * 97 + sum_u_gamma * 1000),
    )


def _window(sequence, *, start_idx: int = 2, base_board_count: int = 2, blockers=()) -> RepairWindow:
    seq_t = tuple(int(pid) for pid in sequence)
    movable = seq_t[start_idx:]
    movable_set = set(movable)
    movable_set.update(int(pid) for pid in blockers)
    fixed_prefix = tuple(pid for pid in seq_t if pid not in movable_set)
    movable_ids = tuple(pid for pid in seq_t if pid in movable_set)
    return RepairWindow(
        start_idx=start_idx,
        start_board=1,
        span=len(movable_ids),
        movable_part_ids=movable_ids,
        blocker_part_ids=tuple(int(pid) for pid in blockers),
        fixed_prefix_sequence=fixed_prefix,
        base_board_count=base_board_count,
        score=(0.0, 0.0, 0.0, float(start_idx)),
        source="test",
    )


def test_elite_archive_replaces_same_tail_signature_with_better_candidate():
    mcts = RecedingHorizonMCTS(_parts(), _cfg(), seed=1)
    weaker = _dummy_snapshot(board_count=3, sum_u_gamma=1.5)
    stronger = _dummy_snapshot(board_count=2, sum_u_gamma=0.9)

    mcts._record_elite_candidate((1, 2, 3, 4), weaker)
    mcts._record_elite_candidate((5, 6, 3, 4), stronger)

    assert len(mcts.elite_archive) == 1
    assert mcts.elite_archive[0].sequence == (5, 6, 3, 4)


def test_repair_accepts_lexicographic_priority():
    mcts = RecedingHorizonMCTS(_parts(), _cfg(), seed=1)

    assert mcts._repair_accepts((-3.0, 0.8), (-2.0, 0.1))
    assert mcts._repair_accepts((-2.0, 0.1), (-2.0, 0.2))
    assert not mcts._repair_accepts((-2.0, 0.2), (-3.0, 0.99))


def test_board_closing_repair_preserves_parts_and_legality():
    parts = _parts()
    cfg = _cfg()
    mcts = RecedingHorizonMCTS(parts, cfg, seed=1)
    base_sequence = tuple(part.uid for part in parts)
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)

    repaired_sequence, repaired_snapshot = mcts._board_closing_repair(base_sequence, base_snapshot)

    assert sorted(repaired_sequence) == sorted(base_sequence)
    validate_solution(
        parts,
        list(repaired_snapshot.boards),
        cfg.BOARD_W,
        cfg.BOARD_H,
        trim=cfg.TRIM,
        safe_gap=cfg.SAFE_GAP,
        touch_tol=cfg.TOUCH_TOL,
    )


def test_memetic_tail_search_returns_original_for_single_part():
    cfg = _cfg()
    part = [Part(uid=1, pid_raw="1", w=4.0, h=4.0, w0=4.0, h0=4.0)]
    mcts = RecedingHorizonMCTS(part, cfg, seed=1)
    base_sequence = (1,)
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)

    polished_sequence, polished_snapshot = mcts._memetic_tail_search(base_sequence, base_snapshot)

    assert polished_sequence == base_sequence
    assert len(polished_snapshot.boards) == 1


def test_region_proposals_are_sorted_and_unique():
    cfg = _cfg()
    cfg.BOARD_W = 7.0
    cfg.BOARD_H = 7.0
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)

    assert len(base_snapshot.boards) >= 2

    proposals = mcts._region_proposals(base_snapshot, base_sequence)

    assert proposals
    assert len(proposals) <= cfg.REGION_PROPOSAL_TOPK
    assert len({proposal.start_idx for proposal in proposals}) == len(proposals)
    assert proposals == sorted(proposals, key=lambda item: (item.score, item.start_idx), reverse=True)


def test_region_rebuild_keeps_prefix_and_preserves_legality():
    cfg = _cfg()
    cfg.BOARD_W = 7.0
    cfg.BOARD_H = 7.0
    cfg.REGION_REBUILD_NODE_LIMIT = 24
    cfg.REGION_REBUILD_ACTIONS_PER_STATE = 8
    cfg.REGION_REBUILD_TIME_SLICE_S = 0.2
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, base_snapshots = mcts.decoder.replay_order(base_sequence)
    proposals = mcts._region_proposals(base_snapshot, base_sequence)

    assert proposals

    start_idx = min(len(base_sequence) - 1, max(1, proposals[0].start_idx))
    proposal = RegionProposal(
        start_idx=start_idx,
        span=len(base_sequence) - start_idx,
        score=proposals[0].score,
        bucket=proposals[0].bucket,
        source="test",
    )

    rebuilt_sequence, rebuilt_snapshot, _, _, _ = mcts._region_rebuild_pass(
        base_sequence,
        base_snapshot,
        base_snapshots,
        proposal,
    )

    assert rebuilt_sequence[:start_idx] == base_sequence[:start_idx]
    assert sorted(rebuilt_sequence) == sorted(base_sequence)
    validate_solution(
        _parts(),
        list(rebuilt_snapshot.boards),
        cfg.BOARD_W,
        cfg.BOARD_H,
        trim=cfg.TRIM,
        safe_gap=cfg.SAFE_GAP,
        touch_tol=cfg.TOUCH_TOL,
    )


def test_region_priority_keeps_board_count_primary():
    mcts = RecedingHorizonMCTS(_parts(), _cfg(), seed=1)
    worse = _dummy_snapshot(board_count=3, sum_u_gamma=2.5)
    better = _dummy_snapshot(board_count=2, sum_u_gamma=0.1)

    assert mcts._region_priority(better) > mcts._region_priority(worse)


def test_update_global_best_ignores_incomplete_sequence():
    mcts = RecedingHorizonMCTS(_parts(), _cfg(), seed=1)
    partial_snapshot = _dummy_snapshot(board_count=1, sum_u_gamma=0.5)

    mcts._update_global_best((1, 2, 3), partial_snapshot)

    assert mcts.global_best_sequence is None
    assert mcts.global_best_snapshot is None


def test_aggregate_totals_reports_tail_metrics():
    totals = aggregate_totals(
        [
            {"board": 1, "n_parts": 2, "used_area": 90.0, "waste_area": 10.0, "board_area": 100.0, "U": 0.9},
            {"board": 2, "n_parts": 2, "used_area": 80.0, "waste_area": 20.0, "board_area": 100.0, "U": 0.8},
            {"board": 3, "n_parts": 1, "used_area": 20.0, "waste_area": 80.0, "board_area": 100.0, "U": 0.2},
        ],
        sparse_util=0.55,
    )

    assert abs(float(totals["U_avg_excl_last"]) - 0.85) < 1e-9
    assert abs(float(totals["U_avg_excl_last2"]) - 0.9) < 1e-9
    assert abs(float(totals["U_last"]) - 0.2) < 1e-9
    assert abs(float(totals["U_worst_excl_last"]) - 0.8) < 1e-9
    assert abs(float(totals["last2_used_equiv"]) - 1.0) < 1e-9
    assert int(totals["sparse_board_count"]) == 1
    assert abs(float(totals["tail_waste_area"]) - 80.0) < 1e-9


def test_geometry_policy_uses_single_spacing_model_when_binary_spacing_enabled():
    cfg = _cfg()
    cfg.GAP = 4.0
    cfg.BINARY_SPACING = True
    cfg.SAFE_GAP = 6.0
    cfg.TOOL_D = 8.0

    policy = resolve_geometry_policy(cfg)

    assert policy.spacing_mode == "binary"
    assert abs(policy.part_inflate_gap - 0.0) < 1e-9
    assert abs(policy.safe_gap - 6.0) < 1e-9


def test_geometry_policy_zero_safe_gap_does_not_fallback_to_tool_d():
    cfg = _cfg()
    cfg.BINARY_SPACING = True
    cfg.SAFE_GAP = 0.0
    cfg.TOOL_D = 6.0

    policy = resolve_geometry_policy(cfg)

    assert abs(policy.safe_gap - 0.0) < 1e-9


def test_read_sample_parts_respects_effective_inflate_gap(tmp_path):
    csv_path = tmp_path / "parts.csv"
    csv_path.write_text("Fleng_mm,Fwidth_mm,PartID\n10,5,p1\n", encoding="utf-8")

    cfg = _cfg()
    cfg.GAP = 2.0
    cfg.BINARY_SPACING = False
    inflate_policy = resolve_geometry_policy(cfg)
    inflate_parts = read_sample_parts(
        str(csv_path),
        cfg.TRIM,
        cfg.GAP,
        tool_d=cfg.TOOL_D,
        inflate_gap=inflate_policy.part_inflate_gap,
    )

    cfg.BINARY_SPACING = True
    cfg.SAFE_GAP = 6.0
    binary_policy = resolve_geometry_policy(cfg)
    binary_parts = read_sample_parts(
        str(csv_path),
        cfg.TRIM,
        cfg.GAP,
        tool_d=cfg.TOOL_D,
        inflate_gap=binary_policy.part_inflate_gap,
    )

    assert abs(inflate_parts[0].w - 12.0) < 1e-9
    assert abs(inflate_parts[0].h - 7.0) < 1e-9
    assert abs(binary_parts[0].w - 10.0) < 1e-9
    assert abs(binary_parts[0].h - 5.0) < 1e-9


def test_preprocess_parts_adds_duplicate_and_shape_labels():
    cfg = _cfg()
    cfg.ALLOW_ROT = True
    parts = [
        Part(uid=1, pid_raw="a", w=10.0, h=5.0, w0=10.0, h0=5.0),
        Part(uid=2, pid_raw="b", w=5.0, h=10.0, w0=5.0, h0=10.0),
        Part(uid=3, pid_raw="c", w=9.0, h=1.0, w0=9.0, h0=1.0),
    ]

    processed, summary = preprocess_parts(parts, cfg)

    assert summary.n_parts == 3
    assert summary.n_distinct_shapes == 2
    assert summary.duplicate_parts == 1
    assert summary.max_dup_count == 2
    assert processed[0].dup_group == processed[1].dup_group
    assert processed[0].dup_count == 2
    assert processed[2].is_strip


def test_preprocess_parts_quantizes_only_when_enabled():
    cfg = _cfg()
    cfg.PREPROCESS_DIM_STEP = 0.5
    parts = [
        Part(uid=1, pid_raw="a", w=10.2, h=5.2, w0=10.2, h0=5.2),
    ]

    processed, summary = preprocess_parts(parts, cfg)

    assert summary.quantized_parts == 0
    assert abs(processed[0].w0 - 10.2) < 1e-9
    assert abs(processed[0].h0 - 5.2) < 1e-9

    cfg.PREPROCESS_ENABLE = True
    processed, summary = preprocess_parts(parts, cfg)

    assert summary.quantized_parts == 1
    assert abs(processed[0].w0 - 10.0) < 1e-9
    assert abs(processed[0].h0 - 5.0) < 1e-9


def test_area_aspect_sequence_prefers_tagged_hard_parts_and_penalizes_late_duplicates():
    cfg = _cfg()
    parts = [
        Part(uid=1, pid_raw="hard", w=5.0, h=5.0, w0=5.0, h0=5.0, is_large=True, dup_count=1, dup_rank=1),
        Part(uid=2, pid_raw="plain", w=5.0, h=5.0, w0=5.0, h0=5.0, dup_count=1, dup_rank=1),
        Part(uid=3, pid_raw="dup", w=5.0, h=5.0, w0=5.0, h0=5.0, dup_count=3, dup_rank=3),
    ]

    mcts = RecedingHorizonMCTS(parts, cfg, seed=31)
    sequence = mcts._area_aspect_sequence()

    assert sequence[0] == 1
    assert sequence[-1] == 3


def test_action_geom_stats_meta_score_uses_preprocess_tags():
    cfg = _cfg()
    parts = [
        Part(uid=1, pid_raw="strip", w=8.0, h=1.0, w0=8.0, h0=1.0, is_strip=True, dup_count=1, dup_rank=1),
        Part(uid=2, pid_raw="plain", w=8.0, h=1.0, w0=8.0, h0=1.0, dup_count=4, dup_rank=4),
    ]

    mcts = RecedingHorizonMCTS(parts, cfg, seed=32)
    snapshot = LayoutSnapshot(boards=PersistentBoards.empty(), sum_u_gamma=0.0, layout_hash=(0, 0), ordered_hash=0)

    tagged = mcts._action_geom_stats(snapshot, 1)
    plain = mcts._action_geom_stats(snapshot, 2)

    assert tagged.meta_score > plain.meta_score


def test_duplicate_penalty_only_applies_in_early_order_bonus():
    cfg = _cfg()
    parts = [
        Part(uid=1, pid_raw="plain", w=5.0, h=5.0, w0=5.0, h0=5.0, is_large=True, dup_count=1, dup_rank=1),
        Part(uid=2, pid_raw="dup", w=5.0, h=5.0, w0=5.0, h0=5.0, is_large=True, dup_count=4, dup_rank=4),
    ]

    mcts = RecedingHorizonMCTS(parts, cfg, seed=33)

    assert mcts._part_order_bonus(1, stage="early") > mcts._part_order_bonus(2, stage="early")
    assert abs(mcts._part_order_bonus(1, stage="mid") - mcts._part_order_bonus(2, stage="mid")) < 1e-12


def test_sequence_meta_score_favors_anchor_and_pattern_over_tail_and_single():
    cfg = _cfg()
    parts = [
        Part(uid=1, pid_raw="large", w=7.0, h=7.0, w0=7.0, h0=7.0, is_large=True),
        Part(uid=2, pid_raw="strip", w=8.0, h=1.0, w0=8.0, h0=1.0, is_strip=True),
    ]

    mcts = RecedingHorizonMCTS(parts, cfg, seed=34)
    seq = (1, 2)

    anchor = mcts._sequence_meta_score(seq, kind="anchor", stage="early")
    pattern = mcts._sequence_meta_score(seq, kind="pattern", stage="mid")
    tail = mcts._sequence_meta_score(seq, kind="tail", stage="late")
    single = mcts._sequence_meta_score(seq, kind="single", stage="late")

    assert anchor > tail
    assert pattern > tail
    assert tail > single


def test_area_lower_bound_uses_effective_trimmed_board_area():
    cfg = _cfg()
    cfg.BOARD_W = 10.0
    cfg.BOARD_H = 10.0
    cfg.TRIM = 1.0

    assert abs(_effective_board_area(cfg) - 64.0) < 1e-9
    assert _area_lower_bound(130.0, cfg) == 3


def test_adaptive_main_profile_selects_small_for_tiny_case():
    cfg = _cfg()
    stats = _case_stats(_parts(), cfg)

    profile_name, overrides = _adaptive_main_profile(stats)
    cfg_variant = _variant_cfg(cfg, "rh_mcts_main", stats)

    assert profile_name == "small"
    assert int(overrides["SOLVER_TIME_LIMIT_S"]) == 120
    assert str(cfg_variant.MAIN_PROFILE_TIER) == "small"
    assert int(cfg_variant.MAIN_PROFILE_N_PARTS) == len(_parts())


def test_adaptive_main_profile_selects_large_for_industrial_scale_case():
    cfg = _cfg()
    parts = [
        Part(uid=i + 1, pid_raw=str(i + 1), w=2.0, h=3.0, w0=2.0, h0=3.0)
        for i in range(250)
    ]
    stats = _case_stats(parts, cfg)

    profile_name, overrides = _adaptive_main_profile(stats)
    cfg_variant = _variant_cfg(cfg, "rh_mcts_main", stats)

    assert profile_name == "large"
    assert overrides == {}
    assert str(cfg_variant.MAIN_PROFILE_TIER) == "large"


def test_variant_cfg_respects_external_time_limit_cap():
    cfg = _cfg()
    cfg.SOLVER_TIME_LIMIT_S = 10.0
    stats = _case_stats(_parts(), cfg)

    cfg_variant = _variant_cfg(cfg, "rh_mcts_main", stats)

    assert float(cfg_variant.SOLVER_TIME_LIMIT_S) == 10.0
    assert float(cfg_variant.POST_MCTS_RESERVE_MAX_S) <= 5.0


def test_variant_cfg_zero_time_limit_means_unbounded_runtime():
    cfg = _cfg()
    cfg.SOLVER_TIME_LIMIT_S = 0.0
    stats = _case_stats(_parts(), cfg)

    cfg_variant = _variant_cfg(cfg, "rh_mcts_main", stats)

    assert float(cfg_variant.SOLVER_TIME_LIMIT_S) == 0.0
    assert float(cfg_variant.POST_MCTS_RESERVE_MIN_S) == 0.0
    assert float(cfg_variant.POST_MCTS_RESERVE_MAX_S) == 0.0
    assert float(cfg_variant.SA_TIME_CAP_S) == 0.0
    assert float(cfg_variant.POST_REPAIR_TIME_LIMIT_S) == 0.0
    assert float(cfg_variant.GLOBAL_PATTERN_TIME_LIMIT_S) == 0.0


def test_variant_cfg_ours_base_disables_post_repair_and_global_memory():
    cfg = _cfg()
    stats = _case_stats(_parts(), cfg)

    cfg_variant = _variant_cfg(cfg, "ours_base", stats)

    assert str(cfg_variant.MAIN_PROFILE_TIER) == "small"
    assert not bool(cfg_variant.POST_REPAIR_ENABLE)
    assert not bool(cfg_variant.GLOBAL_PATTERN_MEMORY_ENABLE)


def test_variant_cfg_ours_noprior_enables_uniform_prior_ablation():
    cfg = _cfg()
    stats = _case_stats(_parts(), cfg)

    cfg_variant = _variant_cfg(cfg, "ours_noprior", stats)

    assert bool(cfg_variant.DISABLE_GEOM_PRIOR)
    assert not bool(cfg_variant.MACRO_ACTION_ENABLE)
    assert abs(float(cfg_variant.ROLLOUT_PRIOR_W)) < 1e-12


def test_variant_cfg_rh_mcts_ref_explicitly_disables_heavy_modules():
    cfg = _cfg()
    stats = _case_stats(_parts(), cfg)

    cfg_variant = _variant_cfg(cfg, "rh_mcts_ref", stats)

    assert str(cfg_variant.SOLVER_ROLE) == "reference_core"
    assert str(cfg_variant.SOLVER_BUNDLE) == "single_action_core"
    assert not bool(cfg_variant.ROOT_DIRICHLET_ENABLE)
    assert not bool(cfg_variant.MACRO_ACTION_ENABLE)
    assert not bool(cfg_variant.GLOBAL_PATTERN_MEMORY_ENABLE)
    assert not bool(cfg_variant.GLOBAL_PATTERN_MASTER_ENABLE)
    assert not bool(cfg_variant.POST_REPAIR_TAIL_STRIKE_ENABLE)
    assert not bool(cfg_variant.POST_REPAIR_DYNAMIC_EXPAND_ENABLE)
    assert not bool(cfg_variant.POST_REPAIR_ALNS_ENABLE)


def test_variant_cfg_rh_mcts_main_keeps_mcts_centered_bundle_without_global_master():
    cfg = _cfg()
    parts = [
        Part(uid=i + 1, pid_raw=str(i + 1), w=2.0, h=3.0, w0=2.0, h0=3.0)
        for i in range(400)
    ]
    stats = _case_stats(parts, cfg)

    cfg_variant = _variant_cfg(cfg, "rh_mcts_main", stats)

    assert str(cfg_variant.SOLVER_ROLE) == "paper_main"
    assert str(cfg_variant.SOLVER_BUNDLE) == "mcts_plus_light_tail"
    assert not bool(cfg_variant.GLOBAL_PATTERN_MASTER_ENABLE)
    assert bool(cfg_variant.GLOBAL_PATTERN_MEMORY_ENABLE)
    assert bool(cfg_variant.MACRO_ACTION_ENABLE)
    assert bool(cfg_variant.POST_REPAIR_TAIL_STRIKE_ENABLE)
    assert bool(cfg_variant.POST_REPAIR_DYNAMIC_EXPAND_ENABLE)
    assert bool(cfg_variant.POST_REPAIR_ALNS_ENABLE)
    assert int(cfg_variant.MCTS_N_SIM) == 8
    assert int(cfg_variant.COMMIT_MIN_ROOT_VISITS) == 64
    assert int(cfg_variant.LOCAL_MAX_ITERS) == 48
    assert int(cfg_variant.SA_ITERS) == 5000
    assert int(cfg_variant.MACRO_MAX_PARTS) == 6
    assert abs(float(cfg_variant.MACRO_PATTERN_UTIL_MIN) - 0.92) < 1e-12
    assert abs(float(cfg_variant.MACRO_EARLY_SINGLE_ONLY_RATIO)) < 1e-12


def test_variant_cfg_rh_mcts_full_keeps_extended_experiment_bundle():
    cfg = _cfg()
    parts = [
        Part(uid=i + 1, pid_raw=str(i + 1), w=2.0, h=3.0, w0=2.0, h0=3.0)
        for i in range(400)
    ]
    stats = _case_stats(parts, cfg)

    cfg_variant = _variant_cfg(cfg, "rh_mcts_full", stats)

    assert str(cfg_variant.SOLVER_ROLE) == "extended_full"
    assert str(cfg_variant.SOLVER_BUNDLE) == "macro_plus_global_master"
    assert bool(cfg_variant.GLOBAL_PATTERN_MASTER_ENABLE)
    assert bool(cfg_variant.GLOBAL_PATTERN_MEMORY_ENABLE)
    assert bool(cfg_variant.MACRO_ACTION_ENABLE)
    assert bool(cfg_variant.POST_REPAIR_TAIL_STRIKE_ENABLE)
    assert bool(cfg_variant.POST_REPAIR_DYNAMIC_EXPAND_ENABLE)
    assert bool(cfg_variant.POST_REPAIR_ALNS_ENABLE)
    assert int(cfg_variant.MCTS_N_SIM) == 8
    assert int(cfg_variant.COMMIT_MIN_ROOT_VISITS) == 64
    assert int(cfg_variant.LOCAL_MAX_ITERS) == 48
    assert int(cfg_variant.SA_ITERS) == 5000


def test_variant_cfg_rh_mcts_beam_enables_beam_bundle():
    cfg = _cfg()
    parts = [
        Part(uid=i + 1, pid_raw=str(i + 1), w=2.0, h=3.0, w0=2.0, h0=3.0)
        for i in range(400)
    ]
    stats = _case_stats(parts, cfg)

    cfg_variant = _variant_cfg(cfg, "rh_mcts_beam", stats)

    assert str(cfg_variant.SOLVER_ROLE) == "beam_main"
    assert str(cfg_variant.SOLVER_BUNDLE) == "beam_macro_rh_mcts"
    assert bool(cfg_variant.BEAM_ENABLE)
    assert int(cfg_variant.BEAM_WIDTH) == 2
    assert int(cfg_variant.BEAM_BRANCH_TOPK) == 2
    assert int(cfg_variant.BEAM_POSTPROCESS_TOPK) == 2
    assert int(cfg_variant.BEAM_MIN_ROOT_VISITS) == 16
    assert int(cfg_variant.BEAM_MIN_CHILD_VISITS) == 1
    assert not bool(cfg_variant.BEAM_REQUIRE_DOMINANCE)
    assert bool(cfg_variant.MACRO_ACTION_ENABLE)
    assert abs(float(cfg_variant.MACRO_EARLY_SINGLE_ONLY_RATIO)) < 1e-12


def test_variant_cfg_rh_mcts_prefix_beam_enables_prefix_bundle():
    cfg = _cfg()
    stats = _case_stats(_parts(), cfg)

    cfg_variant = _variant_cfg(cfg, "rh_mcts_prefix_beam", stats)

    assert str(cfg_variant.SOLVER_ROLE) == "prefix_beam_main"
    assert str(cfg_variant.SOLVER_BUNDLE) == "prefix_beam_oracle"
    assert bool(cfg_variant.PREFIX_BEAM_ENABLE)
    assert int(cfg_variant.PREFIX_BEAM_DEPTH) == 3
    assert int(cfg_variant.PREFIX_BEAM_WIDTH) == 2
    assert int(cfg_variant.PREFIX_BEAM_BRANCH_TOPK) == 3
    assert int(cfg_variant.PREFIX_BEAM_POST_TOPK) == 2
    assert bool(cfg_variant.MACRO_ACTION_ENABLE)
    assert not bool(cfg_variant.POST_REPAIR_ENABLE)
    assert not bool(cfg_variant.GLOBAL_PATTERN_MASTER_ENABLE)
    assert not bool(cfg_variant.BEAM_FOCUSED_REPAIR_ENABLE)


def test_variant_cfg_applies_variant_extra_overrides_after_profiles():
    cfg = _cfg()
    cfg.VARIANT_EXTRA_OVERRIDES = {
        "rh_mcts_beam": {
            "BEAM_WIDTH": 3,
            "BEAM_FOCUSED_REPAIR_TOPK": 1,
        }
    }
    stats = _case_stats(_parts(), cfg)

    cfg_variant = _variant_cfg(cfg, "rh_mcts_beam", stats)

    assert int(cfg_variant.BEAM_WIDTH) == 3
    assert int(cfg_variant.BEAM_FOCUSED_REPAIR_TOPK) == 1


def test_beam_solver_runs_on_toy_case_and_records_meta():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.90
    cfg.MACRO_PATTERN_UTIL_MIN = 0.90
    cfg.MACRO_MEMORY_UTIL_MIN = 0.90
    cfg.BEAM_ENABLE = True
    cfg.BEAM_WIDTH = 2
    cfg.BEAM_BRANCH_TOPK = 2
    cfg.BEAM_POSTPROCESS_TOPK = 2
    cfg.MCTS_N_SIM = 1
    cfg.COMMIT_MIN_ROOT_VISITS = 1
    cfg.COMMIT_MIN_CHILD_VISITS = 1
    cfg.COMMIT_CONFIRM_PASSES = 1
    cfg.COMMIT_MAX_STEPS = 1
    cfg.SOLVER_TIME_LIMIT_S = 2.0
    cfg.SA_ITERS = 10
    cfg.SA_TIME_CAP_S = 0.2
    cfg.POST_REPAIR_TIME_LIMIT_S = 0.2
    cfg.GLOBAL_PATTERN_TIME_LIMIT_S = 0.2
    cfg.POST_REPAIR_ENABLE = False
    cfg.GLOBAL_PATTERN_MASTER_ENABLE = False

    result = solve_with_receding_horizon_mcts(_parts(), cfg, seed=42)

    assert result.sequence
    assert sorted(result.sequence) == sorted(part.uid for part in _parts())
    assert float(result.meta["beam_rounds"]) >= 1.0
    assert float(result.meta["beam_active_states_max"]) >= 1.0
    validate_solution(
        _parts(),
        result.boards,
        cfg.BOARD_W,
        cfg.BOARD_H,
        trim=cfg.TRIM,
        safe_gap=cfg.SAFE_GAP,
        touch_tol=cfg.TOUCH_TOL,
    )


def test_beam_commit_chains_keeps_multiple_branches_without_single_chain_dominance():
    cfg = _cfg()
    cfg.BEAM_ENABLE = True
    cfg.BEAM_WIDTH = 2
    cfg.BEAM_BRANCH_TOPK = 2
    cfg.BEAM_MIN_ROOT_VISITS = 4
    cfg.BEAM_MIN_CHILD_VISITS = 1
    cfg.BEAM_REQUIRE_DOMINANCE = False
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=7)

    root = SearchNode(snapshot=_dummy_snapshot(1, 0.2), remaining_mask=3, visits=8)
    child_a = SearchNode(snapshot=_dummy_snapshot(1, 0.3), remaining_mask=2, parent=root, action_from_parent=1, visits=5, reward_sum=4.0)
    child_b = SearchNode(snapshot=_dummy_snapshot(1, 0.25), remaining_mask=1, parent=root, action_from_parent=2, visits=4, reward_sum=3.0)
    root.children = {1: child_a, 2: child_b}

    assert not mcts._can_commit_node(root, mcts._rank_children(root))

    chains = mcts._beam_commit_chains(root)

    assert len(chains) == 2
    assert {chain[0][0] for chain in chains} == {1, 2}


def test_beam_root_rank_prefers_fewer_boards():
    cfg = _cfg()
    cfg.BEAM_ENABLE = True
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=8)

    compact = SearchNode(snapshot=_dummy_snapshot(2, 0.7), remaining_mask=7, visits=3, reward_sum=1.0, prefix_sequence=(1, 2))
    scattered = SearchNode(snapshot=_dummy_snapshot(3, 0.9), remaining_mask=7, visits=20, reward_sum=10.0, prefix_sequence=(1, 2, 3))

    assert mcts._beam_root_rank(compact) > mcts._beam_root_rank(scattered)


def test_beam_state_signature_dedupes_same_tail_structure_with_different_prefixes():
    cfg = _cfg()
    cfg.BEAM_ENABLE = True
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=9)
    snapshot = _dummy_snapshot(2, 0.6)

    a = SearchNode(snapshot=snapshot, remaining_mask=3, prefix_sequence=(1, 2))
    b = SearchNode(snapshot=snapshot, remaining_mask=3, prefix_sequence=(3, 4, 5))

    assert mcts._beam_state_signature(a) == mcts._beam_state_signature(b)


def test_beam_focus_windows_respect_tail_width_budget():
    cfg = _cfg()
    cfg.BOARD_W = 7.0
    cfg.BOARD_H = 7.0
    cfg.BEAM_ENABLE = True
    cfg.BEAM_FOCUSED_REPAIR_WINDOW_WIDTHS = (2, 3, 4)
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=10)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)

    windows = mcts._beam_focus_windows(base_snapshot, base_sequence)

    assert windows
    assert all(window.source.startswith("beam-focus:") for window in windows)
    assert all(2 <= int(window.base_board_count) <= 4 for window in windows)


def test_prefix_baseline_completion_oracle_completes_legally_and_hits_cache():
    cfg = _cfg()
    cfg.PREFIX_BEAM_ENABLE = True
    cfg.MACRO_ACTION_ENABLE = True
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=11)
    prefix = (1, 2)
    prefix_snapshot, _ = mcts.decoder.replay_order(prefix)
    remaining_mask = mcts.full_remaining_mask
    for pid in prefix:
        remaining_mask &= ~int(mcts.part_bit[int(pid)])

    full_sequence_1, final_snapshot_1, _ = mcts._evaluate_prefix_with_baseline(
        prefix,
        prefix_snapshot,
        remaining_mask,
        prefix_depth=len(prefix),
    )
    full_sequence_2, final_snapshot_2, _ = mcts._evaluate_prefix_with_baseline(
        prefix,
        prefix_snapshot,
        remaining_mask,
        prefix_depth=len(prefix),
    )

    assert sorted(full_sequence_1) == sorted(part.uid for part in _parts())
    assert full_sequence_1 == full_sequence_2
    assert len(final_snapshot_1.boards.materialize()) == len(final_snapshot_2.boards.materialize())
    assert float(mcts.meta["oracle_completion_calls"]) >= 2.0
    assert float(mcts.meta["oracle_completion_cache_hits"]) >= 1.0


def test_prefix_beam_root_rank_prefers_better_oracle_terminal_result_over_visits():
    cfg = _cfg()
    cfg.PREFIX_BEAM_ENABLE = True
    cfg.BEAM_ENABLE = True
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=12)

    compact = SearchNode(snapshot=_dummy_snapshot(2, 0.7), remaining_mask=7, visits=2, reward_sum=1.0, prefix_sequence=(1, 2), depth=2)
    noisy = SearchNode(snapshot=_dummy_snapshot(3, 0.9), remaining_mask=7, visits=20, reward_sum=10.0, prefix_sequence=(1,), depth=1)

    def _fake_eval(prefix_sequence, prefix_snapshot, remaining_mask, *, prefix_depth=None):
        del prefix_snapshot, remaining_mask
        if tuple(prefix_sequence) == (1, 2):
            return tuple(prefix_sequence), _dummy_snapshot(2, 0.8), (-2.0, 0.8)
        return tuple(prefix_sequence), _dummy_snapshot(3, 0.95), (-3.0, 0.95)

    mcts._evaluate_prefix_with_baseline = _fake_eval  # type: ignore[assignment]

    assert mcts._beam_root_rank(compact) > mcts._beam_root_rank(noisy)


def test_baseline_sa_solver_returns_legal_complete_solution():
    cfg = _cfg()
    cfg.SOLVER_TIME_LIMIT_S = 2.0
    cfg.SA_BUDGET_FRAC = 1.0
    result = _solve_baseline_sa(_parts(), cfg, seed=5)

    assert result.meta["solver_kind"] == "baseline_sa"
    assert sorted(result.sequence) == sorted(part.uid for part in _parts())
    validate_solution(
        _parts(),
        result.boards,
        cfg.BOARD_W,
        cfg.BOARD_H,
        trim=cfg.TRIM,
        safe_gap=cfg.SAFE_GAP,
        touch_tol=cfg.TOUCH_TOL,
    )


def test_generate_pattern_candidates_deduplicates_coverage_sets():
    cfg = _cfg()
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)
    window = _window(base_sequence, start_idx=2, base_board_count=max(1, len(base_snapshot.boards) - 1))
    prefix_snapshot, _ = mcts.decoder.replay_order(window.fixed_prefix_sequence)

    generated, patterns = mcts._generate_pattern_candidates(base_snapshot, window, prefix_snapshot)

    assert generated >= len(patterns) >= 1
    assert len({pattern.part_ids for pattern in patterns}) == len(patterns)
    assert all(pattern.part_ids for pattern in patterns)
    assert all(set(pattern.part_ids).issubset(set(window.movable_part_ids)) for pattern in patterns)


def test_tail_strike_windows_anchor_the_last_board():
    cfg = _cfg()
    cfg.BOARD_W = 7.0
    cfg.BOARD_H = 7.0
    cfg.POST_REPAIR_TAIL_STRIKE_ENABLE = True
    cfg.POST_REPAIR_TAIL_STRIKE_WIDTHS = (2, 3, 4)
    cfg.POST_REPAIR_TAIL_STRIKE_TOPK = 3
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)
    boards = base_snapshot.boards.materialize()
    idx_map = {int(pid): idx for idx, pid in enumerate(base_sequence)}

    windows = mcts._tail_strike_windows(base_snapshot, base_sequence, idx_map)

    assert windows
    last_board_ids = {int(pp.uid) for pp in boards[-1].placed}
    assert last_board_ids
    assert all(window.source.startswith("tail-strike:") for window in windows)
    assert all(last_board_ids.issubset(set(window.movable_part_ids)) for window in windows)


def test_tail_strike_window_gets_larger_pattern_budget():
    cfg = _cfg()
    cfg.POST_REPAIR_TAIL_STRIKE_ENABLE = True
    cfg.POST_REPAIR_TAIL_STRIKE_PATTERN_MULT = 3
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    plain = RepairWindow(
        start_idx=0,
        start_board=0,
        span=2,
        movable_part_ids=(1, 2),
        blocker_part_ids=tuple(),
        fixed_prefix_sequence=tuple(),
        base_board_count=2,
        score=(0.0, 0.0, 0.0, 0.0),
        source="plain",
    )
    tail = RepairWindow(
        start_idx=0,
        start_board=0,
        span=2,
        movable_part_ids=(1, 2),
        blocker_part_ids=tuple(),
        fixed_prefix_sequence=tuple(),
        base_board_count=2,
        score=(0.0, 0.0, 0.0, 0.0),
        source="tail-strike:1,2",
    )

    plain_budget = mcts._pattern_budget_for_window(plain)
    tail_budget = mcts._pattern_budget_for_window(tail)

    assert tail_budget[0] >= plain_budget[0] * 3
    assert tail_budget[1] >= plain_budget[1] * 3


def test_dynamic_window_expansion_grows_tail_window():
    cfg = _cfg()
    cfg.BOARD_W = 6.0
    cfg.BOARD_H = 6.0
    cfg.POST_REPAIR_TAIL_STRIKE_ENABLE = True
    cfg.POST_REPAIR_TAIL_STRIKE_WIDTHS = (2,)
    cfg.POST_REPAIR_DYNAMIC_EXPAND_ENABLE = True
    cfg.POST_REPAIR_DYNAMIC_WINDOW_WIDTHS = (3, 4)
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)
    idx_map = {int(pid): idx for idx, pid in enumerate(base_sequence)}
    base_window = mcts._tail_strike_windows(base_snapshot, base_sequence, idx_map)[0]

    chain = mcts._window_expansion_chain(base_snapshot, base_sequence, base_window)

    assert len(chain) >= 2
    assert len(chain[-1].movable_part_ids) >= len(chain[0].movable_part_ids)
    assert "|expand:" in chain[-1].source


def test_dynamic_window_expansion_normalizes_large_window_before_expanding():
    cfg = _cfg()
    cfg.BOARD_W = 6.0
    cfg.BOARD_H = 6.0
    cfg.POST_REPAIR_DYNAMIC_EXPAND_ENABLE = True
    cfg.POST_REPAIR_INITIAL_WINDOW_WIDTH = 2
    cfg.POST_REPAIR_DYNAMIC_WINDOW_WIDTHS = (3, 4)
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)
    idx_map = {int(pid): idx for idx, pid in enumerate(base_sequence)}
    huge = mcts._window_from_movable_set(
        base_snapshot.boards.materialize(),
        base_sequence,
        idx_map,
        set(base_sequence),
        blocker_ids=tuple(),
        score=(1.0, 0.0, 0.0, 0.0),
        source="spill",
    )

    assert huge is not None
    assert huge.base_board_count >= 3

    chain = mcts._window_expansion_chain(base_snapshot, base_sequence, huge)

    assert chain[0].base_board_count == 2
    assert "|seedw:2" in chain[0].source
    assert any("|expand:" in item.source for item in chain[1:])


def test_global_pattern_memory_collects_snapshot_boards():
    cfg = _cfg()
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_LIMIT = 16
    cfg.GLOBAL_PATTERN_MEMORY_TOPK = 8
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)

    mcts._record_snapshot_patterns(base_snapshot, source="test")
    memory_patterns = mcts._global_pattern_memory_candidates()

    assert memory_patterns
    assert int(mcts.meta["global_pattern_memory_patterns"]) == len(mcts.global_pattern_memory)


def test_global_pattern_cover_adds_singleton_fallback_for_every_part():
    mcts = RecedingHorizonMCTS(_parts(), _cfg(), seed=1)
    selected = [
        PatternCandidate((1, 2), (1, 2), 48.0, 0.48, 52.0, "maxrects_baf", "combo12"),
        PatternCandidate((2, 3), (2, 3), 40.0, 0.40, 60.0, "maxrects_baf", "combo23"),
    ]

    covered = mcts._ensure_global_pattern_cover(selected, (1, 2, 3))

    assert {(1,), (2,), (3,)}.issubset({pattern.part_ids for pattern in covered})


def test_global_pattern_candidates_force_memory_patterns_into_master_pool():
    cfg = _cfg()
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_CAP = 1
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_TOPK = 8
    cfg.GLOBAL_PATTERN_MEMORY_FOCUS_TAIL_BOARDS = 8
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)
    memory_pattern = PatternCandidate((1, 4), (1, 4), 33.0, 0.33, 67.0, "maxrects_baf", "memory:test")
    mcts.global_pattern_memory[memory_pattern.part_ids] = memory_pattern

    _, selected = mcts._global_pattern_candidates(base_sequence, base_snapshot)

    assert any(pattern.part_ids == (1, 4) for pattern in selected)
    assert int(mcts.meta["global_pattern_memory_selected"]) >= 1


def test_global_pattern_memory_candidates_apply_focus_and_elite_threshold():
    cfg = _cfg()
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.95
    cfg.GLOBAL_PATTERN_MEMORY_TOPK = 8
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    mcts.global_pattern_memory = {
        (1, 2): PatternCandidate((1, 2), (1, 2), 96.0, 0.96, 4.0, "maxrects_baf", "focus-elite"),
        (3, 4): PatternCandidate((3, 4), (3, 4), 94.0, 0.94, 6.0, "maxrects_baf", "focus-weak"),
        (5, 6): PatternCandidate((5, 6), (5, 6), 97.0, 0.97, 3.0, "maxrects_baf", "off-focus"),
    }

    selected = mcts._global_pattern_memory_candidates(focus_part_ids={1, 4})

    assert [pattern.part_ids for pattern in selected] == [(1, 2)]
    assert int(mcts.meta["global_pattern_memory_relevant"]) == 1


def test_global_pattern_candidates_expand_tail_focus_when_last_board_has_no_memory_hit():
    cfg = _cfg()
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.95
    cfg.GLOBAL_PATTERN_MEMORY_FOCUS_TAIL_BOARDS = 2
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)
    boards = base_snapshot.boards.materialize()
    last_board_ids = {int(pp.uid) for pp in boards[-1].placed}
    if len(boards) >= 2:
        second_last_ids = {int(pp.uid) for pp in boards[-2].placed}
    else:
        second_last_ids = set(last_board_ids)
    bridge_ids = tuple(sorted((second_last_ids - last_board_ids) or second_last_ids))
    mcts.global_pattern_memory = {
        bridge_ids: PatternCandidate(bridge_ids, bridge_ids, 96.0, 0.96, 4.0, "maxrects_baf", "bridge")
    }

    _, selected = mcts._global_pattern_candidates(base_sequence, base_snapshot)

    assert any(pattern.part_ids == bridge_ids for pattern in selected)


def test_pattern_hint_indices_use_board_pattern_or_singleton_fallback():
    mcts = RecedingHorizonMCTS(_parts(), _cfg(), seed=1)
    patterns = [
        PatternCandidate((1, 2), (1, 2), 48.0, 0.48, 52.0, "maxrects_baf", "pair"),
        PatternCandidate((3,), (3,), 16.0, 0.16, 84.0, "maxrects_baf", "s3"),
        PatternCandidate((4,), (4,), 9.0, 0.09, 91.0, "maxrects_baf", "s4"),
    ]

    indices = mcts._pattern_hint_indices(patterns, (1, 2, 3, 4), ((1, 2), (3, 4)))

    assert indices == [0, 1, 2]


def test_alns_operator_score_increases_after_success():
    cfg = _cfg()
    cfg.POST_REPAIR_ALNS_ENABLE = True
    cfg.POST_REPAIR_ALNS_ALPHA = 0.5
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    window = RepairWindow(
        start_idx=0,
        start_board=0,
        span=2,
        movable_part_ids=(1, 2),
        blocker_part_ids=tuple(),
        fixed_prefix_sequence=tuple(),
        base_board_count=2,
        score=(0.0, 0.0, 0.0, 0.0),
        source="tail-strike:1,2",
    )
    before = _dummy_snapshot(board_count=3, sum_u_gamma=0.6)
    after = _dummy_snapshot(board_count=2, sum_u_gamma=0.9)
    old = mcts.post_repair_operator_scores["tail-strike"]

    mcts._update_post_repair_operator_score(window, before, after, attempt_time_s=0.2)

    assert mcts.post_repair_operator_scores["tail-strike"] > old
    assert int(mcts.meta["post_repair_alns_updates"]) == 1


def test_pattern_master_falls_back_when_drop_target_is_infeasible():
    mcts = RecedingHorizonMCTS(_parts(), _cfg(), seed=1)
    window = RepairWindow(
        start_idx=0,
        start_board=0,
        span=2,
        movable_part_ids=(1, 2),
        blocker_part_ids=tuple(),
        fixed_prefix_sequence=tuple(),
        base_board_count=2,
        score=(0.0, 0.0, 0.0, 0.0),
        source="test",
    )
    patterns = [
        PatternCandidate((1,), (1,), 24.0, 0.24, 76.0, "maxrects_baf", "s1"),
        PatternCandidate((2,), (2,), 24.0, 0.24, 76.0, "maxrects_baf", "s2"),
    ]

    selected, status = mcts._solve_pattern_master(window, patterns, time_limit_s=1.0)

    assert {pattern.part_ids for pattern in selected} == {(1,), (2,)}
    assert "drop:" in status
    assert "min:" in status


def test_pattern_master_branch_and_bound_beats_greedy_cover_count():
    mcts = RecedingHorizonMCTS(_parts(), _cfg(), seed=1)
    window = RepairWindow(
        start_idx=0,
        start_board=0,
        span=6,
        movable_part_ids=(1, 2, 3, 4, 5, 6),
        blocker_part_ids=tuple(),
        fixed_prefix_sequence=tuple(),
        base_board_count=3,
        score=(0.0, 0.0, 0.0, 0.0),
        source="test",
    )
    patterns = [
        PatternCandidate((1, 2, 3, 4), (1, 2, 3, 4), 70.0, 0.95, 5.0, "maxrects_baf", "greedy-a"),
        PatternCandidate((5,), (5,), 12.0, 0.80, 20.0, "maxrects_baf", "greedy-b"),
        PatternCandidate((6,), (6,), 11.0, 0.79, 21.0, "maxrects_baf", "greedy-c"),
        PatternCandidate((1, 2, 3), (1, 2, 3), 40.0, 0.70, 30.0, "maxrects_baf", "exact-a"),
        PatternCandidate((4, 5, 6), (4, 5, 6), 42.0, 0.69, 31.0, "maxrects_baf", "exact-b"),
    ]

    selected, status = mcts._solve_pattern_master(window, patterns, time_limit_s=1.0)

    assert len(selected) == 2
    assert {pattern.part_ids for pattern in selected} == {(1, 2, 3), (4, 5, 6)}
    assert status
    assert "drop:" in status or "min:" in status


def test_sequence_from_selected_patterns_replays_legally():
    cfg = _cfg()
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)
    window = _window(base_sequence, start_idx=2, base_board_count=max(1, len(base_snapshot.boards) - 1))
    prefix_snapshot, _ = mcts.decoder.replay_order(window.fixed_prefix_sequence)
    _, patterns = mcts._generate_pattern_candidates(base_snapshot, window, prefix_snapshot)
    selected, _ = mcts._solve_pattern_master(window, patterns, time_limit_s=1.0)

    rebuilt = mcts._sequence_from_selected_patterns(window.fixed_prefix_sequence, selected, base_sequence)

    assert rebuilt is not None
    rebuilt_sequence, rebuilt_snapshot, _ = rebuilt
    assert sorted(rebuilt_sequence) == sorted(base_sequence)
    validate_solution(
        _parts(),
        list(rebuilt_snapshot.boards),
        cfg.BOARD_W,
        cfg.BOARD_H,
        trim=cfg.TRIM,
        safe_gap=cfg.SAFE_GAP,
        touch_tol=cfg.TOUCH_TOL,
    )


def test_constructive_warmstarts_prime_root_cache():
    cfg = _cfg()
    cfg.BASELINE_RESTARTS = 3
    cfg.RAND_TOPK = 2
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)

    mcts._prime_constructive_warmstarts()

    assert mcts.constructive_warmstarts
    assert int(mcts.meta["warmstart_candidates_kept"]) >= 1
    root = SearchNode(
        snapshot=LayoutSnapshot(boards=PersistentBoards.empty(), sum_u_gamma=0.0, layout_hash=(0, 0), ordered_hash=0),
        remaining_mask=mcts.full_remaining_mask,
    )
    mcts._warm_start(root)

    assert root.visits > 0


def test_constructive_warmstarts_include_true_baseline_sequence():
    cfg = _cfg()
    cfg.BASELINE_RESTARTS = 0
    cfg.GLOBAL_PATTERN_INCLUDE_BASELINE = True
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)

    mcts._prime_constructive_warmstarts()

    assert int(mcts.meta["baseline_seed_injected"]) == 1
    assert any(candidate.sequence == mcts._area_aspect_sequence() for candidate in mcts.constructive_warmstarts)


def test_seed_rollout_cache_reuses_constructive_suffix():
    cfg = _cfg()
    cfg.BASELINE_RESTARTS = 2
    cfg.RAND_TOPK = 2
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    mcts._prime_constructive_warmstarts()

    seed_sequence = mcts.constructive_warmstarts[0].sequence
    prefix_len = 2
    prefix_snapshot, _ = mcts.decoder.replay_order(seed_sequence[:prefix_len])
    remaining_mask = int(mcts.full_remaining_mask)
    for pid in seed_sequence[:prefix_len]:
        remaining_mask &= ~int(mcts.part_bit[int(pid)])

    final_snapshot, suffix = mcts._rollout(prefix_snapshot, remaining_mask)

    assert tuple(suffix)
    assert sorted(tuple(suffix)) == sorted(seed_sequence[prefix_len:])
    assert len(final_snapshot.boards) >= 1


def test_root_dirichlet_noise_perturbs_root_priors():
    cfg = _cfg()
    cfg.ROOT_DIRICHLET_ENABLE = True
    cfg.ROOT_DIRICHLET_ALPHA = 0.30
    cfg.ROOT_DIRICHLET_EPS = 0.25
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=7)
    root = SearchNode(
        snapshot=LayoutSnapshot(boards=PersistentBoards.empty(), sum_u_gamma=0.0, layout_hash=(0, 0), ordered_hash=0),
        remaining_mask=mcts.full_remaining_mask,
    )

    original = {}
    for pid in mcts.part_ids:
        original[int(pid)] = mcts._action_geom_stats(root.snapshot, int(pid)).prior_value(cfg)
    mcts._ensure_candidate_actions(root, 3)

    assert root.root_noise_applied
    assert int(mcts.meta["root_dirichlet_applications"]) == 1
    assert len(root.candidate_actions) == len(mcts.part_ids)
    assert any(abs(float(root.priors[int(pid)]) - float(original[int(pid)])) > 1e-9 for pid in mcts.part_ids)


def test_rollout_rank_weights_follow_configured_rcl_profile():
    cfg = _cfg()
    cfg.ROLLOUT_RCL_WEIGHTS = (0.60, 0.30, 0.10)
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=3)

    weights = mcts._rollout_rank_weights(3)

    assert len(weights) == 3
    assert abs(weights[0] - 0.60) < 1e-9
    assert abs(weights[1] - 0.30) < 1e-9
    assert abs(weights[2] - 0.10) < 1e-9


def test_macro_action_generation_yields_multi_part_candidates():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.90
    cfg.MACRO_PATTERN_UTIL_MIN = 0.90
    cfg.MACRO_PATTERN_TOPK = 2
    cfg.MACRO_ANCHOR_TOPK = 2
    cfg.MACRO_TAIL_TOPK = 1
    cfg.MACRO_SINGLE_TOPK = 2
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=5)
    pattern = PatternCandidate((1, 3), (1, 3), 40.0, 0.96, 4.0, "maxrects_baf", "memory")
    mcts.global_pattern_memory[pattern.part_ids] = pattern

    keys, action_meta, _ = mcts._generate_macro_actions(
        LayoutSnapshot(boards=PersistentBoards.empty(), sum_u_gamma=0.0, layout_hash=(0, 0), ordered_hash=0),
        mcts.full_remaining_mask,
        6,
    )

    assert keys
    assert any(len(action.sequence) >= 2 for action in action_meta.values())
    assert any(action.kind in {"pattern", "anchor"} for action in action_meta.values())


def test_macro_expand_removes_multiple_parts_and_tracks_prefix_sequence():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.90
    cfg.MACRO_PATTERN_UTIL_MIN = 0.90
    cfg.MACRO_PATTERN_TOPK = 2
    cfg.MACRO_ANCHOR_TOPK = 2
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=5)
    pattern = PatternCandidate((4, 6), (4, 6), 40.0, 0.96, 4.0, "maxrects_baf", "memory")
    mcts.global_pattern_memory[pattern.part_ids] = pattern
    prefix = (1, 2, 3)
    snapshot, _ = mcts.decoder.replay_order(prefix)
    remaining_mask = int(mcts.full_remaining_mask)
    for pid in prefix:
        remaining_mask &= ~int(mcts.part_bit[int(pid)])
    root = SearchNode(
        snapshot=snapshot,
        remaining_mask=remaining_mask,
        prefix_sequence=prefix,
    )

    mcts._ensure_candidate_actions(root, 6)
    macro_key = next(key for key, action in root.action_meta.items() if len(action.sequence) >= 2)
    child = mcts._expand(root, macro_key)
    macro = root.action_meta[macro_key]

    assert child.prefix_sequence == prefix + macro.sequence
    for pid in macro.sequence:
        assert child.remaining_mask & int(mcts.part_bit[int(pid)]) == 0


def test_macro_root_uses_single_stage_candidates_early():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.90
    cfg.MACRO_PATTERN_TOPK = 4
    cfg.MACRO_ANCHOR_TOPK = 4
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=6)
    pattern = PatternCandidate((1, 3), (1, 3), 40.0, 0.98, 2.0, "maxrects_baf", "memory")
    mcts.global_pattern_memory[pattern.part_ids] = pattern
    root = SearchNode(
        snapshot=LayoutSnapshot(boards=PersistentBoards.empty(), sum_u_gamma=0.0, layout_hash=(0, 0), ordered_hash=0),
        remaining_mask=mcts.full_remaining_mask,
    )

    mcts._ensure_candidate_actions(root, 6)

    assert root.candidate_actions
    assert all(len(root.action_meta[int(action)].sequence) == 1 for action in root.candidate_actions)


def test_single_stage_macro_gate_can_be_disabled_at_root():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.MACRO_EARLY_SINGLE_ONLY_RATIO = 0.0
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=15)

    assert not mcts._use_single_stage_macro(mcts.full_remaining_mask)


def test_macro_early_stage_can_use_real_patterns_after_single_only_window():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.MACRO_STAGE_EARLY_RATIO = 0.80
    cfg.MACRO_EARLY_SINGLE_ONLY_RATIO = 0.20
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.90
    cfg.MACRO_PATTERN_UTIL_MIN = 0.90
    cfg.MACRO_MEMORY_UTIL_MIN = 0.90
    cfg.MACRO_PATTERN_TOPK = 2
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=16)
    mcts.global_pattern_memory[(4, 6)] = PatternCandidate((4, 6), (4, 6), 40.0, 0.97, 3.0, "maxrects_baf", "memory")
    prefix = (1, 2)
    snapshot, _ = mcts.decoder.replay_order(prefix)
    remaining_mask = int(mcts.full_remaining_mask)
    for pid in prefix:
        remaining_mask &= ~int(mcts.part_bit[int(pid)])
    node = SearchNode(snapshot=snapshot, remaining_mask=remaining_mask, prefix_sequence=prefix)

    assert mcts._search_stage(remaining_mask) == "early"
    assert not mcts._use_single_stage_macro(remaining_mask)

    mcts._ensure_candidate_actions(node, 4)

    assert node.candidate_actions
    assert any(node.action_meta[int(action)].kind == "pattern" for action in node.candidate_actions)
    assert node.action_meta[int(node.candidate_actions[0])].kind == "pattern"


def test_detach_root_resets_single_frontier_after_leaving_single_only_window():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.MACRO_STAGE_EARLY_RATIO = 0.80
    cfg.MACRO_EARLY_SINGLE_ONLY_RATIO = 0.20
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=26)
    prefix = (1, 2)
    snapshot, _ = mcts.decoder.replay_order(prefix)
    remaining_mask = int(mcts.full_remaining_mask)
    for pid in prefix:
        remaining_mask &= ~int(mcts.part_bit[int(pid)])
    node = SearchNode(snapshot=snapshot, remaining_mask=remaining_mask, prefix_sequence=prefix)
    node.candidate_actions = [3, 4]
    node.candidate_width = 2
    node.action_meta = {
        3: MacroAction(key=3, kind="single", part_ids=(3,), sequence=(3,), prior=1.0, rank_key=(1.0,), source="single"),
        4: MacroAction(key=4, kind="single", part_ids=(4,), sequence=(4,), prior=1.0, rank_key=(0.9,), source="single"),
    }
    node.children = {
        3: SearchNode(snapshot=snapshot, remaining_mask=remaining_mask, prefix_sequence=prefix + (3,)),
    }

    assert mcts._search_stage(remaining_mask) == "early"
    assert not mcts._use_single_stage_macro(remaining_mask)

    root = mcts._detach_root(node)

    assert root.parent is None
    assert root.candidate_actions == []
    assert root.candidate_width == 0
    assert root.action_meta == {}
    assert root.children == {}
    assert int(mcts.meta["macro_frontier_resets"]) == 1


def test_macro_memory_actions_fall_back_when_tail_focus_has_no_hit():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.90
    cfg.MACRO_PATTERN_UTIL_MIN = 0.90
    cfg.MACRO_MEMORY_UTIL_MIN = 0.90
    cfg.MACRO_PATTERN_TOPK = 2
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=7)
    pattern = PatternCandidate((4, 6), (4, 6), 40.0, 0.97, 3.0, "maxrects_baf", "memory")
    mcts.global_pattern_memory[pattern.part_ids] = pattern
    snapshot = LayoutSnapshot(boards=PersistentBoards.empty(), sum_u_gamma=0.0, layout_hash=(0, 0), ordered_hash=0)
    remaining_mask = int(mcts.full_remaining_mask)

    actions = mcts._macro_memory_actions(snapshot, remaining_mask, 2)

    assert actions
    assert any(tuple(seq) == pattern.sequence for _, seq, _ in actions)
    assert int(mcts.meta["macro_memory_fallback_hits"]) == 1


def test_macro_memory_actions_can_hit_tail_focus_for_remaining_parts():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.90
    cfg.MACRO_PATTERN_UTIL_MIN = 0.90
    cfg.MACRO_MEMORY_UTIL_MIN = 0.90
    cfg.MACRO_PATTERN_TOPK = 2
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=17)
    pattern = PatternCandidate((4, 6), (4, 6), 40.0, 0.97, 3.0, "maxrects_baf", "memory")
    mcts.global_pattern_memory[pattern.part_ids] = pattern
    prefix = (1, 2, 3)
    snapshot, _ = mcts.decoder.replay_order(prefix)
    remaining_mask = int(mcts.full_remaining_mask)
    for pid in prefix:
        remaining_mask &= ~int(mcts.part_bit[int(pid)])

    focus = mcts._tail_focus_remaining_part_ids(snapshot, remaining_mask, tail_boards=1, topk=4)
    actions = mcts._macro_memory_actions(snapshot, remaining_mask, 2)

    assert focus
    assert pattern.part_ids[0] in focus or pattern.part_ids[1] in focus
    assert actions
    assert int(mcts.meta["macro_memory_focus_hits"]) >= 1


def test_macro_stage_limits_suppress_anchor_when_pattern_pool_is_strong():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.MACRO_STAGE_LATE_RATIO = 0.30
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=18)
    prefix = (1, 2, 3)
    snapshot, _ = mcts.decoder.replay_order(prefix)
    remaining_mask = int(mcts.full_remaining_mask)
    for pid in prefix:
        remaining_mask &= ~int(mcts.part_bit[int(pid)])

    limits = mcts._macro_stage_limits(snapshot, remaining_mask, pattern_candidates=3)

    assert limits["pattern"] >= 4
    assert limits["anchor"] == 0


def test_macro_stage_limits_allow_patterns_in_early_stage_when_available():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.MACRO_PATTERN_TOPK = 3
    cfg.MACRO_STAGE_EARLY_RATIO = 0.90
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=19)
    snapshot = LayoutSnapshot(boards=PersistentBoards.empty(), sum_u_gamma=0.0, layout_hash=(0, 0), ordered_hash=0)
    remaining_mask = int(mcts.full_remaining_mask)

    limits = mcts._macro_stage_limits(snapshot, remaining_mask, pattern_candidates=4)

    assert mcts._search_stage(remaining_mask) == "early"
    assert limits["pattern"] >= 1
    assert limits["single"] <= 2


def test_macro_late_stage_prioritizes_pattern_actions_when_available():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.MACRO_STAGE_LATE_RATIO = 0.30
    cfg.GLOBAL_PATTERN_MEMORY_ENABLE = True
    cfg.GLOBAL_PATTERN_MEMORY_MIN_UTIL = 0.0
    cfg.GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL = 0.90
    cfg.MACRO_PATTERN_UTIL_MIN = 0.90
    cfg.MACRO_MEMORY_UTIL_MIN = 0.90
    cfg.MACRO_ACTION_TOPK = 4
    cfg.MACRO_PATTERN_TOPK = 2
    cfg.MACRO_ANCHOR_TOPK = 4
    cfg.MACRO_TAIL_TOPK = 3
    cfg.MACRO_SINGLE_TOPK = 2
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=8)
    mcts.global_pattern_memory[(4, 6)] = PatternCandidate((4, 6), (4, 6), 40.0, 0.97, 3.0, "maxrects_baf", "memory1")
    mcts.global_pattern_memory[(4, 5)] = PatternCandidate((4, 5), (4, 5), 38.0, 0.95, 5.0, "maxrects_baf", "memory2")
    prefix = (1, 2, 3)
    snapshot, _ = mcts.decoder.replay_order(prefix)
    remaining_mask = int(mcts.full_remaining_mask)
    for pid in prefix:
        remaining_mask &= ~int(mcts.part_bit[int(pid)])

    actions, action_meta, _ = mcts._generate_macro_actions(snapshot, remaining_mask, 4)
    selected = [action_meta[int(key)] for key in actions]

    assert selected
    assert selected[0].kind == "pattern"
    assert all(item.kind != "anchor" for item in selected)
    assert sum(1 for item in selected if item.kind == "pattern") >= 2


def test_root_dirichlet_depth_gate_skips_deeper_roots():
    cfg = _cfg()
    cfg.ROOT_DIRICHLET_ENABLE = True
    cfg.ROOT_DIRICHLET_ALPHA = 0.30
    cfg.ROOT_DIRICHLET_EPS = 0.25
    cfg.ROOT_DIRICHLET_MAX_DEPTH = 1
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=11)
    root = SearchNode(
        snapshot=LayoutSnapshot(boards=PersistentBoards.empty(), sum_u_gamma=0.0, layout_hash=(0, 0), ordered_hash=0),
        remaining_mask=mcts.full_remaining_mask,
        depth=2,
    )

    original = {}
    for pid in mcts.part_ids:
        original[int(pid)] = mcts._action_geom_stats(root.snapshot, int(pid)).prior_value(cfg)
    mcts._ensure_candidate_actions(root, 3)

    assert root.root_noise_applied
    assert int(mcts.meta["root_dirichlet_applications"]) == 0
    assert all(abs(float(root.priors[int(pid)]) - float(original[int(pid)])) < 1e-9 for pid in mcts.part_ids)


def test_post_mcts_reserve_budget_respects_fraction_and_cap():
    cfg = _cfg()
    cfg.SOLVER_TIME_LIMIT_S = 100.0
    cfg.POST_MCTS_RESERVE_FRAC = 0.20
    cfg.POST_MCTS_RESERVE_MIN_S = 5.0
    cfg.POST_MCTS_RESERVE_MAX_S = 12.0
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=13)

    assert abs(mcts._post_mcts_reserve_s() - 12.0) < 1e-9


def test_macro_main_keeps_baseline_floor_on_toy_case():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.BASELINE_RESTARTS = 2
    cfg.MCTS_N_SIM = 1
    cfg.SOLVER_TIME_LIMIT_S = 1.5
    cfg.POST_REPAIR_ENABLE = False
    cfg.GLOBAL_PATTERN_MASTER_ENABLE = False
    baseline = RecedingHorizonMCTS(_parts(), _cfg(), seed=21).solve()
    main = RecedingHorizonMCTS(_parts(), cfg, seed=21).solve()

    assert main.score_global >= baseline.score_global


def test_sequence_sa_lns_preserves_parts_and_legality():
    cfg = _cfg()
    cfg.BASELINE_RESTARTS = 2
    cfg.SA_ITERS = 8
    cfg.LNS_ENABLE = True
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)

    annealed_sequence, annealed_snapshot = mcts._sequence_sa_lns(base_sequence, base_snapshot)

    assert sorted(annealed_sequence) == sorted(base_sequence)
    validate_solution(
        _parts(),
        list(annealed_snapshot.boards),
        cfg.BOARD_W,
        cfg.BOARD_H,
        trim=cfg.TRIM,
        safe_gap=cfg.SAFE_GAP,
        touch_tol=cfg.TOUCH_TOL,
    )


def test_global_pattern_master_repair_replays_legally():
    cfg = _cfg()
    cfg.BOARD_W = 7.0
    cfg.BOARD_H = 7.0
    cfg.GLOBAL_PATTERN_MASTER_ENABLE = True
    cfg.GLOBAL_PATTERN_INCLUDE_BASELINE = True
    cfg.GLOBAL_LAYOUT_POOL_LIMIT = 8
    cfg.GLOBAL_PATTERN_LAYOUT_TOPK = 4
    cfg.GLOBAL_PATTERN_CAP = 64
    cfg.GLOBAL_PATTERN_TIME_LIMIT_S = 1.0
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=5)
    base_sequence = tuple(part.uid for part in _parts())
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)
    alt_sequence = tuple(reversed(base_sequence))
    alt_snapshot, _ = mcts.decoder.replay_order(alt_sequence)

    mcts._record_layout_candidate(base_sequence, base_snapshot, source="base")
    mcts._record_layout_candidate(alt_sequence, alt_snapshot, source="alt")
    rebuilt_sequence, rebuilt_snapshot = mcts._global_pattern_master_repair(base_sequence, base_snapshot)

    assert int(mcts.meta["global_pattern_attempts"]) == 1
    assert int(mcts.meta["global_pattern_patterns_generated"]) >= int(mcts.meta["global_pattern_patterns_kept"]) >= 1
    assert sorted(rebuilt_sequence) == sorted(base_sequence)
    validate_solution(
        _parts(),
        list(rebuilt_snapshot.boards),
        cfg.BOARD_W,
        cfg.BOARD_H,
        trim=cfg.TRIM,
        safe_gap=cfg.SAFE_GAP,
        touch_tol=cfg.TOUCH_TOL,
    )


def _write_parts_csv(path: Path) -> None:
    path.write_text(
        "upi,Fleng,Fwidth\n"
        "1,6,4\n"
        "2,4,6\n"
        "3,4,4\n"
        "4,3,3\n",
        encoding="utf-8",
    )


def test_run_one_seed_writes_resume_totals_file(tmp_path):
    cfg = _cfg()
    parts_csv = tmp_path / "parts.csv"
    _write_parts_csv(parts_csv)
    cfg.SAMPLE_CSV = str(parts_csv)
    cfg.CASE_NAME = "toy_case"
    cfg.OUT_ROOT = str(tmp_path)
    cfg.VARIANTS = ("baseline_greedy",)
    parts = _parts()[:4]
    out_case_dir = tmp_path / cfg.CASE_NAME

    row = run_one_seed(parts, cfg, seed=1000, out_case_dir=out_case_dir, variant="baseline_greedy")

    totals_path = out_case_dir / "baseline_greedy" / "seed_1000" / "totals_seed.json"
    assert totals_path.is_file()
    cached = json.loads(totals_path.read_text(encoding="utf-8"))
    assert int(cached["seed"]) == 1000
    assert str(cached["variant"]) == "baseline_greedy"
    assert int(cached["N_board"]) == int(row["N_board"])


def test_run_stepf_reuses_cached_seed_results(tmp_path, monkeypatch):
    cfg = _cfg()
    parts_csv = tmp_path / "parts.csv"
    _write_parts_csv(parts_csv)
    cfg.SAMPLE_CSV = str(parts_csv)
    cfg.CASE_NAME = "resume_case"
    cfg.OUT_ROOT = str(tmp_path)
    cfg.VARIANTS = ("baseline_greedy",)
    cfg.N_SEEDS = 1
    cfg.SEED0 = 1000
    cfg.N_JOBS = 1

    run_stepF(cfg)

    out_case_dir = Path(cfg.OUT_ROOT) / cfg.CASE_NAME
    cfg_variant = _variant_cfg(cfg, "baseline_greedy", _case_stats(_parts()[:4], cfg))
    cached = _load_cached_seed_totals(_parts()[:4], cfg_variant, out_case_dir, "baseline_greedy", 1000)
    assert cached is not None

    def _should_not_run(*args, **kwargs):
        raise AssertionError("run_one_seed should not be called for an already cached seed")

    monkeypatch.setattr("inrp.runner.run_one_seed", _should_not_run)
    run_stepF(cfg)


def test_beam_root_rank_prefers_pattern_heavier_paths_when_bonus_enabled():
    cfg = _cfg()
    cfg.BEAM_ENABLE = True
    cfg.BEAM_ROOT_PATTERN_BONUS = 0.5
    cfg.BEAM_ROOT_SINGLE_PENALTY = 0.2
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    snapshot = _dummy_snapshot(board_count=2, sum_u_gamma=1.5)
    pattern_root = SearchNode(
        snapshot=snapshot,
        remaining_mask=0,
        path_pattern_steps=3,
        path_single_steps=1,
    )
    single_root = SearchNode(
        snapshot=snapshot,
        remaining_mask=0,
        path_pattern_steps=0,
        path_single_steps=4,
    )

    assert mcts._beam_root_rank(pattern_root) > mcts._beam_root_rank(single_root)


def test_beam_completed_rank_prefers_pattern_heavier_finalists_when_bonus_enabled():
    cfg = _cfg()
    cfg.BEAM_ENABLE = True
    cfg.BEAM_COMPLETED_PATTERN_BONUS = 0.6
    cfg.BEAM_COMPLETED_SINGLE_PENALTY = 0.25
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    snapshot = _dummy_snapshot(board_count=2, sum_u_gamma=1.5)

    assert mcts._beam_completed_rank(snapshot, (3, 0, 0, 1)) > mcts._beam_completed_rank(snapshot, (0, 0, 0, 4))


def test_rollout_macro_prefers_pattern_in_early_stage_over_single_fallback():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.BEAM_ENABLE = True
    cfg.ROLLOUT_TOPR = 1
    cfg.ROLLOUT_DETERMINISTIC_TAIL = 8
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    snapshot = _dummy_snapshot(board_count=1, sum_u_gamma=0.8)
    remaining_mask = int(mcts.part_bit[1] | mcts.part_bit[2])

    def _fake_stage(_remaining_mask):
        return "early"

    def _fake_generate(_snapshot, _remaining, _limit):
        pattern_snapshot = _dummy_snapshot(board_count=1, sum_u_gamma=0.9)
        single_snapshot = _dummy_snapshot(board_count=1, sum_u_gamma=0.7)
        action_meta = {
            1001: MacroAction(
                key=1001,
                kind="pattern",
                part_ids=(1, 2),
                sequence=(1, 2),
                prior=0.55,
                rank_key=(0.55, 0.0),
                source="test-pattern",
                pred_util=0.9,
            ),
            1: MacroAction(
                key=1,
                kind="single",
                part_ids=(1,),
                sequence=(1,),
                prior=0.85,
                rank_key=(0.85, 0.0),
                source="test-single",
                pred_util=0.7,
            ),
        }
        decoded_map = {
            1001: AppliedActionResult(1001, (1, 2), pattern_snapshot, (0.0,), 0.9),
            1: AppliedActionResult(1, (1,), single_snapshot, (0.0,), 0.7),
        }
        return [1001, 1], action_meta, decoded_map

    mcts._search_stage = _fake_stage  # type: ignore[assignment]
    mcts._generate_macro_actions = _fake_generate  # type: ignore[assignment]

    final_snapshot, seq = mcts._rollout_macro(snapshot, remaining_mask)

    assert tuple(seq) == (1, 2)
    assert len(final_snapshot.boards) == 1
    assert int(mcts.meta["macro_rollout_pattern_steps"]) == 1
    assert int(mcts.meta["macro_rollout_single_fallback_steps"]) == 0
    assert int(mcts.meta["macro_rollout_early_single_fallbacks"]) == 0


def test_beam_reward_prefers_lower_tail_waste_and_open_risk():
    cfg = _cfg()
    cfg.MACRO_ACTION_ENABLE = True
    cfg.BEAM_ENABLE = True
    cfg.BEAM_REWARD_ENABLE = True
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    good = _dummy_snapshot(board_count=2, sum_u_gamma=1.2)
    bad = _dummy_snapshot(board_count=2, sum_u_gamma=1.21)

    def _fake_metrics(snapshot, _remaining):
        if snapshot.ordered_hash == good.ordered_hash:
            return {
                "tail_waste": 10.0,
                "sparse_count": 0.0,
                "open_risk": 0.0,
                "hard_tail_overlap": 2.0,
            }
        return {
            "tail_waste": 90.0,
            "sparse_count": 2.0,
            "open_risk": 3.0,
            "hard_tail_overlap": 0.0,
        }

    mcts._beam_closure_metrics = _fake_metrics  # type: ignore[assignment]

    assert mcts._reward_scalar(good, remaining_mask=3) > mcts._reward_scalar(bad, remaining_mask=3)


def test_beam_commit_prefers_higher_beam_rank_over_higher_visits():
    cfg = _cfg()
    cfg.BEAM_ENABLE = True
    cfg.BEAM_MIN_ROOT_VISITS = 1
    cfg.BEAM_MIN_CHILD_VISITS = 1
    cfg.BEAM_BRANCH_TOPK = 2
    cfg.COMMIT_MAX_STEPS = 1
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    root = SearchNode(snapshot=_dummy_snapshot(1, 0.5), remaining_mask=3, visits=4)
    child_a = SearchNode(snapshot=_dummy_snapshot(1, 0.5), remaining_mask=1, parent=root, visits=10, reward_sum=5.0)
    child_b = SearchNode(snapshot=_dummy_snapshot(1, 0.5), remaining_mask=2, parent=root, visits=2, reward_sum=1.0)
    root.children = {1: child_a, 2: child_b}

    def _fake_rank(node):
        return (1.0,) if node is child_b else (0.0,)

    mcts._beam_root_rank = _fake_rank  # type: ignore[assignment]

    chains = mcts._beam_commit_chains(root)

    assert chains
    assert chains[0][0][0] == 2


def test_beam_force_commit_rejects_closure_below_threshold():
    cfg = _cfg()
    cfg.BEAM_ENABLE = True
    cfg.BEAM_FORCE_MIN_CLOSURE_SCORE = 1.0
    mcts = RecedingHorizonMCTS(_parts(), cfg, seed=1)
    root = SearchNode(snapshot=_dummy_snapshot(1, 0.5), remaining_mask=3, visits=1)
    child = SearchNode(snapshot=_dummy_snapshot(1, 0.5), remaining_mask=1, parent=root, visits=1, reward_sum=1.0)
    root.children = {1: child}

    def _fake_metrics(_snapshot, _remaining):
        return {
            "last_util": 0.0,
            "tail_waste": 0.0,
            "sparse_count": 0.0,
            "hard_tail_overlap": 0.0,
            "open_risk": 0.0,
            "u_avg": 0.0,
            "u_global": 0.0,
            "avg_u_gamma": 0.0,
            "tail_capacity_eq": 0.0,
            "remaining_area_eq": 0.0,
        }

    mcts._beam_closure_metrics = _fake_metrics  # type: ignore[assignment]

    assert mcts._beam_commit_chains(root, force=True) == []


def test_autonomous_beam_loop_chooses_prefix_depth_push_when_baseline_is_tied_but_not_beaten():
    case_rows = {
        "data6": {
            "comparison": {"N_board_main": "151", "N_board_baseline": "151", "N_board_ref": "153"},
            "beam_meta": {
                "beam_active_states_max": 2.0,
                "prefix_candidates_generated": 4.0,
                "oracle_completion_calls": 10.0,
                "oracle_completion_cache_hits": 6.0,
            },
            "elapsed_s": 240.0,
            "beam_totals": {},
        },
        "data3": {
            "comparison": {"N_board_main": "176", "N_board_baseline": "175", "N_board_ref": "177"},
            "beam_meta": {
                "beam_active_states_max": 2.0,
                "prefix_candidates_generated": 4.0,
                "oracle_completion_calls": 12.0,
                "oracle_completion_cache_hits": 8.0,
            },
            "elapsed_s": 250.0,
            "beam_totals": {},
        },
        "I1_parts": {
            "comparison": {"N_board_main": "70", "N_board_baseline": "71", "N_board_ref": "70"},
            "beam_meta": {
                "beam_active_states_max": 2.0,
                "prefix_candidates_generated": 3.0,
                "oracle_completion_calls": 8.0,
                "oracle_completion_cache_hits": 5.0,
            },
            "elapsed_s": 220.0,
            "beam_totals": {},
        },
    }

    assert _choose_next_bundle(case_rows, used_bundles=(), current_overrides={}) == "prefix_depth_push"


def test_autonomous_beam_loop_chooses_prefix_branch_push_when_branching_is_too_low():
    case_rows = {
        "data6": {
            "comparison": {"N_board_main": "153", "N_board_baseline": "151", "N_board_ref": "153"},
            "beam_meta": {
                "beam_active_states_max": 1.0,
                "prefix_candidates_generated": 1.0,
                "oracle_completion_calls": 6.0,
                "oracle_completion_cache_hits": 2.0,
            },
            "elapsed_s": 210.0,
            "beam_totals": {},
        },
        "data3": {
            "comparison": {"N_board_main": "177", "N_board_baseline": "175", "N_board_ref": "177"},
            "beam_meta": {
                "beam_active_states_max": 1.0,
                "prefix_candidates_generated": 1.0,
                "oracle_completion_calls": 6.0,
                "oracle_completion_cache_hits": 2.0,
            },
            "elapsed_s": 215.0,
            "beam_totals": {},
        },
        "I1_parts": {
            "comparison": {"N_board_main": "70", "N_board_baseline": "71", "N_board_ref": "70"},
            "beam_meta": {
                "beam_active_states_max": 1.0,
                "prefix_candidates_generated": 1.0,
                "oracle_completion_calls": 4.0,
                "oracle_completion_cache_hits": 1.0,
            },
            "elapsed_s": 205.0,
            "beam_totals": {},
        },
    }

    assert _choose_next_bundle(case_rows, used_bundles=(), current_overrides={}) == "prefix_branch_push"


def test_autonomous_beam_loop_chooses_oracle_cache_push_when_runtime_is_high_and_cache_hits_are_low():
    case_rows = {
        "data6": {
            "comparison": {"N_board_main": "153", "N_board_baseline": "151", "N_board_ref": "153"},
            "beam_meta": {
                "beam_active_states_max": 2.0,
                "prefix_candidates_generated": 3.0,
                "oracle_completion_calls": 20.0,
                "oracle_completion_cache_hits": 2.0,
            },
            "elapsed_s": 290.0,
            "beam_totals": {},
        },
        "data3": {
            "comparison": {"N_board_main": "177", "N_board_baseline": "175", "N_board_ref": "177"},
            "beam_meta": {
                "beam_active_states_max": 2.0,
                "prefix_candidates_generated": 3.0,
                "oracle_completion_calls": 18.0,
                "oracle_completion_cache_hits": 2.0,
            },
            "elapsed_s": 286.0,
            "beam_totals": {},
        },
        "I1_parts": {
            "comparison": {"N_board_main": "70", "N_board_baseline": "71", "N_board_ref": "70"},
            "beam_meta": {
                "beam_active_states_max": 2.0,
                "prefix_candidates_generated": 3.0,
                "oracle_completion_calls": 12.0,
                "oracle_completion_cache_hits": 1.0,
            },
            "elapsed_s": 250.0,
            "beam_totals": {},
        },
    }

    assert _choose_next_bundle(case_rows, used_bundles=(), current_overrides={}) == "oracle_cache_push"


def test_autonomous_beam_promotion_requires_gate_and_hard_case_win():
    ready_rows = {
        "data6": {"comparison": {"N_board_main": "153", "N_board_baseline": "153", "N_board_ref": "154"}},
        "data3": {"comparison": {"N_board_main": "176", "N_board_baseline": "176", "N_board_ref": "177"}},
        "I1_parts": {"comparison": {"N_board_main": "70", "N_board_baseline": "71", "N_board_ref": "70"}},
    }
    blocked_rows = {
        "data6": {"comparison": {"N_board_main": "153", "N_board_baseline": "151", "N_board_ref": "153"}},
        "data3": {"comparison": {"N_board_main": "176", "N_board_baseline": "175", "N_board_ref": "176"}},
        "I1_parts": {"comparison": {"N_board_main": "70", "N_board_baseline": "71", "N_board_ref": "70"}},
    }

    assert _promotion_ready(ready_rows)
    assert not _promotion_ready(blocked_rows)
