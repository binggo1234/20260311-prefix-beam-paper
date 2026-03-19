from __future__ import annotations

import csv
import copy
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .dataio import read_sample_parts
from .geometry import effective_safe_gap, effective_trim, resolve_geometry_policy
from .mcts import OmnipotentDecoder, RecedingHorizonMCTS, SearchResult, solve_with_receding_horizon_mcts
from .packer import board_used_area
from .preprocess import preprocess_parts
from .repro import dump_config, set_global_seed
from .validate import validate_solution

logger = logging.getLogger(__name__)

RH_MCTS_REF_OVERRIDES: Dict[str, Any] = {
    "SOLVER_ROLE": "reference_core",
    "SOLVER_BUNDLE": "single_action_core",
    "ROOT_DIRICHLET_ENABLE": False,
    "MACRO_ACTION_ENABLE": False,
    "GLOBAL_PATTERN_MEMORY_ENABLE": False,
    "GLOBAL_PATTERN_MASTER_ENABLE": False,
    "GLOBAL_PATTERN_INCLUDE_BASELINE": False,
    "POST_REPAIR_TAIL_STRIKE_ENABLE": False,
    "POST_REPAIR_DYNAMIC_EXPAND_ENABLE": False,
    "POST_REPAIR_ALNS_ENABLE": False,
}

RH_MCTS_MAIN_OVERRIDES: Dict[str, Any] = {
    "SOLVER_ROLE": "paper_main",
    "SOLVER_BUNDLE": "mcts_plus_light_tail",
    "MCTS_N_SIM": 8,
    "MCTS_PUCT_C": 3.0,
    "PW_K0": 12,
    "PW_KMAX": 64,
    "DYN_POOL_MULT": 2,
    "DYN_POOL_MIN": 12,
    "DYN_POOL_MAX": 64,
    "DYN_TAIL_SAMPLES": 4,
    "COMMIT_MIN_ROOT_VISITS": 64,
    "COMMIT_MIN_CHILD_VISITS": 3,
    "COMMIT_CONFIRM_PASSES": 2,
    "COMMIT_MAX_STEPS": 1,
    "COMMIT_MAX_ROUND_SIMS": 0,
    "ROOT_DIRICHLET_ENABLE": True,
    "ROOT_DIRICHLET_ALPHA": 0.30,
    "ROOT_DIRICHLET_EPS": 0.20,
    "ROOT_DIRICHLET_MAX_DEPTH": 2,
    "ROLLOUT_RCL_TOPR": 3,
    "ROLLOUT_RCL_WEIGHTS": (0.60, 0.30, 0.10),
    "ROLLOUT_GREEDY_FRAC": 0.0,
    "ROLLOUT_DETERMINISTIC_TAIL": 8,
    "LOCAL_MAX_ITERS": 48,
    "LOCAL_FAIL_LIMIT": 8,
    "LOCAL_BLOCKER_TOPK": 5,
    "LOCAL_INSERT_MAX_POS": 12,
    "SOLVER_TIME_LIMIT_S": 900.0,
    "POST_MCTS_RESERVE_FRAC": 0.05,
    "POST_MCTS_RESERVE_MIN_S": 12.0,
    "POST_MCTS_RESERVE_MAX_S": 12.0,
    "SA_TIME_CAP_S": 60.0,
    "BASELINE_RESTARTS": 16,
    "RAND_TOPK": 4,
    "SA_ITERS": 5000,
    "SA_T0": 0.05,
    "SA_ALPHA": 0.997,
    "LNS_ENABLE": True,
    "LNS_PROB": 0.25,
    "LNS_DESTROY_FRAC": 0.10,
    "POST_REPAIR_WINDOWS_TOPK": 3,
    "POST_REPAIR_BLOCKER_BOARDS": 3,
    "POST_REPAIR_BLOCKER_TOPK": 20,
    "POST_REPAIR_PATTERNS_PER_MODE": 96,
    "POST_REPAIR_PATTERN_CAP": 512,
    "POST_REPAIR_PATTERN_MIN_UTIL": 0.85,
    "POST_REPAIR_PATTERN_STRONG_UTIL": 0.90,
    "POST_REPAIR_RANDOM_ORDERS": 32,
    "POST_REPAIR_RANDOM_TOPK": 6,
    "POST_REPAIR_EVICT_ENABLE": True,
    "POST_REPAIR_EVICT_HOST_BOARDS": 3,
    "POST_REPAIR_EVICT_CANDIDATES": 8,
    "POST_REPAIR_EVICT_TOPK": 6,
    "POST_REPAIR_TAIL_STRIKE_ENABLE": True,
    "POST_REPAIR_TAIL_STRIKE_WIDTHS": (2, 3, 4),
    "POST_REPAIR_TAIL_STRIKE_TOPK": 4,
    "POST_REPAIR_TAIL_STRIKE_PATTERN_MULT": 2,
    "POST_REPAIR_DYNAMIC_EXPAND_ENABLE": True,
    "POST_REPAIR_ALNS_ENABLE": True,
    "GLOBAL_PATTERN_MASTER_ENABLE": False,
    "GLOBAL_PATTERN_INCLUDE_BASELINE": True,
    "GLOBAL_LAYOUT_POOL_LIMIT": 128,
    "GLOBAL_PATTERN_LAYOUT_TOPK": 24,
    "GLOBAL_PATTERN_CAP": 6000,
    "GLOBAL_PATTERN_MIN_UTIL": 0.80,
    "GLOBAL_PATTERN_STRONG_UTIL": 0.90,
    "GLOBAL_PATTERN_TIME_LIMIT_S": 30.0,
    "GLOBAL_PATTERN_USE_BSSF": True,
    "GLOBAL_PATTERN_MEMORY_ENABLE": True,
    "GLOBAL_PATTERN_MEMORY_MIN_UTIL": 0.95,
    "GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL": 0.95,
    "GLOBAL_PATTERN_MEMORY_LIMIT": 24000,
    "GLOBAL_PATTERN_MEMORY_TOPK": 6000,
    "GLOBAL_PATTERN_MEMORY_FOCUS_TAIL_BOARDS": 3,
    "MACRO_ACTION_ENABLE": True,
    "MACRO_ACTION_TOPK": 12,
    "MACRO_MIN_PARTS": 2,
    "MACRO_MAX_PARTS": 6,
    "MACRO_PATTERN_TOPK": 4,
    "MACRO_ANCHOR_TOPK": 4,
    "MACRO_TAIL_TOPK": 3,
    "MACRO_SINGLE_TOPK": 3,
    "MACRO_PATTERN_UTIL_MIN": 0.92,
    "MACRO_MEMORY_UTIL_MIN": 0.95,
    "MACRO_EARLY_SINGLE_ONLY_RATIO": 0.0,
    "MACRO_STAGE_EARLY_RATIO": 0.35,
    "MACRO_STAGE_LATE_RATIO": 0.80,
    "MACRO_TAIL_TRIGGER_UTIL": 0.75,
    "MACRO_SPARSE_TRIGGER_UTIL": 0.60,
}

RH_MCTS_FULL_OVERRIDES: Dict[str, Any] = {
    "SOLVER_ROLE": "extended_full",
    "SOLVER_BUNDLE": "macro_plus_global_master",
    "MCTS_N_SIM": 8,
    "MCTS_PUCT_C": 3.0,
    "PW_K0": 12,
    "PW_KMAX": 64,
    "DYN_POOL_MULT": 2,
    "DYN_POOL_MIN": 12,
    "DYN_POOL_MAX": 64,
    "DYN_TAIL_SAMPLES": 4,
    "COMMIT_MIN_ROOT_VISITS": 64,
    "COMMIT_MIN_CHILD_VISITS": 3,
    "COMMIT_CONFIRM_PASSES": 2,
    "COMMIT_MAX_STEPS": 1,
    "COMMIT_MAX_ROUND_SIMS": 0,
    "ROOT_DIRICHLET_ENABLE": True,
    "ROOT_DIRICHLET_ALPHA": 0.30,
    "ROOT_DIRICHLET_EPS": 0.25,
    "ROOT_DIRICHLET_MAX_DEPTH": 3,
    "ROLLOUT_RCL_TOPR": 3,
    "ROLLOUT_RCL_WEIGHTS": (0.60, 0.30, 0.10),
    "ROLLOUT_GREEDY_FRAC": 0.0,
    "ROLLOUT_DETERMINISTIC_TAIL": 8,
    "LOCAL_MAX_ITERS": 48,
    "LOCAL_FAIL_LIMIT": 8,
    "LOCAL_BLOCKER_TOPK": 5,
    "LOCAL_INSERT_MAX_POS": 12,
    "SOLVER_TIME_LIMIT_S": 900.0,
    "POST_MCTS_RESERVE_FRAC": 0.05,
    "POST_MCTS_RESERVE_MIN_S": 12.0,
    "POST_MCTS_RESERVE_MAX_S": 12.0,
    "SA_TIME_CAP_S": 60.0,
    "BASELINE_RESTARTS": 16,
    "RAND_TOPK": 4,
    "SA_ITERS": 5000,
    "SA_T0": 0.05,
    "SA_ALPHA": 0.997,
    "LNS_ENABLE": True,
    "LNS_PROB": 0.25,
    "LNS_DESTROY_FRAC": 0.10,
    "POST_REPAIR_WINDOWS_TOPK": 3,
    "POST_REPAIR_BLOCKER_BOARDS": 3,
    "POST_REPAIR_BLOCKER_TOPK": 20,
    "POST_REPAIR_PATTERNS_PER_MODE": 96,
    "POST_REPAIR_PATTERN_CAP": 512,
    "POST_REPAIR_PATTERN_MIN_UTIL": 0.85,
    "POST_REPAIR_PATTERN_STRONG_UTIL": 0.90,
    "POST_REPAIR_RANDOM_ORDERS": 32,
    "POST_REPAIR_RANDOM_TOPK": 6,
    "POST_REPAIR_EVICT_ENABLE": True,
    "POST_REPAIR_EVICT_HOST_BOARDS": 3,
    "POST_REPAIR_EVICT_CANDIDATES": 8,
    "POST_REPAIR_EVICT_TOPK": 6,
    "POST_REPAIR_TAIL_STRIKE_ENABLE": True,
    "POST_REPAIR_TAIL_STRIKE_WIDTHS": (2, 3, 4),
    "POST_REPAIR_TAIL_STRIKE_TOPK": 4,
    "POST_REPAIR_TAIL_STRIKE_PATTERN_MULT": 2,
    "POST_REPAIR_DYNAMIC_EXPAND_ENABLE": True,
    "POST_REPAIR_INITIAL_WINDOW_WIDTH": 4,
    "POST_REPAIR_DYNAMIC_WINDOW_WIDTHS": (3, 5, 8, 10),
    "POST_REPAIR_DYNAMIC_FAST_SOLVE_S": 3.0,
    "POST_REPAIR_ALNS_ENABLE": True,
    "POST_REPAIR_ALNS_ALPHA": 0.20,
    "GLOBAL_PATTERN_MASTER_ENABLE": True,
    "GLOBAL_PATTERN_INCLUDE_BASELINE": True,
    "GLOBAL_LAYOUT_POOL_LIMIT": 128,
    "GLOBAL_PATTERN_LAYOUT_TOPK": 24,
    "GLOBAL_PATTERN_CAP": 6000,
    "GLOBAL_PATTERN_MIN_UTIL": 0.80,
    "GLOBAL_PATTERN_STRONG_UTIL": 0.90,
    "GLOBAL_PATTERN_TIME_LIMIT_S": 180.0,
    "GLOBAL_PATTERN_USE_BSSF": True,
    "GLOBAL_PATTERN_MEMORY_ENABLE": True,
    "GLOBAL_PATTERN_MEMORY_MIN_UTIL": 0.95,
    "GLOBAL_PATTERN_MEMORY_MASTER_MIN_UTIL": 0.95,
    "GLOBAL_PATTERN_MEMORY_LIMIT": 24000,
    "GLOBAL_PATTERN_MEMORY_TOPK": 6000,
    "GLOBAL_PATTERN_MEMORY_FOCUS_TAIL_BOARDS": 3,
    "MACRO_ACTION_ENABLE": True,
    "MACRO_ACTION_TOPK": 12,
    "MACRO_MIN_PARTS": 2,
    "MACRO_MAX_PARTS": 6,
    "MACRO_PATTERN_TOPK": 4,
    "MACRO_ANCHOR_TOPK": 4,
    "MACRO_TAIL_TOPK": 3,
    "MACRO_SINGLE_TOPK": 3,
    "MACRO_PATTERN_UTIL_MIN": 0.92,
    "MACRO_MEMORY_UTIL_MIN": 0.95,
    "MACRO_EARLY_SINGLE_ONLY_RATIO": 0.0,
    "MACRO_STAGE_EARLY_RATIO": 0.35,
    "MACRO_STAGE_LATE_RATIO": 0.80,
    "MACRO_TAIL_TRIGGER_UTIL": 0.75,
    "MACRO_SPARSE_TRIGGER_UTIL": 0.60,
}

RH_MCTS_BEAM_OVERRIDES: Dict[str, Any] = {
    "SOLVER_ROLE": "beam_main",
    "SOLVER_BUNDLE": "beam_macro_rh_mcts",
    "BEAM_ENABLE": True,
    "BEAM_WIDTH": 2,
    "BEAM_BRANCH_TOPK": 2,
    "BEAM_POSTPROCESS_TOPK": 2,
    "BEAM_MIN_ROOT_VISITS": 16,
    "BEAM_MIN_CHILD_VISITS": 1,
    "BEAM_REQUIRE_DOMINANCE": False,
    "BEAM_FOCUSED_REPAIR_ENABLE": True,
    "BEAM_FOCUSED_REPAIR_TOPK": 2,
    "BEAM_FOCUSED_REPAIR_WINDOW_WIDTHS": (2, 3, 4),
    "BEAM_DIVERSITY_TAIL_BOARDS": 2,
    "BEAM_DIVERSITY_SPARSE_TOPK": 2,
    "BEAM_DIVERSITY_UTIL_BUCKET": 0.02,
    "BEAM_CLOSURE_LAST_UTIL_W": 3.0,
    "BEAM_CLOSURE_TAIL_WASTE_W": 1.5,
    "BEAM_CLOSURE_SPARSE_W": 1.5,
    "BEAM_CLOSURE_HARD_FIT_W": 1.2,
    "BEAM_CLOSURE_OPEN_RISK_W": 2.0,
    "BEAM_CLOSURE_UAVG_W": 0.8,
    "BEAM_CLOSURE_UGLOBAL_W": 0.6,
    "BEAM_REWARD_ENABLE": True,
    "BEAM_REWARD_TAIL_WASTE_W": 0.35,
    "BEAM_REWARD_SPARSE_W": 0.25,
    "BEAM_REWARD_OPEN_RISK_W": 0.25,
    "BEAM_REWARD_HARD_FIT_W": 0.20,
    "BEAM_FORCE_MIN_CLOSURE_SCORE": -6.0,
    "BEAM_FORCE_REQUIRE_PATTERN_OR_TAIL": False,
    "BEAM_ROOT_PATTERN_BONUS": 0.0,
    "BEAM_ROOT_SINGLE_PENALTY": 0.0,
    "BEAM_ROOT_TAIL_PENALTY": 0.0,
    "BEAM_COMPLETED_PATTERN_BONUS": 0.0,
    "BEAM_COMPLETED_SINGLE_PENALTY": 0.0,
    "BEAM_COMPLETED_TAIL_PENALTY": 0.0,
    "BEAM_FOCUSED_REPAIR_LAST_UTIL_W": 1.0,
    "BEAM_FOCUSED_REPAIR_TAIL_WASTE_W": 1.0,
    "BEAM_FOCUSED_REPAIR_SMALL_WINDOW_BONUS": 0.25,
}

RH_MCTS_PREFIX_BEAM_OVERRIDES: Dict[str, Any] = {
    "SOLVER_ROLE": "prefix_beam_main",
    "SOLVER_BUNDLE": "prefix_beam_multi_oracle_light_repair",
    "PREFIX_BEAM_ENABLE": True,
    "PREFIX_BEAM_DEPTH": 3,
    "PREFIX_BEAM_WIDTH": 2,
    "PREFIX_BEAM_BRANCH_TOPK": 3,
    "PREFIX_BEAM_POST_TOPK": 2,
    "PREFIX_BEAM_USE_MACRO": True,
    "PREFIX_BEAM_ALLOW_SINGLE_FALLBACK": True,
    "PREFIX_BEAM_ORACLE_CACHE_ENABLE": True,
    "PREFIX_BEAM_ORACLE_MODES": ("baseline", "long_edge", "strip_bias"),
    "PREFIX_BEAM_LIGHT_REPAIR_ENABLE": True,
    "PREFIX_BEAM_LIGHT_REPAIR_TOPK": 1,
    "BEAM_ENABLE": True,
    "BEAM_WIDTH": 2,
    "BEAM_BRANCH_TOPK": 3,
    "BEAM_POSTPROCESS_TOPK": 2,
    "BEAM_MIN_ROOT_VISITS": 16,
    "BEAM_MIN_CHILD_VISITS": 1,
    "BEAM_REQUIRE_DOMINANCE": False,
    "BEAM_FOCUSED_REPAIR_ENABLE": False,
    "BEAM_REWARD_ENABLE": False,
    "ROOT_DIRICHLET_ENABLE": True,
    "ROOT_DIRICHLET_ALPHA": 0.30,
    "ROOT_DIRICHLET_EPS": 0.20,
    "ROOT_DIRICHLET_MAX_DEPTH": 2,
    "MACRO_ACTION_ENABLE": True,
    "MACRO_ACTION_TOPK": 12,
    "MACRO_PATTERN_TOPK": 4,
    "MACRO_ANCHOR_TOPK": 4,
    "MACRO_TAIL_TOPK": 2,
    "MACRO_SINGLE_TOPK": 2,
    "MACRO_PATTERN_UTIL_MIN": 0.92,
    "MACRO_MEMORY_UTIL_MIN": 0.95,
    "GLOBAL_PATTERN_MEMORY_ENABLE": True,
    "GLOBAL_PATTERN_MASTER_ENABLE": False,
    "GLOBAL_PATTERN_INCLUDE_BASELINE": True,
    "POST_REPAIR_ENABLE": False,
    "POST_REPAIR_TAIL_STRIKE_ENABLE": False,
    "POST_REPAIR_DYNAMIC_EXPAND_ENABLE": False,
    "POST_REPAIR_ALNS_ENABLE": False,
}

RH_MCTS_PROFILE_BASES: Dict[str, str] = {
    "rh_mcts": "rh_mcts_ref",
    "rh_mcts_main": "rh_mcts_ref",
    "rh_mcts_beam": "rh_mcts_main",
    "rh_mcts_prefix_beam": "rh_mcts_main",
    "rh_mcts_full": "rh_mcts_main",
    "ours_full": "rh_mcts_full",
    "ours_base": "rh_mcts_main",
    "ours_noprior": "ours_base",
}

RH_MCTS_PROFILE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "baseline_greedy": {"SOLVER_KIND": "baseline_greedy"},
    "baseline_sa": {
        "SOLVER_KIND": "baseline_sa",
        "MACRO_ACTION_ENABLE": False,
        "ROOT_DIRICHLET_ENABLE": False,
        "POST_REPAIR_ENABLE": False,
        "GLOBAL_PATTERN_MASTER_ENABLE": False,
        "GLOBAL_PATTERN_MEMORY_ENABLE": False,
        "SA_ITERS": 6000,
        "SA_TIME_CAP_S": 300.0,
        "SA_BUDGET_FRAC": 1.0,
    },
    "rh_mcts": {},
    "rh_mcts_ref": RH_MCTS_REF_OVERRIDES,
    "rh_mcts_main": RH_MCTS_MAIN_OVERRIDES,
    "rh_mcts_beam": RH_MCTS_BEAM_OVERRIDES,
    "rh_mcts_prefix_beam": RH_MCTS_PREFIX_BEAM_OVERRIDES,
    "rh_mcts_full": RH_MCTS_FULL_OVERRIDES,
    "ours_full": {},
    "ours_base": {
        "SOLVER_ROLE": "ablation_base",
        "SOLVER_BUNDLE": "main_without_repair",
        "POST_REPAIR_ENABLE": False,
        "GLOBAL_PATTERN_MASTER_ENABLE": False,
        "GLOBAL_PATTERN_MEMORY_ENABLE": False,
        "GLOBAL_PATTERN_INCLUDE_BASELINE": False,
    },
    "ours_noprior": {
        "SOLVER_ROLE": "ablation_noprior",
        "SOLVER_BUNDLE": "uniform_prior_control",
        "MACRO_ACTION_ENABLE": False,
        "DISABLE_GEOM_PRIOR": True,
        "ROOT_DIRICHLET_ENABLE": False,
        "ROLLOUT_PRIOR_W": 0.0,
        "POST_REPAIR_ENABLE": False,
        "GLOBAL_PATTERN_MASTER_ENABLE": False,
        "GLOBAL_PATTERN_MEMORY_ENABLE": False,
        "GLOBAL_PATTERN_INCLUDE_BASELINE": False,
    },
}

RH_MCTS_MAIN_ADAPTIVE_PROFILES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "small",
        "max_n_parts": 60,
        "overrides": {
            "MCTS_N_SIM": 2,
            "PW_K0": 8,
            "PW_KMAX": 32,
            "DYN_POOL_MIN": 8,
            "DYN_POOL_MAX": 24,
            "LOCAL_MAX_ITERS": 8,
            "SOLVER_TIME_LIMIT_S": 120.0,
            "POST_MCTS_RESERVE_MIN_S": 3.0,
            "POST_MCTS_RESERVE_MAX_S": 3.0,
            "SA_TIME_CAP_S": 8.0,
            "BASELINE_RESTARTS": 8,
            "SA_ITERS": 300,
            "POST_REPAIR_WINDOWS_TOPK": 2,
            "POST_REPAIR_PATTERNS_PER_MODE": 24,
            "POST_REPAIR_PATTERN_CAP": 160,
            "POST_REPAIR_RANDOM_ORDERS": 8,
            "POST_REPAIR_RANDOM_TOPK": 4,
            "GLOBAL_LAYOUT_POOL_LIMIT": 24,
            "GLOBAL_PATTERN_LAYOUT_TOPK": 8,
            "GLOBAL_PATTERN_MEMORY_TOPK": 256,
            "MACRO_ACTION_TOPK": 8,
            "MACRO_PATTERN_TOPK": 3,
            "MACRO_ANCHOR_TOPK": 2,
            "MACRO_TAIL_TOPK": 2,
            "MACRO_SINGLE_TOPK": 2,
        },
    },
    {
        "name": "medium",
        "max_n_parts": 180,
        "overrides": {
            "PW_K0": 10,
            "PW_KMAX": 48,
            "DYN_POOL_MIN": 10,
            "DYN_POOL_MAX": 40,
            "LOCAL_MAX_ITERS": 12,
            "SOLVER_TIME_LIMIT_S": 360.0,
            "POST_MCTS_RESERVE_MIN_S": 6.0,
            "POST_MCTS_RESERVE_MAX_S": 6.0,
            "SA_TIME_CAP_S": 16.0,
            "BASELINE_RESTARTS": 12,
            "SA_ITERS": 700,
            "POST_REPAIR_WINDOWS_TOPK": 3,
            "POST_REPAIR_PATTERNS_PER_MODE": 48,
            "POST_REPAIR_PATTERN_CAP": 256,
            "POST_REPAIR_RANDOM_ORDERS": 16,
            "GLOBAL_LAYOUT_POOL_LIMIT": 48,
            "GLOBAL_PATTERN_LAYOUT_TOPK": 12,
            "GLOBAL_PATTERN_MEMORY_TOPK": 1024,
            "MACRO_ACTION_TOPK": 10,
            "MACRO_PATTERN_TOPK": 4,
            "MACRO_ANCHOR_TOPK": 3,
            "MACRO_TAIL_TOPK": 2,
            "MACRO_SINGLE_TOPK": 2,
        },
    },
)

MCTS_MAIN_VARIANTS = {
    "rh_mcts_main",
    "rh_mcts_beam",
    "rh_mcts_prefix_beam",
    "rh_mcts_full",
    "ours_full",
    "ours_base",
    "ours_noprior",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        return

    preferred = [
        "case",
        "variant",
        "seed",
        "board",
        "uid",
        "pid_raw",
        "N_input",
        "N_board",
        "n_parts",
        "used_area",
        "waste_area",
        "board_area",
        "part_area_sum",
        "board_area_effective",
        "LB_area",
        "N_board_gap_to_LB",
        "U",
        "U_avg",
        "U_avg_excl_last",
        "U_avg_excl_last2",
        "U_global",
        "U_last",
        "U_worst_excl_last",
        "U_best_board",
        "U_worst_board",
        "last2_used_equiv",
        "sparse_board_count",
        "tail_waste_area",
        "U_avg_pct",
        "U_avg_excl_last_pct",
        "U_avg_excl_last2_pct",
        "U_global_pct",
        "U_last_pct",
        "U_worst_excl_last_pct",
        "U_best_board_pct",
        "U_worst_board_pct",
        "runtime_s",
        "x",
        "y",
        "w",
        "h",
        "w0",
        "h0",
        "rotated",
    ]

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = []
    for key in preferred:
        if key in all_keys:
            fieldnames.append(key)
            all_keys.remove(key)
    fieldnames.extend(sorted(all_keys))

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _read_rows_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _safe_gap(cfg) -> float:
    return effective_safe_gap(cfg)


def _compute_board_metrics(board) -> Dict[str, Any]:
    board_area = float(board.W) * float(board.H)
    used_area = board_used_area(board)
    waste_area = max(0.0, board_area - used_area)
    util = used_area / board_area if board_area > 0 else 0.0
    return {
        "board": int(board.bid),
        "n_parts": len(board.placed),
        "used_area": used_area,
        "waste_area": waste_area,
        "board_area": board_area,
        "U": util,
    }


def _effective_board_area(cfg) -> float:
    trim = effective_trim(cfg)
    usable_w = max(0.0, float(getattr(cfg, "BOARD_W", 0.0)) - 2.0 * trim)
    usable_h = max(0.0, float(getattr(cfg, "BOARD_H", 0.0)) - 2.0 * trim)
    return usable_w * usable_h


def _area_lower_bound(part_area_sum: float, cfg) -> int:
    board_area_effective = _effective_board_area(cfg)
    if board_area_effective <= 0.0:
        return 0
    return int(math.ceil(max(0.0, float(part_area_sum)) / board_area_effective))


def _placement_rows(boards: Sequence) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for board in boards:
        for pp in board.placed:
            rows.append(
                {
                    "board": int(board.bid),
                    "uid": int(pp.uid),
                    "pid_raw": str(pp.pid_raw),
                    "x": float(pp.rect.x),
                    "y": float(pp.rect.y),
                    "w": float(pp.rect.w),
                    "h": float(pp.rect.h),
                    "w0": float(pp.w0),
                    "h0": float(pp.h0),
                    "rotated": int(bool(pp.rotated)),
                }
            )
    return rows


def _sequence_rows(sequence: Sequence[int]) -> List[Dict[str, Any]]:
    return [{"step": i + 1, "uid": int(uid)} for i, uid in enumerate(sequence)]


def _solver_payload(result: SearchResult) -> Tuple[Sequence, Dict[str, Any], Optional[Sequence[int]]]:
    return result.boards, dict(result.meta), result.sequence


def _normalize_variants(cfg) -> List[str]:
    raw = getattr(cfg, "VARIANTS", ("rh_mcts",))
    if isinstance(raw, str):
        items = [raw]
    else:
        items = list(raw)
    out: List[str] = []
    seen = set()
    for item in items:
        variant = str(item).strip()
        if not variant or variant in seen:
            continue
        seen.add(variant)
        out.append(variant)
    return out or ["rh_mcts"]


def _clone_cfg(cfg):
    if hasattr(cfg, "__dict__"):
        data = {key: copy.deepcopy(getattr(cfg, key)) for key in dir(cfg) if key.isupper() and not key.startswith("_")}
        return SimpleNamespace(**data)
    return SimpleNamespace(**_cfg_dict(cfg))


def _profile_overrides(variant: str) -> Dict[str, Any]:
    variant_key = str(variant)
    merged: Dict[str, Any] = {}
    base = RH_MCTS_PROFILE_BASES.get(variant_key)
    if base:
        merged.update(_profile_overrides(base))
    merged.update(copy.deepcopy(RH_MCTS_PROFILE_OVERRIDES.get(variant_key, {})))
    return merged


def _case_stats(parts: Sequence, cfg) -> Dict[str, Any]:
    part_area_sum = sum(float(part.w0) * float(part.h0) for part in parts)
    duplicate_parts = sum(max(0, int(getattr(part, "dup_count", 1)) - 1) for part in parts if int(getattr(part, "dup_rank", 1)) == 1)
    return {
        "n_parts": len(parts),
        "part_area_sum": float(part_area_sum),
        "lb_area": _area_lower_bound(float(part_area_sum), cfg),
        "n_distinct_shapes": len({str(getattr(part, "dup_group", "")) for part in parts if str(getattr(part, "dup_group", ""))}),
        "duplicate_parts": int(duplicate_parts),
        "large_part_count": sum(1 for part in parts if bool(getattr(part, "is_large", False))),
        "strip_part_count": sum(1 for part in parts if bool(getattr(part, "is_strip", False))),
    }


def _adaptive_main_profile(case_stats: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    if not case_stats:
        return "large", {}
    n_parts = int(case_stats.get("n_parts", 0) or 0)
    for profile in RH_MCTS_MAIN_ADAPTIVE_PROFILES:
        if n_parts <= int(profile.get("max_n_parts", 0) or 0):
            return str(profile["name"]), copy.deepcopy(profile["overrides"])
    return "large", {}


def _cap_runtime_budget(cfg_base, cfg_variant) -> None:
    base_limit = float(getattr(cfg_base, "SOLVER_TIME_LIMIT_S", getattr(cfg_variant, "SOLVER_TIME_LIMIT_S", 0.0)) or 0.0)
    variant_limit = float(getattr(cfg_variant, "SOLVER_TIME_LIMIT_S", base_limit) or 0.0)
    if base_limit <= 0.0 or not math.isfinite(base_limit):
        cfg_variant.SOLVER_TIME_LIMIT_S = 0.0
        cfg_variant.POST_MCTS_RESERVE_MIN_S = 0.0
        cfg_variant.POST_MCTS_RESERVE_MAX_S = 0.0
        cfg_variant.SA_TIME_CAP_S = 0.0
        cfg_variant.POST_REPAIR_TIME_LIMIT_S = 0.0
        cfg_variant.GLOBAL_PATTERN_TIME_LIMIT_S = 0.0
        return
    if base_limit > 0.0 and variant_limit > 0.0:
        solver_limit = min(base_limit, variant_limit)
        cfg_variant.SOLVER_TIME_LIMIT_S = solver_limit
        reserve_cap = max(0.0, solver_limit * 0.5)
        cfg_variant.POST_MCTS_RESERVE_MIN_S = min(
            float(getattr(cfg_variant, "POST_MCTS_RESERVE_MIN_S", 0.0) or 0.0),
            reserve_cap,
        )
        cfg_variant.POST_MCTS_RESERVE_MAX_S = min(
            float(getattr(cfg_variant, "POST_MCTS_RESERVE_MAX_S", 0.0) or 0.0),
            reserve_cap,
        )
        post_repair_limit = float(getattr(cfg_variant, "POST_REPAIR_TIME_LIMIT_S", 0.0) or 0.0)
        if post_repair_limit > 0.0:
            cfg_variant.POST_REPAIR_TIME_LIMIT_S = min(post_repair_limit, solver_limit)
        global_pattern_limit = float(getattr(cfg_variant, "GLOBAL_PATTERN_TIME_LIMIT_S", 0.0) or 0.0)
        if global_pattern_limit > 0.0:
            cfg_variant.GLOBAL_PATTERN_TIME_LIMIT_S = min(global_pattern_limit, solver_limit)


def _variant_cfg(cfg, variant: str, case_stats: Optional[Dict[str, Any]] = None):
    cfg_variant = _clone_cfg(cfg)
    for key, value in _profile_overrides(str(variant)).items():
        setattr(cfg_variant, key, copy.deepcopy(value))
    profile_name = "base"
    if str(variant) in MCTS_MAIN_VARIANTS:
        profile_name, adaptive_overrides = _adaptive_main_profile(case_stats)
        for key, value in adaptive_overrides.items():
            setattr(cfg_variant, key, copy.deepcopy(value))
    extra_variant_overrides = copy.deepcopy(getattr(cfg, "VARIANT_EXTRA_OVERRIDES", {}) or {})
    variant_extra = extra_variant_overrides.get(str(variant), {})
    for key, value in variant_extra.items():
        setattr(cfg_variant, str(key), copy.deepcopy(value))
    cfg_variant.MAIN_PROFILE_TIER = str(profile_name)
    if case_stats:
        cfg_variant.MAIN_PROFILE_N_PARTS = int(case_stats.get("n_parts", 0) or 0)
        cfg_variant.MAIN_PROFILE_LB_AREA = int(case_stats.get("lb_area", 0) or 0)
    _cap_runtime_budget(cfg, cfg_variant)
    cfg_variant.VARIANTS = (str(variant),)
    return cfg_variant


def _comparison_table_rows(totals_all: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for row in totals_all:
        by_seed.setdefault(int(row["seed"]), {})[str(row["variant"])] = row

    for seed in sorted(by_seed):
        variants = by_seed[seed]
        baseline = variants.get("baseline_greedy")
        ref = variants.get("rh_mcts_ref") or variants.get("rh_mcts")
        main = variants.get("rh_mcts_prefix_beam") or variants.get("rh_mcts_main") or variants.get("rh_mcts_beam")
        if ref is None or main is None:
            continue
        row = {
            "case": main["case"],
            "seed": seed,
            "LB_area": main.get("LB_area", ref.get("LB_area", 0)),
            "variant_ref": ref["variant"],
            "variant_main": main["variant"],
            "N_board_ref": ref.get("N_board", 0),
            "N_board_ref_gap_to_LB": ref.get("N_board_gap_to_LB", 0),
            "N_board_main": main.get("N_board", 0),
            "N_board_main_gap_to_LB": main.get("N_board_gap_to_LB", 0),
            "N_board_delta": float(main.get("N_board", 0)) - float(ref.get("N_board", 0)),
            "U_global_ref": ref.get("U_global", 0.0),
            "U_global_main": main.get("U_global", 0.0),
            "U_global_delta": float(main.get("U_global", 0.0)) - float(ref.get("U_global", 0.0)),
            "U_avg_excl_last_ref": ref.get("U_avg_excl_last", 0.0),
            "U_avg_excl_last_main": main.get("U_avg_excl_last", 0.0),
            "U_avg_excl_last_delta": float(main.get("U_avg_excl_last", 0.0))
            - float(ref.get("U_avg_excl_last", 0.0)),
            "runtime_s_ref": ref.get("runtime_s", 0.0),
            "runtime_s_main": main.get("runtime_s", 0.0),
            "runtime_s_delta": float(main.get("runtime_s", 0.0)) - float(ref.get("runtime_s", 0.0)),
        }
        if baseline is not None:
            row.update(
                {
                    "variant_baseline": baseline["variant"],
                    "N_board_baseline": baseline.get("N_board", 0),
                    "N_board_baseline_gap_to_LB": baseline.get("N_board_gap_to_LB", 0),
                    "U_global_baseline": baseline.get("U_global", 0.0),
                    "runtime_s_baseline": baseline.get("runtime_s", 0.0),
                    "N_board_main_vs_baseline": float(main.get("N_board", 0)) - float(baseline.get("N_board", 0)),
                    "U_global_main_vs_baseline": float(main.get("U_global", 0.0))
                    - float(baseline.get("U_global", 0.0)),
                }
            )
        rows.append(row)
    return rows


def _mean_excl_tail(utils: List[float], tail_count: int) -> float:
    if not utils:
        return 0.0
    keep = max(1, len(utils) - max(1, int(tail_count)))
    return mean(utils[:keep])


def _min_excl_tail(utils: List[float], tail_count: int) -> float:
    if not utils:
        return 0.0
    keep = max(1, len(utils) - max(1, int(tail_count)))
    return min(utils[:keep])


def aggregate_totals(board_rows: List[Dict[str, Any]], *, sparse_util: float = 0.55) -> Dict[str, Any]:
    if not board_rows:
        return {
            "N_board": 0,
            "n_parts_sum": 0,
            "used_area_sum": 0.0,
            "waste_area_sum": 0.0,
            "board_area_sum": 0.0,
            "U_avg": 0.0,
            "U_avg_excl_last": 0.0,
            "U_avg_excl_last2": 0.0,
            "U_global": 0.0,
            "U_last": 0.0,
            "U_worst_excl_last": 0.0,
            "U_best_board": 0.0,
            "U_worst_board": 0.0,
            "last2_used_equiv": 0.0,
            "sparse_board_count": 0,
            "tail_waste_area": 0.0,
        }

    utils = [float(row["U"]) for row in board_rows]
    used_area_sum = sum(float(row["used_area"]) for row in board_rows)
    waste_area_sum = sum(float(row["waste_area"]) for row in board_rows)
    board_area_sum = sum(float(row["board_area"]) for row in board_rows)
    tail_rows = board_rows[-min(2, len(board_rows)) :]
    board_area_ref = float(board_rows[-1]["board_area"]) if board_rows else 0.0
    last2_used_equiv = (
        sum(float(row["used_area"]) for row in tail_rows) / board_area_ref if board_area_ref > 0.0 else 0.0
    )

    return {
        "N_board": len(board_rows),
        "n_parts_sum": int(sum(int(row["n_parts"]) for row in board_rows)),
        "used_area_sum": used_area_sum,
        "waste_area_sum": waste_area_sum,
        "board_area_sum": board_area_sum,
        "U_avg": mean(utils),
        "U_avg_excl_last": _mean_excl_tail(utils, 1),
        "U_avg_excl_last2": _mean_excl_tail(utils, 2),
        "U_global": (used_area_sum / board_area_sum) if board_area_sum > 0 else 0.0,
        "U_last": utils[-1],
        "U_worst_excl_last": _min_excl_tail(utils, 1),
        "U_best_board": max(utils),
        "U_worst_board": min(utils),
        "last2_used_equiv": last2_used_equiv,
        "sparse_board_count": int(sum(1 for u in utils if u < float(sparse_util))),
        "tail_waste_area": float(board_rows[-1]["waste_area"]),
    }


def _utilization_table_rows(totals_all: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in sorted(totals_all, key=lambda item: (str(item["variant"]), int(item["seed"]))):
        rows.append(
            {
                "case": row["case"],
                "variant": row["variant"],
                "seed": row["seed"],
                "N_input": row["N_input"],
                "N_board": row["N_board"],
                "U_avg": row.get("U_avg", 0.0),
                "U_avg_pct": 100.0 * float(row.get("U_avg", 0.0)),
                "U_avg_excl_last": row.get("U_avg_excl_last", 0.0),
                "U_avg_excl_last_pct": 100.0 * float(row.get("U_avg_excl_last", 0.0)),
                "U_avg_excl_last2": row.get("U_avg_excl_last2", 0.0),
                "U_avg_excl_last2_pct": 100.0 * float(row.get("U_avg_excl_last2", 0.0)),
                "U_global": row.get("U_global", 0.0),
                "U_global_pct": 100.0 * float(row.get("U_global", 0.0)),
                "U_last": row.get("U_last", 0.0),
                "U_last_pct": 100.0 * float(row.get("U_last", 0.0)),
                "U_worst_excl_last": row.get("U_worst_excl_last", 0.0),
                "U_worst_excl_last_pct": 100.0 * float(row.get("U_worst_excl_last", 0.0)),
                "U_best_board": row.get("U_best_board", 0.0),
                "U_best_board_pct": 100.0 * float(row.get("U_best_board", 0.0)),
                "U_worst_board": row.get("U_worst_board", 0.0),
                "U_worst_board_pct": 100.0 * float(row.get("U_worst_board", 0.0)),
                "last2_used_equiv": row.get("last2_used_equiv", 0.0),
                "sparse_board_count": row.get("sparse_board_count", 0),
                "tail_waste_area": row.get("tail_waste_area", 0.0),
                "runtime_s": row.get("runtime_s", 0.0),
            }
        )
    return rows


def _utilization_summary_rows(totals_all: List[Dict[str, Any]], variants: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metrics = [
        "U_avg",
        "U_avg_excl_last",
        "U_avg_excl_last2",
        "U_global",
        "U_last",
        "U_worst_excl_last",
        "U_best_board",
        "U_worst_board",
    ]

    for variant in variants:
        data = [row for row in totals_all if row["variant"] == variant]
        out: Dict[str, Any] = {"variant": variant, "n": len(data)}
        for metric in metrics:
            xs = [100.0 * float(row.get(metric, 0.0)) for row in data]
            out[f"{metric}_pct_mean"] = mean(xs) if xs else 0.0
            out[f"{metric}_pct_std"] = stdev(xs) if len(xs) > 1 else 0.0
        rows.append(out)
    return rows


def _summary_stats(totals_all: List[Dict[str, Any]], variants: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metrics = [
        "N_board",
        "U_avg",
        "U_avg_excl_last",
        "U_avg_excl_last2",
        "U_global",
        "U_last",
        "U_worst_excl_last",
        "used_area_sum",
        "waste_area_sum",
        "last2_used_equiv",
        "sparse_board_count",
        "tail_waste_area",
        "runtime_s",
    ]

    for variant in variants:
        data = [row for row in totals_all if row["variant"] == variant]
        out: Dict[str, Any] = {"variant": variant, "n": len(data)}
        for metric in metrics:
            xs = [float(row.get(metric, 0.0)) for row in data]
            out[f"{metric}_mean"] = mean(xs) if xs else 0.0
            out[f"{metric}_std"] = stdev(xs) if len(xs) > 1 else 0.0
        rows.append(out)
    return rows


def _solve_rh_mcts(parts, cfg, seed: int) -> SearchResult:
    return solve_with_receding_horizon_mcts(parts, cfg, seed)


def _baseline_sequence(parts: Sequence) -> Tuple[int, ...]:
    ranked = sorted(
        parts,
        key=lambda part: (
            float(part.w0) * float(part.h0),
            max(float(part.w0), float(part.h0)) / max(1e-9, min(float(part.w0), float(part.h0))),
            float(max(part.w0, part.h0)),
            -int(part.uid),
        ),
        reverse=True,
    )
    return tuple(int(part.uid) for part in ranked)


def _solve_baseline_greedy(parts, cfg, seed: int) -> SearchResult:
    del seed
    decoder = OmnipotentDecoder({int(part.uid): part for part in parts}, cfg)
    sequence = _baseline_sequence(parts)
    snapshot, _ = decoder.replay_order(sequence)
    return SearchResult(
        boards=list(snapshot.boards),
        sequence=sequence,
        score_global=tuple(),
        scalar_reward=0.0,
        meta={
            "solver_kind": "baseline_greedy",
            "baseline_order": "area_aspect_desc",
            "decode_cache_hits": float(decoder.decode_hits),
            "decode_cache_misses": float(decoder.decode_misses),
            "decode_cache_entries": float(len(decoder.decode_cache)),
            "micro_cache_hits": float(decoder.micro_hits),
            "micro_cache_misses": float(decoder.micro_misses),
            "micro_cache_entries": float(len(decoder.micro_cache)),
        },
    )


def _solve_baseline_sa(parts, cfg, seed: int) -> SearchResult:
    mcts = RecedingHorizonMCTS(parts, cfg, seed)
    mcts.solve_deadline = time.perf_counter() + mcts.solver_time_limit_s if mcts.solver_time_limit_s > 0.0 else math.inf
    base_sequence = mcts._area_aspect_sequence()
    base_snapshot, _ = mcts.decoder.replay_order(base_sequence)
    annealed_sequence, annealed_snapshot = mcts._sequence_sa_lns(base_sequence, base_snapshot)
    final_sequence = tuple(int(pid) for pid in annealed_sequence)
    final_snapshot, _ = OmnipotentDecoder(mcts.parts_by_id, cfg).replay_order(final_sequence)
    mcts.meta["solver_kind"] = "baseline_sa"
    mcts.meta["baseline_order"] = "area_aspect_desc"
    mcts.meta["result_source"] = "sa_lns"
    mcts.meta["decode_cache_hits"] = float(mcts.decoder.decode_hits)
    mcts.meta["decode_cache_misses"] = float(mcts.decoder.decode_misses)
    mcts.meta["decode_cache_entries"] = float(len(mcts.decoder.decode_cache))
    mcts.meta["micro_cache_hits"] = float(mcts.decoder.micro_hits)
    mcts.meta["micro_cache_misses"] = float(mcts.decoder.micro_misses)
    mcts.meta["micro_cache_entries"] = float(len(mcts.decoder.micro_cache))
    return SearchResult(
        boards=list(final_snapshot.boards),
        sequence=final_sequence,
        score_global=mcts._terminal_score(final_snapshot),
        scalar_reward=mcts._reward_scalar(final_snapshot, 0),
        meta=dict(mcts.meta),
    )


def _seed_totals_path(out_seed_dir: Path) -> Path:
    return out_seed_dir / "totals_seed.json"


def _estimate_seed_runtime_s(out_seed_dir: Path) -> float:
    mtimes = []
    for name in ("boards_metrics.csv", "placements.csv", "sequence.csv", "search_meta.json", "totals_seed.json"):
        path = out_seed_dir / name
        if path.is_file():
            mtimes.append(path.stat().st_mtime)
    if len(mtimes) < 2:
        return 0.0
    return max(0.0, float(max(mtimes) - min(mtimes)))


def _reconstruct_seed_totals(parts, cfg, out_case_dir: Path, variant: str, seed: int) -> Optional[Dict[str, Any]]:
    out_seed_dir = out_case_dir / variant / f"seed_{seed}"
    boards_path = out_seed_dir / "boards_metrics.csv"
    placements_path = out_seed_dir / "placements.csv"
    if not boards_path.is_file() or not placements_path.is_file():
        return None

    board_rows = _read_rows_csv(boards_path)
    if not board_rows:
        return None

    solver_meta: Dict[str, Any] = {}
    meta_path = out_seed_dir / "search_meta.json"
    if meta_path.is_file():
        try:
            solver_meta = _read_json(meta_path)
        except Exception:
            solver_meta = {}

    totals_row = aggregate_totals(board_rows, sparse_util=float(getattr(cfg, "SPILL_SPARSE_UTIL", 0.55)))
    totals_row.update(
        {
            "case": getattr(cfg, "CASE_NAME", ""),
            "variant": str(variant),
            "seed": int(seed),
            "N_input": len(parts),
            "runtime_s": float(solver_meta.get("runtime_s", _estimate_seed_runtime_s(out_seed_dir))),
        }
    )
    totals_row["part_area_sum"] = float(totals_row.get("used_area_sum", 0.0))
    totals_row["board_area_effective"] = _effective_board_area(cfg)
    totals_row["LB_area"] = _area_lower_bound(float(totals_row["part_area_sum"]), cfg)
    totals_row["N_board_gap_to_LB"] = float(totals_row.get("N_board", 0)) - float(totals_row["LB_area"])
    totals_row.update(solver_meta)
    totals_row["result_source"] = str(totals_row.get("result_source", "resume_cache"))
    totals_row["resume_cache_hit"] = 1
    _write_json(_seed_totals_path(out_seed_dir), totals_row)
    return totals_row


def _load_cached_seed_totals(parts, cfg, out_case_dir: Path, variant: str, seed: int) -> Optional[Dict[str, Any]]:
    out_seed_dir = out_case_dir / variant / f"seed_{seed}"
    totals_path = _seed_totals_path(out_seed_dir)
    if totals_path.is_file():
        try:
            row = _read_json(totals_path)
            if row:
                row["resume_cache_hit"] = 1
                return row
        except Exception:
            pass
    return _reconstruct_seed_totals(parts, cfg, out_case_dir, variant, seed)


def run_one_seed(parts, cfg, seed: int, out_case_dir: Path, variant: str) -> Dict[str, Any]:
    set_global_seed(seed)
    variant = str(variant)

    out_seed_dir = out_case_dir / variant / f"seed_{seed}"
    _ensure_dir(out_seed_dir)
    cached = _load_cached_seed_totals(parts, cfg, out_case_dir, variant, seed)
    if cached is not None:
        return cached

    t0 = time.perf_counter()
    solver_kind = str(getattr(cfg, "SOLVER_KIND", "rh_mcts"))
    if solver_kind == "baseline_greedy":
        solver_result = _solve_baseline_greedy(parts, cfg, seed)
    elif solver_kind == "baseline_sa":
        solver_result = _solve_baseline_sa(parts, cfg, seed)
    else:
        solver_result = _solve_rh_mcts(parts, cfg, seed)
    boards, solver_meta, sequence = _solver_payload(solver_result)
    solver_meta = dict(solver_meta)
    solver_meta.setdefault("solver_kind", solver_kind)
    solver_meta.setdefault("solver_role", str(getattr(cfg, "SOLVER_ROLE", "")))
    solver_meta.setdefault("solver_bundle", str(getattr(cfg, "SOLVER_BUNDLE", "")))
    solver_meta.setdefault("preprocess_distinct_shapes", int(getattr(cfg, "PREPROCESS_N_DISTINCT_SHAPES", 0) or 0))
    solver_meta.setdefault("preprocess_duplicate_parts", int(getattr(cfg, "PREPROCESS_DUPLICATE_PARTS", 0) or 0))
    solver_meta.setdefault("preprocess_large_part_count", int(getattr(cfg, "PREPROCESS_LARGE_PART_COUNT", 0) or 0))
    solver_meta.setdefault("preprocess_strip_part_count", int(getattr(cfg, "PREPROCESS_STRIP_PART_COUNT", 0) or 0))
    solver_meta.setdefault("preprocess_quantized_parts", int(getattr(cfg, "PREPROCESS_QUANTIZED_PARTS", 0) or 0))
    solver_meta.setdefault("main_profile_tier", str(getattr(cfg, "MAIN_PROFILE_TIER", "base")))
    solver_meta.setdefault("main_profile_n_parts", int(getattr(cfg, "MAIN_PROFILE_N_PARTS", len(parts)) or len(parts)))
    solver_meta.setdefault("main_profile_lb_area", int(getattr(cfg, "MAIN_PROFILE_LB_AREA", 0) or 0))
    validate_solution(
        parts,
        boards,
        cfg.BOARD_W,
        cfg.BOARD_H,
        trim=effective_trim(cfg),
        safe_gap=_safe_gap(cfg),
        touch_tol=float(getattr(cfg, "TOUCH_TOL", 1e-6) or 0.0),
    )

    board_rows = [_compute_board_metrics(board) for board in boards]
    placement_rows = _placement_rows(boards)
    _write_rows_csv(out_seed_dir / "boards_metrics.csv", board_rows)
    _write_rows_csv(out_seed_dir / "placements.csv", placement_rows)
    if sequence:
        _write_rows_csv(out_seed_dir / "sequence.csv", _sequence_rows(sequence))
    if solver_meta:
        with open(out_seed_dir / "search_meta.json", "w", encoding="utf-8") as fh:
            json.dump(solver_meta, fh, ensure_ascii=False, indent=2)

    totals_row = aggregate_totals(board_rows, sparse_util=float(getattr(cfg, "SPILL_SPARSE_UTIL", 0.55)))
    totals_row.update(
        {
            "case": cfg.CASE_NAME,
            "variant": variant,
            "seed": seed,
            "N_input": len(parts),
            "runtime_s": time.perf_counter() - t0,
        }
    )
    totals_row["part_area_sum"] = float(totals_row.get("used_area_sum", 0.0))
    totals_row["board_area_effective"] = _effective_board_area(cfg)
    totals_row["LB_area"] = _area_lower_bound(float(totals_row["part_area_sum"]), cfg)
    totals_row["N_board_gap_to_LB"] = float(totals_row.get("N_board", 0)) - float(totals_row["LB_area"])
    totals_row.update(solver_meta)
    totals_row["resume_cache_hit"] = 0
    _write_json(_seed_totals_path(out_seed_dir), totals_row)
    return totals_row


def _cfg_dict(cfg) -> Dict[str, Any]:
    return {key: getattr(cfg, key) for key in dir(cfg) if key.isupper() and not key.startswith("_")}


def _stamp_preprocess_summary(cfg, summary) -> None:
    payload = summary.to_dict()
    for key, value in payload.items():
        setattr(cfg, f"PREPROCESS_{str(key).upper()}", value)


def _load_case_parts(cfg):
    geometry = resolve_geometry_policy(cfg)
    raw_parts = read_sample_parts(
        cfg.SAMPLE_CSV,
        cfg.TRIM,
        cfg.GAP,
        tool_d=getattr(cfg, "TOOL_D", 0.0),
        inflate_gap=geometry.part_inflate_gap,
    )
    parts, preprocess_summary = preprocess_parts(raw_parts, cfg)
    return parts, geometry, preprocess_summary


def _run_one_seed_worker(cfg_dict: Dict[str, Any], seed: int, out_case_dir: str, variant: str) -> Dict[str, Any]:
    cfg = SimpleNamespace(**cfg_dict)
    parts, _, preprocess_summary = _load_case_parts(cfg)
    _stamp_preprocess_summary(cfg, preprocess_summary)
    out_case_path = Path(out_case_dir)
    return run_one_seed(parts, cfg, seed, out_case_path, variant)


def run_stepF(cfg) -> None:
    out_root = Path(cfg.OUT_ROOT)
    out_case_dir = out_root / cfg.CASE_NAME
    _ensure_dir(out_case_dir)

    parts_case, geometry, preprocess_summary = _load_case_parts(cfg)
    case_stats = _case_stats(parts_case, cfg)
    _stamp_preprocess_summary(cfg, preprocess_summary)
    _write_json(out_case_dir / "preprocess_summary.json", preprocess_summary.to_dict())

    if bool(getattr(cfg, "DUMP_CONFIG", True)):
        dump_config(cfg, out_case_dir, extra={"pipeline": "nesting_only"})
        variants = _normalize_variants(cfg)
        variant_cfgs = {variant: _variant_cfg(cfg, variant, case_stats) for variant in variants}
        for variant, cfg_variant in variant_cfgs.items():
            dump_config(
                cfg_variant,
                out_case_dir / variant,
                extra={"pipeline": "nesting_only", "variant": variant, "profile": variant},
            )
    else:
        variants = _normalize_variants(cfg)
        variant_cfgs = {variant: _variant_cfg(cfg, variant, case_stats) for variant in variants}

    part_area_sum = float(case_stats["part_area_sum"])
    board_area_effective = _effective_board_area(cfg)
    lb_area = int(case_stats["lb_area"])

    logger.info("[CASE] %s", cfg.CASE_NAME)
    logger.info(
        "[CFG] board=%sx%s trim=%s gap=%s inflate_gap=%s safe_gap=%s spacing_mode=%s seeds=%s seed0=%s solver=%s",
        cfg.BOARD_W,
        cfg.BOARD_H,
        geometry.trim,
        cfg.GAP,
        geometry.part_inflate_gap,
        geometry.safe_gap,
        geometry.spacing_mode,
        cfg.N_SEEDS,
        cfg.SEED0,
        ",".join(variants),
    )
    logger.info(
        "[PRE] parts=%s distinct_shapes=%s duplicate_parts=%s large=%s strips=%s dim_step=%s",
        preprocess_summary.n_parts,
        preprocess_summary.n_distinct_shapes,
        preprocess_summary.duplicate_parts,
        preprocess_summary.large_part_count,
        preprocess_summary.strip_part_count,
        preprocess_summary.dim_step,
    )
    if "rh_mcts_main" in variant_cfgs:
        logger.info(
            "[PROFILE] rh_mcts_main tier=%s n_parts=%s lb_area=%s",
            getattr(variant_cfgs["rh_mcts_main"], "MAIN_PROFILE_TIER", "base"),
            getattr(variant_cfgs["rh_mcts_main"], "MAIN_PROFILE_N_PARTS", len(parts_case)),
            getattr(variant_cfgs["rh_mcts_main"], "MAIN_PROFILE_LB_AREA", lb_area),
        )
    logger.info(
        "[LB] part_area=%.3f board_area_effective=%.3f LB_area=%s",
        part_area_sum,
        board_area_effective,
        lb_area,
    )

    totals_all: List[Dict[str, Any]] = []
    n_jobs = int(getattr(cfg, "N_JOBS", 1) or 1)
    pending_tasks: List[Tuple[str, Any, int]] = []

    for variant in variants:
        cfg_variant = variant_cfgs[variant]
        _stamp_preprocess_summary(cfg_variant, preprocess_summary)
        for i in range(int(cfg.N_SEEDS)):
            seed = cfg.SEED0 + i
            cached_row = _load_cached_seed_totals(parts_case, cfg_variant, out_case_dir, variant, seed)
            if cached_row is not None:
                totals_all.append(cached_row)
            else:
                pending_tasks.append((variant, cfg_variant, seed))

    if totals_all:
        logger.info("[RESUME] recovered %s completed variant/seed results from cache", len(totals_all))

    if n_jobs > 1 and pending_tasks:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(
                    _run_one_seed_worker,
                    _cfg_dict(cfg_variant),
                    seed,
                    str(out_case_dir),
                    variant,
                )
                for variant, cfg_variant, seed in pending_tasks
            ]
            for fut in as_completed(futures):
                row = fut.result()
                totals_all.append(row)
                logger.info(
                    "[OK] %s seed=%s N_board=%s U_global=%.4f runtime=%.3fs",
                    row["variant"],
                    row["seed"],
                    row["N_board"],
                    row["U_global"],
                    row["runtime_s"],
                )
    elif pending_tasks:
        parts = parts_case
        for variant, cfg_variant, seed in pending_tasks:
            row = run_one_seed(parts, cfg_variant, seed, out_case_dir, variant)
            totals_all.append(row)
            logger.info(
                "[OK] %s seed=%s N_board=%s U_global=%.4f runtime=%.3fs",
                row["variant"],
                row["seed"],
                row["N_board"],
                row["U_global"],
                row["runtime_s"],
            )

    for variant in variants:
        rows_variant = sorted(
            [row for row in totals_all if str(row["variant"]) == variant],
            key=lambda row: int(row["seed"]),
        )
        _write_rows_csv(out_case_dir / f"totals_{variant}.csv", rows_variant)
    _write_rows_csv(out_case_dir / "summary_stats.csv", _summary_stats(totals_all, variants))
    _write_rows_csv(out_case_dir / "utilization_table.csv", _utilization_table_rows(totals_all))
    _write_rows_csv(out_case_dir / "utilization_summary.csv", _utilization_summary_rows(totals_all, variants))
    _write_rows_csv(out_case_dir / "comparison_table.csv", _comparison_table_rows(totals_all))
