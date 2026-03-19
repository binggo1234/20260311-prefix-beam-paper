from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.cfg import CFG
from inrp.runner import run_stepF


REPRESENTATIVE_CASES: Tuple[Dict[str, str], ...] = (
    {
        "case_id": "data3",
        "family": "industrial_xlsx6",
        "csv_path": str((ROOT / "data" / "industrial_xlsx6" / "data3.csv").resolve()),
    },
    {
        "case_id": "data6",
        "family": "industrial_xlsx6",
        "csv_path": str((ROOT / "data" / "industrial_xlsx6" / "data6.csv").resolve()),
    },
    {
        "case_id": "I1_parts",
        "family": "industrial_cases",
        "csv_path": str((ROOT / "data" / "industrial_cases" / "I1_parts.csv").resolve()),
    },
)

REPRESENTATIVE_GATES: Dict[str, int] = {
    "data6": 153,
    "data3": 176,
    "I1_parts": 70,
}

HISTORICAL_REF_TARGETS: Dict[str, int] = {
    "data6": 150,
    "data3": 174,
    "I1_parts": 70,
}

TARGET_VARIANT = "rh_mcts_prefix_beam"

BUNDLES: Dict[str, Dict[str, Any]] = {
    "prefix_depth_push": {
        "PREFIX_BEAM_DEPTH": 4,
        "PREFIX_BEAM_POST_TOPK": 3,
    },
    "prefix_branch_push": {
        "PREFIX_BEAM_WIDTH": 3,
        "PREFIX_BEAM_BRANCH_TOPK": 4,
        "BEAM_WIDTH": 3,
        "BEAM_BRANCH_TOPK": 4,
        "BEAM_MIN_ROOT_VISITS": 8,
    },
    "oracle_cache_push": {
        "PREFIX_BEAM_ORACLE_CACHE_ENABLE": True,
        "PREFIX_BEAM_POST_TOPK": 3,
        "MCTS_N_SIM": 6,
    },
}


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            key_s = str(key)
            if key_s not in seen:
                fieldnames.append(key_s)
                seen.add(key_s)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _base_cfg(
    *,
    case_name: str,
    csv_path: str,
    out_root: Path,
    solver_time_limit_s: float,
    seed0: int,
    n_seeds: int,
    inner_jobs: int,
    post_repair_cp_workers: int,
    beam_overrides: Dict[str, Any],
) -> CFG:
    cfg = CFG()
    cfg.CASE_NAME = str(case_name)
    cfg.SAMPLE_CSV = str(csv_path)
    cfg.OUT_ROOT = str(out_root)
    cfg.VARIANTS = ("baseline_greedy", "rh_mcts_ref", TARGET_VARIANT)
    cfg.N_SEEDS = int(n_seeds)
    cfg.SEED0 = int(seed0)
    cfg.N_JOBS = int(inner_jobs)
    cfg.SOLVER_TIME_LIMIT_S = float(solver_time_limit_s)
    cfg.POST_REPAIR_CP_NUM_WORKERS = int(post_repair_cp_workers)
    cfg.VARIANT_EXTRA_OVERRIDES = {TARGET_VARIANT: deepcopy(beam_overrides)}
    return cfg


def _load_case_outcome(out_case_dir: Path) -> Dict[str, Any]:
    rows = _read_rows(out_case_dir / "comparison_table.csv")
    row = rows[0] if rows else {}
    beam_meta = _read_json(out_case_dir / TARGET_VARIANT / "seed_1000" / "search_meta.json")
    ref_meta = _read_json(out_case_dir / "rh_mcts_ref" / "seed_1000" / "search_meta.json")
    baseline_totals = _read_rows(out_case_dir / "totals_baseline_greedy.csv")
    beam_totals = _read_rows(out_case_dir / f"totals_{TARGET_VARIANT}.csv")
    ref_totals = _read_rows(out_case_dir / "totals_rh_mcts_ref.csv")
    return {
        "comparison": row,
        "beam_meta": beam_meta,
        "ref_meta": ref_meta,
        "baseline_totals": baseline_totals[0] if baseline_totals else {},
        "beam_totals": beam_totals[0] if beam_totals else {},
        "ref_totals": ref_totals[0] if ref_totals else {},
    }


def _representative_score(case_rows: Dict[str, Dict[str, Any]]) -> Tuple:
    ordered = []
    pass_count = 0
    for case_id in ("data6", "data3", "I1_parts"):
        row = case_rows[case_id]
        beam_n = _as_int(row["comparison"].get("N_board_main"))
        gate = int(REPRESENTATIVE_GATES[case_id])
        if beam_n <= gate:
            pass_count += 1
        ordered.append(-beam_n)
    u_global_mean = sum(_as_float(case_rows[c]["comparison"].get("U_global_main")) for c in case_rows) / max(1.0, float(len(case_rows)))
    return (pass_count, *ordered, u_global_mean)


def _representative_passed(case_rows: Dict[str, Dict[str, Any]]) -> bool:
    for case_id, gate in REPRESENTATIVE_GATES.items():
        beam_n = _as_int(case_rows[case_id]["comparison"].get("N_board_main"), 10**9)
        if beam_n > int(gate):
            return False
    return True


def _matches_historical_ref(case_rows: Dict[str, Dict[str, Any]]) -> bool:
    for case_id, target in HISTORICAL_REF_TARGETS.items():
        beam_n = _as_int(case_rows[case_id]["comparison"].get("N_board_main"), 10**9)
        if beam_n > int(target):
            return False
    return True


def _promotion_ready(case_rows: Dict[str, Dict[str, Any]]) -> bool:
    if not _representative_passed(case_rows):
        return False
    if any(
        _as_int(case_rows[case_id]["comparison"].get("N_board_main"), 10**9)
        > _as_int(case_rows[case_id]["comparison"].get("N_board_baseline"), 10**9)
        for case_id in REPRESENTATIVE_GATES
    ):
        return False
    hard_cases = ("data6", "data3")
    return any(
        _as_int(case_rows[case_id]["comparison"].get("N_board_main"), 10**9)
        < _as_int(case_rows[case_id]["comparison"].get("N_board_ref"), 10**9)
        for case_id in hard_cases
    )


def _choose_next_bundle(
    case_rows: Dict[str, Dict[str, Any]],
    used_bundles: Sequence[str],
    current_overrides: Dict[str, Any],
) -> Optional[str]:
    if _matches_historical_ref(case_rows):
        return None

    hard_cases = ("data6", "data3")
    ties_baseline_without_win = (
        any(
            _as_int(case_rows[case_id]["comparison"].get("N_board_main"))
            == _as_int(case_rows[case_id]["comparison"].get("N_board_baseline"))
            for case_id in hard_cases
        )
        and not any(
            _as_int(case_rows[case_id]["comparison"].get("N_board_main"))
            < _as_int(case_rows[case_id]["comparison"].get("N_board_baseline"))
            for case_id in hard_cases
        )
    )
    current_beam_width = _as_float(current_overrides.get("PREFIX_BEAM_WIDTH", current_overrides.get("BEAM_WIDTH", 2.0)), 2.0)
    low_branching = any(
        _as_float(case_rows[case_id]["beam_meta"].get("beam_active_states_max")) < max(2.0, current_beam_width)
        or _as_float(case_rows[case_id]["beam_meta"].get("prefix_candidates_generated")) < 2.0
        for case_id in hard_cases
    )
    cache_pressure = any(
        _as_float(case_rows[case_id]["elapsed_s"]) >= 0.80 * 300.0
        and _as_float(case_rows[case_id]["beam_meta"].get("oracle_completion_calls")) >= 4.0
        and _as_float(case_rows[case_id]["beam_meta"].get("oracle_completion_cache_hits"))
        <= 0.25 * _as_float(case_rows[case_id]["beam_meta"].get("oracle_completion_calls"), 1.0)
        for case_id in hard_cases
    )

    priority: List[str] = []
    if ties_baseline_without_win:
        priority.append("prefix_depth_push")
    if low_branching:
        priority.append("prefix_branch_push")
    if cache_pressure:
        priority.append("oracle_cache_push")
    priority.extend(["prefix_depth_push", "prefix_branch_push", "oracle_cache_push"])

    used = {str(item) for item in used_bundles}
    for name in priority:
        if name not in used:
            return name
    return None


def _run_representative_round(
    *,
    round_idx: int,
    suite_dir: Path,
    beam_overrides: Dict[str, Any],
    solver_time_limit_s: float,
    seed0: int,
    n_seeds: int,
    inner_jobs: int,
    post_repair_cp_workers: int,
) -> Dict[str, Dict[str, Any]]:
    round_dir = suite_dir / f"round_{round_idx:02d}"
    case_rows: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    for case in REPRESENTATIVE_CASES:
        case_name = f"{round_dir.name}__{case['case_id']}"
        cfg = _base_cfg(
            case_name=case_name,
            csv_path=case["csv_path"],
            out_root=round_dir,
            solver_time_limit_s=solver_time_limit_s,
            seed0=seed0,
            n_seeds=n_seeds,
            inner_jobs=inner_jobs,
            post_repair_cp_workers=post_repair_cp_workers,
            beam_overrides=beam_overrides,
        )
        started = time.perf_counter()
        run_stepF(cfg)
        elapsed = time.perf_counter() - started
        out_case_dir = round_dir / case_name
        outcome = _load_case_outcome(out_case_dir)
        outcome["elapsed_s"] = round(float(elapsed), 6)
        case_rows[str(case["case_id"])] = outcome
        comparison = outcome["comparison"]
        summary_rows.append(
            {
                "round": int(round_idx),
                "case_id": str(case["case_id"]),
                "N_board_baseline": _as_int(comparison.get("N_board_baseline")),
                "N_board_ref": _as_int(comparison.get("N_board_ref")),
                "N_board_beam": _as_int(comparison.get("N_board_main")),
                "U_global_beam": _as_float(comparison.get("U_global_main")),
                "beam_candidates_generated": _as_float(outcome["beam_meta"].get("beam_candidates_generated")),
                "prefix_candidates_generated": _as_float(outcome["beam_meta"].get("prefix_candidates_generated")),
                "beam_active_states_max": _as_float(outcome["beam_meta"].get("beam_active_states_max")),
                "beam_force_rounds": _as_float(outcome["beam_meta"].get("beam_force_rounds")),
                "macro_pattern_actions": _as_float(outcome["beam_meta"].get("macro_pattern_actions")),
                "macro_rollout_pattern_steps": _as_float(outcome["beam_meta"].get("macro_rollout_pattern_steps")),
                "macro_rollout_single_fallback_steps": _as_float(outcome["beam_meta"].get("macro_rollout_single_fallback_steps")),
                "oracle_completion_calls": _as_float(outcome["beam_meta"].get("oracle_completion_calls")),
                "oracle_completion_cache_hits": _as_float(outcome["beam_meta"].get("oracle_completion_cache_hits")),
                "oracle_completion_mode_evals": _as_float(outcome["beam_meta"].get("oracle_completion_mode_evals")),
                "prefix_depth_reached": _as_float(outcome["beam_meta"].get("prefix_depth_reached")),
                "result_source": str(outcome["beam_meta"].get("result_source", "")),
                "oracle_completion_last_best_mode": str(outcome["beam_meta"].get("oracle_completion_last_best_mode", "")),
                "post_repair_boards_saved": _as_float(outcome["beam_meta"].get("post_repair_boards_saved")),
                "elapsed_s": round(float(elapsed), 6),
                "out_case_dir": str(out_case_dir),
            }
        )
    _write_rows(round_dir / "representative_summary.csv", summary_rows)
    return case_rows


def _short_suite_validation(
    *,
    suite_dir: Path,
    beam_overrides: Dict[str, Any],
    solver_time_limit_s: float,
    seed0: int,
    n_seeds: int,
    inner_jobs: int,
    post_repair_cp_workers: int,
) -> None:
    cases = sorted((ROOT / "data" / "industrial_xlsx6").glob("data*.csv"))
    out_dir = suite_dir / "promotion_short_industrial_xlsx6"
    rows: List[Dict[str, Any]] = []
    for path in cases:
        case_name = f"promotion_short__{path.stem}"
        cfg = _base_cfg(
            case_name=case_name,
            csv_path=str(path.resolve()),
            out_root=out_dir,
            solver_time_limit_s=solver_time_limit_s,
            seed0=seed0,
            n_seeds=n_seeds,
            inner_jobs=inner_jobs,
            post_repair_cp_workers=post_repair_cp_workers,
            beam_overrides=beam_overrides,
        )
        run_stepF(cfg)
        outcome = _load_case_outcome(out_dir / case_name)
        comparison = outcome["comparison"]
        rows.append(
            {
                "case_id": path.stem,
                "N_board_baseline": _as_int(comparison.get("N_board_baseline")),
                "N_board_ref": _as_int(comparison.get("N_board_ref")),
                "N_board_beam": _as_int(comparison.get("N_board_main")),
                "U_global_beam": _as_float(comparison.get("U_global_main")),
            }
        )
    _write_rows(out_dir / "suite_summary.csv", rows)


def _formal_suite_validation(
    *,
    suite_dir: Path,
    beam_overrides: Dict[str, Any],
    seed0: int,
    inner_jobs: int,
    post_repair_cp_workers: int,
) -> None:
    cases = sorted((ROOT / "data" / "industrial_xlsx6").glob("data*.csv"))
    out_dir = suite_dir / "promotion_formal_industrial_xlsx6"
    rows: List[Dict[str, Any]] = []
    for path in cases:
        case_name = f"promotion_formal__{path.stem}"
        cfg = _base_cfg(
            case_name=case_name,
            csv_path=str(path.resolve()),
            out_root=out_dir,
            solver_time_limit_s=3600.0,
            seed0=seed0,
            n_seeds=3,
            inner_jobs=inner_jobs,
            post_repair_cp_workers=post_repair_cp_workers,
            beam_overrides=beam_overrides,
        )
        run_stepF(cfg)
        outcome = _load_case_outcome(out_dir / case_name)
        comparison = outcome["comparison"]
        rows.append(
            {
                "case_id": path.stem,
                "N_board_baseline": _as_int(comparison.get("N_board_baseline")),
                "N_board_ref": _as_int(comparison.get("N_board_ref")),
                "N_board_beam": _as_int(comparison.get("N_board_main")),
                "U_global_beam": _as_float(comparison.get("U_global_main")),
            }
        )
    _write_rows(out_dir / "suite_summary.csv", rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous beam-first optimization loop.")
    parser.add_argument("--exp-name", default="autonomous_beam_loop")
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--inner-jobs", type=int, default=1)
    parser.add_argument("--solver-time-limit-s", type=float, default=300.0)
    parser.add_argument("--post-repair-cp-workers", type=int, default=2)
    parser.add_argument("--skip-promotion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    suite_dir = (ROOT / args.out_root).resolve() / args.exp_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    used_bundles: List[str] = []
    beam_overrides: Dict[str, Any] = {}
    round_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    best_score: Optional[Tuple] = None
    best_round: Optional[int] = None
    best_overrides: Dict[str, Any] = {}
    passed_repr = False
    promotion_ready = False
    rounds_without_progress = 0

    for round_idx in range(1, max(1, int(args.max_rounds)) + 1):
        case_rows = _run_representative_round(
            round_idx=round_idx,
            suite_dir=suite_dir,
            beam_overrides=beam_overrides,
            solver_time_limit_s=float(args.solver_time_limit_s),
            seed0=int(args.seed0),
            n_seeds=int(args.n_seeds),
            inner_jobs=int(args.inner_jobs),
            post_repair_cp_workers=int(args.post_repair_cp_workers),
        )
        rep_score = _representative_score(case_rows)
        round_rows.append(
            {
                "round": int(round_idx),
                "score": json.dumps(rep_score, ensure_ascii=False),
                "passed_repr": int(_representative_passed(case_rows)),
                "promotion_ready": int(_promotion_ready(case_rows)),
                "matches_historical_ref": int(_matches_historical_ref(case_rows)),
                "beam_overrides": json.dumps(beam_overrides, ensure_ascii=False),
            }
        )
        if best_score is None or rep_score > best_score:
            best_score = rep_score
            best_round = int(round_idx)
            best_overrides = deepcopy(beam_overrides)
            rounds_without_progress = 0
        else:
            rounds_without_progress += 1

        if _promotion_ready(case_rows):
            passed_repr = True
            promotion_ready = True
            decision_rows.append(
                {
                    "round": int(round_idx),
                    "decision": "promote",
                    "reason": "promotion_ready",
                    "beam_overrides": json.dumps(beam_overrides, ensure_ascii=False),
                }
            )
            break
        if rounds_without_progress >= 2:
            decision_rows.append(
                {
                    "round": int(round_idx),
                    "decision": "stop",
                    "reason": "no_meaningful_progress",
                    "beam_overrides": json.dumps(beam_overrides, ensure_ascii=False),
                }
            )
            break

        next_bundle = _choose_next_bundle(case_rows, used_bundles, beam_overrides)
        if not next_bundle:
            decision_rows.append(
                {
                    "round": int(round_idx),
                    "decision": "stop",
                    "reason": "no_unused_bundle_or_historical_target_matched",
                    "beam_overrides": json.dumps(beam_overrides, ensure_ascii=False),
                }
            )
            break
        used_bundles.append(str(next_bundle))
        if str(next_bundle) == "behavior_realign":
            beam_overrides = deepcopy(BUNDLES[str(next_bundle)])
        else:
            beam_overrides.update(deepcopy(BUNDLES[str(next_bundle)]))
        decision_rows.append(
            {
                "round": int(round_idx),
                "decision": str(next_bundle),
                "reason": "deterministic_rule",
                "beam_overrides": json.dumps(beam_overrides, ensure_ascii=False),
            }
        )

    _write_rows(suite_dir / "round_summary.csv", round_rows)
    _write_rows(suite_dir / "decision_log.csv", decision_rows)
    with open(suite_dir / "best_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "best_round": best_round,
                "best_score": best_score,
                "beam_overrides": best_overrides,
                "used_bundles": used_bundles,
                "passed_repr": passed_repr,
                "promotion_ready": promotion_ready,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    if promotion_ready and not args.skip_promotion:
        _short_suite_validation(
            suite_dir=suite_dir,
            beam_overrides=best_overrides,
            solver_time_limit_s=float(args.solver_time_limit_s),
            seed0=int(args.seed0),
            n_seeds=int(args.n_seeds),
            inner_jobs=int(args.inner_jobs),
            post_repair_cp_workers=int(args.post_repair_cp_workers),
        )
        _formal_suite_validation(
            suite_dir=suite_dir,
            beam_overrides=best_overrides,
            seed0=int(args.seed0),
            inner_jobs=int(args.inner_jobs),
            post_repair_cp_workers=int(args.post_repair_cp_workers),
        )


if __name__ == "__main__":
    main()
