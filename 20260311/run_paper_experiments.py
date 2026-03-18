from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.cfg import CFG
from inrp.runner import run_stepF
from run_benchmark_suite import _discover_cases, _filter_cases


EXTERNAL_VARIANTS: List[Tuple[str, str]] = [
    ("baseline_greedy", "PureGreedy"),
    ("baseline_sa", "BaselineSA"),
    ("rh_mcts_main", "OursMain"),
]

ABLATION_VARIANTS: List[Tuple[str, str]] = [
    ("baseline_greedy", "Greedy"),
    ("ours_base", "OursBase"),
    ("ours_noprior", "OursNoPrior"),
    ("rh_mcts_main", "OursMain"),
]


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    text = str(value).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except Exception:
        return float(default)


def _gap_to_lb_pct(n_board: Any, lb_area: Any) -> float:
    lb = _as_float(lb_area, 0.0)
    if lb <= 0.0:
        return 0.0
    return 100.0 * (_as_float(n_board, 0.0) - lb) / lb


def _label_map(items: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    return {str(variant): str(label) for variant, label in items}


def _load_totals_rows(out_case_dir: Path, variants: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        path = out_case_dir / f"totals_{variant}.csv"
        if not path.is_file():
            continue
        for row in _read_rows(path):
            out = {str(k): v for k, v in row.items()}
            out["variant"] = str(variant)
            rows.append(out)
    return rows


def _case_output_name(exp_name: str, case: Dict[str, Any], suffix: str = "") -> str:
    base = f"{exp_name}__{case['case_id']}"
    if suffix:
        base += f"__{suffix}"
    return base


def _configure_case(
    cfg: CFG,
    case: Dict[str, Any],
    *,
    case_name: str,
    out_root: Path,
    variants: Sequence[str],
    n_seeds: int,
    seed0: int,
    inner_jobs: int,
    solver_time_limit_s: float,
    post_repair_cp_workers: int,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    cfg.CASE_NAME = str(case_name)
    cfg.SAMPLE_CSV = str(case["csv_path"])
    cfg.OUT_ROOT = str(out_root)
    cfg.VARIANTS = tuple(str(variant) for variant in variants)
    cfg.N_SEEDS = int(n_seeds)
    cfg.SEED0 = int(seed0)
    cfg.N_JOBS = int(inner_jobs)
    cfg.SOLVER_TIME_LIMIT_S = float(solver_time_limit_s)
    cfg.POST_REPAIR_CP_NUM_WORKERS = int(post_repair_cp_workers)

    if case.get("board_w") is not None:
        cfg.BOARD_W = float(case["board_w"])
    if case.get("board_h") is not None:
        cfg.BOARD_H = float(case["board_h"])
    if case.get("allow_rot") is not None:
        cfg.ALLOW_ROT = bool(case["allow_rot"])
    if case.get("trim") is not None:
        cfg.TRIM = float(case["trim"])
    if case.get("gap") is not None:
        cfg.GAP = float(case["gap"])
    if case.get("tool_d") is not None:
        cfg.TOOL_D = float(case["tool_d"])
    if case.get("safe_gap") is not None:
        cfg.SAFE_GAP = float(case["safe_gap"])
    if case.get("binary_spacing") is not None:
        cfg.BINARY_SPACING = bool(case["binary_spacing"])

    if extra_overrides:
        for key, value in extra_overrides.items():
            setattr(cfg, str(key), value)


def _collect_long_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    exp_name: str,
    suite: str,
    case: Dict[str, Any],
    variant_labels: Dict[str, str],
    extra_fields: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        variant = str(row.get("variant", ""))
        lb_area = _as_float(row.get("LB_area"), 0.0)
        item = {
            "experiment": exp_name,
            "suite": suite,
            "case_id": str(case["case_id"]),
            "family": str(case["family"]),
            "source": str(case["source"]),
            "csv_path": str(case["csv_path"]),
            "n_items": int(case.get("n_items", 0)),
            "variant": variant,
            "variant_label": variant_labels.get(variant, variant),
            "seed": int(float(row.get("seed", 0) or 0)),
            "N_input": int(float(row.get("N_input", 0) or 0)),
            "N_board": int(float(row.get("N_board", 0) or 0)),
            "LB_area": int(lb_area),
            "gap_to_lb_pct": _gap_to_lb_pct(row.get("N_board"), row.get("LB_area")),
            "U_global": _as_float(row.get("U_global")),
            "U_avg_excl_last": _as_float(row.get("U_avg_excl_last")),
            "runtime_s": _as_float(row.get("runtime_s")),
            "solver_kind": str(row.get("solver_kind", "")),
            "result_source": str(row.get("result_source", "")),
            "main_profile_tier": str(row.get("main_profile_tier", "")),
            "simulations": _as_float(row.get("simulations")),
        }
        if extra_fields:
            item.update(extra_fields)
        out.append(item)
    return out


def _summary_rows(rows: Iterable[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    metrics = ("N_board", "gap_to_lb_pct", "U_global", "U_avg_excl_last", "runtime_s", "simulations")
    for key in sorted(groups):
        data = groups[key]
        item = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        item["n"] = len(data)
        for metric in metrics:
            xs = [_as_float(row.get(metric)) for row in data]
            item[f"{metric}_mean"] = mean(xs) if xs else 0.0
            item[f"{metric}_std"] = stdev(xs) if len(xs) > 1 else 0.0
            if metric == "N_board":
                item[f"{metric}_best"] = min(xs) if xs else 0.0
        out.append(item)
    return out


def _external_wide_rows(rows: Iterable[Dict[str, Any]], variant_labels: Dict[str, str]) -> List[Dict[str, Any]]:
    by_case_seed: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["case_id"]), int(row["seed"]))
        base = by_case_seed.setdefault(
            key,
            {
                "case_id": str(row["case_id"]),
                "family": str(row["family"]),
                "seed": int(row["seed"]),
                "LB_area": int(row["LB_area"]),
                "n_items": int(row["n_items"]),
            },
        )
        label = variant_labels.get(str(row["variant"]), str(row["variant"]))
        for metric in ("N_board", "gap_to_lb_pct", "U_global", "runtime_s"):
            base[f"{metric}_{label}"] = row.get(metric)
    return [by_case_seed[key] for key in sorted(by_case_seed)]


def _ablation_wide_rows(rows: Iterable[Dict[str, Any]], variant_labels: Dict[str, str]) -> List[Dict[str, Any]]:
    wide = _external_wide_rows(rows, variant_labels)
    for row in wide:
        if "N_board_OursMain" in row and "N_board_OursBase" in row:
            row["delta_main_vs_base"] = _as_float(row["N_board_OursMain"]) - _as_float(row["N_board_OursBase"])
        if "N_board_OursMain" in row and "N_board_OursNoPrior" in row:
            row["delta_main_vs_noprior"] = _as_float(row["N_board_OursMain"]) - _as_float(row["N_board_OursNoPrior"])
    return wide


def _run_one_case(
    args: argparse.Namespace,
    suite_dir: Path,
    case: Dict[str, Any],
    *,
    variants: Sequence[str],
    variant_labels: Dict[str, str],
    suffix: str = "",
    extra_overrides: Optional[Dict[str, Any]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    case_name = _case_output_name(args.exp_name, case, suffix)
    cfg = CFG()
    _configure_case(
        cfg,
        case,
        case_name=case_name,
        out_root=suite_dir,
        variants=variants,
        n_seeds=args.n_seeds,
        seed0=args.seed0,
        inner_jobs=args.inner_jobs,
        solver_time_limit_s=args.solver_time_limit_s,
        post_repair_cp_workers=args.post_repair_cp_workers,
        extra_overrides=extra_overrides,
    )
    started = time.perf_counter()
    run_stepF(cfg)
    elapsed = time.perf_counter() - started
    out_case_dir = suite_dir / case_name
    totals_rows = _load_totals_rows(out_case_dir, variants)
    long_rows = _collect_long_rows(
        totals_rows,
        exp_name=args.exp_name,
        suite=args.suite,
        case=case,
        variant_labels=variant_labels,
        extra_fields=extra_fields,
    )
    status = {
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "status": "completed" if long_rows else "no_totals",
        "elapsed_s": round(float(elapsed), 6),
        "out_case_dir": str(out_case_dir),
        "n_rows": len(long_rows),
    }
    if extra_fields:
        status.update(extra_fields)
    return long_rows, status


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--suite",
        choices=("public_compatible", "industrial_new5", "industrial_cases", "industrial_xlsx6", "demo", "all_known"),
        default="public_compatible",
    )
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--case-filter", default="")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--inner-jobs", type=int, default=1)
    parser.add_argument("--solver-time-limit-s", type=float, default=300.0)
    parser.add_argument("--post-repair-cp-workers", type=int, default=8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-ready experiment batches.")
    sub = parser.add_subparsers(dest="experiment", required=True)

    external = sub.add_parser("external", help="External baseline comparison.")
    _common_args(external)

    ablation = sub.add_parser("ablation", help="Internal ablation study.")
    _common_args(ablation)

    robustness = sub.add_parser("robustness", help="Seed robustness experiment.")
    _common_args(robustness)
    robustness.add_argument("--variants", nargs="+", default=["rh_mcts_main"])

    sensitivity = sub.add_parser("sensitivity", help="MCTS simulation-budget sensitivity.")
    _common_args(sensitivity)
    sensitivity.add_argument("--variants", nargs="+", default=["rh_mcts_main"])
    sensitivity.add_argument("--mcts-n-sim-values", nargs="+", type=int, default=[1, 2, 3, 5])
    return parser.parse_args()


def _resolve_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return _filter_cases(
        _discover_cases(args.suite),
        case_filter=args.case_filter,
        max_items=args.max_items,
        case_limit=args.case_limit,
    )


def _run_external(args: argparse.Namespace, suite_dir: Path, cases: Sequence[Dict[str, Any]]) -> None:
    variant_pairs = EXTERNAL_VARIANTS
    variants = [variant for variant, _ in variant_pairs]
    variant_labels = _label_map(variant_pairs)
    results: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    for case in cases:
        long_rows, status = _run_one_case(args, suite_dir, case, variants=variants, variant_labels=variant_labels)
        results.extend(long_rows)
        status_rows.append(status)
        _write_rows(suite_dir / "case_status.csv", status_rows)
        _write_rows(suite_dir / "results_long.csv", results)
    _write_rows(suite_dir / "summary_by_case_variant.csv", _summary_rows(results, ("case_id", "variant_label")))
    _write_rows(suite_dir / "comparison_wide.csv", _external_wide_rows(results, variant_labels))


def _run_ablation(args: argparse.Namespace, suite_dir: Path, cases: Sequence[Dict[str, Any]]) -> None:
    variant_pairs = ABLATION_VARIANTS
    variants = [variant for variant, _ in variant_pairs]
    variant_labels = _label_map(variant_pairs)
    results: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    for case in cases:
        long_rows, status = _run_one_case(args, suite_dir, case, variants=variants, variant_labels=variant_labels)
        results.extend(long_rows)
        status_rows.append(status)
        _write_rows(suite_dir / "case_status.csv", status_rows)
        _write_rows(suite_dir / "results_long.csv", results)
    _write_rows(suite_dir / "summary_by_case_variant.csv", _summary_rows(results, ("case_id", "variant_label")))
    _write_rows(suite_dir / "ablation_wide.csv", _ablation_wide_rows(results, variant_labels))


def _run_robustness(args: argparse.Namespace, suite_dir: Path, cases: Sequence[Dict[str, Any]]) -> None:
    variants = [str(variant) for variant in args.variants]
    variant_labels = {variant: variant for variant in variants}
    results: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    for case in cases:
        long_rows, status = _run_one_case(args, suite_dir, case, variants=variants, variant_labels=variant_labels)
        results.extend(long_rows)
        status_rows.append(status)
        _write_rows(suite_dir / "case_status.csv", status_rows)
        _write_rows(suite_dir / "results_long.csv", results)
    _write_rows(suite_dir / "robustness_summary.csv", _summary_rows(results, ("case_id", "variant_label")))


def _run_sensitivity(args: argparse.Namespace, suite_dir: Path, cases: Sequence[Dict[str, Any]]) -> None:
    variants = [str(variant) for variant in args.variants]
    variant_labels = {variant: variant for variant in variants}
    results: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    for case in cases:
        for value in [int(x) for x in args.mcts_n_sim_values]:
            long_rows, status = _run_one_case(
                args,
                suite_dir,
                case,
                variants=variants,
                variant_labels=variant_labels,
                suffix=f"nsim_{value}",
                extra_overrides={"MCTS_N_SIM": int(value)},
                extra_fields={"mcts_n_sim": int(value)},
            )
            results.extend(long_rows)
            status_rows.append(status)
            _write_rows(suite_dir / "case_status.csv", status_rows)
            _write_rows(suite_dir / "results_long.csv", results)
    _write_rows(suite_dir / "sensitivity_summary.csv", _summary_rows(results, ("case_id", "variant_label", "mcts_n_sim")))


def main() -> None:
    args = _parse_args()
    if not args.exp_name:
        args.exp_name = f"{args.experiment}_{args.suite}"

    cases = _resolve_cases(args)
    if not cases:
        raise SystemExit("No cases matched the requested suite/filter.")

    suite_dir = (ROOT / args.out_root).resolve() / args.exp_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            "experiment": args.experiment,
            "suite": args.suite,
            "exp_name": args.exp_name,
            "case_id": str(case["case_id"]),
            "family": str(case["family"]),
            "source": str(case["source"]),
            "csv_path": str(case["csv_path"]),
            "n_items": int(case.get("n_items", 0)),
        }
        for case in cases
    ]
    _write_rows(suite_dir / "experiment_manifest.csv", manifest)

    if args.experiment == "external":
        _run_external(args, suite_dir, cases)
    elif args.experiment == "ablation":
        _run_ablation(args, suite_dir, cases)
    elif args.experiment == "robustness":
        _run_robustness(args, suite_dir, cases)
    elif args.experiment == "sensitivity":
        _run_sensitivity(args, suite_dir, cases)
    else:
        raise SystemExit(f"Unsupported experiment: {args.experiment}")


if __name__ == "__main__":
    main()
