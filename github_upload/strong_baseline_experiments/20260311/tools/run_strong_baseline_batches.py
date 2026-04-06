from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_benchmark_suite import _discover_cases, _filter_cases
from run_paper_experiments import (
    _external_wide_rows,
    _run_one_case,
    _summary_rows,
    _write_rows,
)


PILOT_COMPARE_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("baseline_greedy", "Greedy"),
    ("rh_mcts_ref", "StdMCTS"),
    ("baseline_pilot", "Pilot"),
    ("rh_mcts_prefix_beam", "OursMain"),
)

FULL_BEAM_COMPARE_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("baseline_greedy", "Greedy"),
    ("rh_mcts_ref", "StdMCTS"),
    ("baseline_full_beam", "FullBeam"),
    ("rh_mcts_prefix_beam", "OursMain"),
)

TIME_SCAN_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("rh_mcts_ref", "StdMCTS"),
    ("baseline_pilot", "Pilot"),
    ("baseline_full_beam", "FullBeam"),
    ("rh_mcts_prefix_beam", "OursMain"),
)

REPEATABILITY_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("rh_mcts_ref", "StdMCTS"),
    ("baseline_pilot", "Pilot"),
    ("rh_mcts_prefix_beam", "OursMain"),
)

DEFAULT_PUBLIC_TIME_CASE_IDS: Tuple[str, ...] = (
    "cgcut1",
    "cgcut2",
    "cgcut3",
    "gcut1",
    "gcut5",
    "gcut6",
    "gcut13",
    "ngcut1",
    "ngcut5",
    "ngcut9",
)

DEFAULT_INDUSTRIAL_PARAM_CASE_IDS: Tuple[str, ...] = ("data2", "data4", "data6")
DEFAULT_PUBLIC_PARAM_CASE_IDS: Tuple[str, ...] = ("cgcut1", "gcut6", "gcut13", "ngcut3", "ngcut9")


def _common_args(parser: argparse.ArgumentParser, *, default_suite: str) -> None:
    parser.add_argument(
        "--suite",
        choices=("public_compatible", "industrial_new5", "industrial_cases", "industrial_xlsx6", "demo", "all_known"),
        default=default_suite,
    )
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--case-filter", default="")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--case-ids", nargs="+", default=[])
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--inner-jobs", type=int, default=1)
    parser.add_argument("--solver-time-limit-s", type=float, default=300.0)
    parser.add_argument("--post-repair-cp-workers", type=int, default=8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stronger external baseline experiment batches.")
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="Run one strong-baseline comparison group.")
    _common_args(compare, default_suite="public_compatible")
    compare.add_argument("--baseline-kind", choices=("pilot", "full_beam"), required=True)

    time_scan = sub.add_parser("time_scan", help="Run time-budget scans over strong baselines.")
    _common_args(time_scan, default_suite="industrial_xlsx6")
    time_scan.add_argument("--budgets", nargs="+", type=float, default=[10.0, 30.0, 60.0, 120.0, 300.0])

    repeatability = sub.add_parser("repeatability", help="Run industrial repeatability batches.")
    _common_args(repeatability, default_suite="industrial_xlsx6")
    repeatability.set_defaults(n_seeds=5, solver_time_limit_s=300.0)

    param_grid = sub.add_parser("param_grid", help="Run coarse parameter grids for strong baselines.")
    _common_args(param_grid, default_suite="industrial_xlsx6")
    param_grid.add_argument("--target", choices=("pilot", "full_beam"), required=True)
    param_grid.add_argument("--pilot-lookahead-values", nargs="+", type=int, default=[4, 8, 16])
    param_grid.add_argument("--beam-width-values", nargs="+", type=int, default=[2, 4, 8, 16])
    param_grid.add_argument("--beam-candidate-values", nargs="+", type=int, default=[8, 16, 32])
    param_grid.set_defaults(n_seeds=3, solver_time_limit_s=300.0)
    return parser.parse_args()


def _resolve_cases(args: argparse.Namespace, *, default_case_ids: Sequence[str] = ()) -> List[Dict[str, Any]]:
    cases = _filter_cases(
        _discover_cases(args.suite),
        case_filter=args.case_filter,
        max_items=args.max_items,
        case_limit=args.case_limit,
    )
    selected = [str(case_id) for case_id in args.case_ids] or [str(case_id) for case_id in default_case_ids]
    if selected:
        allowed = set(selected)
        cases = [case for case in cases if str(case["case_id"]) in allowed]
    return cases


def _variant_config(variant_pairs: Sequence[Tuple[str, str]]) -> Tuple[List[str], Dict[str, str]]:
    variants = [str(variant) for variant, _ in variant_pairs]
    variant_labels = {str(variant): str(label) for variant, label in variant_pairs}
    return variants, variant_labels


def _manifest_row(
    command: str,
    args: argparse.Namespace,
    case: Dict[str, Any],
    *,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "command": str(command),
        "suite": str(args.suite),
        "exp_name": str(args.exp_name),
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "source": str(case["source"]),
        "csv_path": str(case["csv_path"]),
        "n_items": int(case.get("n_items", 0)),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _run_jobs(
    args: argparse.Namespace,
    suite_dir: Path,
    cases: Sequence[Dict[str, Any]],
    *,
    command_name: str,
    variant_pairs: Sequence[Tuple[str, str]],
    jobs: Sequence[Dict[str, Any]],
    summary_filename: str,
    group_keys: Sequence[str],
    wide_filename: str = "",
) -> None:
    variants, variant_labels = _variant_config(variant_pairs)
    manifest: List[Dict[str, Any]] = []
    for case in cases:
        for job in jobs:
            manifest.append(_manifest_row(command_name, args, case, extra_fields=job.get("manifest_fields")))
    _write_rows(suite_dir / "experiment_manifest.csv", manifest)

    results: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    for case in cases:
        for job in jobs:
            run_args = copy.copy(args)
            if "solver_time_limit_s" in job:
                run_args.solver_time_limit_s = float(job["solver_time_limit_s"])
            long_rows, status = _run_one_case(
                run_args,
                suite_dir,
                case,
                variants=variants,
                variant_labels=variant_labels,
                suffix=str(job.get("suffix", "")),
                extra_overrides=job.get("extra_overrides"),
                extra_fields=job.get("extra_fields"),
            )
            results.extend(long_rows)
            status_rows.append(status)
            _write_rows(suite_dir / "case_status.csv", status_rows)
            _write_rows(suite_dir / "results_long.csv", results)

    _write_rows(suite_dir / summary_filename, _summary_rows(results, tuple(str(key) for key in group_keys)))
    if wide_filename:
        _write_rows(suite_dir / wide_filename, _external_wide_rows(results, variant_labels))


def _default_time_scan_case_ids(args: argparse.Namespace) -> Sequence[str]:
    if str(args.suite) == "public_compatible":
        return DEFAULT_PUBLIC_TIME_CASE_IDS
    return ()


def _default_param_case_ids(args: argparse.Namespace) -> Sequence[str]:
    if str(args.suite) == "public_compatible":
        return DEFAULT_PUBLIC_PARAM_CASE_IDS
    if str(args.suite) == "industrial_xlsx6":
        return DEFAULT_INDUSTRIAL_PARAM_CASE_IDS
    return ()


def _run_compare(args: argparse.Namespace) -> None:
    cases = _resolve_cases(args)
    if not cases:
        raise SystemExit("No cases matched the requested suite/filter/case_ids.")
    suite_dir = (ROOT / args.out_root).resolve() / args.exp_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    variant_pairs = PILOT_COMPARE_VARIANTS if str(args.baseline_kind) == "pilot" else FULL_BEAM_COMPARE_VARIANTS
    _run_jobs(
        args,
        suite_dir,
        cases,
        command_name=f"compare_{args.baseline_kind}",
        variant_pairs=variant_pairs,
        jobs=[{}],
        summary_filename="summary_by_case_variant.csv",
        group_keys=("case_id", "variant_label"),
        wide_filename="comparison_wide.csv",
    )


def _run_time_scan(args: argparse.Namespace) -> None:
    cases = _resolve_cases(args, default_case_ids=_default_time_scan_case_ids(args))
    if not cases:
        raise SystemExit("No cases matched the requested suite/filter/case_ids.")
    suite_dir = (ROOT / args.out_root).resolve() / args.exp_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    for budget in [float(value) for value in args.budgets]:
        jobs.append(
            {
                "suffix": f"budget_{int(budget) if float(budget).is_integer() else budget}",
                "solver_time_limit_s": float(budget),
                "extra_fields": {"budget_s": float(budget)},
                "manifest_fields": {"budget_s": float(budget)},
            }
        )

    _run_jobs(
        args,
        suite_dir,
        cases,
        command_name="time_scan",
        variant_pairs=TIME_SCAN_VARIANTS,
        jobs=jobs,
        summary_filename="time_scan_summary.csv",
        group_keys=("case_id", "variant_label", "budget_s"),
    )


def _run_repeatability(args: argparse.Namespace) -> None:
    cases = _resolve_cases(args)
    if not cases:
        raise SystemExit("No cases matched the requested suite/filter/case_ids.")
    suite_dir = (ROOT / args.out_root).resolve() / args.exp_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    _run_jobs(
        args,
        suite_dir,
        cases,
        command_name="repeatability",
        variant_pairs=REPEATABILITY_VARIANTS,
        jobs=[{}],
        summary_filename="repeatability_summary.csv",
        group_keys=("case_id", "variant_label"),
        wide_filename="comparison_wide.csv",
    )


def _pilot_param_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for lookahead in [int(value) for value in args.pilot_lookahead_values]:
        jobs.append(
            {
                "suffix": f"pilot_lookahead_{lookahead}",
                "extra_overrides": {
                    "VARIANT_EXTRA_OVERRIDES": {
                        "baseline_pilot": {
                            "PREFIX_BEAM_BRANCH_TOPK": int(lookahead),
                            "BEAM_BRANCH_TOPK": int(lookahead),
                        }
                    }
                },
                "extra_fields": {
                    "tuning_target": "pilot",
                    "pilot_lookahead": int(lookahead),
                },
                "manifest_fields": {
                    "tuning_target": "pilot",
                    "pilot_lookahead": int(lookahead),
                },
            }
        )
    return jobs


def _full_beam_param_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for beam_width in [int(value) for value in args.beam_width_values]:
        for candidate_topk in [int(value) for value in args.beam_candidate_values]:
            jobs.append(
                {
                    "suffix": f"full_beam_w{beam_width}_topk{candidate_topk}",
                    "extra_overrides": {
                        "VARIANT_EXTRA_OVERRIDES": {
                            "baseline_full_beam": {
                                "BEAM_WIDTH": int(beam_width),
                                "BEAM_BRANCH_TOPK": int(candidate_topk),
                                "BEAM_POSTPROCESS_TOPK": min(int(beam_width), 4),
                            }
                        }
                    },
                    "extra_fields": {
                        "tuning_target": "full_beam",
                        "beam_width": int(beam_width),
                        "beam_candidate_topk": int(candidate_topk),
                    },
                    "manifest_fields": {
                        "tuning_target": "full_beam",
                        "beam_width": int(beam_width),
                        "beam_candidate_topk": int(candidate_topk),
                    },
                }
            )
    return jobs


def _run_param_grid(args: argparse.Namespace) -> None:
    cases = _resolve_cases(args, default_case_ids=_default_param_case_ids(args))
    if not cases:
        raise SystemExit("No cases matched the requested suite/filter/case_ids.")
    suite_dir = (ROOT / args.out_root).resolve() / args.exp_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    if str(args.target) == "pilot":
        variant_pairs = (("baseline_pilot", "Pilot"),)
        jobs = _pilot_param_jobs(args)
        group_keys = ("case_id", "variant_label", "tuning_target", "pilot_lookahead")
    else:
        variant_pairs = (("baseline_full_beam", "FullBeam"),)
        jobs = _full_beam_param_jobs(args)
        group_keys = ("case_id", "variant_label", "tuning_target", "beam_width", "beam_candidate_topk")

    _run_jobs(
        args,
        suite_dir,
        cases,
        command_name="param_grid",
        variant_pairs=variant_pairs,
        jobs=jobs,
        summary_filename="param_grid_summary.csv",
        group_keys=group_keys,
    )


def main() -> None:
    args = _parse_args()
    if args.command == "compare":
        _run_compare(args)
    elif args.command == "time_scan":
        _run_time_scan(args)
    elif args.command == "repeatability":
        _run_repeatability(args)
    elif args.command == "param_grid":
        _run_param_grid(args)
    else:
        raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
