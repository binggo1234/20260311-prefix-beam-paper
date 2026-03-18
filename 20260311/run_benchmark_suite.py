from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.cfg import CFG
from inrp.runner import run_stepF


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(str(key))
                seen.add(str(key))
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return text in {"1", "true", "yes", "y"}


def _count_csv_rows(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except Exception:
        return 0


def _discover_public_compatible() -> List[Dict[str, Any]]:
    meta_path = ROOT / "data" / "public_binpacktwo" / "instance_meta.csv"
    rows = _read_rows(meta_path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("enabled", "")).strip() != "1":
            continue
        if str(row.get("pipeline_compatible", "")).strip() != "1":
            continue
        instance = str(row.get("instance", "")).strip()
        if not instance:
            continue
        csv_path = ROOT / "data" / "public_binpacktwo" / "instances" / f"{instance}.csv"
        if not csv_path.is_file():
            continue
        out.append(
            {
                "case_id": instance,
                "family": str(row.get("family", "public")).strip() or "public",
                "source": "public_binpacktwo",
                "csv_path": str(csv_path.resolve()),
                "board_w": _as_float(row.get("board_w")),
                "board_h": _as_float(row.get("board_h")),
                "allow_rot": _as_bool(row.get("allow_rot")),
                "trim": 0.0,
                "gap": 0.0,
                "tool_d": 0.0,
                "safe_gap": 0.0,
                "binary_spacing": False,
                "n_items": int(float(row.get("n_items", 0) or 0)),
            }
        )
    return out


def _discover_csv_glob(base_dir: Path, pattern: str, *, family: str, source: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted(base_dir.glob(pattern)):
        if not path.is_file():
            continue
        out.append(
            {
                "case_id": path.stem,
                "family": family,
                "source": source,
                "csv_path": str(path.resolve()),
                "board_w": None,
                "board_h": None,
                "allow_rot": None,
                "trim": None,
                "gap": None,
                "tool_d": None,
                "safe_gap": None,
                "binary_spacing": None,
                "n_items": _count_csv_rows(path),
            }
        )
    return out


def _discover_cases(suite: str) -> List[Dict[str, Any]]:
    suite_key = str(suite).strip().lower()
    cases: List[Dict[str, Any]] = []
    if suite_key in {"public_compatible", "all_known"}:
        cases.extend(_discover_public_compatible())
    if suite_key in {"industrial_new5", "all_known"}:
        cases.extend(
            _discover_csv_glob(
                ROOT / "data" / "industrial_new5",
                "industrial_data*.csv",
                family="industrial_new5",
                source="industrial_new5",
            )
        )
    if suite_key in {"industrial_cases", "all_known"}:
        cases.extend(
            _discover_csv_glob(
                ROOT / "data" / "industrial_cases",
                "I*_parts.csv",
                family="industrial_cases",
                source="industrial_cases",
            )
        )
    if suite_key in {"industrial_xlsx6", "all_known"}:
        cases.extend(
            _discover_csv_glob(
                ROOT / "data" / "industrial_xlsx6",
                "data*.csv",
                family="industrial_xlsx6",
                source="industrial_xlsx6",
            )
        )
    if suite_key in {"demo", "all_known"}:
        demo_path = ROOT / "data" / "sample_parts.csv"
        if demo_path.is_file():
            cases.append(
                {
                    "case_id": "sample_parts",
                    "family": "demo",
                    "source": "demo",
                    "csv_path": str(demo_path.resolve()),
                    "board_w": None,
                    "board_h": None,
                    "allow_rot": None,
                    "trim": None,
                    "gap": None,
                    "tool_d": None,
                    "safe_gap": None,
                    "binary_spacing": None,
                    "n_items": _count_csv_rows(demo_path),
                }
            )

    dedup: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        dedup[str(case["csv_path"]).lower()] = case
    ordered = sorted(
        dedup.values(),
        key=lambda item: (str(item["family"]), int(item.get("n_items", 0)), str(item["case_id"])),
    )
    return ordered


def _filter_cases(
    cases: Iterable[Dict[str, Any]],
    *,
    case_filter: str,
    max_items: int,
    case_limit: int,
) -> List[Dict[str, Any]]:
    out = list(cases)
    if case_filter:
        needle = str(case_filter).strip().lower()
        out = [
            case
            for case in out
            if needle in str(case["case_id"]).lower() or needle in str(case["family"]).lower()
        ]
    if max_items > 0:
        out = [case for case in out if int(case.get("n_items", 0)) <= int(max_items)]
    if case_limit > 0:
        out = out[:case_limit]
    return out


def _case_output_name(suite_name: str, case: Dict[str, Any]) -> str:
    return f"{suite_name}__{case['case_id']}"


def _load_case_comparison(out_case_dir: Path) -> List[Dict[str, Any]]:
    table = out_case_dir / "comparison_table.csv"
    if not table.is_file():
        return []
    return [{str(k): v for k, v in row.items()} for row in _read_rows(table)]


def _annotate_case_rows(
    args: argparse.Namespace,
    case: Dict[str, Any],
    rows: List[Dict[str, Any]],
    *,
    suite_runtime_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = {str(k): v for k, v in row.items()}
        item["suite"] = args.suite_name
        item["case_id"] = str(case["case_id"])
        item["family"] = str(case["family"])
        item["source"] = str(case["source"])
        item["csv_path"] = str(case["csv_path"])
        item["n_items"] = int(case.get("n_items", 0))
        if suite_runtime_s is not None:
            item["suite_runtime_s"] = round(float(suite_runtime_s), 6)
        out.append(item)
    return out


def _status_row(
    args: argparse.Namespace,
    case: Dict[str, Any],
    *,
    ordinal: int,
    status: str,
    suite_dir: Path,
    started_s: Optional[float] = None,
    elapsed_s: Optional[float] = None,
    comparison_rows: int = 0,
    error: str = "",
) -> Dict[str, Any]:
    out_case_dir = suite_dir / _case_output_name(args.suite_name, case)
    return {
        "suite": args.suite_name,
        "ordinal": int(ordinal),
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "source": str(case["source"]),
        "csv_path": str(case["csv_path"]),
        "n_items": int(case.get("n_items", 0)),
        "status": str(status),
        "started_s": "" if started_s is None else round(float(started_s), 6),
        "elapsed_s": "" if elapsed_s is None else round(float(elapsed_s), 6),
        "comparison_rows": int(comparison_rows),
        "has_comparison_table": int((out_case_dir / "comparison_table.csv").is_file()),
        "out_case_dir": str(out_case_dir),
        "error": str(error),
    }


def _write_suite_outputs(
    suite_dir: Path,
    manifest_rows: List[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
    status_rows: List[Dict[str, Any]],
) -> None:
    suite_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(suite_dir / "suite_manifest.csv", manifest_rows)
    _write_rows(suite_dir / "suite_case_status.csv", status_rows)
    if comparison_rows:
        _write_rows(suite_dir / "suite_comparison_table.csv", comparison_rows)


def _run_case(args: argparse.Namespace, suite_dir: Path, case: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = CFG()
    cfg.CASE_NAME = _case_output_name(args.suite_name, case)
    cfg.SAMPLE_CSV = str(case["csv_path"])
    cfg.OUT_ROOT = str(suite_dir)
    cfg.VARIANTS = tuple(args.variants)
    cfg.N_SEEDS = int(args.n_seeds)
    cfg.SEED0 = int(args.seed0)
    cfg.N_JOBS = int(args.inner_jobs)
    cfg.SOLVER_TIME_LIMIT_S = float(args.solver_time_limit_s)
    cfg.POST_REPAIR_CP_NUM_WORKERS = int(args.post_repair_cp_workers)

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

    started = time.perf_counter()
    run_stepF(cfg)
    elapsed = time.perf_counter() - started

    rows = _load_case_comparison(suite_dir / cfg.CASE_NAME)
    return _annotate_case_rows(args, case, rows, suite_runtime_s=elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch benchmark suite over available CSV datasets.")
    parser.add_argument(
        "--suite",
        choices=("public_compatible", "industrial_new5", "industrial_cases", "industrial_xlsx6", "demo", "all_known"),
        default="public_compatible",
    )
    parser.add_argument("--suite-name", default="")
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--variants", nargs="+", default=["baseline_greedy", "rh_mcts_ref", "rh_mcts_main"])
    parser.add_argument("--solver-time-limit-s", type=float, default=300.0)
    parser.add_argument("--post-repair-cp-workers", type=int, default=8)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--inner-jobs", type=int, default=1)
    parser.add_argument("--case-jobs", type=int, default=1)
    parser.add_argument("--case-filter", default="")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.suite_name:
        args.suite_name = f"{args.suite}_suite"

    suite_dir = (ROOT / args.out_root).resolve() / args.suite_name
    cases = _filter_cases(
        _discover_cases(args.suite),
        case_filter=args.case_filter,
        max_items=args.max_items,
        case_limit=args.case_limit,
    )
    if not cases:
        raise SystemExit("No cases matched the requested suite/filter.")

    manifest_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    runnable: List[tuple[int, Dict[str, Any]]] = []
    for idx, case in enumerate(cases, start=1):
        manifest_rows.append(
            {
                "suite": args.suite_name,
                "ordinal": idx,
                "case_id": case["case_id"],
                "family": case["family"],
                "source": case["source"],
                "csv_path": case["csv_path"],
                "n_items": case["n_items"],
                "board_w": case["board_w"],
                "board_h": case["board_h"],
                "allow_rot": case["allow_rot"],
            }
        )
        existing_rows = _annotate_case_rows(
            args,
            case,
            _load_case_comparison(suite_dir / _case_output_name(args.suite_name, case)),
        ) if bool(args.resume) else []
        if existing_rows:
            comparison_rows.extend(existing_rows)
            status_rows.append(
                _status_row(
                    args,
                    case,
                    ordinal=idx,
                    status="completed",
                    suite_dir=suite_dir,
                    comparison_rows=len(existing_rows),
                )
            )
        else:
            status_rows.append(_status_row(args, case, ordinal=idx, status="pending", suite_dir=suite_dir))
            runnable.append((idx, case))

    _write_suite_outputs(suite_dir, manifest_rows, comparison_rows, status_rows)

    case_jobs = max(1, int(args.case_jobs))
    if case_jobs <= 1:
        for idx, case in runnable:
            started = time.perf_counter()
            status_rows[idx - 1] = _status_row(
                args,
                case,
                ordinal=idx,
                status="running",
                suite_dir=suite_dir,
                started_s=started,
            )
            _write_suite_outputs(suite_dir, manifest_rows, comparison_rows, status_rows)
            try:
                rows = _run_case(args, suite_dir, case)
                comparison_rows.extend(rows)
                status = "completed" if rows else "no_comparison"
                status_rows[idx - 1] = _status_row(
                    args,
                    case,
                    ordinal=idx,
                    status=status,
                    suite_dir=suite_dir,
                    started_s=started,
                    elapsed_s=time.perf_counter() - started,
                    comparison_rows=len(rows),
                )
            except KeyboardInterrupt:
                status_rows[idx - 1] = _status_row(
                    args,
                    case,
                    ordinal=idx,
                    status="interrupted",
                    suite_dir=suite_dir,
                    started_s=started,
                    elapsed_s=time.perf_counter() - started,
                    error="KeyboardInterrupt",
                )
                _write_suite_outputs(suite_dir, manifest_rows, comparison_rows, status_rows)
                raise
            except Exception as exc:
                status_rows[idx - 1] = _status_row(
                    args,
                    case,
                    ordinal=idx,
                    status="failed",
                    suite_dir=suite_dir,
                    started_s=started,
                    elapsed_s=time.perf_counter() - started,
                    error=repr(exc),
                )
                if args.fail_fast:
                    _write_suite_outputs(suite_dir, manifest_rows, comparison_rows, status_rows)
                    raise
            _write_suite_outputs(suite_dir, manifest_rows, comparison_rows, status_rows)
        return

    futures = {}
    case_info: Dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=case_jobs) as ex:
        pending = list(runnable)
        while pending or futures:
            while pending and len(futures) < case_jobs:
                idx, case = pending.pop(0)
                started = time.perf_counter()
                case_info[idx] = (case, started)
                status_rows[idx - 1] = _status_row(
                    args,
                    case,
                    ordinal=idx,
                    status="running",
                    suite_dir=suite_dir,
                    started_s=started,
                )
                _write_suite_outputs(suite_dir, manifest_rows, comparison_rows, status_rows)
                futures[ex.submit(_run_case, args, suite_dir, case)] = idx

            done = next(as_completed(list(futures)))
            idx = futures.pop(done)
            case, started = case_info[idx]
            try:
                rows = done.result()
                comparison_rows.extend(rows)
                status = "completed" if rows else "no_comparison"
                status_rows[idx - 1] = _status_row(
                    args,
                    case,
                    ordinal=idx,
                    status=status,
                    suite_dir=suite_dir,
                    started_s=started,
                    elapsed_s=time.perf_counter() - started,
                    comparison_rows=len(rows),
                )
            except KeyboardInterrupt:
                status_rows[idx - 1] = _status_row(
                    args,
                    case,
                    ordinal=idx,
                    status="interrupted",
                    suite_dir=suite_dir,
                    started_s=started,
                    elapsed_s=time.perf_counter() - started,
                    error="KeyboardInterrupt",
                )
                _write_suite_outputs(suite_dir, manifest_rows, comparison_rows, status_rows)
                raise
            except Exception as exc:
                status_rows[idx - 1] = _status_row(
                    args,
                    case,
                    ordinal=idx,
                    status="failed",
                    suite_dir=suite_dir,
                    started_s=started,
                    elapsed_s=time.perf_counter() - started,
                    error=repr(exc),
                )
                if args.fail_fast:
                    _write_suite_outputs(suite_dir, manifest_rows, comparison_rows, status_rows)
                    raise
            _write_suite_outputs(suite_dir, manifest_rows, comparison_rows, status_rows)


if __name__ == "__main__":
    main()
