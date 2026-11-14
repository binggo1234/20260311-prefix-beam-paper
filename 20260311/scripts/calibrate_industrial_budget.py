#!/usr/bin/env python3
"""One-shot industrial-scale budget propagation calibration under a watchdog."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA6 = PROJECT_ROOT / "data" / "industrial_xlsx6" / "data6.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_industrial_csv(path: Path):
    from inrp.handoff import PackingPart

    parts = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not {"Fleng", "Fwidth"}.issubset(reader.fieldnames):
            raise ValueError("industrial calibration CSV must contain Fleng and Fwidth")
        for row_number, row in enumerate(reader, 1):
            length = float(row["Fleng"])
            width = float(row["Fwidth"])
            if not all(math.isfinite(value) and value > 0.0 for value in (length, width)):
                raise ValueError(f"invalid dimensions at data row {row_number}")
            part_id = f"{path.stem}-row-{row_number:04d}"
            parts.append(
                PackingPart(
                    part_id=part_id,
                    source_part_id=part_id,
                    nominal_width_mm=length,
                    nominal_height_mm=width,
                    allowed_rotations_deg=(0, 90),
                )
            )
    if not parts:
        raise ValueError("industrial calibration CSV is empty")
    return tuple(parts)


def _worker(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from inrp.budget import TotalBudget, atomic_write_json_with_budget
    from inrp.handoff import PackingRequest, solve_packing_handoff_with_budget

    budget = TotalBudget(args.total_s, args.closing_reserve_s)
    parts = _read_industrial_csv(args.input_csv)
    budget.mark("input_read")
    request = PackingRequest(
        case_id=f"{args.input_csv.stem}-industrial-scale-budget-calibration-{args.solver_variant}",
        parts=parts,
        board_width_mm=2440.0,
        board_height_mm=1220.0,
        trim_mm=5.0,
        tool_diameter_mm=6.0,
        allow_rotation=True,
        seed=args.seed,
        time_limit_s=args.total_s,
        solver_variant=args.solver_variant,
    )
    budget.mark("input_adapted")
    handoff = solve_packing_handoff_with_budget(request, budget)
    solver_meta = dict(handoff.provenance.get("solver_meta", {}))
    implementation_paths = (
        PROJECT_ROOT / "src" / "inrp" / "budget.py",
        PROJECT_ROOT / "src" / "inrp" / "handoff.py",
        PROJECT_ROOT / "src" / "inrp" / "mcts.py",
        PROJECT_ROOT / "src" / "inrp" / "watchdog.py",
        Path(__file__).resolve(),
    )
    payload = {
        "status": "usable_calibration_only",
        "claim_scope": "budget propagation and industrial-scale closing cost only",
        "excluded_claims": [
            "manufacturing validity",
            "algorithm quality",
            "algorithm speed superiority",
            "formal comparative evidence",
        ],
        "source_kind": "historical_chapter2_data_model_declared",
        "source_csv": str(args.input_csv.resolve()),
        "source_sha256": _sha256(args.input_csv),
        "dimension_basis": "existing chapter2 Fleng/Fwidth; not manufacturing-confirmed",
        "rotation_rule": "model_declared 0/90; not manufacturing-confirmed",
        "stock_and_tool_rule": "model_declared 2440x1220, trim5, tool6",
        "case_id": request.case_id,
        "seed": request.seed,
        "solver_variant": request.solver_variant,
        "n_parts": len(parts),
        "panel_count": handoff.panel_count,
        "score_global": list(handoff.score_global),
        "runtime_s_before_write": handoff.runtime_s,
        "solver_meta_summary": {
            key: solver_meta.get(key)
            for key in (
                "formal_total_budget",
                "post_mcts_reserve_s",
                "solver_timeout_hit",
                "result_source",
                "formal_optional_postprocess_skipped",
                "simulations",
                "prefix_depth_reached",
            )
            if key in solver_meta
        },
        "implementation_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in implementation_paths
        },
        "placements": [asdict(item) for item in handoff.placements],
        "stamps_before_write": budget.summary()["stamps"],
    }
    atomic_write_json_with_budget(args.result_path, payload, budget)
    summary = {
        "status": "worker_completed",
        "result_sha256": _sha256(args.result_path),
        "budget": budget.summary(),
    }
    args.worker_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


def _parent(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from inrp.watchdog import run_command_with_watchdog

    if args.output_dir.exists():
        raise FileExistsError(f"calibration output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "minimal_result.json"
    worker_summary_path = args.output_dir / "worker_summary.json"
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--input-csv",
        str(args.input_csv.resolve()),
        "--result-path",
        str(result_path),
        "--worker-summary-path",
        str(worker_summary_path),
        "--total-s",
        str(args.total_s),
        "--closing-reserve-s",
        str(args.closing_reserve_s),
        "--seed",
        str(args.seed),
        "--solver-variant",
        str(args.solver_variant),
    )
    watchdog = run_command_with_watchdog(
        command,
        hard_timeout_s=args.total_s + args.startup_allowance_s,
        terminate_grace_s=args.terminate_grace_s,
        cwd=PROJECT_ROOT,
        env=os.environ,
        stdout_path=args.output_dir / "worker.stdout.log",
        stderr_path=args.output_dir / "worker.stderr.log",
    )
    watchdog_path = args.output_dir / "watchdog.json"
    watchdog_path.write_text(json.dumps(watchdog.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    worker_summary = None
    if worker_summary_path.is_file():
        worker_summary = json.loads(worker_summary_path.read_text(encoding="utf-8"))
    report = {
        "status": "calibration_completed" if watchdog.status == "completed" and result_path.is_file() else "calibration_failed",
        "claim_scope": "one-shot industrial-scale budget propagation calibration",
        "statistical_unit": "one calibration run; excluded from effect analysis",
        "input_csv": str(args.input_csv.resolve()),
        "input_sha256": _sha256(args.input_csv),
        "total_s": args.total_s,
        "closing_reserve_s": args.closing_reserve_s,
        "solver_variant": args.solver_variant,
        "startup_allowance_s_parent_only": args.startup_allowance_s,
        "watchdog": watchdog.to_dict(),
        "minimal_result_exists": result_path.is_file(),
        "worker_summary": worker_summary,
    }
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "watchdog_status": watchdog.status,
        "parent_wall_s": watchdog.parent_wall_s,
        "minimal_result_exists": result_path.is_file(),
    }, ensure_ascii=False))
    return 0 if report["status"] == "calibration_completed" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_DATA6)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--worker-summary-path", type=Path)
    parser.add_argument("--total-s", type=float, default=30.0)
    parser.add_argument("--closing-reserve-s", type=float, default=5.0)
    parser.add_argument("--startup-allowance-s", type=float, default=3.0)
    parser.add_argument("--terminate-grace-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--solver-variant",
        choices=("baseline_greedy", "rh_mcts_prefix_beam"),
        default="baseline_greedy",
    )
    args = parser.parse_args()
    if args.worker:
        if args.result_path is None or args.worker_summary_path is None:
            parser.error("--worker requires --result-path and --worker-summary-path")
    elif args.output_dir is None:
        parser.error("output_dir is required")
    return args


def main() -> int:
    args = parse_args()
    return _worker(args) if args.worker else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
