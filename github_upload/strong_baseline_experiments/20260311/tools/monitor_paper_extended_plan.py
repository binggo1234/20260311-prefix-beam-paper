from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


TASKS = [
    ("industrial_main", "exp_main_ind_all_1seed_300s", "robustness_summary.csv"),
    ("public_main", "exp_main_pub_all_1seed_300s", "robustness_summary.csv"),
    ("industrial_ablation", "exp_ablation_ind_3seed_600s", "ablation_wide.csv"),
    ("public_ablation", "exp_ablation_pub_1seed_300s", "ablation_wide.csv"),
    ("industrial_strong_ablation", "exp_ablation_strong_ind", "robustness_summary.csv"),
    ("public_strong_ablation", "exp_ablation_strong_pub", "robustness_summary.csv"),
    ("industrial_strong_pilot", "exp_strong_pilot_ind_1seed_300s", "comparison_wide.csv"),
    ("public_strong_pilot", "exp_strong_pilot_pub_1seed_300s", "comparison_wide.csv"),
    ("industrial_strong_full_beam", "exp_strong_full_beam_ind_1seed_300s", "comparison_wide.csv"),
    ("public_strong_full_beam", "exp_strong_full_beam_pub_1seed_300s", "comparison_wide.csv"),
    ("industrial_strong_budget", "exp_strong_budget_ind_3seed", "time_scan_summary.csv"),
    ("public_strong_budget", "exp_strong_budget_pub10_1seed", "time_scan_summary.csv"),
    ("industrial_strong_repeatability", "exp_strong_repeat_ind_5seed_300s", "repeatability_summary.csv"),
    ("industrial_strong_pilot_param", "exp_strong_param_pilot_ind", "param_grid_summary.csv"),
    ("public_strong_pilot_param", "exp_strong_param_pilot_pub5", "param_grid_summary.csv"),
    ("industrial_strong_full_beam_param", "exp_strong_param_full_beam_ind", "param_grid_summary.csv"),
    ("public_strong_full_beam_param", "exp_strong_param_full_beam_pub5", "param_grid_summary.csv"),
    ("industrial_robustness", "exp_robust_ind_5seed_1200s", "robustness_summary.csv"),
    ("industrial_nsim", "exp_nsim_ind", "sensitivity_summary.csv"),
    ("prefix_depth_grid", "exp_prefix_depth_grid", "param_grid_summary.csv"),
    ("prefix_width_grid", "exp_prefix_width_grid", "param_grid_summary.csv"),
    ("prefix_branch_grid", "exp_prefix_branch_grid", "param_grid_summary.csv"),
]

FIGURE_TASKS = [
    ("layout_industrial_large", Path("figures/layout_data2_compare.png")),
    ("layout_industrial_medium", Path("figures/layout_data6_compare.png")),
    ("layout_public", Path("figures/layout_gcut6_compare.png")),
    ("summary_ablation", Path("figures/ablation_industrial_xlsx6.png")),
    ("summary_robustness", Path("figures/robustness_data3_data6.png")),
    ("summary_sensitivity", Path("figures/sensitivity_data3_data6.png")),
]

TERMINAL_HINTS = {"completed", "no_totals", "failed", "interrupted"}


def _append(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _status_counts(rows: List[Dict[str, str]]) -> str:
    counts = Counter(str(row.get("status", "")).strip() for row in rows)
    return "; ".join(f"{k}={counts[k]}" for k in sorted(counts))


def _task_state(root: Path, task_name: str, exp_dir_name: str, summary_file: str) -> Dict[str, str]:
    exp_dir = root / exp_dir_name
    summary_path = exp_dir / summary_file
    status_path = exp_dir / "case_status.csv"
    result_long = exp_dir / "results_long.csv"

    status = "pending"
    detail = ""
    if summary_path.is_file():
        status = "completed"
        detail = summary_file
    elif status_path.is_file():
        rows = _read_rows(status_path)
        if rows:
            terminalish = all(str(r.get("status", "")).strip() in TERMINAL_HINTS for r in rows)
            status = "completed" if terminalish and result_long.is_file() else "running"
            detail = _status_counts(rows)
        else:
            status = "running"
            detail = "status file exists"
    elif exp_dir.is_dir():
        status = "running"
        detail = "directory created"
    return {
        "task": task_name,
        "kind": "experiment",
        "path": str(exp_dir),
        "status": status,
        "detail": detail,
        "updated": datetime.fromtimestamp(exp_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if exp_dir.exists() else "",
    }


def _figure_state(root: Path, task_name: str, rel_path: Path) -> Dict[str, str]:
    path = root / rel_path
    status = "completed" if path.is_file() else "pending"
    if path.parent.exists() and not path.is_file():
        status = "running"
    return {
        "task": task_name,
        "kind": "figure",
        "path": str(path),
        "status": status,
        "detail": rel_path.name if path.is_file() else "",
        "updated": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if path.exists() else "",
    }


def _current_task_from_log(log_path: Path) -> str:
    if not log_path.is_file():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    current = ""
    for line in lines:
        text = line.strip()
        if text.startswith("===") and text.endswith("==="):
            current = text.strip("= ").strip()
    return current


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: List[Dict[str, str]], current_task: str) -> None:
    lines = ["# Extended Plan Progress", ""]
    if current_task:
        lines.append(f"Current task: **{current_task}**")
        lines.append("")
    lines.append("| Task | Kind | Status | Detail |")
    lines.append("|---|---|---|---|")
    for row in rows:
        lines.append(f"| {row['task']} | {row['kind']} | {row['status']} | {row['detail']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor the one-shot extended paper experiment suite.")
    parser.add_argument("--root", required=True, help="Output root passed to run_paper_extended_plan.ps1")
    parser.add_argument("--interval-s", type=float, default=120.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "monitor.log"
    summary_csv = root / "progress_summary.csv"
    summary_md = root / "progress_summary.md"
    launcher_log = root / "launcher.log"

    _append(log_path, f"[{datetime.now():%Y-%m-%d %H:%M:%S}] MONITOR_START root={root}")

    while True:
        rows: List[Dict[str, str]] = []
        for task_name, exp_dir_name, summary_file in TASKS:
            rows.append(_task_state(root, task_name, exp_dir_name, summary_file))
        for task_name, rel_path in FIGURE_TASKS:
            rows.append(_figure_state(root, task_name, rel_path))

        current_task = _current_task_from_log(launcher_log)
        _write_csv(summary_csv, rows)
        _write_md(summary_md, rows, current_task)

        counts = Counter(row["status"] for row in rows)
        stamp = f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"
        _append(log_path, f"{stamp} HEARTBEAT current={current_task or 'n/a'} statuses={dict(counts)}")

        if rows and all(row["status"] == "completed" for row in rows):
            _append(log_path, f"{stamp} COMPLETE all tasks finished")
            return 0

        time.sleep(max(5.0, float(args.interval_s)))


if __name__ == "__main__":
    sys.exit(main())
