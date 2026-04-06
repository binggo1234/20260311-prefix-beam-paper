from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Task:
    name: str
    argv: List[str]
    deps: List[str] = field(default_factory=list)


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def _tasks(out_root: str, post_workers: int, robustness_seconds: int, public_case: str, industrial_large: str, industrial_medium: str) -> List[Task]:
    def rp(*args: str) -> List[str]:
        return [sys.executable, str(ROOT / "run_paper_experiments.py"), *args]

    def grid(*args: str) -> List[str]:
        return [sys.executable, str(ROOT / "tools" / "run_prefix_param_grid.py"), *args]

    def strong(*args: str) -> List[str]:
        return [sys.executable, str(ROOT / "tools" / "run_strong_baseline_batches.py"), *args]

    def layout(*args: str) -> List[str]:
        return [sys.executable, str(ROOT / "tools" / "render_layout_compare.py"), *args]

    def summary(*args: str) -> List[str]:
        return [sys.executable, str(ROOT / "tools" / "render_paper_summary_charts.py"), *args]

    out_root_abs = str((ROOT / out_root).resolve())

    return [
        Task(
            "industrial_main",
            rp(
                "robustness",
                "--suite", "industrial_xlsx6",
                "--exp-name", "exp_main_ind_all_1seed_300s",
                "--out-root", out_root,
                "--variants", "baseline_greedy", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "public_main",
            rp(
                "robustness",
                "--suite", "public_compatible",
                "--exp-name", "exp_main_pub_all_1seed_300s",
                "--out-root", out_root,
                "--variants", "baseline_greedy", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "industrial_ablation",
            rp(
                "ablation",
                "--suite", "industrial_xlsx6",
                "--exp-name", "exp_ablation_ind_3seed_600s",
                "--out-root", out_root,
                "--n-seeds", "3",
                "--seed0", "1000",
                "--solver-time-limit-s", "600",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "public_ablation",
            rp(
                "ablation",
                "--suite", "public_compatible",
                "--exp-name", "exp_ablation_pub_1seed_300s",
                "--out-root", out_root,
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "industrial_strong_ablation",
            rp(
                "robustness",
                "--suite", "industrial_xlsx6",
                "--exp-name", "exp_ablation_strong_ind",
                "--out-root", out_root,
                "--variants", "rh_mcts_ref", "rh_mcts_full", "ours_nomacro", "ours_nolightrepair", "rh_mcts_prefix_beam",
                "--n-seeds", "3",
                "--seed0", "1000",
                "--solver-time-limit-s", "600",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "public_strong_ablation",
            rp(
                "robustness",
                "--suite", "public_compatible",
                "--exp-name", "exp_ablation_strong_pub",
                "--out-root", out_root,
                "--variants", "rh_mcts_ref", "rh_mcts_full", "ours_nomacro", "ours_nolightrepair", "rh_mcts_prefix_beam",
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "industrial_strong_pilot",
            strong(
                "compare",
                "--baseline-kind", "pilot",
                "--suite", "industrial_xlsx6",
                "--exp-name", "exp_strong_pilot_ind_1seed_300s",
                "--out-root", out_root,
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "public_strong_pilot",
            strong(
                "compare",
                "--baseline-kind", "pilot",
                "--suite", "public_compatible",
                "--exp-name", "exp_strong_pilot_pub_1seed_300s",
                "--out-root", out_root,
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "industrial_strong_full_beam",
            strong(
                "compare",
                "--baseline-kind", "full_beam",
                "--suite", "industrial_xlsx6",
                "--exp-name", "exp_strong_full_beam_ind_1seed_300s",
                "--out-root", out_root,
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "public_strong_full_beam",
            strong(
                "compare",
                "--baseline-kind", "full_beam",
                "--suite", "public_compatible",
                "--exp-name", "exp_strong_full_beam_pub_1seed_300s",
                "--out-root", out_root,
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "industrial_strong_budget",
            strong(
                "time_scan",
                "--suite", "industrial_xlsx6",
                "--exp-name", "exp_strong_budget_ind_3seed",
                "--out-root", out_root,
                "--n-seeds", "3",
                "--seed0", "1000",
                "--budgets", "10", "30", "60", "120", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "public_strong_budget",
            strong(
                "time_scan",
                "--suite", "public_compatible",
                "--exp-name", "exp_strong_budget_pub10_1seed",
                "--out-root", out_root,
                "--n-seeds", "1",
                "--seed0", "1000",
                "--budgets", "10", "30", "60", "120", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "industrial_strong_repeatability",
            strong(
                "repeatability",
                "--suite", "industrial_xlsx6",
                "--exp-name", "exp_strong_repeat_ind_5seed_300s",
                "--out-root", out_root,
                "--n-seeds", "5",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "industrial_strong_pilot_param",
            strong(
                "param_grid",
                "--target", "pilot",
                "--suite", "industrial_xlsx6",
                "--exp-name", "exp_strong_param_pilot_ind",
                "--out-root", out_root,
                "--n-seeds", "3",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "public_strong_pilot_param",
            strong(
                "param_grid",
                "--target", "pilot",
                "--suite", "public_compatible",
                "--exp-name", "exp_strong_param_pilot_pub5",
                "--out-root", out_root,
                "--n-seeds", "3",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "industrial_strong_full_beam_param",
            strong(
                "param_grid",
                "--target", "full_beam",
                "--suite", "industrial_xlsx6",
                "--exp-name", "exp_strong_param_full_beam_ind",
                "--out-root", out_root,
                "--n-seeds", "3",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "public_strong_full_beam_param",
            strong(
                "param_grid",
                "--target", "full_beam",
                "--suite", "public_compatible",
                "--exp-name", "exp_strong_param_full_beam_pub5",
                "--out-root", out_root,
                "--n-seeds", "3",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "industrial_robustness",
            rp(
                "robustness",
                "--suite", "industrial_xlsx6",
                "--exp-name", f"exp_robust_ind_5seed_{robustness_seconds}s",
                "--out-root", out_root,
                "--variants", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
                "--n-seeds", "5",
                "--seed0", "1000",
                "--solver-time-limit-s", str(robustness_seconds),
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "industrial_nsim",
            rp(
                "sensitivity",
                "--suite", "industrial_xlsx6",
                "--exp-name", "exp_nsim_ind",
                "--out-root", out_root,
                "--variants", "rh_mcts_prefix_beam",
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
                "--mcts-n-sim-values", "1", "2", "3", "5", "8",
            ),
        ),
        Task(
            "prefix_depth_grid",
            grid(
                "--suite", "all_known",
                "--exp-name", "exp_prefix_depth_grid",
                "--out-root", out_root,
                "--variants", "rh_mcts_prefix_beam",
                "--case-ids", "data2", "data6", "data3", "gcut6", "cgcut1",
                "--parameter", "PREFIX_BEAM_DEPTH",
                "--values", "1", "2", "3", "4", "5",
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "prefix_width_grid",
            grid(
                "--suite", "all_known",
                "--exp-name", "exp_prefix_width_grid",
                "--out-root", out_root,
                "--variants", "rh_mcts_prefix_beam",
                "--case-ids", "data2", "data6", "data3", "gcut6", "cgcut1",
                "--parameter", "PREFIX_BEAM_WIDTH",
                "--values", "1", "2", "3", "4",
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "prefix_branch_grid",
            grid(
                "--suite", "all_known",
                "--exp-name", "exp_prefix_branch_grid",
                "--out-root", out_root,
                "--variants", "rh_mcts_prefix_beam",
                "--case-ids", "data2", "data6", "data3", "gcut6", "cgcut1",
                "--parameter", "PREFIX_BEAM_BRANCH_TOPK",
                "--values", "2", "3", "4", "5",
                "--n-seeds", "1",
                "--seed0", "1000",
                "--solver-time-limit-s", "300",
                "--post-repair-cp-workers", str(post_workers),
            ),
        ),
        Task(
            "layout_industrial_large",
            layout(
                "--run-dir", str((ROOT / out_root / "exp_main_ind_all_1seed_300s").resolve()),
                "--case-id", industrial_large,
                "--variants", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
                "--seed", "1000",
                "--out", str((ROOT / out_root / "figures" / f"layout_{industrial_large}_compare.png").resolve()),
            ),
            deps=["industrial_main"],
        ),
        Task(
            "layout_industrial_medium",
            layout(
                "--run-dir", str((ROOT / out_root / "exp_main_ind_all_1seed_300s").resolve()),
                "--case-id", industrial_medium,
                "--variants", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
                "--seed", "1000",
                "--out", str((ROOT / out_root / "figures" / f"layout_{industrial_medium}_compare.png").resolve()),
            ),
            deps=["industrial_main"],
        ),
        Task(
            "layout_public",
            layout(
                "--run-dir", str((ROOT / out_root / "exp_main_pub_all_1seed_300s").resolve()),
                "--case-id", public_case,
                "--variants", "baseline_sa", "rh_mcts_ref", "rh_mcts_prefix_beam",
                "--seed", "1000",
                "--out", str((ROOT / out_root / "figures" / f"layout_{public_case}_compare.png").resolve()),
            ),
            deps=["public_main"],
        ),
        Task(
            "summary_figures",
            summary(
                "--result-root", out_root_abs,
                "--out-dir", str((ROOT / out_root / "figures").resolve()),
                "--ablation-exp", "exp_ablation_ind_3seed_600s",
                "--robustness-exp", f"exp_robust_ind_5seed_{robustness_seconds}s",
                "--robustness-exp-2", f"exp_robust_ind_5seed_{robustness_seconds}s",
                "--sensitivity-exp", "exp_nsim_ind",
                "--sensitivity-exp-2", "exp_nsim_ind",
            ),
            deps=["industrial_ablation", "industrial_robustness", "industrial_nsim"],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the extended paper experiment plan in parallel lanes.")
    parser.add_argument("--out-root", default="outputs/paper_extended_suite_20260321")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--post-repair-cp-workers", type=int, default=2)
    parser.add_argument("--robustness-seconds", type=int, default=1200)
    parser.add_argument("--industrial-large-case", default="data2")
    parser.add_argument("--industrial-medium-case", default="data6")
    parser.add_argument("--public-case", default="gcut6")
    parser.add_argument("--poll-s", type=float, default=15.0)
    args = parser.parse_args()

    out_root = (ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    task_log_dir = out_root / "task_logs"
    task_log_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = out_root / "launcher.log"
    launcher_err = out_root / "launcher.err.log"

    tasks = {task.name: task for task in _tasks(args.out_root, args.post_repair_cp_workers, args.robustness_seconds, args.public_case, args.industrial_large_case, args.industrial_medium_case)}
    pending = list(tasks.keys())
    running: Dict[str, subprocess.Popen] = {}
    completed: Dict[str, int] = {}
    failed: Dict[str, int] = {}

    _append(launcher_log, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] PARALLEL_START out_root={out_root}")

    while pending or running:
        launched = False
        while len(running) < max(1, int(args.max_parallel)):
            ready_name = None
            for name in pending:
                deps = tasks[name].deps
                if any(dep in failed for dep in deps):
                    ready_name = None
                    failed[name] = -1
                    pending.remove(name)
                    _append(launcher_err, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SKIP {name} due to failed dependency")
                    break
                if all(dep in completed for dep in deps):
                    ready_name = name
                    break
            if ready_name is None:
                break

            task = tasks[ready_name]
            pending.remove(ready_name)
            stdout_path = task_log_dir / f"{ready_name}.stdout.log"
            stderr_path = task_log_dir / f"{ready_name}.stderr.log"
            stdout_fh = open(stdout_path, "a", encoding="utf-8")
            stderr_fh = open(stderr_path, "a", encoding="utf-8")
            proc = subprocess.Popen(task.argv, cwd=str(ROOT.parent), stdout=stdout_fh, stderr=stderr_fh)
            proc._stdout_fh = stdout_fh  # type: ignore[attr-defined]
            proc._stderr_fh = stderr_fh  # type: ignore[attr-defined]
            running[ready_name] = proc
            _append(launcher_log, f"=== {ready_name} ===")
            _append(launcher_log, " ".join(task.argv))
            launched = True

        finished = []
        for name, proc in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            proc._stdout_fh.close()  # type: ignore[attr-defined]
            proc._stderr_fh.close()  # type: ignore[attr-defined]
            if rc == 0:
                completed[name] = rc
                _append(launcher_log, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] COMPLETE {name}")
            else:
                failed[name] = rc
                _append(launcher_err, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAIL {name} rc={rc}")
            finished.append(name)
        for name in finished:
            del running[name]

        if pending or running:
            time.sleep(max(2.0, float(args.poll_s)))
        elif not launched:
            break

    if failed:
        _append(launcher_err, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] PARALLEL_DONE failed={sorted(failed.keys())}")
        return 1

    _append(launcher_log, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] PARALLEL_DONE all tasks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
