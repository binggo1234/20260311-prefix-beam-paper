from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.cfg import CFG
from inrp.runner import run_stepF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a formal nesting case.")
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--out-root", default="20260311/outputs")
    parser.add_argument("--variants", nargs="+", default=["baseline_greedy", "rh_mcts_main"])
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--solver-time-limit-s", type=float, default=900.0)
    parser.add_argument("--post-repair-cp-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = CFG()
    cfg.CASE_NAME = str(args.case_name)
    cfg.SAMPLE_CSV = str(args.sample_csv)
    cfg.OUT_ROOT = str(args.out_root)
    cfg.VARIANTS = tuple(args.variants)
    cfg.N_SEEDS = int(args.n_seeds)
    cfg.SEED0 = int(args.seed0)
    cfg.N_JOBS = int(args.n_jobs)
    cfg.SOLVER_TIME_LIMIT_S = float(args.solver_time_limit_s)
    cfg.POST_REPAIR_CP_NUM_WORKERS = int(args.post_repair_cp_workers)
    run_stepF(cfg)


if __name__ == "__main__":
    main()
