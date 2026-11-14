#!/usr/bin/env python3
"""Freeze the immutable run table for a formal industrial attempt."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.formal_batch import freeze_manifest


METHODS = ("rh_mcts_prefix_beam", "rh_mcts_prefix_beam_random_completion")
SEEDS = tuple(range(1000, 1005))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--schedule-seed", type=int, default=20260906)
    parser.add_argument("--attempt-number", type=int, required=True)
    args = parser.parse_args()
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    tracked = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if tracked:
        raise RuntimeError("commit the implementation before freezing the formal manifest")
    digest = freeze_manifest(
        args.readiness,
        args.output,
        data_root=args.data_root,
        code_commit=commit,
        methods=METHODS,
        seeds=SEEDS,
        schedule_seed=args.schedule_seed,
        attempt_number=args.attempt_number,
    )
    print(f"rows=110 sha256={digest} code_commit={commit} schedule_seed={args.schedule_seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
