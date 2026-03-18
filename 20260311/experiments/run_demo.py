"""Run the nesting-only demo on a CSV of rectangular parts."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.cfg import CFG
from inrp.runner import run_stepF


def _read_instance_meta(parts_path: Path) -> Optional[Dict[str, str]]:
    instance = parts_path.stem
    for parent in parts_path.parents:
        meta_path = parent / "instance_meta.csv"
        if not meta_path.is_file():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if str(row.get("instance", "")).strip() == instance:
                        out = {str(k): str(v).strip() for k, v in row.items() if k is not None}
                        out["_meta_path"] = str(meta_path)
                        return out
        except Exception as exc:
            logging.warning("Failed to read instance metadata from %s: %s", meta_path, exc)
            return None
    return None


def _as_float(value, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if not s:
        return bool(default)
    return s in {"1", "true", "yes", "y"}


def _apply_dataset_defaults(cfg: CFG, args, parts_path: Path) -> None:
    meta = _read_instance_meta(parts_path)

    if meta is not None:
        cfg.BOARD_W = float(args.board_w) if args.board_w is not None else _as_float(meta.get("board_w"), cfg.BOARD_W)
        cfg.BOARD_H = float(args.board_h) if args.board_h is not None else _as_float(meta.get("board_h"), cfg.BOARD_H)
        cfg.ALLOW_ROT = _as_bool(meta.get("allow_rot"), cfg.ALLOW_ROT)

        cfg.TRIM = float(args.trim) if args.trim is not None else 0.0
        cfg.GAP = float(args.gap) if args.gap is not None else 0.0
        cfg.TOOL_D = float(args.tool_d) if args.tool_d is not None else 0.0
        if args.safe_gap is not None:
            cfg.SAFE_GAP = float(args.safe_gap)
        elif args.tool_d is not None:
            cfg.SAFE_GAP = float(args.tool_d)
        else:
            cfg.SAFE_GAP = 0.0
            cfg.BINARY_SPACING = False

        logging.info(
            "[DATA] instance=%s meta=%s board=%sx%s allow_rot=%s trim=%s safe_gap=%s",
            parts_path.stem,
            meta.get("_meta_path", ""),
            cfg.BOARD_W,
            cfg.BOARD_H,
            int(bool(cfg.ALLOW_ROT)),
            cfg.TRIM,
            cfg.SAFE_GAP,
        )
        return

    if args.board_w is not None:
        cfg.BOARD_W = float(args.board_w)
    if args.board_h is not None:
        cfg.BOARD_H = float(args.board_h)
    if args.trim is not None:
        cfg.TRIM = float(args.trim)
    if args.gap is not None:
        cfg.GAP = float(args.gap)
    if args.tool_d is not None:
        cfg.TOOL_D = float(args.tool_d)
    if args.safe_gap is not None:
        cfg.SAFE_GAP = float(args.safe_gap)
    elif args.tool_d is not None:
        cfg.SAFE_GAP = float(args.tool_d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, default="demo_case")
    ap.add_argument("--parts_csv", type=str, default=None)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--board_w", type=float, default=None)
    ap.add_argument("--board_h", type=float, default=None)
    ap.add_argument("--trim", type=float, default=None)
    ap.add_argument("--gap", type=float, default=None)
    ap.add_argument(
        "--tool_d",
        type=float,
        default=None,
        help="Used only as the default SAFE_GAP when binary spacing is enabled.",
    )
    ap.add_argument("--safe_gap", type=float, default=None)
    ap.add_argument("--mcts_sims", type=int, default=None)
    ap.add_argument("--local_search", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    repo = ROOT
    cfg = CFG()
    cfg.CASE_NAME = args.case
    cfg.SEED0 = args.seed0
    cfg.N_SEEDS = args.n_seeds
    cfg.N_JOBS = args.n_jobs
    if args.mcts_sims is not None:
        cfg.MCTS_N_SIM = args.mcts_sims
    if args.local_search:
        cfg.LOCAL_SEARCH_ENABLE = True

    if args.parts_csv:
        cfg.SAMPLE_CSV = args.parts_csv

    parts_path = Path(cfg.SAMPLE_CSV)
    if not parts_path.is_absolute():
        parts_path = repo / parts_path
    cfg.SAMPLE_CSV = str(parts_path.resolve())
    cfg.OUT_ROOT = str((repo / cfg.OUT_ROOT).resolve())
    _apply_dataset_defaults(cfg, args, parts_path.resolve())

    run_stepF(cfg)


if __name__ == "__main__":
    main()
