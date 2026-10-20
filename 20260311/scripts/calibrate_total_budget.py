#!/usr/bin/env python3
"""Calibrate closing costs on the synthetic six-piece shared-input fixture."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "shared_six_parts.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--total-s", type=float, default=2.0)
    parser.add_argument("--closing-reserve-s", type=float, default=0.5)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"calibration output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from inrp.budget import TotalBudget, atomic_write_json_with_budget
    from inrp.handoff import solve_packing_handoff_with_budget
    from thesis_input.adapters import to_packing_request

    records = []
    for index in range(args.repeats):
        budget = TotalBudget(args.total_s, args.closing_reserve_s)
        with args.fixture.open("r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        budget.mark("input_read")
        request = to_packing_request(bundle, seed=1000 + index)
        budget.mark("input_adapted")
        handoff = solve_packing_handoff_with_budget(request, budget)
        payload = {
            "status": "usable",
            "case_id": request.case_id,
            "seed": request.seed,
            "panel_count": handoff.panel_count,
            "sequence": list(handoff.sequence),
            "score_global": list(handoff.score_global),
            "runtime_s_before_write": handoff.runtime_s,
            "placements": [asdict(item) for item in handoff.placements],
        }
        result_path = args.output_dir / f"run_{index + 1:02d}.json"
        atomic_write_json_with_budget(result_path, payload, budget)
        summary = budget.summary()
        records.append(
            {
                "run": index + 1,
                "seed": request.seed,
                "panel_count": handoff.panel_count,
                "elapsed_s": summary["elapsed_s"],
                "remaining_s": summary["remaining_s"],
                "stamps": summary["stamps"],
                "result_sha256": _sha256(result_path),
            }
        )

    elapsed = [float(row["elapsed_s"]) for row in records]
    ordered = sorted(elapsed)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    report = {
        "status": "calibration_only_not_effect_evidence",
        "fixture": str(args.fixture.resolve()),
        "fixture_sha256": _sha256(args.fixture),
        "source_kind": "synthetic_model_declared",
        "repeats": args.repeats,
        "total_s": args.total_s,
        "closing_reserve_s": args.closing_reserve_s,
        "elapsed_s_mean": mean(elapsed),
        "elapsed_s_max": max(elapsed),
        "elapsed_s_p95_nearest_rank": ordered[p95_index],
        "all_runs_usable": all(row["panel_count"] > 0 for row in records),
        "records": records,
    }
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "repeats",
        "elapsed_s_mean",
        "elapsed_s_max",
        "elapsed_s_p95_nearest_rank",
        "all_runs_usable",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
