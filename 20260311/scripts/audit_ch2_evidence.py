#!/usr/bin/env python3
"""Independent source and nominal-placement checks for the Chapter 2 batch."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.formal_audit import audit_batch


def audit_geometry(result: dict, source: Path) -> dict[str, float | int]:
    request = result["request"]
    handoff = result["handoff"]
    with source.open(encoding="gb18030", newline="") as handle:
        raw_parts = list(csv.DictReader(handle))
    requested_parts = request["parts"]
    if len(raw_parts) != len(requested_parts):
        raise ValueError("source and request have different part counts")
    expected = {}
    for index, (raw, part) in enumerate(zip(raw_parts, requested_parts), 1):
        source_id = raw["Upi"].strip()
        part_id = f"{source_id}__copy_{index:06d}"
        if (
            part["source_part_id"] != source_id
            or part["part_id"] != part_id
            or float(part["nominal_width_mm"]) != float(raw["Fleng"])
            or float(part["nominal_height_mm"]) != float(raw["Fwidth"])
        ):
            raise ValueError(f"source/request identity or dimensions differ at part {index}")
        if part_id in expected:
            raise ValueError(f"duplicate source part identity: {part_id}")
        expected[part_id] = part
    placements = handoff["placements"]
    placed_ids = [placement["part_id"] for placement in placements]
    if Counter(placed_ids) != Counter(expected.keys()):
        raise ValueError("missing, duplicated, or foreign placed part")
    if Counter(handoff["sequence"]) != Counter(expected.keys()):
        raise ValueError("missing, duplicated, or foreign part in sequence")
    if request != handoff["provenance"]["request"]:
        raise ValueError("handoff provenance request differs from result request")

    radius = float(request["tool_diameter_mm"]) / 2.0
    trim = float(request["trim_mm"])
    width = float(request["board_width_mm"])
    height = float(request["board_height_mm"])
    panels: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    nominal_area = 0.0
    tolerance = 1e-7
    for placement in placements:
        part = expected[placement["part_id"]]
        rotated = bool(placement["rotated"])
        orientation = 90 if rotated else 0
        if orientation not in part["allowed_rotations_deg"]:
            raise ValueError("placement rotation is forbidden")
        part_width = float(part["nominal_height_mm"] if rotated else part["nominal_width_mm"])
        part_height = float(part["nominal_width_mm"] if rotated else part["nominal_height_mm"])
        if (
            abs(float(placement["nominal_width_mm"]) - part_width) > tolerance
            or abs(float(placement["nominal_height_mm"]) - part_height) > tolerance
            or placement["source_part_id"] != part["source_part_id"]
        ):
            raise ValueError("placement orientation, dimensions, or source identity differ")
        x = float(placement["x_mm"])
        y = float(placement["y_mm"])
        box = (x - radius, y - radius, x + part_width + radius, y + part_height + radius)
        if (
            box[0] < trim - tolerance or box[1] < trim - tolerance
            or box[2] > width - trim + tolerance or box[3] > height - trim + tolerance
        ):
            raise ValueError("part envelope crosses the trimmed board boundary")
        panel_id = int(placement["panel_id"])
        if panel_id < 1:
            raise ValueError("nonpositive panel identity")
        for other in panels[panel_id]:
            overlap_x = min(box[2], other[2]) - max(box[0], other[0])
            overlap_y = min(box[3], other[3]) - max(box[1], other[1])
            if overlap_x > tolerance and overlap_y > tolerance:
                raise ValueError("tool envelopes overlap")
        panels[panel_id].append(box)
        nominal_area += part_width * part_height
    if len(panels) != int(handoff["panel_count"]):
        raise ValueError("panel count differs from distinct panel identities")
    utilization = nominal_area / (int(handoff["panel_count"]) * width * height)
    return {"part_count": len(expected), "panel_count": len(panels), "global_utilization": utilization}


def audit_completion_trace(result: dict) -> dict[str, float | int | None]:
    method = result["request"]["solver_variant"]
    if method == "baseline_greedy":
        return {"mode_attempts": None, "mode_completed": None, "mode_timeouts": None, "partial_returned": None}
    meta = result["handoff"]["provenance"]["solver_meta"]
    attempts = int(meta["oracle_completion_mode_evals"])
    completed = int(meta["oracle_completion_mode_completed"])
    timed_out = int(meta["oracle_completion_mode_timeouts"])
    partial = int(meta.get("oracle_completion_partial_returned", 0))
    if min(attempts, completed, timed_out, partial) < 0 or completed + timed_out != attempts:
        raise ValueError("completion attempt accounting is inconsistent")
    if partial > timed_out:
        raise ValueError("partial return count exceeds timed-out modes")
    events = meta["global_best_events"]
    previous_score = None
    previous_elapsed = -1.0
    for event in events:
        score = tuple(float(value) for value in event["score"])
        elapsed = float(event["elapsed_s"])
        if elapsed < previous_elapsed or elapsed < 0.0 or elapsed > float(result["budget"]["elapsed_s"]) + 1.0:
            raise ValueError("best-event time is outside the run or decreases")
        if previous_score is not None and score <= previous_score:
            raise ValueError("global best score did not improve")
        previous_elapsed, previous_score = elapsed, score
    return {
        "mode_attempts": attempts,
        "mode_completed": completed,
        "mode_timeouts": timed_out,
        "partial_returned": partial,
        "best_event_count": len(events),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit, layouts = audit_batch(
        args.manifest, args.out_root, expected_attempt=args.attempt, expected_count=121,
        data_root=args.data_root, authorization_path=args.authorization,
    )
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    records = []
    for row, layout in zip(manifest_rows, layouts):
        result = json.loads(Path(layout["result_path"]).read_text(encoding="utf-8"))
        geometry = audit_geometry(result, args.data_root / row["source_csv"])
        trace = audit_completion_trace(result)
        records.append({"block_index": int(row["block_index"]), "method": row["method"], **geometry, **trace})
    audit["independent_geometry_results"] = len(records)
    audit["independent_completion_traces"] = sum(row["mode_attempts"] is not None for row in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"audit": audit, "records": records}, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "passed", "rows": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
