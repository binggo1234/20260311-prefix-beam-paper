from __future__ import annotations

import copy
import csv
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_ch2_evidence import audit_completion_trace, audit_geometry


def _fixture(tmp_path: Path):
    source = tmp_path / "parts.csv"
    with source.open("w", encoding="gb18030", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Upi", "Fleng", "Fwidth"))
        writer.writeheader()
        writer.writerows((
            {"Upi": "A", "Fleng": "100", "Fwidth": "100"},
            {"Upi": "B", "Fleng": "100", "Fwidth": "100"},
        ))
    parts = [
        {"part_id": f"{key}__copy_{index:06d}", "source_part_id": key,
         "nominal_width_mm": 100.0, "nominal_height_mm": 100.0,
         "allowed_rotations_deg": [0, 90]}
        for index, key in enumerate(("A", "B"), 1)
    ]
    request = {
        "parts": parts, "tool_diameter_mm": 6.0, "trim_mm": 5.0,
        "board_width_mm": 2440.0, "board_height_mm": 1220.0,
        "solver_variant": "rh_mcts_prefix_beam",
    }
    placements = [
        {"panel_id": 1, "part_id": part["part_id"], "source_part_id": part["source_part_id"],
         "x_mm": x, "y_mm": 8.0, "nominal_width_mm": 100.0,
         "nominal_height_mm": 100.0, "rotated": False}
        for part, x in zip(parts, (8.0, 114.0))
    ]
    result = {
        "request": request,
        "handoff": {"placements": placements, "panel_count": 1,
                    "sequence": [part["part_id"] for part in parts],
                    "provenance": {"request": request, "solver_meta": {
                        "oracle_completion_mode_evals": 2,
                        "oracle_completion_mode_completed": 1,
                        "oracle_completion_mode_timeouts": 1,
                        "oracle_completion_partial_returned": 1,
                        "global_best_events": [{"elapsed_s": 0.2, "score": [-1.0]}],
                    }}},
        "budget": {"elapsed_s": 1.0},
    }
    return source, result


def test_ch2_independent_geometry_and_trace_accept_valid_layout(tmp_path: Path):
    source, result = _fixture(tmp_path)
    assert audit_geometry(result, source)["part_count"] == 2
    assert audit_completion_trace(result)["mode_completed"] == 1


@pytest.mark.parametrize("damage,expected", [
    ("missing", "missing"), ("overlap", "overlap"),
    ("boundary", "boundary"), ("identity", "source/request"),
    ("rotation", "rotation"),
])
def test_ch2_independent_geometry_rejects_corruption(tmp_path: Path, damage: str, expected: str):
    source, result = _fixture(tmp_path)
    result = copy.deepcopy(result)
    if damage == "missing":
        result["handoff"]["placements"].pop()
    elif damage == "overlap":
        result["handoff"]["placements"][1]["x_mm"] = 113.0
    elif damage == "boundary":
        result["handoff"]["placements"][0]["x_mm"] = 7.0
    elif damage == "identity":
        result["request"]["parts"][0]["source_part_id"] = "wrong"
    elif damage == "rotation":
        result["request"]["parts"][0]["allowed_rotations_deg"] = [0]
        result["handoff"]["placements"][0]["rotated"] = True
    with pytest.raises(ValueError, match=expected):
        audit_geometry(result, source)


def test_ch2_trace_rejects_inconsistent_mode_accounting(tmp_path: Path):
    _, result = _fixture(tmp_path)
    result["handoff"]["provenance"]["solver_meta"]["oracle_completion_mode_completed"] = 2
    with pytest.raises(ValueError, match="accounting"):
        audit_completion_trace(result)
