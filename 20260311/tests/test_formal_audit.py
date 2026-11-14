from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from inrp.formal_audit import audit_batch
from inrp.formal_batch import BUDGET_PROTOCOL


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _completed_one_row_batch(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.csv"
    row = {
        "attempt_number": "6",
        "block_index": "1",
        "task_id": "case",
        "batch_id": "batch",
        "analysis_role": "main",
        "seed": "1000",
        "method": "rh_mcts_prefix_beam",
        "budget_protocol": BUDGET_PROTOCOL,
        "source_csv": "source.csv",
        "source_sha256": "",
        "code_commit": "a" * 40,
    }
    source = tmp_path / "source.csv"
    source.write_text("source", encoding="utf-8")
    row["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    root = tmp_path / "results"
    snapshot = root / "pmset_assertions_at_start.txt"
    snapshot.parent.mkdir()
    snapshot.write_text("asserted", encoding="utf-8")
    _write_json(root / "batch_identity.json", {"formal_attempt": 6, "manifest_sha256": manifest_sha256})
    _write_json(root / "batch_state.json", {"status": "completed", "completed_count": 1, "total_count": 1})
    _write_json(
        root / "power_assertion.json",
        {
            "status": "asserted",
            "pmset_snapshot": snapshot.name,
            "pmset_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        },
    )
    attempt = root / "run_001" / "attempt_001"
    placement = {
        "panel_id": 1,
        "part_id": "p",
        "source_part_id": "p",
        "x_mm": 8.0,
        "y_mm": 8.0,
        "nominal_width_mm": 100.0,
        "nominal_height_mm": 200.0,
        "rotated": False,
    }
    utilization = 100.0 * 200.0 / (2440.0 * 1220.0)
    result = {
        "status": "usable",
        "formal_attempt": 6,
        "manifest_sha256": manifest_sha256,
        "manifest_row": row,
        "source_sha256": row["source_sha256"],
        "request": {
            "seed": 1000,
            "solver_variant": "rh_mcts_prefix_beam",
            "time_limit_s": 420.0,
            "parts": [{"part_id": "p"}],
        },
        "handoff": {
            "feasible": True,
            "panel_count": 1,
            "placements": [placement],
            "score_global": [-1.0, 0.1, 0.1, 0.0, utilization, 0.1],
            "runtime_s": 1.25,
        },
        "budget": {"total_s": 420.0, "closing_reserve_s": 60.0, "elapsed_s": 1.5},
    }
    result_path = attempt / "result.json"
    _write_json(result_path, result)
    _write_json(
        attempt / "watchdog.json",
        {"status": "completed", "hard_timeout_s": 425.0, "parent_wall_s": 1.75},
    )
    _write_json(
        root / "run_001" / "success.json",
        {
            "manifest_sha256": manifest_sha256,
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "attempt_dir": "attempt_001",
        },
    )
    return manifest, root


def test_completion_audit_emits_bound_layout_row(tmp_path: Path) -> None:
    manifest, root = _completed_one_row_batch(tmp_path)

    authorization = tmp_path / "authorization.json"
    _write_json(
        authorization,
        {
            "status": "authorized",
            "attempt_number": 6,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "code_commit": "a" * 40,
        },
    )
    audit, layouts = audit_batch(
        manifest,
        root,
        expected_attempt=6,
        expected_count=1,
        data_root=tmp_path,
        authorization_path=authorization,
    )

    assert audit["status"] == "passed"
    assert audit["usable_results"] == 1
    assert audit["authorization_sha256"] == hashlib.sha256(authorization.read_bytes()).hexdigest()
    assert layouts[0]["panel_count"] == 1
    assert len(layouts[0]["layout_fingerprint_sha256"]) == 64
    assert Path(layouts[0]["result_path"]).is_absolute()


def test_completion_audit_rejects_selective_rerun(tmp_path: Path) -> None:
    manifest, root = _completed_one_row_batch(tmp_path)
    (root / "run_001" / "attempt_002").mkdir()

    with pytest.raises(RuntimeError, match="selective rerun"):
        audit_batch(manifest, root, expected_attempt=6, expected_count=1)
