"""Completion and shared-layout audit for a frozen formal batch."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .formal_batch import BUDGET_PROTOCOL, sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _layout_fingerprint(placements: list[dict[str, Any]]) -> str:
    canonical = sorted(
        placements,
        key=lambda item: (int(item["panel_id"]), str(item["part_id"])),
    )
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def audit_batch(
    manifest_path: Path,
    output_root: Path,
    *,
    expected_attempt: int = 6,
    expected_count: int = 110,
    data_root: Path | None = None,
    authorization_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = Path(manifest_path)
    root = Path(output_root)
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == expected_count, f"manifest row count is {len(rows)}, expected {expected_count}")
    indexes = [int(row["block_index"]) for row in rows]
    _require(indexes == list(range(1, expected_count + 1)), "manifest indexes are not contiguous")
    keys = {(row["task_id"], row["seed"], row["method"]) for row in rows}
    _require(len(keys) == expected_count, "manifest task/seed/method keys are not unique")
    _require({int(row["attempt_number"]) for row in rows} == {expected_attempt}, "attempt mismatch")
    _require({row["budget_protocol"] for row in rows} == {BUDGET_PROTOCOL}, "budget protocol mismatch")
    commits = {row["code_commit"] for row in rows}
    _require(len(commits) == 1, "code commit mismatch within manifest")
    manifest_sha256 = sha256_file(manifest)
    authorization_sha256 = None
    if authorization_path is not None:
        authorization = _load_json(Path(authorization_path))
        _require(authorization.get("status") == "authorized", "authorization status mismatch")
        _require(int(authorization.get("attempt_number", 0)) == expected_attempt, "authorization attempt mismatch")
        _require(authorization.get("manifest_sha256") == manifest_sha256, "authorization manifest mismatch")
        _require(authorization.get("code_commit") in commits, "authorization code commit mismatch")
        authorization_sha256 = sha256_file(Path(authorization_path))

    identity = _load_json(root / "batch_identity.json")
    _require(identity == {"formal_attempt": expected_attempt, "manifest_sha256": manifest_sha256}, "identity mismatch")
    state = _load_json(root / "batch_state.json")
    _require(state.get("status") == "completed", "batch state is not completed")
    _require(int(state.get("completed_count", -1)) == expected_count, "completed count mismatch")
    _require(int(state.get("total_count", -1)) == expected_count, "total count mismatch")
    power = _load_json(root / "power_assertion.json")
    _require(power.get("status") == "asserted", "caffeinate assertion was not recorded")
    power_snapshot = root / str(power["pmset_snapshot"])
    _require(sha256_file(power_snapshot) == power["pmset_snapshot_sha256"], "power assertion snapshot hash mismatch")

    observed_run_dirs = sorted(path.name for path in root.glob("run_*"))
    _require(observed_run_dirs == [f"run_{idx:03d}" for idx in indexes], "run directory set mismatch")
    layouts: list[dict[str, Any]] = []
    for row in rows:
        idx = int(row["block_index"])
        run_dir = root / f"run_{idx:03d}"
        attempts = sorted(path.name for path in run_dir.glob("attempt_*"))
        _require(attempts == ["attempt_001"], f"block {idx} has selective rerun evidence: {attempts}")
        attempt_dir = run_dir / "attempt_001"
        result_path = attempt_dir / "result.json"
        result_sha256 = sha256_file(result_path)
        success = _load_json(run_dir / "success.json")
        _require(success.get("manifest_sha256") == manifest_sha256, f"block {idx} success manifest mismatch")
        _require(success.get("result_sha256") == result_sha256, f"block {idx} result hash mismatch")
        _require(success.get("attempt_dir") == "attempt_001", f"block {idx} success attempt mismatch")
        watchdog = _load_json(attempt_dir / "watchdog.json")
        _require(watchdog.get("status") == "completed", f"block {idx} watchdog not completed")
        _require(float(watchdog.get("hard_timeout_s", -1.0)) == 425.0, f"block {idx} watchdog budget mismatch")
        result = _load_json(result_path)
        _require(result.get("status") == "usable", f"block {idx} result is not usable")
        _require(int(result.get("formal_attempt", 0)) == expected_attempt, f"block {idx} attempt mismatch")
        _require(result.get("manifest_sha256") == manifest_sha256, f"block {idx} manifest mismatch")
        _require(result.get("manifest_row") == row, f"block {idx} manifest row mismatch")
        _require(result.get("source_sha256") == row["source_sha256"], f"block {idx} source mismatch")
        if data_root is not None:
            source = Path(data_root) / row["source_csv"]
            _require(sha256_file(source) == row["source_sha256"], f"block {idx} current input hash mismatch")
        request = result.get("request") or {}
        _require(int(request.get("seed", -1)) == int(row["seed"]), f"block {idx} seed mismatch")
        _require(request.get("solver_variant") == row["method"], f"block {idx} method mismatch")
        _require(float(request.get("time_limit_s", -1.0)) == 420.0, f"block {idx} total budget mismatch")
        budget = result.get("budget") or {}
        _require(float(budget.get("total_s", -1.0)) == 420.0, f"block {idx} budget total mismatch")
        _require(float(budget.get("closing_reserve_s", -1.0)) == 60.0, f"block {idx} closing reserve mismatch")
        handoff = result.get("handoff") or {}
        placements = handoff.get("placements") or []
        panel_count = int(handoff.get("panel_count", 0))
        _require(handoff.get("feasible") is True and panel_count > 0, f"block {idx} infeasible handoff")
        _require(len(placements) == len(request.get("parts") or []), f"block {idx} placement count mismatch")
        _require(len({int(item["panel_id"]) for item in placements}) == panel_count, f"block {idx} panel count mismatch")
        used_area = sum(float(item["nominal_width_mm"]) * float(item["nominal_height_mm"]) for item in placements)
        utilization = used_area / (panel_count * 2440.0 * 1220.0)
        score_global = handoff.get("score_global") or []
        if row["method"] == "baseline_greedy":
            _require(not score_global, f"block {idx} Greedy unexpectedly has search score")
        else:
            _require(
                len(score_global) > 4 and abs(float(score_global[4]) - utilization) < 1e-9,
                f"block {idx} utilization mismatch",
            )
        layouts.append(
            {
                "block_index": idx,
                "task_id": row["task_id"],
                "batch_id": row["batch_id"],
                "analysis_role": row["analysis_role"],
                "seed": int(row["seed"]),
                "method": row["method"],
                "layout_fingerprint_sha256": _layout_fingerprint(placements),
                "panel_count": panel_count,
                "global_utilization": utilization,
                "solver_runtime_s": float(handoff["runtime_s"]),
                "budget_elapsed_s": float(budget["elapsed_s"]),
                "watchdog_parent_wall_s": float(watchdog["parent_wall_s"]),
                "result_path": str(result_path.resolve()),
                "result_sha256": result_sha256,
                "source_sha256": row["source_sha256"],
                "manifest_sha256": manifest_sha256,
                "code_commit": row["code_commit"],
                "budget_protocol": row["budget_protocol"],
                "formal_attempt": expected_attempt,
            }
        )
    audit = {
        "audit_type": "formal_batch_completion",
        "status": "passed",
        "formal_attempt": expected_attempt,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "code_commit": commits.pop(),
        "budget_protocol": BUDGET_PROTOCOL,
        "unique_manifest_rows": len(keys),
        "usable_results": len(layouts),
        "selective_reruns": 0,
        "power_assertion_status": "asserted",
    }
    if authorization_path is not None:
        audit["authorization"] = str(Path(authorization_path).resolve())
        audit["authorization_sha256"] = authorization_sha256
    return audit, layouts
