"""Read-only health classification for a running formal batch.

Process elapsed time is deliberately retained only as diagnostic text. A fatal
verdict requires persisted batch/worker evidence or missing live processes.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _process_rows(process_table: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for raw in process_table.splitlines():
        fields = raw.strip().split(None, 3)
        if len(fields) < 3:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        rows[pid] = {
            "ppid": ppid,
            "etime": fields[2],
            "command": fields[3] if len(fields) == 4 else "",
        }
    return rows


def evaluate_batch_health(
    output_root: Path,
    *,
    process_table: str,
    pmset_assertions: str = "",
    recent_power_events: list[str] | None = None,
) -> dict[str, Any]:
    """Classify current evidence without changing or signaling any process."""

    root = Path(output_root)
    state = _read_json(root / "batch_state.json")
    if state is None:
        return {
            "verdict": "fatal",
            "reasons": ["batch_state_missing_or_invalid"],
            "etime_is_failure_signal": False,
        }
    status = str(state.get("status", ""))
    if status == "completed":
        return {"verdict": "completed", "reasons": [], "etime_is_failure_signal": False}
    if status in {"interrupted", "incomplete"}:
        return {
            "verdict": "fatal",
            "reasons": [f"batch_state_{status}"],
            "etime_is_failure_signal": False,
        }

    reasons: list[str] = []
    assertion = _read_json(root / "power_assertion.json")
    launcher_pid = int(state.get("launcher_pid", 0) or 0)
    worker_pid = int(state.get("worker_pid", 0) or 0)
    if not assertion or assertion.get("status") != "asserted":
        reasons.append("caffeinate_assertion_missing_or_invalid")
    elif int(assertion.get("launcher_pid", 0) or 0) != launcher_pid:
        reasons.append("caffeinate_launcher_pid_mismatch")
    assertion_pattern = rf"caffeinate asserting on behalf of .*\(pid\s+{launcher_pid}\)"
    if pmset_assertions and re.search(assertion_pattern, pmset_assertions, re.IGNORECASE) is None:
        reasons.append("live_caffeinate_assertion_missing")

    attempt_dir_value = state.get("attempt_dir")
    attempt_dir = Path(str(attempt_dir_value)) if attempt_dir_value else None
    watchdog = _read_json(attempt_dir / "watchdog.json") if attempt_dir else None
    result = _read_json(attempt_dir / "result.json") if attempt_dir else None
    if watchdog and watchdog.get("status") != "completed":
        reasons.append(str(watchdog.get("status") or "watchdog_invalid"))
    if result and result.get("status") != "usable":
        reasons.append(f"result_{result.get('status', 'invalid')}")

    processes = _process_rows(process_table)
    launcher_alive = launcher_pid in processes
    worker_alive = worker_pid in processes
    if not launcher_alive:
        reasons.append("launcher_process_missing")
    if worker_pid and not worker_alive and watchdog is None:
        reasons.append("worker_process_missing_without_watchdog")

    verdict = "fatal" if reasons else ("healthy_running" if worker_alive else "healthy_transition")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "status": status,
        "launcher_pid": launcher_pid,
        "worker_pid": worker_pid,
        "launcher_alive": launcher_alive,
        "worker_alive": worker_alive,
        "launcher_etime": processes.get(launcher_pid, {}).get("etime"),
        "worker_etime": processes.get(worker_pid, {}).get("etime"),
        "recent_power_events": list(recent_power_events or ()),
        "etime_is_failure_signal": False,
    }
