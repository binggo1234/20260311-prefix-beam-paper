#!/usr/bin/env python3
"""Quiet, read-only monitor for the formal batch evidence directory."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.formal_monitor import evaluate_batch_health


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _command_output(command: tuple[str, ...]) -> str:
    return subprocess.run(command, text=True, capture_output=True, check=True).stdout


def _recent_power_events(limit: int = 80) -> list[str]:
    process = subprocess.Popen(("/usr/bin/pmset", "-g", "log"), stdout=subprocess.PIPE, text=True)
    assert process.stdout is not None
    selected = deque(maxlen=limit)
    for line in process.stdout:
        if " Sleep " in line or " Wake " in line or "DarkWake" in line:
            selected.append(line.rstrip())
    if process.wait() != 0:
        return []
    return list(selected)


def inspect_once(output_root: Path, *, include_power_log: bool = False) -> dict:
    process_table = _command_output(("/bin/ps", "-axo", "pid=,ppid=,etime=,command="))
    assertions = _command_output(("/usr/bin/pmset", "-g", "assertions"))
    return evaluate_batch_health(
        output_root,
        process_table=process_table,
        pmset_assertions=assertions,
        recent_power_events=_recent_power_events() if include_power_log else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    previous = None
    while True:
        health = inspect_once(args.out_root)
        if health["verdict"] == "fatal":
            health = inspect_once(args.out_root, include_power_log=True)
        _atomic_json(args.out_root / "monitor_state.json", health)
        current = (health["verdict"], tuple(health.get("reasons", ())))
        if current != previous or health["verdict"] in {"fatal", "completed"}:
            print(json.dumps(health, ensure_ascii=False, sort_keys=True), flush=True)
        if args.once or health["verdict"] in {"fatal", "completed"}:
            return 0 if health["verdict"] in {"healthy_running", "healthy_transition", "completed"} else 2
        previous = current
        time.sleep(max(5.0, args.interval_s))


if __name__ == "__main__":
    raise SystemExit(main())
