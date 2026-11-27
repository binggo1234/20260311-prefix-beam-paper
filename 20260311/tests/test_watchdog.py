from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import pytest

from inrp.watchdog import run_command_with_watchdog


def test_watchdog_reports_completed_worker(tmp_path: Path) -> None:
    output = tmp_path / "worker.json"
    code = (
        "import json, pathlib; "
        f"pathlib.Path({str(output)!r}).write_text(json.dumps({{'ok': True}}))"
    )

    result = run_command_with_watchdog(
        (sys.executable, "-c", code),
        hard_timeout_s=2.0,
    )

    assert result.status == "completed"
    assert result.returncode == 0
    assert json.loads(output.read_text()) == {"ok": True}


def test_watchdog_reports_worker_pid_at_start() -> None:
    observed = []

    result = run_command_with_watchdog(
        (sys.executable, "-c", "pass"),
        hard_timeout_s=2.0,
        on_started=observed.append,
    )

    assert result.status == "completed"
    assert result.child_pid == observed[0]
    assert observed[0] > 0


def test_watchdog_terminates_overrunning_worker(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    code = (
        "import pathlib, time; time.sleep(2); "
        f"pathlib.Path({str(marker)!r}).write_text('late')"
    )
    started = time.perf_counter()

    result = run_command_with_watchdog(
        (sys.executable, "-c", code),
        hard_timeout_s=0.1,
        terminate_grace_s=0.1,
    )

    assert result.status in {"watchdog_terminated", "watchdog_killed"}
    assert result.returncode is not None and result.returncode < 0
    assert time.perf_counter() - started < 1.0
    assert not marker.exists()


def test_watchdog_cleans_child_if_parent_path_is_interrupted(tmp_path: Path) -> None:
    observed = []

    def interrupt(pid: int) -> None:
        observed.append(pid)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_command_with_watchdog(
            (sys.executable, "-c", "import time; time.sleep(10)"),
            hard_timeout_s=20.0,
            terminate_grace_s=0.1,
            on_started=interrupt,
        )

    with pytest.raises(ProcessLookupError):
        os.kill(observed[0], 0)
