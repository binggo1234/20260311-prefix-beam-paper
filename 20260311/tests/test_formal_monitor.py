from __future__ import annotations

import json
from pathlib import Path

from inrp.formal_monitor import evaluate_batch_health


def test_large_etime_is_not_a_failure_when_parent_and_worker_are_alive(tmp_path: Path) -> None:
    attempt = tmp_path / "run_008" / "attempt_001"
    attempt.mkdir(parents=True)
    (tmp_path / "power_assertion.json").write_text(
        json.dumps({"status": "asserted", "launcher_pid": 111}), encoding="utf-8"
    )
    (tmp_path / "batch_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "launcher_pid": 111,
                "worker_pid": 222,
                "current_block_index": 8,
                "attempt_dir": str(attempt),
            }
        ),
        encoding="utf-8",
    )
    process_table = "111 1 99:59:59 python runner.py\n222 111 99:59:00 python worker.py\n"

    health = evaluate_batch_health(tmp_path, process_table=process_table)

    assert health["verdict"] == "healthy_running"
    assert health["etime_is_failure_signal"] is False


def test_bad_watchdog_is_fatal_even_if_etime_is_short(tmp_path: Path) -> None:
    attempt = tmp_path / "run_001" / "attempt_001"
    attempt.mkdir(parents=True)
    (tmp_path / "power_assertion.json").write_text(
        json.dumps({"status": "asserted", "launcher_pid": 111}), encoding="utf-8"
    )
    (tmp_path / "batch_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "launcher_pid": 111,
                "worker_pid": 222,
                "current_block_index": 1,
                "attempt_dir": str(attempt),
            }
        ),
        encoding="utf-8",
    )
    (attempt / "watchdog.json").write_text(
        json.dumps({"status": "watchdog_terminated"}), encoding="utf-8"
    )

    health = evaluate_batch_health(tmp_path, process_table="111 1 00:01 python\n")

    assert health["verdict"] == "fatal"
    assert "watchdog_terminated" in health["reasons"]
