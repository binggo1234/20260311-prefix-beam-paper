import argparse
import csv
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


TERMINAL_STATUSES = {"completed", "failed", "no_comparison", "interrupted"}


def _read_status_rows(status_path: Path):
    with status_path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _status_summary(rows) -> str:
    counts = Counter(str(row.get("status", "")).strip() for row in rows)
    return "; ".join(f"{name}={counts[name]}" for name in sorted(counts))


def _append_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _python_memory_gb() -> float:
    try:
        import psutil  # type: ignore

        total = 0
        for proc in psutil.process_iter(["name", "memory_info"]):
            name = str(proc.info.get("name") or "").lower()
            if "python" not in name:
                continue
            mem = proc.info.get("memory_info")
            if mem is not None:
                total += int(mem.rss)
        return round(total / (1024 ** 3), 2)
    except Exception:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight monitor for a benchmark suite output directory.")
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--interval-s", type=float, default=60.0)
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    status_path = suite_dir / "suite_case_status.csv"
    err_path = suite_dir / "launcher.err.log"
    log_path = suite_dir / "monitor.log"

    _append_log(log_path, f"[{datetime.now():%Y-%m-%d %H:%M:%S}] MONITOR_START suite_dir={suite_dir}")

    while True:
        ts = f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"
        if not status_path.is_file():
            _append_log(log_path, f"{ts} HEARTBEAT | status_file_missing")
            time.sleep(max(1.0, float(args.interval_s)))
            continue

        try:
            rows = _read_status_rows(status_path)
            summary = _status_summary(rows)
            err_bytes = err_path.stat().st_size if err_path.is_file() else 0
            py_ws_gb = _python_memory_gb()
            _append_log(
                log_path,
                f"{ts} HEARTBEAT | statuses: {summary} | py_ws_gb={py_ws_gb:.2f} | err_bytes={err_bytes}",
            )
            if rows and all(str(row.get("status", "")).strip() in TERMINAL_STATUSES for row in rows):
                _append_log(log_path, f"{ts} COMPLETE | statuses: {summary} | err_bytes={err_bytes}")
                return 0
        except Exception as exc:
            _append_log(log_path, f"{ts} HEARTBEAT | status_read_error={exc!r}")

        time.sleep(max(1.0, float(args.interval_s)))


if __name__ == "__main__":
    sys.exit(main())
