"""Parent-process containment for formal experiment workers."""

from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class WatchdogResult:
    status: str
    returncode: Optional[int]
    parent_wall_s: float
    hard_timeout_s: float
    terminate_grace_s: float
    child_pid: int
    command: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def run_command_with_watchdog(
    command: Sequence[str],
    *,
    hard_timeout_s: float,
    terminate_grace_s: float = 1.0,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
    on_started: Optional[Callable[[int], None]] = None,
) -> WatchdogResult:
    """Run one worker and terminate its process group after a hard parent timeout.

    The parent wall clock is a containment limit, not the algorithm runtime. The
    child remains responsible for starting and recording the formal total budget
    at the declared input boundary.
    """

    argv = tuple(str(item) for item in command)
    if not argv:
        raise ValueError("command must not be empty")
    timeout = float(hard_timeout_s)
    grace = float(terminate_grace_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("hard_timeout_s must be positive and finite")
    if not math.isfinite(grace) or grace < 0.0:
        raise ValueError("terminate_grace_s must be nonnegative and finite")

    stdout_target = subprocess.DEVNULL
    stderr_target = subprocess.DEVNULL
    stdout_fh = None
    stderr_fh = None
    try:
        if stdout_path is not None:
            stdout_file = Path(stdout_path)
            stdout_file.parent.mkdir(parents=True, exist_ok=True)
            stdout_fh = stdout_file.open("wb")
            stdout_target = stdout_fh
        if stderr_path is not None:
            stderr_file = Path(stderr_path)
            stderr_file.parent.mkdir(parents=True, exist_ok=True)
            stderr_fh = stderr_file.open("wb")
            stderr_target = stderr_fh

        started = time.perf_counter()
        process = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=stdout_target,
            stderr=stderr_target,
            start_new_session=True,
        )
        try:
            if on_started is not None:
                on_started(int(process.pid))
            try:
                returncode = process.wait(timeout=timeout)
                status = "completed" if returncode == 0 else "worker_failed"
            except subprocess.TimeoutExpired:
                status = "watchdog_terminated"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    returncode = process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    status = "watchdog_killed"
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            raise
        parent_wall_s = time.perf_counter() - started
        return WatchdogResult(
            status=status,
            returncode=returncode,
            parent_wall_s=float(parent_wall_s),
            hard_timeout_s=timeout,
            terminate_grace_s=grace,
            child_pid=int(process.pid),
            command=argv,
        )
    finally:
        if stdout_fh is not None:
            stdout_fh.close()
        if stderr_fh is not None:
            stderr_fh.close()
