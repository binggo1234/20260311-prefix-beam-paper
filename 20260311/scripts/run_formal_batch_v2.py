#!/usr/bin/env python3
"""Resumable, evidence-preserving launcher for a frozen formal batch."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.budget import TotalBudget, atomic_write_json_with_budget
from inrp.formal_batch import completed_block_indexes, load_packing_parts, sha256_file
from inrp.handoff import PackingRequest, solve_packing_handoff_with_budget
from inrp.watchdog import run_command_with_watchdog


TOTAL_BUDGET_S = 420.0
CLOSING_RESERVE_S = 60.0
HARD_TIMEOUT_S = 425.0
TERMINATE_GRACE_S = 2.0
CAFFEINATE = "/usr/bin/caffeinate"
CAFFEINATE_FLAGS = "-dimsu"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _caffeinate_command(argv: tuple[str, ...]) -> tuple[str, ...]:
    return (CAFFEINATE, CAFFEINATE_FLAGS, sys.executable, *argv, "--caffeinated")


def _pmset_asserts_for_pid(output: str, pid: int) -> bool:
    pattern = rf"caffeinate asserting on behalf of .*\(pid\s+{int(pid)}\)"
    return re.search(pattern, output, flags=re.IGNORECASE) is not None


def _capture_power_assertion(output_root: Path) -> dict[str, Any]:
    if sys.platform != "darwin":
        raise RuntimeError("formal parent requires macOS caffeinate")
    if os.environ.get("INRP_CAFFEINATE_WRAPPED") != "1":
        raise RuntimeError("formal parent did not enter through the caffeinate wrapper")
    last_output = ""
    for _ in range(20):
        completed = subprocess.run(
            ("/usr/bin/pmset", "-g", "assertions"),
            text=True,
            capture_output=True,
            check=True,
        )
        last_output = completed.stdout
        if _pmset_asserts_for_pid(last_output, os.getpid()):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("pmset did not confirm caffeinate assertion for launcher PID")
    snapshot_path = output_root / "pmset_assertions_at_start.txt"
    _atomic_text(snapshot_path, last_output)
    evidence = {
        "status": "asserted",
        "captured_at": _utc_now(),
        "launcher_pid": os.getpid(),
        "caffeinate_executable": CAFFEINATE,
        "caffeinate_flags": CAFFEINATE_FLAGS,
        "pmset_snapshot": snapshot_path.name,
        "pmset_snapshot_sha256": sha256_file(snapshot_path),
    }
    _atomic_json(output_root / "power_assertion.json", evidence)
    return evidence


def _load_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = list(range(1, len(rows) + 1))
    observed = [int(row["block_index"]) for row in rows]
    if not rows or observed != expected:
        raise ValueError("manifest block indexes must be contiguous and one-based")
    return rows


def _manifest_attempt_number(rows: list[dict[str, str]]) -> int:
    attempts = {int(row["attempt_number"]) for row in rows}
    if len(attempts) != 1:
        raise ValueError("manifest must contain exactly one attempt number")
    attempt_number = attempts.pop()
    if attempt_number < 1:
        raise ValueError("manifest attempt number must be positive")
    return attempt_number


def _verify_code_commit(expected: str) -> None:
    actual = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(f"code commit mismatch: expected={expected} actual={actual}")
    tracked = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if tracked:
        raise RuntimeError("tracked working tree must be clean for a formal run")


def _verify_head_commit(expected: str) -> None:
    actual = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(f"code commit mismatch: expected={expected} actual={actual}")


def _tracked_diff_sha256() -> str:
    patch = subprocess.run(
        ("git", "diff", "--binary", "HEAD"), cwd=PROJECT_ROOT, capture_output=True, check=True
    ).stdout
    return hashlib.sha256(patch).hexdigest()


def _worker(
    row: dict[str, str],
    *,
    data_root: Path,
    output: Path,
    manifest_sha256: str,
    attempt_number: int,
    diagnostic_only: bool = False,
) -> int:
    source = data_root / row["source_csv"]
    if sha256_file(source) != row["source_sha256"]:
        raise RuntimeError(f"source hash mismatch before worker start: {source}")
    budget = TotalBudget(TOTAL_BUDGET_S, CLOSING_RESERVE_S)
    budget.mark("formal_input_boundary")
    parts = load_packing_parts(source)
    budget.mark("input_read_and_adapted")
    request = PackingRequest(
        case_id=row["task_id"],
        parts=parts,
        board_width_mm=2440.0,
        board_height_mm=1220.0,
        trim_mm=5.0,
        tool_diameter_mm=6.0,
        allow_rotation=True,
        seed=int(row["seed"]),
        time_limit_s=TOTAL_BUDGET_S,
        solver_variant=row["method"],
    )
    handoff = solve_packing_handoff_with_budget(request, budget)
    payload = {
        "status": "usable",
        "formal_attempt": int(attempt_number),
        "manifest_sha256": manifest_sha256,
        "manifest_row": row,
        "source_sha256": sha256_file(source),
        "request": asdict(request),
        "handoff": asdict(handoff),
        "budget": budget.summary(),
    }
    if diagnostic_only:
        payload.update(
            {
                "diagnostic_only": True,
                "formal_statistics_eligible": False,
                "tracked_worktree_diff_sha256": _tracked_diff_sha256(),
            }
        )
    atomic_write_json_with_budget(output, payload, budget)
    return 0


def _validate_worker_result(
    payload: Mapping[str, Any],
    row: Mapping[str, str],
    manifest_sha256: str,
    attempt_number: int,
) -> None:
    if payload.get("status") != "usable":
        raise RuntimeError(f"worker result is not usable: {payload.get('status')!r}")
    if int(payload.get("formal_attempt", 0)) != int(attempt_number):
        raise RuntimeError("worker result formal attempt mismatch")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("worker result manifest mismatch")
    if payload.get("manifest_row") != dict(row):
        raise RuntimeError("worker result manifest row mismatch")
    if payload.get("source_sha256") != row["source_sha256"]:
        raise RuntimeError("worker result source hash mismatch")
    request = payload.get("request") or {}
    if request.get("case_id") != row["task_id"]:
        raise RuntimeError("worker result case mismatch")
    if int(request.get("seed", -1)) != int(row["seed"]):
        raise RuntimeError("worker result seed mismatch")
    if request.get("solver_variant") != row["method"]:
        raise RuntimeError("worker result method mismatch")
    if float(request.get("time_limit_s", -1.0)) != TOTAL_BUDGET_S:
        raise RuntimeError("worker result total budget mismatch")
    budget = payload.get("budget") or {}
    if float(budget.get("total_s", -1.0)) != TOTAL_BUDGET_S:
        raise RuntimeError("worker budget total mismatch")
    if float(budget.get("closing_reserve_s", -1.0)) != CLOSING_RESERVE_S:
        raise RuntimeError("worker closing reserve mismatch")
    handoff = payload.get("handoff") or {}
    if not handoff.get("feasible") or int(handoff.get("panel_count", 0)) <= 0:
        raise RuntimeError("worker handoff is not feasible")
    if not handoff.get("placements"):
        raise RuntimeError("worker handoff has no placements")


def _next_attempt_dir(run_dir: Path) -> Path:
    indexes = []
    for path in run_dir.glob("attempt_*"):
        try:
            indexes.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    attempt = run_dir / f"attempt_{max(indexes, default=0) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def _validate_authorization(
    path: Path,
    manifest_sha256: str,
    attempt_number: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "authorized" or payload.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("authorization does not match the frozen manifest")
    if int(payload.get("attempt_number", 0)) != int(attempt_number):
        raise RuntimeError(f"authorization is not for formal attempt {attempt_number}")
    return payload


def _run_parent(args: argparse.Namespace) -> int:
    manifest_sha256 = sha256_file(args.manifest)
    rows = _load_rows(args.manifest)
    attempt_number = _manifest_attempt_number(rows)
    required_rows = (
        1 if args.startup_acceptance else
        121 if any(row.get("method") == "baseline_greedy" for row in rows) else 110
    )
    if len(rows) != required_rows:
        run_kind = "startup acceptance" if args.startup_acceptance else f"formal attempt {attempt_number}"
        raise RuntimeError(f"{run_kind} requires exactly {required_rows} rows, got {len(rows)}")
    _validate_authorization(args.authorization, manifest_sha256, attempt_number)
    expected_commits = {row["code_commit"] for row in rows}
    if len(expected_commits) != 1:
        raise RuntimeError("manifest contains multiple code commits")
    expected_commit = expected_commits.pop()
    if args.startup_acceptance:
        _verify_head_commit(expected_commit)
    else:
        _verify_code_commit(expected_commit)

    args.out_root.mkdir(parents=True, exist_ok=True)
    identity_path = args.out_root / "batch_identity.json"
    identity = {"formal_attempt": attempt_number, "manifest_sha256": manifest_sha256}
    if args.startup_acceptance:
        identity["diagnostic_only"] = True
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("output directory belongs to a different manifest")
    else:
        unexpected = tuple(args.out_root.iterdir())
        if unexpected:
            raise RuntimeError("new output directory is nonempty and has no batch identity")
        _atomic_json(identity_path, identity)

    _capture_power_assertion(args.out_root)

    completed = completed_block_indexes(args.out_root, manifest_sha256=manifest_sha256)
    state_path = args.out_root / "batch_state.json"
    current_block_index = None
    current_attempt_dir = None
    try:
        for row in rows:
            block_index = int(row["block_index"])
            if block_index in completed:
                continue
            run_dir = args.out_root / f"run_{block_index:03d}"
            attempt_dir = _next_attempt_dir(run_dir)
            current_block_index = block_index
            current_attempt_dir = attempt_dir
            running_state = {
                "status": "running",
                "launcher_pid": os.getpid(),
                "current_block_index": block_index,
                "completed_count": len(completed),
                "started_at": _utc_now(),
                "attempt_dir": str(attempt_dir),
                "power_assertion_sha256": sha256_file(args.out_root / "power_assertion.json"),
            }
            _atomic_json(state_path, running_state)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--manifest",
                str(args.manifest),
                "--data-root",
                str(args.data_root),
                "--worker-row",
                str(block_index),
                "--worker-output",
                str(attempt_dir / "result.json"),
                "--manifest-sha256",
                manifest_sha256,
            ]
            if args.startup_acceptance:
                command.append("--diagnostic-worker")
            watchdog = run_command_with_watchdog(
                command,
                hard_timeout_s=HARD_TIMEOUT_S,
                terminate_grace_s=TERMINATE_GRACE_S,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": str(SRC)},
                stdout_path=attempt_dir / "worker.stdout.log",
                stderr_path=attempt_dir / "worker.stderr.log",
                on_started=lambda pid, state=running_state: _atomic_json(
                    state_path, {**state, "worker_pid": int(pid), "worker_started_at": _utc_now()}
                ),
            )
            _atomic_json(attempt_dir / "watchdog.json", watchdog.to_dict())
            result_path = attempt_dir / "result.json"
            if watchdog.status != "completed":
                raise RuntimeError(f"block {block_index} watchdog status is {watchdog.status}")
            if not result_path.is_file():
                raise RuntimeError(f"block {block_index} completed without result.json")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            _validate_worker_result(payload, row, manifest_sha256, attempt_number)
            _atomic_json(
                run_dir / "success.json",
                {
                    "block_index": block_index,
                    "manifest_sha256": manifest_sha256,
                    "attempt_dir": attempt_dir.name,
                    "result_sha256": sha256_file(result_path),
                    "completed_at": _utc_now(),
                },
            )
            completed.add(block_index)
            _atomic_json(
                state_path,
                {
                    "status": "running",
                    "launcher_pid": os.getpid(),
                    "last_block_index": block_index,
                    "last_worker_status": watchdog.status,
                    "completed_count": len(completed),
                    "updated_at": _utc_now(),
                },
            )
    except BaseException as exc:
        _atomic_json(
            state_path,
            {
                "status": "interrupted",
                "launcher_pid": os.getpid(),
                "current_block_index": current_block_index,
                "attempt_dir": str(current_attempt_dir) if current_attempt_dir else None,
                "completed_count": len(completed),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at": _utc_now(),
            },
        )
        raise

    all_complete = len(completed) == len(rows)
    final_status = "acceptance_completed" if args.startup_acceptance and all_complete else (
        "completed" if all_complete else "incomplete"
    )
    _atomic_json(
        state_path,
        {
            "status": final_status,
            "launcher_pid": os.getpid(),
            "completed_count": len(completed),
            "total_count": len(rows),
            "updated_at": _utc_now(),
        },
    )
    return 0 if all_complete else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--worker-row", type=int)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--caffeinated", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--startup-acceptance", action="store_true")
    parser.add_argument("--diagnostic-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    needs_caffeinate = args.worker_row is None or args.startup_acceptance
    if needs_caffeinate and not args.caffeinated:
        if sys.platform != "darwin":
            raise RuntimeError("formal parent requires macOS caffeinate")
        if not Path(CAFFEINATE).is_file():
            raise RuntimeError(f"caffeinate is unavailable: {CAFFEINATE}")
        env = {**os.environ, "INRP_CAFFEINATE_WRAPPED": "1"}
        os.execvpe(CAFFEINATE, _caffeinate_command(tuple(sys.argv)), env)
        raise AssertionError("os.execvpe returned unexpectedly")
    rows = _load_rows(args.manifest)
    attempt_number = _manifest_attempt_number(rows)
    if args.worker_row is not None:
        if args.worker_output is None or args.manifest_sha256 is None:
            parser.error("worker mode requires --worker-output and --manifest-sha256")
        row = rows[args.worker_row - 1]
        if int(row["block_index"]) != args.worker_row:
            raise RuntimeError("worker row does not match block index")
        actual_manifest_sha256 = sha256_file(args.manifest)
        if actual_manifest_sha256 != args.manifest_sha256:
            raise RuntimeError("worker manifest SHA-256 argument does not match manifest bytes")
        if args.startup_acceptance:
            _verify_head_commit(row["code_commit"])
            if not args.caffeinated:
                raise RuntimeError("startup acceptance did not enter through caffeinate")
            _capture_power_assertion(args.worker_output.parent)
        elif args.diagnostic_worker:
            _verify_head_commit(row["code_commit"])
        else:
            _verify_code_commit(row["code_commit"])
        return _worker(
            row,
            data_root=args.data_root,
            output=args.worker_output,
            manifest_sha256=args.manifest_sha256,
            attempt_number=attempt_number,
            diagnostic_only=args.startup_acceptance or args.diagnostic_worker,
        )
    if args.out_root is None or args.authorization is None:
        parser.error("parent mode requires --out-root and --authorization")
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
