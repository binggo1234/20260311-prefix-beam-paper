"""Frozen-input helpers for the second formal industrial batch."""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence, Set, Tuple

from .handoff import PackingPart


DIMENSION_PRECISION_POLICY = "source_decimal_max_0.1_mm_no_rounding"
MANIFEST_SCHEMA_VERSION = "inrp-formal-run-manifest-v2"
BUDGET_PROTOCOL = "v4_420s_total_60s_close_425s_watchdog"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dimension(value: str, *, row_number: int, column: str) -> float:
    text = str(value).strip()
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid {column} at CSV row {row_number}: {text!r}") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"{column} must be positive and finite at CSV row {row_number}")
    if number.as_tuple().exponent < -1:
        raise ValueError(
            f"{column} exceeds frozen 0.1 mm source precision at CSV row {row_number}: {text!r}"
        )
    return float(number)


def load_packing_parts(path: Path) -> Tuple[PackingPart, ...]:
    """Load exact production fields without aliasing or dimension rounding."""

    source = Path(path)
    with source.open(encoding="gb18030", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        required = ("Upi", "Fleng", "Fwidth")
        if any(name not in fieldnames for name in required):
            raise ValueError(f"production CSV must contain exact columns {required}; got {fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("production CSV contains no parts")

    seen: Set[str] = set()
    parts = []
    for row_number, row in enumerate(rows, start=2):
        source_id = str(row["Upi"]).strip()
        if not source_id:
            raise ValueError(f"blank Upi at CSV row {row_number}")
        if source_id in seen:
            raise ValueError(f"duplicate Upi at CSV row {row_number}: {source_id}")
        seen.add(source_id)
        width = _dimension(row["Fleng"], row_number=row_number, column="Fleng")
        height = _dimension(row["Fwidth"], row_number=row_number, column="Fwidth")
        copy_index = row_number - 1
        parts.append(
            PackingPart(
                part_id=f"{source_id}__copy_{copy_index:06d}",
                source_part_id=source_id,
                nominal_width_mm=width,
                nominal_height_mm=height,
                allowed_rotations_deg=(0, 90),
            )
        )
    return tuple(parts)


def freeze_manifest(
    readiness_path: Path,
    output_path: Path,
    *,
    data_root: Path,
    code_commit: str,
    methods: Sequence[str],
    seeds: Sequence[int],
    schedule_seed: int,
    attempt_number: int,
) -> str:
    """Create a new manifest once; an existing target is never replaced."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    if len(code_commit) != 40 or any(c not in "0123456789abcdef" for c in code_commit.lower()):
        raise ValueError("code_commit must be a 40-character hexadecimal commit")
    if int(attempt_number) < 1:
        raise ValueError("attempt_number must be positive")
    methods = tuple(str(item) for item in methods)
    seeds = tuple(int(item) for item in seeds)
    if not methods or not seeds:
        raise ValueError("methods and seeds must not be empty")

    with Path(readiness_path).open(encoding="utf-8", newline="") as handle:
        tasks = list(csv.DictReader(handle))
    if not tasks:
        raise ValueError("readiness table contains no tasks")

    root = Path(data_root)
    rng = random.Random(int(schedule_seed))
    rows = []
    for task in tasks:
        if "manufacturing_gate" in task and not task["manufacturing_gate"].startswith("passed"):
            raise ValueError(f"manufacturing gate is not passed for {task['task_id']}")
        source = root / task["source_csv"]
        actual_sha256 = sha256_file(source)
        if actual_sha256 != task["source_sha256"]:
            raise ValueError(f"source hash mismatch before freeze: {source}")
        load_packing_parts(source)
        for seed in seeds:
            method_order = list(methods)
            rng.shuffle(method_order)
            for rank, method in enumerate(method_order, start=1):
                rows.append(
                    {
                        "manifest_schema": MANIFEST_SCHEMA_VERSION,
                        "attempt_number": int(attempt_number),
                        "block_index": len(rows) + 1,
                        "task_id": task["task_id"],
                        "batch_id": task["batch_id"],
                        "analysis_role": task["analysis_role"],
                        "seed": seed,
                        "method_order": rank,
                        "method": method,
                        "budget_protocol": BUDGET_PROTOCOL,
                        "dimension_precision_policy": DIMENSION_PRECISION_POLICY,
                        "source_csv": task["source_csv"],
                        "source_sha256": actual_sha256,
                        "code_commit": code_commit.lower(),
                        "schedule_seed": int(schedule_seed),
                        "status": "frozen_pending_authorization",
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(output)


def completed_block_indexes(output_root: Path, *, manifest_sha256: str) -> Set[int]:
    """Return only successes cryptographically bound to the active manifest."""

    completed: Set[int] = set()
    for marker in Path(output_root).glob("run_*/success.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("manifest_sha256") != manifest_sha256:
                continue
            block_index = int(payload["block_index"])
            if marker.parent.name != f"run_{block_index:03d}":
                continue
            attempt_dir = str(payload["attempt_dir"])
            if not attempt_dir.startswith("attempt_") or "/" in attempt_dir:
                continue
            result_path = marker.parent / attempt_dir / "result.json"
            if not result_path.is_file():
                continue
            if sha256_file(result_path) != payload.get("result_sha256"):
                continue
            completed.add(block_index)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return completed
