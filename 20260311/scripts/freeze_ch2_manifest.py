#!/usr/bin/env python3
"""Freeze the paired Chapter 2 repair comparison and one Greedy reference per task."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.formal_batch import (
    BUDGET_PROTOCOL,
    DIMENSION_PRECISION_POLICY,
    MANIFEST_SCHEMA_VERSION,
    load_packing_parts,
    sha256_file,
)


DETERMINISTIC = "rh_mcts_prefix_beam"
RANDOM = "rh_mcts_prefix_beam_random_completion"
GREEDY = "baseline_greedy"
SEEDS = tuple(range(1000, 1005))


def build_ch2_rows(
    old_rows: list[dict[str, str]], *, code_commit: str, schedule_seed: int, attempt_number: int
) -> list[dict[str, str | int]]:
    if len(old_rows) != 110:
        raise ValueError(f"old comparison needs 110 rows, got {len(old_rows)}")
    if len(code_commit) != 40 or any(ch not in "0123456789abcdef" for ch in code_commit.lower()):
        raise ValueError("code commit must be a 40-character hexadecimal SHA")
    if attempt_number < 1:
        raise ValueError("attempt number must be positive")
    by_task: dict[str, list[dict[str, str]]] = {}
    for row in old_rows:
        by_task.setdefault(row["task_id"], []).append(row)
    if len(by_task) != 11 or Counter(rows[0]["analysis_role"] for rows in by_task.values()) != {
        "main": 10, "boundary": 1
    }:
        raise ValueError("old comparison must contain 10 main tasks and one boundary task")
    if len({rows[0]["batch_id"] for rows in by_task.values()}) != 4:
        raise ValueError("old comparison must contain four source batches")
    required_old = {(seed, method) for seed in SEEDS for method in (DETERMINISTIC, RANDOM)}
    for task_id, task_rows in by_task.items():
        if len(task_rows) != 10 or {(int(row["seed"]), row["method"]) for row in task_rows} != required_old:
            raise ValueError(f"incomplete or duplicated task/seed/method cells: {task_id}")
        for name in ("batch_id", "analysis_role", "source_csv", "source_sha256"):
            if len({row[name] for row in task_rows}) != 1:
                raise ValueError(f"inconsistent {name}: {task_id}")
        if {row["budget_protocol"] for row in task_rows} != {BUDGET_PROTOCOL}:
            raise ValueError(f"inconsistent old budget protocol: {task_id}")
        if {row["dimension_precision_policy"] for row in task_rows} != {DIMENSION_PRECISION_POLICY}:
            raise ValueError(f"inconsistent old dimension policy: {task_id}")

    rng = random.Random(schedule_seed)
    blocks = [(task_id, seed) for task_id in by_task for seed in SEEDS]
    rng.shuffle(blocks)
    new_rows: list[dict[str, str | int]] = []
    for task_id, seed in blocks:
        source = by_task[task_id][0]
        methods = [DETERMINISTIC, RANDOM]
        if seed == 1000:
            methods.append(GREEDY)
        rng.shuffle(methods)
        for order, method in enumerate(methods, 1):
            new_rows.append(
                {
                    "manifest_schema": MANIFEST_SCHEMA_VERSION,
                    "attempt_number": attempt_number,
                    "block_index": len(new_rows) + 1,
                    "task_id": task_id,
                    "batch_id": source["batch_id"],
                    "analysis_role": source["analysis_role"],
                    "seed": seed,
                    "method_order": order,
                    "method": method,
                    "budget_protocol": BUDGET_PROTOCOL,
                    "dimension_precision_policy": DIMENSION_PRECISION_POLICY,
                    "source_csv": source["source_csv"],
                    "source_sha256": source["source_sha256"],
                    "code_commit": code_commit.lower(),
                    "schedule_seed": schedule_seed,
                    "status": "frozen_pending_authorization",
                }
            )
    if len(new_rows) != 121:
        raise AssertionError("Chapter 2 matrix must contain exactly 121 rows")
    return new_rows


def inspect_sources(rows: list[dict[str, str | int]], data_root: Path) -> dict[str, object]:
    tasks: dict[str, dict[str, object]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in tasks:
            continue
        source = data_root / str(row["source_csv"])
        if sha256_file(source) != row["source_sha256"]:
            raise ValueError(f"source hash mismatch: {source}")
        parts = load_packing_parts(source)
        with source.open(encoding="gb18030", newline="") as handle:
            raw = list(csv.DictReader(handle))
        if len(raw) != len(parts):
            raise ValueError(f"adapted part count differs from source: {task_id}")
        quantities = {item.get("Quantity", "").strip() for item in raw}
        thicknesses = {item.get("Thickness", "").strip() for item in raw}
        materials = {item.get("Material", "").strip() for item in raw}
        if quantities != {"1"} or thicknesses != {"18"} or len(materials) != 1 or "" in materials:
            raise ValueError(f"manufacturing field condition differs from frozen study: {task_id}")
        tasks[task_id] = {
            "batch_id": row["batch_id"],
            "analysis_role": row["analysis_role"],
            "source_csv": row["source_csv"],
            "source_sha256": row["source_sha256"],
            "part_count": len(parts),
            "quantity": "1",
            "thickness_mm": "18",
            "material": next(iter(materials)),
        }
    main_parts = sum(int(task["part_count"]) for task in tasks.values() if task["analysis_role"] == "main")
    if main_parts != 12629:
        raise ValueError(f"main task part total differs from thesis: {main_parts}")
    return {"task_count": len(tasks), "main_part_count": main_parts, "tasks": tasks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-inventory", required=True, type=Path)
    parser.add_argument("--attempt-number", required=True, type=int)
    parser.add_argument("--schedule-seed", type=int, default=20260928)
    args = parser.parse_args()
    for target in (args.manifest, args.source_inventory):
        if target.exists():
            raise FileExistsError(target)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("commit the implementation before freezing the manifest")
    with args.old_manifest.open(encoding="utf-8", newline="") as handle:
        old_rows = list(csv.DictReader(handle))
    rows = build_ch2_rows(
        old_rows, code_commit=commit, schedule_seed=args.schedule_seed, attempt_number=args.attempt_number
    )
    inventory = inspect_sources(rows, args.data_root)
    inventory.update(
        {
            "old_manifest_sha256": sha256_file(args.old_manifest),
            "code_commit": commit,
            "schedule_seed": args.schedule_seed,
            "run_count": len(rows),
        }
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.source_inventory.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with args.source_inventory.open("x", encoding="utf-8") as handle:
        json.dump(inventory, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"rows": len(rows), "manifest_sha256": sha256_file(args.manifest), "code_commit": commit}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
