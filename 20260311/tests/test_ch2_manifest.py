from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_ch2_manifest import DETERMINISTIC, GREEDY, RANDOM, build_ch2_rows
from inrp.formal_batch import BUDGET_PROTOCOL, DIMENSION_PRECISION_POLICY


def _old_rows() -> list[dict[str, str]]:
    # Exercise the frozen matrix contract without requiring private orders.
    return [
        {
            "task_id": f"synthetic-{task}",
            "batch_id": f"batch-{task % 4}",
            "analysis_role": "boundary" if task == 10 else "main",
            "source_csv": f"synthetic-{task}.csv",
            "source_sha256": f"{task:064x}",
            "seed": str(seed),
            "method": method,
            "budget_protocol": BUDGET_PROTOCOL,
            "dimension_precision_policy": DIMENSION_PRECISION_POLICY,
        }
        for task in range(11)
        for seed in range(1000, 1005)
        for method in (DETERMINISTIC, RANDOM)
    ]


def test_ch2_schedule_has_all_pairs_and_one_greedy_per_task():
    rows = build_ch2_rows(_old_rows(), code_commit="a" * 40, schedule_seed=20260928, attempt_number=7)
    counts = Counter(row["method"] for row in rows)
    assert counts == {DETERMINISTIC: 55, RANDOM: 55, GREEDY: 11}
    assert len({(row["task_id"], row["seed"], row["method"]) for row in rows}) == 121
    assert [row["block_index"] for row in rows] == list(range(1, 122))
    assert len({row["task_id"] for row in rows if row["analysis_role"] == "main"}) == 10
    blocks = defaultdict(list)
    for row in rows:
        blocks[(row["task_id"], row["seed"])].append(row)
    assert len(blocks) == 55
    for (_, seed), block in blocks.items():
        assert {row["method"] for row in block} == (
            {DETERMINISTIC, RANDOM, GREEDY} if seed == 1000 else {DETERMINISTIC, RANDOM}
        )
        assert [row["block_index"] for row in block] == list(
            range(block[0]["block_index"], block[0]["block_index"] + len(block))
        )
    assert rows == build_ch2_rows(_old_rows(), code_commit="a" * 40, schedule_seed=20260928, attempt_number=7)


def test_ch2_schedule_rejects_missing_old_control():
    rows = _old_rows()
    with pytest.raises(ValueError, match="110 rows"):
        build_ch2_rows(rows[:-1], code_commit="a" * 40, schedule_seed=20260928, attempt_number=7)
