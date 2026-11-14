from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.formal_batch import (
    BUDGET_PROTOCOL,
    DIMENSION_PRECISION_POLICY,
    completed_block_indexes,
    freeze_manifest,
    load_packing_parts,
)
from inrp.handoff import PackingHandoff


def _load_formal_runner():
    path = ROOT / "scripts" / "run_formal_batch_v2.py"
    spec = importlib.util.spec_from_file_location("run_formal_batch_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="gb18030", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Upi", "Fleng", "Fwidth"))
        writer.writeheader()
        writer.writerows(rows)


def test_loader_preserves_exact_upi_and_source_tenth_millimetres(tmp_path: Path) -> None:
    source = tmp_path / "parts.csv"
    _write_csv(
        source,
        [
            {"Upi": "1319926641097", "Fleng": "449.3", "Fwidth": "821.5"},
            {"Upi": "1319926641098", "Fleng": "100", "Fwidth": "50"},
        ],
    )

    parts = load_packing_parts(source)

    assert DIMENSION_PRECISION_POLICY == "source_decimal_max_0.1_mm_no_rounding"
    assert [part.source_part_id for part in parts] == ["1319926641097", "1319926641098"]
    assert [(part.nominal_width_mm, part.nominal_height_mm) for part in parts] == [
        (449.3, 821.5),
        (100.0, 50.0),
    ]


def test_loader_rejects_case_substitution_duplicate_upi_and_extra_precision(tmp_path: Path) -> None:
    wrong_header = tmp_path / "wrong.csv"
    wrong_header.write_text("UPI,Fleng,Fwidth\n1,10,20\n", encoding="gb18030")
    with pytest.raises(ValueError, match="exact columns"):
        load_packing_parts(wrong_header)

    duplicate = tmp_path / "duplicate.csv"
    _write_csv(
        duplicate,
        [
            {"Upi": "same", "Fleng": "10", "Fwidth": "20"},
            {"Upi": "same", "Fleng": "11", "Fwidth": "21"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate Upi"):
        load_packing_parts(duplicate)

    too_precise = tmp_path / "too-precise.csv"
    _write_csv(too_precise, [{"Upi": "one", "Fleng": "10.25", "Fwidth": "20"}])
    with pytest.raises(ValueError, match="0.1 mm"):
        load_packing_parts(too_precise)


def test_manifest_freeze_is_immutable_and_binds_source_and_code(tmp_path: Path) -> None:
    source = tmp_path / "parts.csv"
    _write_csv(source, [{"Upi": "one", "Fleng": "10.5", "Fwidth": "20"}])
    readiness = tmp_path / "readiness.csv"
    with readiness.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_id", "batch_id", "analysis_role", "source_csv", "source_sha256"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "case",
                "batch_id": "batch",
                "analysis_role": "main",
                "source_csv": source.name,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    output = tmp_path / "manifest.csv"

    digest = freeze_manifest(
        readiness,
        output,
        data_root=tmp_path,
        code_commit="a" * 40,
        methods=("method-a", "method-b"),
        seeds=(1000,),
        schedule_seed=7,
        attempt_number=3,
    )

    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert len(rows) == 2
    assert {row["attempt_number"] for row in rows} == {"3"}
    assert {row["budget_protocol"] for row in rows} == {
        "v4_420s_total_60s_close_425s_watchdog"
    }
    assert BUDGET_PROTOCOL == "v4_420s_total_60s_close_425s_watchdog"
    assert {row["dimension_precision_policy"] for row in rows} == {DIMENSION_PRECISION_POLICY}
    assert {row["code_commit"] for row in rows} == {"a" * 40}
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        freeze_manifest(
            readiness,
            output,
            data_root=tmp_path,
            code_commit="a" * 40,
            methods=("method-a",),
            seeds=(1000,),
            schedule_seed=7,
            attempt_number=3,
        )


def test_runner_derives_third_attempt_and_requires_matching_authorization(tmp_path: Path) -> None:
    runner = _load_formal_runner()
    rows = [
        {"attempt_number": "3", "block_index": "1"},
        {"attempt_number": "3", "block_index": "2"},
    ]
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "status": "authorized",
                "manifest_sha256": "manifest-digest",
                "attempt_number": 3,
            }
        ),
        encoding="utf-8",
    )

    assert runner._manifest_attempt_number(rows) == 3
    assert runner._validate_authorization(authorization, "manifest-digest", 3)["attempt_number"] == 3

    with pytest.raises(RuntimeError, match="attempt 4"):
        runner._validate_authorization(authorization, "manifest-digest", 4)


def test_formal_worker_reserves_sixty_seconds_for_lifecycle_close(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_formal_runner()
    source = tmp_path / "parts.csv"
    _write_csv(source, [{"Upi": "one", "Fleng": "10", "Fwidth": "20"}])
    observed = []

    def fake_solve(request, budget):
        observed.append((budget.total_s, budget.closing_reserve_s, budget.search_deadline - budget.started_at))
        return PackingHandoff(
            schema_version="test",
            feasible=True,
            panel_count=0,
            placements=tuple(),
            sequence=tuple(),
            score_global=tuple(),
            runtime_s=0.0,
            provenance={"case_id": request.case_id},
        )

    monkeypatch.setattr(runner, "solve_packing_handoff_with_budget", fake_solve)
    output = tmp_path / "result.json"
    runner._worker(
        {
            "source_csv": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "task_id": "case",
            "seed": "1000",
            "method": "rh_mcts_prefix_beam",
        },
        data_root=tmp_path,
        output=output,
        manifest_sha256="manifest-digest",
        attempt_number=5,
    )

    assert observed == [(420.0, 60.0, 360.0)]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["budget"]["closing_reserve_s"] == 60.0


def test_resume_accepts_only_success_bound_to_current_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("block_index\n1\n2\n", encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    run1 = tmp_path / "run_001"
    result1 = run1 / "attempt_001" / "result.json"
    result1.parent.mkdir(parents=True)
    result1.write_text('{"ok":true}', encoding="utf-8")
    (run1 / "success.json").write_text(
        json.dumps(
            {
                "block_index": 1,
                "manifest_sha256": digest,
                "attempt_dir": "attempt_001",
                "result_sha256": hashlib.sha256(result1.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    run2 = tmp_path / "run_002"
    result2 = run2 / "attempt_001" / "result.json"
    result2.parent.mkdir(parents=True)
    result2.write_text('{"ok":true}', encoding="utf-8")
    (run2 / "success.json").write_text(
        json.dumps(
            {
                "block_index": 2,
                "manifest_sha256": "stale",
                "attempt_dir": "attempt_001",
                "result_sha256": hashlib.sha256(result2.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    assert completed_block_indexes(tmp_path, manifest_sha256=digest) == {1}

    result1.write_text('{"tampered":true}', encoding="utf-8")
    assert completed_block_indexes(tmp_path, manifest_sha256=digest) == set()


def test_caffeinate_command_wraps_parent_and_pmset_assertion_targets_launcher() -> None:
    runner = _load_formal_runner()

    command = runner._caffeinate_command(("runner.py", "--manifest", "manifest.csv"))

    assert command[:2] == ("/usr/bin/caffeinate", "-dimsu")
    assert command[-1] == "--caffeinated"
    pmset = """Listed by owning process:
   pid 222(caffeinate): PreventUserIdleSystemSleep named: \"caffeinate command-line tool\"
        Details: caffeinate asserting on behalf of 'python' (pid 123)
"""
    assert runner._pmset_asserts_for_pid(pmset, 123)
    assert not runner._pmset_asserts_for_pid(pmset, 124)


def test_worker_result_validation_rejects_unusable_and_binding_errors(tmp_path: Path) -> None:
    runner = _load_formal_runner()
    row = {
        "attempt_number": "6",
        "block_index": "1",
        "task_id": "case",
        "seed": "1000",
        "method": "rh_mcts_prefix_beam",
        "source_sha256": "source-digest",
        "budget_protocol": BUDGET_PROTOCOL,
    }
    payload = {
        "status": "usable",
        "formal_attempt": 6,
        "manifest_sha256": "manifest-digest",
        "manifest_row": row,
        "source_sha256": "source-digest",
        "request": {
            "case_id": "case",
            "seed": 1000,
            "solver_variant": "rh_mcts_prefix_beam",
            "time_limit_s": 420.0,
        },
        "handoff": {"feasible": True, "panel_count": 1, "placements": [{"part_id": "p"}]},
        "budget": {"total_s": 420.0, "closing_reserve_s": 60.0},
    }

    runner._validate_worker_result(payload, row, "manifest-digest", 6)
    payload["status"] = "failed"
    with pytest.raises(RuntimeError, match="not usable"):
        runner._validate_worker_result(payload, row, "manifest-digest", 6)
    payload["status"] = "usable"
    payload["request"]["seed"] = 1001
    with pytest.raises(RuntimeError, match="seed"):
        runner._validate_worker_result(payload, row, "manifest-digest", 6)


def test_parent_stops_before_next_row_after_watchdog_failure(tmp_path: Path, monkeypatch) -> None:
    runner = _load_formal_runner()
    manifest = tmp_path / "manifest.csv"
    rows = [
        {
            "attempt_number": "6",
            "block_index": str(index),
            "code_commit": "a" * 40,
        }
        for index in range(1, 111)
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps({"status": "authorized", "attempt_number": 6, "manifest_sha256": digest}),
        encoding="utf-8",
    )
    out_root = tmp_path / "results"
    calls = []

    monkeypatch.setattr(runner, "_verify_code_commit", lambda expected: None)

    def fake_power(root: Path):
        runner._atomic_json(root / "power_assertion.json", {"status": "asserted"})

    def fake_watchdog(command, **kwargs):
        calls.append(command)
        kwargs["on_started"](999)
        return SimpleNamespace(
            status="watchdog_terminated",
            to_dict=lambda: {"status": "watchdog_terminated", "hard_timeout_s": 425.0},
        )

    monkeypatch.setattr(runner, "_capture_power_assertion", fake_power)
    monkeypatch.setattr(runner, "run_command_with_watchdog", fake_watchdog)
    args = SimpleNamespace(
        manifest=manifest,
        authorization=authorization,
        out_root=out_root,
        data_root=tmp_path,
        startup_acceptance=False,
    )

    with pytest.raises(RuntimeError, match="watchdog_terminated"):
        runner._run_parent(args)

    assert len(calls) == 1
    assert (out_root / "run_001").is_dir()
    assert not (out_root / "run_002").exists()
    state = json.loads((out_root / "batch_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert state["completed_count"] == 0
