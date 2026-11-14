from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CUT_ROOT = Path("/Users/binggo/Desktop/graduate/cut_eval")
JOINT_ROOT = Path("/Users/binggo/Desktop/graduate/joint_nesting_cuteval_hde_validation")
for source_root in (ROOT / "src", CUT_ROOT / "src", JOINT_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import thesis_input.minibatch as minibatch
from inrp.handoff import PackingHandoff, PackingPlacement
from thesis_input.adapters import to_nominal_layouts, to_packing_request
from thesis_input.contract import problem_hash, source_manifest_hash, validate_bundle
from thesis_input.minibatch import (
    SelectionRecord,
    run_minibatch,
    semantic_candidate_hash,
    semantic_layout_hash,
    select_delayed,
    select_independent,
    verify_source_identity,
)


FIXTURE = ROOT / "tests" / "fixtures" / "shared_six_parts.json"
pytestmark = pytest.mark.skipif(
    not CUT_ROOT.is_dir() or not JOINT_ROOT.is_dir(),
    reason="optional cross-chapter source repositories are unavailable",
)


@pytest.fixture
def pinned_sources() -> None:
    try:
        verify_source_identity(CUT_ROOT, JOINT_ROOT)
    except (AssertionError, FileNotFoundError) as error:
        pytest.skip(f"optional pinned integration sources unavailable: {error}")


def _record(
    name: str,
    *,
    board_count: int = 1,
    tail: float = 0.2,
    remnant: float = 0.5,
    planning: float | None = 10.0,
    feasible: bool = True,
) -> SelectionRecord:
    return SelectionRecord(
        solution_hash=name,
        candidate_index=0,
        board_count=board_count,
        tail_fraction=tail,
        remnant_ratio=remnant,
        planning_value_seconds=planning,
        feasible=feasible,
    )


def _fixed_handoff(bundle: dict, *, panel_id: int = 7) -> PackingHandoff:
    request = to_packing_request(bundle, seed=17)
    rows = (
        ("p01__copy_001", 8, 8, 120, 60, False),
        ("p02__copy_001", 134, 8, 120, 60, False),
        ("p03__copy_001", 8, 74, 80, 60, False),
        ("p04__copy_001", 94, 74, 80, 60, False),
        ("p05__copy_001", 180, 74, 40, 40, False),
        ("p06__copy_001", 226, 74, 40, 40, False),
    )
    placements = tuple(
        PackingPlacement(
            panel_id,
            part_id,
            part_id.split("__", 1)[0],
            x,
            y,
            width,
            height,
            rotated,
        )
        for part_id, x, y, width, height, rotated in rows
    )
    return PackingHandoff(
        "inrp-packing-handoff-v1",
        True,
        1,
        placements,
        tuple(item.part_id for item in placements),
        (1.0,),
        0.01,
        {
            "request": asdict(request),
            "thesis_input_problem_hash": problem_hash(bundle),
        },
    )


def test_source_identity_rejects_wrong_root(tmp_path: Path) -> None:
    with pytest.raises((AssertionError, FileNotFoundError), match="repository|commit|Git"):
        verify_source_identity(tmp_path, JOINT_ROOT)


def test_source_identity_rejects_dirty_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_git(root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return (
                minibatch.EXPECTED_CUT_COMMIT
                if root == CUT_ROOT
                else minibatch.EXPECTED_JOINT_COMMIT
            )
        if args == ("status", "--porcelain"):
            return " M changed.py" if root == CUT_ROOT else ""
        if args == ("rev-parse", "HEAD^{tree}"):
            return "tree"
        raise AssertionError(args)

    monkeypatch.setattr(minibatch, "_git", fake_git)
    with pytest.raises(AssertionError, match="not clean"):
        verify_source_identity(CUT_ROOT, JOINT_ROOT)


def test_module_identity_rejects_wrong_submodule_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        minibatch.importlib,
        "import_module",
        lambda _: SimpleNamespace(__file__=str(tmp_path / "wrong.py")),
    )
    with pytest.raises(AssertionError, match="unexpected path"):
        minibatch._module_identity("cut_eval.handoff", CUT_ROOT)


def test_output_refuses_existing_source_and_data_locations(tmp_path: Path, pinned_sources) -> None:
    existing = tmp_path / "already-there"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_minibatch(FIXTURE, existing, cut_root=CUT_ROOT, joint_root=JOINT_ROOT)

    with pytest.raises(ValueError, match="source or data"):
        run_minibatch(
            FIXTURE,
            ROOT / "data" / "minibatch-output",
            cut_root=CUT_ROOT,
            joint_root=JOINT_ROOT,
        )

    for protected in (
        CUT_ROOT / "src" / "minibatch-output",
        CUT_ROOT / "data" / "minibatch-output",
        JOINT_ROOT / "src" / "minibatch-output",
        JOINT_ROOT / "data" / "minibatch-output",
        JOINT_ROOT / "evidence" / "minibatch-output",
    ):
        with pytest.raises(ValueError, match="source or data|old evidence"):
            minibatch._prepare_output(
                protected,
                project_root=ROOT,
                cut_root=CUT_ROOT,
                joint_root=JOINT_ROOT,
            )


def test_selection_handles_singleton_ties_cost_and_invalid_candidates() -> None:
    singleton = (_record("only"),)
    assert select_delayed(singleton).selected_hashes == ("only",)
    assert select_independent(singleton).selected_hashes == ("only",)

    tied = (
        _record("left", tail=0.1, remnant=0.4, planning=7.0),
        _record("right", tail=0.2, remnant=0.6, planning=7.0),
    )
    assert select_delayed(tied).selected_hashes == ("left", "right")
    assert select_independent(tied).selected_hashes == ("left", "right")

    varied = (
        _record("slow", tail=0.1, remnant=0.4, planning=9.0),
        _record("fast", tail=0.2, remnant=0.6, planning=4.0),
        _record("invalid", tail=0.0, remnant=1.0, planning=None, feasible=False),
        _record("more-boards", board_count=2, tail=0.0, remnant=1.0, planning=1.0),
    )
    delayed = select_delayed(varied)
    independent = select_independent(varied)
    assert set(delayed.material_front_hashes) == {"slow", "fast"}
    assert delayed.selected_hashes == ("fast",)
    assert independent.selected_hashes == delayed.selected_hashes
    assert "invalid" not in independent.valid_hashes


def test_delayed_front_uses_joint_helper_not_independent_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        _record("joint-only", tail=0.1, remnant=0.4),
        _record("would-win-locally", tail=0.2, remnant=0.7),
    )
    import joint_nesting_cuteval.hierarchical_delayed_evaluation as delayed_module

    observed = []

    def divergent_joint_front(candidates, tolerance):
        observed.append((tuple(candidates), tolerance))
        return (candidates[0],)

    monkeypatch.setattr(delayed_module, "material_front", divergent_joint_front)
    delayed = minibatch._joint_material_front(records)
    independent = minibatch._independent_material_front(records)

    assert observed
    assert tuple(record.solution_hash for record in delayed) == ("joint-only",)
    assert {record.solution_hash for record in independent} == {
        "joint-only",
        "would-win-locally",
    }


def test_semantic_candidate_and_evaluation_identity_include_rotation_rules() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    changed_raw = deepcopy(raw)
    changed_raw["parts"][2]["allowed_rotations_deg"]["value"] = [0]
    base = validate_bundle(raw)
    changed = validate_bundle(changed_raw)

    base_layouts = to_nominal_layouts(base, _fixed_handoff(base))
    changed_layouts = to_nominal_layouts(changed, _fixed_handoff(changed))
    base_candidate, _, _, _ = minibatch._candidate_from_layouts(
        base, base_layouts, seed=17
    )
    changed_candidate, _, _, _ = minibatch._candidate_from_layouts(
        changed, changed_layouts, seed=17
    )

    assert base_candidate.solution_hash == changed_candidate.solution_hash
    assert base_layouts[0].parts == changed_layouts[0].parts
    assert problem_hash(base) != problem_hash(changed)
    assert base_layouts[0].source_hash == semantic_layout_hash(
        problem_hash(base), base_candidate.panels[0].geometry_hash
    )
    assert changed_layouts[0].source_hash == semantic_layout_hash(
        problem_hash(changed), changed_candidate.panels[0].geometry_hash
    )
    assert semantic_candidate_hash(
        problem_hash(base), base_candidate.solution_hash
    ) != semantic_candidate_hash(problem_hash(changed), changed_candidate.solution_hash)
    assert semantic_layout_hash(
        problem_hash(base), base_candidate.panels[0].geometry_hash
    ) != semantic_layout_hash(
        problem_hash(changed), changed_candidate.panels[0].geometry_hash
    )

    base_evaluation_layout = minibatch._evaluation_layout(
        base_candidate,
        base_candidate.panels[0],
        problem_hash_value=problem_hash(base),
        source_method="test:fixed-geometry",
    )
    changed_evaluation_layout = minibatch._evaluation_layout(
        changed_candidate,
        changed_candidate.panels[0],
        problem_hash_value=problem_hash(changed),
        source_method="test:fixed-geometry",
    )
    assert base_evaluation_layout.parts == changed_evaluation_layout.parts
    assert base_evaluation_layout.source_hash != changed_evaluation_layout.source_hash
    assert base_evaluation_layout.layout_id != changed_evaluation_layout.layout_id


def test_semantic_candidate_dedup_rejects_differing_geometry_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def candidate(x_mm: float):
        placement = SimpleNamespace(
            part_id="p01",
            source_part_id="p01",
            x_mm=x_mm,
            y_mm=8.0,
            nominal_width_mm=120.0,
            nominal_height_mm=60.0,
            rotated=False,
        )
        panel = SimpleNamespace(placements=(placement,))
        return SimpleNamespace(solution_hash="joint-geometry", panels=(panel,))

    monkeypatch.setattr(minibatch, "semantic_candidate_hash", lambda *_: "collision")
    candidates_by_hash = {}
    geometry_by_hash = {}
    minibatch._register_candidate(
        candidates_by_hash,
        geometry_by_hash,
        candidate(8.0),
        problem_hash_value="problem",
    )

    with pytest.raises(AssertionError, match="collision"):
        minibatch._register_candidate(
            candidates_by_hash,
            geometry_by_hash,
            candidate(9.0),
            problem_hash_value="problem",
        )


def test_actual_six_piece_integration(tmp_path: Path, pinned_sources) -> None:
    result = run_minibatch(
        FIXTURE,
        tmp_path / "six-piece-output",
        cut_root=CUT_ROOT,
        joint_root=JOINT_ROOT,
    )

    assert result["status"] == "completed"
    assert result["seeds"] == [1000, 1001]
    assert result["selection"]["identity_sets_match"] is True
    assert result["source_manifest_hash"] == source_manifest_hash(
        json.loads(FIXTURE.read_text(encoding="utf-8"))
    )
    output = tmp_path / "six-piece-output"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    seed_runs = json.loads((output / "seed_runs.json").read_text(encoding="utf-8"))
    source = json.loads((output / "source_identity.json").read_text(encoding="utf-8"))
    audits = json.loads((output / "audit_records.json").read_text(encoding="utf-8"))
    delayed = json.loads(
        (output / "selection_public_delayed.json").read_text(encoding="utf-8")
    )
    independent = json.loads(
        (output / "selection_public_independent.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((output / "candidate_metrics.json").read_text(encoding="utf-8"))

    assert [run["seed"] for run in seed_runs] == [1000, 1001]
    assert all(run["request"]["seed"] == run["seed"] for run in seed_runs)
    assert all(
        run["handoff"]["provenance"]["thesis_input_problem_hash"]
        == result["thesis_input_problem_hash"]
        for run in seed_runs
    )
    assert all(
        run["source_manifest_hash"] == result["source_manifest_hash"]
        and run["problem_hash"] == result["thesis_input_problem_hash"]
        and run["joint_geometry_hash"]
        and run["semantic_candidate_hash"]
        and run["semantic_layout_hashes"]
        for run in seed_runs
    )
    assert all(
        set(run["semantic_layout_hashes"])
        == {layout["source_hash"] for layout in run["nominal_layouts"]}
        for run in seed_runs
    )
    assert source["source_manifest_hash"] == result["source_manifest_hash"]
    assert source["problem_hash"] == result["thesis_input_problem_hash"]
    assert source["candidate_identities"]
    assert all(
        identity["semantic_candidate_hash"] in metrics
        and identity["joint_geometry_hash"]
        and all(
            layout["semantic_layout_hash"]
            and layout["joint_panel_geometry_hash"]
            for layout in identity["layouts"]
        )
        for identity in source["candidate_identities"]
    )
    assert source["imports"]["inrp.handoff"]["file"].startswith(str(ROOT / "src"))
    assert all(
        "nc_text" in panel
        and panel["source_identity"]["semantic_layout_hash"]
        and panel["source_identity"]["source_method"]
        == "thesis_input:minibatch_full_audit"
        for panels in audits.values()
        for panel in panels
    )
    assert all(
        "nc_text" not in response
        for response in delayed["public_handoff_responses"].values()
    )
    assert all(
        "audit" not in response
        for response in independent["public_handoff_responses"].values()
    )
    assert all(
        value["candidate_original_frozen_hash"]
        == value["candidate_reconstructed_frozen_hash"]
        for value in metrics.values()
    )
    assert all(
        identity == value["semantic_candidate_hash"]
        and value["joint_geometry_hash"] == value["candidate"]["solution_hash"]
        and value["problem_hash"] == result["thesis_input_problem_hash"]
        and value["source_manifest_hash"] == result["source_manifest_hash"]
        for identity, value in metrics.items()
    )
    for public in (delayed, independent):
        for identity, response in public["public_handoff_responses"].items():
            assert identity == response["source_identity"]["semantic_candidate_hash"]
            assert response["source_identity"]["problem_hash"] == result[
                "thesis_input_problem_hash"
            ]
            assert all(
                panel["source_identity"]["semantic_layout_hash"]
                for panel in response["panel_responses"]
            )
    for identity, audit_panels in audits.items():
        audit_layout_hashes = {
            panel["source_identity"]["semantic_layout_hash"]
            for panel in audit_panels
        }
        public_layout_hashes = {
            panel["source_identity"]["semantic_layout_hash"]
            for panel in independent["public_handoff_responses"][identity][
                "panel_responses"
            ]
        }
        assert audit_layout_hashes == public_layout_hashes == set(
            metrics[identity]["semantic_layout_hashes"]
        )
    for name, digest in manifest.items():
        assert digest == minibatch.hashlib.sha256((output / name).read_bytes()).hexdigest()


def test_failure_keeps_progressive_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pinned_sources
) -> None:
    def infeasible_public(*args, **kwargs):
        return {"feasible": False, "planning_value_seconds": None, "panel_responses": []}

    monkeypatch.setattr(minibatch, "_public_candidate_evaluation", infeasible_public)
    output = tmp_path / "failed-six-piece-output"
    with pytest.raises(RuntimeError, match="no feasible candidates"):
        run_minibatch(FIXTURE, output, cut_root=CUT_ROOT, joint_root=JOINT_ROOT)

    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    for name in (
        "normalized_input.json",
        "source_identity.json",
        "seed_runs.json",
        "audit_records.json",
        "candidate_metrics.json",
        "selection_public_delayed.json",
        "selection_public_independent.json",
    ):
        assert (output / name).is_file()
    assert "failure.json" in manifest
