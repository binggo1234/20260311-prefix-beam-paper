"""Six-piece synthetic integration runner for the shared thesis input.

The runner deliberately keeps CutEval's public handoff separate from the full
CutEvaluator audit: selection records contain only public handoff fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from .adapters import semantic_layout_hash
from .contract import load_bundle, problem_hash, source_manifest_hash


RUNNER_VERSION = "shared-input-six-piece-minibatch-v1"
EXPECTED_CUT_COMMIT = "d816dbd8fb3718625b0c2f0eb4bc85cb97a2eb30"
EXPECTED_JOINT_COMMIT = "d74b222285308ed3d43ffbe23b61dd35bdd96ef2"
SEEDS = (1000, 1001)
TOLERANCE = 1e-9


@dataclass(frozen=True)
class SelectionRecord:
    """Selection uses a semantic composite in ``solution_hash``.

    ``joint_geometry_hash`` is retained only as subordinate geometry
    provenance and never serves as a deduplication or selection key here.
    """

    solution_hash: str
    candidate_index: int
    board_count: int
    tail_fraction: float
    remnant_ratio: float
    planning_value_seconds: float | None
    feasible: bool
    joint_geometry_hash: str = ""


@dataclass(frozen=True)
class SelectionResult:
    valid_hashes: tuple[str, ...]
    material_front_hashes: tuple[str, ...]
    selected_hashes: tuple[str, ...]


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _module_identity(module_name: str, root: Path) -> dict[str, str]:
    module = importlib.import_module(module_name)
    module_file = Path(str(module.__file__)).resolve()
    expected_source = (root / "src").resolve()
    if not _path_is_within(module_file, expected_source):
        raise AssertionError(
            f"{module_name} imported from unexpected path: {module_file}"
        )
    return {"module": module_name, "file": str(module_file)}


def verify_source_identity(cut_root: Path, joint_root: Path) -> dict[str, Any]:
    """Verify immutable source revisions, without altering already loaded modules."""

    cut_root = Path(cut_root).resolve()
    joint_root = Path(joint_root).resolve()
    try:
        cut_commit = _git(cut_root, "rev-parse", "HEAD")
        joint_commit = _git(joint_root, "rev-parse", "HEAD")
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise FileNotFoundError("pinned source Git repository is unavailable") from error
    if cut_commit != EXPECTED_CUT_COMMIT:
        raise AssertionError(f"CutEval commit mismatch: {cut_commit}")
    if joint_commit != EXPECTED_JOINT_COMMIT:
        raise AssertionError(f"joint source commit mismatch: {joint_commit}")
    cut_status = _git(cut_root, "status", "--porcelain")
    joint_status = _git(joint_root, "status", "--porcelain")
    if cut_status:
        raise AssertionError("CutEval repository is not clean")
    if joint_status:
        raise AssertionError("joint repository is not clean")
    return {
        "cut_root": str(cut_root),
        "cut_commit": cut_commit,
        "cut_tree": _git(cut_root, "rev-parse", "HEAD^{tree}"),
        "cut_status": cut_status,
        "joint_root": str(joint_root),
        "joint_commit": joint_commit,
        "joint_tree": _git(joint_root, "rev-parse", "HEAD^{tree}"),
        "joint_status": joint_status,
    }


def _verify_loaded_modules(
    project_root: Path, cut_root: Path, joint_root: Path
) -> dict[str, dict[str, str]]:
    """Check caller-pinned imports; never replace a module from another checkout."""

    return {
        "cut_eval": _module_identity("cut_eval", Path(cut_root)),
        "cut_eval.handoff": _module_identity("cut_eval.handoff", Path(cut_root)),
        "cut_eval.evaluator": _module_identity("cut_eval.evaluator", Path(cut_root)),
        "joint_nesting_cuteval": _module_identity(
            "joint_nesting_cuteval", Path(joint_root)
        ),
        "joint_nesting_cuteval.chapter1_transfer_gate": _module_identity(
            "joint_nesting_cuteval.chapter1_transfer_gate", Path(joint_root)
        ),
        "joint_nesting_cuteval.packing": _module_identity(
            "joint_nesting_cuteval.packing", Path(joint_root)
        ),
        "joint_nesting_cuteval.case_freeze": _module_identity(
            "joint_nesting_cuteval.case_freeze", Path(joint_root)
        ),
        "joint_nesting_cuteval.tail_metrics": _module_identity(
            "joint_nesting_cuteval.tail_metrics", Path(joint_root)
        ),
        "joint_nesting_cuteval.final_validation": _module_identity(
            "joint_nesting_cuteval.final_validation", Path(joint_root)
        ),
        "joint_nesting_cuteval.upstream_contract": _module_identity(
            "joint_nesting_cuteval.upstream_contract", Path(joint_root)
        ),
        "joint_nesting_cuteval.hierarchical_delayed_evaluation": _module_identity(
            "joint_nesting_cuteval.hierarchical_delayed_evaluation", Path(joint_root)
        ),
        "joint_nesting_cuteval.hierarchical_process_selection": _module_identity(
            "joint_nesting_cuteval.hierarchical_process_selection", Path(joint_root)
        ),
        "inrp.handoff": _module_identity("inrp.handoff", Path(project_root)),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def semantic_candidate_hash(
    problem_hash_value: str, joint_geometry_hash: str
) -> str:
    """Bind the joint package's canonical geometry hash to problem semantics."""

    return _canonical_hash(
        {
            "problem_hash": problem_hash_value,
            "joint_geometry_hash": joint_geometry_hash,
        }
    )


def _candidate_geometry(candidate: Any) -> list[list[dict[str, Any]]]:
    """Materialize exact canonical geometry for hash-collision rejection."""

    panels = []
    for panel in candidate.panels:
        placements = [
            {
                "part_id": placement.part_id,
                "source_part_id": placement.source_part_id,
                "x_mm": placement.x_mm,
                "y_mm": placement.y_mm,
                "nominal_width_mm": placement.nominal_width_mm,
                "nominal_height_mm": placement.nominal_height_mm,
                "rotated": placement.rotated,
            }
            for placement in sorted(panel.placements, key=lambda item: item.part_id)
        ]
        panels.append(placements)
    return sorted(panels, key=_canonical_hash)


def _register_candidate(
    candidates_by_hash: dict[str, Any],
    geometry_by_hash: dict[str, list[list[dict[str, Any]]]],
    candidate: Any,
    *,
    problem_hash_value: str,
) -> str:
    identity = semantic_candidate_hash(
        problem_hash_value, candidate.solution_hash
    )
    geometry = _candidate_geometry(candidate)
    if identity in candidates_by_hash:
        if geometry_by_hash[identity] != geometry:
            raise AssertionError("semantic candidate hash collision")
        if candidates_by_hash[identity].solution_hash != candidate.solution_hash:
            raise AssertionError("semantic candidate hash collision")
    else:
        candidates_by_hash[identity] = candidate
        geometry_by_hash[identity] = geometry
    return identity


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _prepare_output(
    output_dir: Path, *, project_root: Path, cut_root: Path, joint_root: Path
) -> Path:
    output = Path(output_dir).resolve()
    prohibited = tuple(
        location
        for root in (Path(project_root), Path(cut_root), Path(joint_root))
        for location in (root / "src", root / "data", root / "evidence")
    )
    if any(_path_is_within(output, root) for root in prohibited):
        raise ValueError("minibatch output cannot be in a source or data location")
    if "evidence" in {part.lower() for part in output.parts}:
        raise ValueError("minibatch output cannot overwrite an old evidence location")
    if output.exists():
        raise FileExistsError(f"minibatch output already exists: {output}")
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        raise FileExistsError(f"minibatch staging output already exists: {staging}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    return staging


def _valid_records(records: Iterable[SelectionRecord]) -> tuple[SelectionRecord, ...]:
    return tuple(
        record
        for record in records
        if record.feasible
        and record.planning_value_seconds is not None
        and record.planning_value_seconds > 0.0
    )


def _deduplicate_records(records: Iterable[SelectionRecord]) -> tuple[SelectionRecord, ...]:
    unique: dict[str, SelectionRecord] = {}
    for record in records:
        previous = unique.setdefault(record.solution_hash, record)
        if previous != record:
            raise AssertionError("selection record hash collision")
    return tuple(unique.values())


def _material_dominates(first: SelectionRecord, second: SelectionRecord) -> bool:
    return (
        first.board_count == second.board_count
        and first.tail_fraction <= second.tail_fraction + TOLERANCE
        and first.remnant_ratio >= second.remnant_ratio - TOLERANCE
        and (
            first.tail_fraction < second.tail_fraction - TOLERANCE
            or first.remnant_ratio > second.remnant_ratio + TOLERANCE
        )
    )


def _independent_material_front(records: Sequence[SelectionRecord]) -> tuple[SelectionRecord, ...]:
    """Standalone material-front reconstruction for the exhaustive reference."""

    if not records:
        return ()
    minimum_boards = min(record.board_count for record in records)
    stratum = tuple(record for record in records if record.board_count == minimum_boards)
    return tuple(
        sorted(
            (
                record
                for record in stratum
                if not any(
                    other.solution_hash != record.solution_hash
                    and _material_dominates(other, record)
                    for other in stratum
                )
            ),
            key=lambda record: (
                record.tail_fraction,
                -record.remnant_ratio,
                record.solution_hash,
            ),
        )
    )


def _joint_material_front(records: Sequence[SelectionRecord]) -> tuple[SelectionRecord, ...]:
    """Use joint H0 material screening before delayed public evaluation."""

    if not records:
        return ()
    from joint_nesting_cuteval.hierarchical_delayed_evaluation import (
        H0Candidate,
        material_front,
    )

    by_hash = {record.solution_hash: record for record in records}
    h0 = tuple(
        H0Candidate(
            order_id="shared-six-piece",
            candidate_index=record.candidate_index,
            solution_hash=record.solution_hash,
            board_count=record.board_count,
            tail_fraction=record.tail_fraction,
            remnant_ratio=record.remnant_ratio,
            planning_value_seconds=(
                float(record.planning_value_seconds)
                if record.planning_value_seconds is not None
                else 1.0
            ),
            evaluation_wall_seconds=0.0,
        )
        for record in records
    )
    return tuple(by_hash[item.solution_hash] for item in material_front(h0, TOLERANCE))


def select_delayed(records: Sequence[SelectionRecord]) -> SelectionResult:
    """Use the joint H0/H1 helpers for the delayed public-oracle path."""

    valid = _deduplicate_records(_valid_records(records))
    if not valid:
        return SelectionResult((), (), ())
    from joint_nesting_cuteval.hierarchical_process_selection import process_selection

    front_records = _joint_material_front(valid)
    from joint_nesting_cuteval.hierarchical_delayed_evaluation import H0Candidate

    h0_by_hash = {
        record.solution_hash: H0Candidate(
            order_id="shared-six-piece",
            candidate_index=record.candidate_index,
            solution_hash=record.solution_hash,
            board_count=record.board_count,
            tail_fraction=record.tail_fraction,
            remnant_ratio=record.remnant_ratio,
            planning_value_seconds=float(record.planning_value_seconds),
            evaluation_wall_seconds=0.0,
        )
        for record in front_records
    }
    front = tuple(h0_by_hash[record.solution_hash] for record in front_records)
    selected = process_selection(front, TOLERANCE)
    return SelectionResult(
        valid_hashes=tuple(sorted(record.solution_hash for record in valid)),
        material_front_hashes=tuple(candidate.solution_hash for candidate in front),
        selected_hashes=tuple(candidate.solution_hash for candidate in selected),
    )


def select_independent(records: Sequence[SelectionRecord]) -> SelectionResult:
    """Independently reconstruct material and process selection from public fields."""

    valid = _deduplicate_records(_valid_records(records))
    front = _independent_material_front(valid)
    if not front:
        return SelectionResult((), (), ())
    minimum = min(float(record.planning_value_seconds) for record in front)
    selected = tuple(
        record
        for record in front
        if float(record.planning_value_seconds) <= minimum + TOLERANCE
    )
    return SelectionResult(
        valid_hashes=tuple(sorted(record.solution_hash for record in valid)),
        material_front_hashes=tuple(record.solution_hash for record in front),
        selected_hashes=tuple(record.solution_hash for record in selected),
    )


def _selection_classification(
    records: Sequence[SelectionRecord], selection: SelectionResult
) -> dict[str, Any]:
    by_hash = {record.solution_hash: record for record in records}
    front = tuple(by_hash[item] for item in selection.material_front_hashes)
    if not front:
        return {
            "potential_process_selection": False,
            "process_selection_active": False,
            "reference_changed": False,
            "minimum_tail_reference_hashes": [],
            "maximum_remnant_reference_hashes": [],
        }
    minimum_tail = min(record.tail_fraction for record in front)
    maximum_remnant = max(record.remnant_ratio for record in front)
    minimum_tail_hashes = {
        record.solution_hash
        for record in front
        if record.tail_fraction <= minimum_tail + TOLERANCE
    }
    maximum_remnant_hashes = {
        record.solution_hash
        for record in front
        if record.remnant_ratio >= maximum_remnant - TOLERANCE
    }
    selected = set(selection.selected_hashes)
    return {
        "potential_process_selection": len(front) > 1,
        "process_selection_active": len(selected) < len(front),
        "reference_changed": (
            bool(selected)
            and selected != minimum_tail_hashes
            and selected != maximum_remnant_hashes
        ),
        "minimum_tail_reference_hashes": sorted(minimum_tail_hashes),
        "maximum_remnant_reference_hashes": sorted(maximum_remnant_hashes),
    }


def _candidate_from_layouts(bundle: dict, layouts, *, seed: int):
    from joint_nesting_cuteval.case_freeze import SourcePart
    from joint_nesting_cuteval.packing import (
        CandidatePanel,
        CandidatePlacement,
        CaseInventory,
        candidate_solution_from_panels,
    )

    source_by_id = {
        instance["instance_id"]: instance["source_part_id"]
        for instance in bundle["instances"]
    }
    inventory = CaseInventory(
        case_id=bundle["task_id"],
        order_id=bundle["task_id"],
        board_width_mm=float(bundle["stock"]["length_mm"]["value"]),
        board_height_mm=float(bundle["stock"]["width_mm"]["value"]),
        trim_mm=float(bundle["stock"]["margin_mm"]["left_mm"]["value"]),
        tool_diameter_mm=float(bundle["tool"]["diameter_mm"]["value"]),
        parts=tuple(
            SourcePart(
                order_id=bundle["task_id"],
                part_id=instance["instance_id"],
                source_part_id=instance["source_part_id"],
                nominal_width_mm=float(instance["length_mm"]["value"]),
                nominal_height_mm=float(instance["width_mm"]["value"]),
            )
            for instance in bundle["instances"]
        ),
        selected_payload_sha256=problem_hash(bundle),
    )
    panels = tuple(
        CandidatePanel(
            geometry_hash=layout.source_hash,
            placements=tuple(
                CandidatePlacement(
                    part_id=part.part_id,
                    source_part_id=source_by_id[part.part_id],
                    x_mm=float(part.x_mm),
                    y_mm=float(part.y_mm),
                    nominal_width_mm=float(part.width_mm),
                    nominal_height_mm=float(part.height_mm),
                    rotated=bool(part.rotated),
                )
                for part in layout.parts
            ),
        )
        for layout in layouts
    )
    candidate = candidate_solution_from_panels(inventory, panels, provenance_seed=seed)
    replay = candidate_solution_from_panels(inventory, candidate.panels, provenance_seed=seed)
    original_frozen_hash = _canonical_hash(candidate)
    reconstructed_frozen_hash = _canonical_hash(replay)
    if (
        replay.solution_hash != candidate.solution_hash
        or reconstructed_frozen_hash != original_frozen_hash
    ):
        raise AssertionError("candidate replay hash changed")
    return candidate, inventory, original_frozen_hash, reconstructed_frozen_hash


def _evaluation_layout(
    candidate,
    panel,
    *,
    problem_hash_value: str,
    source_method: str,
):
    """Create a CutEval input whose source identity includes problem semantics."""

    from joint_nesting_cuteval.final_validation import panel_to_layout

    identity = semantic_layout_hash(problem_hash_value, panel.geometry_hash)
    layout = panel_to_layout(
        candidate,
        panel,
        source_method=source_method,
        source_seed=str(candidate.seed),
    )
    stable_layout_id = f"{candidate.case_id}-{identity}"
    return replace(
        layout,
        panel_id=stable_layout_id,
        layout_id=stable_layout_id,
        source_hash=identity,
    )


def _layout_source_identity(
    candidate,
    panel,
    *,
    problem_hash_value: str,
    semantic_candidate_identity: str,
    source_method: str,
) -> dict[str, str]:
    return {
        "problem_hash": problem_hash_value,
        "semantic_candidate_hash": semantic_candidate_identity,
        "joint_geometry_hash": candidate.solution_hash,
        "semantic_layout_hash": semantic_layout_hash(
            problem_hash_value, panel.geometry_hash
        ),
        "joint_panel_geometry_hash": panel.geometry_hash,
        "source_method": source_method,
    }


def _public_candidate_evaluation(
    api,
    candidate,
    oracle,
    tool,
    profile,
    expected_schema: str,
    *,
    problem_hash_value: str,
    semantic_candidate_identity: str,
):
    panels = []
    planning = 0.0
    feasible = True
    source_method = "thesis_input:minibatch_public_handoff"
    for panel in candidate.panels:
        response = oracle.evaluate(
            _evaluation_layout(
                candidate,
                panel,
                problem_hash_value=problem_hash_value,
                source_method=source_method,
            ),
            tool,
            profile,
        )
        if response.schema_version != expected_schema:
            raise AssertionError("CutEval public handoff schema changed")
        public_response = {
            "source_identity": _layout_source_identity(
                candidate,
                panel,
                problem_hash_value=problem_hash_value,
                semantic_candidate_identity=semantic_candidate_identity,
                source_method=source_method,
            ),
            "schema_version": response.schema_version,
            "feasible": response.feasible,
            "planning_value_seconds": response.planning_value_seconds,
            "event_vector": asdict(response.event_vector),
            "provenance": asdict(response.provenance),
        }
        panels.append(public_response)
        if not response.feasible or response.planning_value_seconds is None:
            feasible = False
        else:
            planning += float(response.planning_value_seconds)
    return {
        "source_identity": {
            "problem_hash": problem_hash_value,
            "semantic_candidate_hash": semantic_candidate_identity,
            "joint_geometry_hash": candidate.solution_hash,
        },
        "feasible": feasible,
        "planning_value_seconds": planning if feasible else None,
        "panel_responses": panels,
    }


def _audit_candidate(
    api,
    candidate,
    tool,
    profile,
    *,
    problem_hash_value: str,
    semantic_candidate_identity: str,
):
    """Save full CutEvaluator evidence without exposing it to selection."""

    evaluator = api.CutEvaluator(api.EndpointAwareHeuristicPlanner(max_order_evaluations=100_000))
    records = []
    source_method = "thesis_input:minibatch_full_audit"
    for panel in candidate.panels:
        result = evaluator.evaluate(
            _evaluation_layout(
                candidate,
                panel,
                problem_hash_value=problem_hash_value,
                source_method=source_method,
            ),
            tool,
            profile,
        )
        records.append(
            {
                "source_identity": _layout_source_identity(
                    candidate,
                    panel,
                    problem_hash_value=problem_hash_value,
                    semantic_candidate_identity=semantic_candidate_identity,
                    source_method=source_method,
                ),
                "panel_geometry_hash": panel.geometry_hash,
                "feasible": result.feasible,
                "planning_value_seconds": result.planning_value_seconds,
                "event_vector": asdict(result.event_vector),
                "audit": asdict(result.audit),
                "provenance": asdict(result.provenance),
                "optimization": asdict(result.optimization),
                "nc_text": result.nc_text,
            }
        )
    return records


def _file_manifest(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("ortools", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_minibatch(
    bundle_path: Path,
    output_dir: Path,
    *,
    cut_root: Path,
    joint_root: Path,
) -> dict:
    """Run the real two-seed six-piece integration and atomically save its evidence."""

    source_identity = verify_source_identity(cut_root, joint_root)
    from inrp.handoff import solve_packing_handoff
    from joint_nesting_cuteval.chapter1_transfer_gate import _oracle
    from joint_nesting_cuteval.tail_metrics import evaluate_tail_metrics
    from joint_nesting_cuteval.upstream_contract import load_contract
    from thesis_input.adapters import to_nominal_layouts, to_packing_request

    api = importlib.import_module("cut_eval")
    project_root = Path(__file__).resolve().parents[2]
    imported_modules = _verify_loaded_modules(project_root, cut_root, joint_root)
    staging = _prepare_output(
        output_dir,
        project_root=project_root,
        cut_root=Path(cut_root),
        joint_root=Path(joint_root),
    )
    final_output = Path(output_dir).resolve()
    bundle: dict | None = None
    input_hash: str | None = None
    manifest_hash: str | None = None
    seed_runs: list[dict[str, Any]] = []
    candidates_by_hash: dict[str, Any] = {}
    geometry_by_hash: dict[str, list[list[dict[str, Any]]]] = {}
    inventory: Any = None
    audit_records: dict[str, Any] = {}
    metrics_by_hash: dict[str, Any] = {}
    delayed_public: dict[str, Any] = {}
    independent_public: dict[str, Any] = {}
    try:
        bundle = load_bundle(Path(bundle_path))
        input_hash = problem_hash(bundle)
        manifest_hash = source_manifest_hash(bundle)
        _write_json(staging / "normalized_input.json", bundle)
        source_identity_record = {
            "source": source_identity,
            "imports": imported_modules,
            "python": {
                "version": sys.version,
                "implementation": platform.python_implementation(),
                "library_versions": _library_versions(),
            },
            "input_bundle_path": str(Path(bundle_path).resolve()),
            "source_manifest_hash": manifest_hash,
            "problem_hash": input_hash,
            "thesis_input_problem_hash": input_hash,
            "candidate_identities": [],
        }
        _write_json(staging / "source_identity.json", source_identity_record)
        contract_path = Path(joint_root) / "configs" / "cuteval_upstream_contract_v2.json"
        contract = load_contract(contract_path)
        delayed_oracle, tool, profile = _oracle(api, contract)
        independent_oracle, independent_tool, independent_profile = _oracle(api, contract)
        expected_schema = str(contract["expected_handoff_schema"])

        for seed in SEEDS:
            request = to_packing_request(bundle, seed)
            handoff = solve_packing_handoff(request)
            actual_provenance = dict(handoff.provenance)
            actual_provenance["thesis_input_problem_hash"] = input_hash
            handoff = replace(handoff, provenance=actual_provenance)
            layouts = to_nominal_layouts(bundle, handoff)
            (
                candidate,
                inventory,
                original_frozen_hash,
                reconstructed_frozen_hash,
            ) = _candidate_from_layouts(bundle, layouts, seed=seed)
            # Seed provenance is intentionally not part of the semantic geometry
            # hash, so identical layouts from the two real solves are deduplicated.
            semantic_identity = _register_candidate(
                candidates_by_hash,
                geometry_by_hash,
                candidate,
                problem_hash_value=input_hash,
            )
            seed_runs.append(
                {
                    "seed": seed,
                    "source_manifest_hash": manifest_hash,
                    "problem_hash": input_hash,
                    "request": asdict(request),
                    "handoff": asdict(handoff),
                    "nominal_layouts": [asdict(layout) for layout in layouts],
                    "joint_geometry_hash": candidate.solution_hash,
                    "semantic_candidate_hash": semantic_identity,
                    "reconstructed_joint_geometry_hash": candidate.solution_hash,
                    "reconstructed_semantic_candidate_hash": semantic_identity,
                    "semantic_layout_hashes": [
                        semantic_layout_hash(input_hash, panel.geometry_hash)
                        for panel in candidate.panels
                    ],
                    "candidate_original_frozen_hash": original_frozen_hash,
                    "candidate_reconstructed_frozen_hash": reconstructed_frozen_hash,
                }
            )
            _write_json(staging / "seed_runs.json", seed_runs)

        candidate_items = tuple(candidates_by_hash.items())
        source_identity_record["candidate_identities"] = [
            {
                "semantic_candidate_hash": semantic_identity,
                "joint_geometry_hash": candidate.solution_hash,
                "layouts": [
                    {
                        "semantic_layout_hash": semantic_layout_hash(
                            input_hash, panel.geometry_hash
                        ),
                        "joint_panel_geometry_hash": panel.geometry_hash,
                    }
                    for panel in candidate.panels
                ],
            }
            for semantic_identity, candidate in candidate_items
        ]
        _write_json(staging / "source_identity.json", source_identity_record)
        from joint_nesting_cuteval.packing import candidate_solution_from_panels

        geometric_records = []
        for index, (semantic_identity, candidate) in enumerate(candidate_items):
            audit_records[semantic_identity] = _audit_candidate(
                api,
                candidate,
                tool,
                profile,
                problem_hash_value=input_hash,
                semantic_candidate_identity=semantic_identity,
            )
            _write_json(staging / "audit_records.json", audit_records)
            tail = evaluate_tail_metrics(candidate)
            original_frozen_hash = _canonical_hash(candidate)
            reconstructed = candidate_solution_from_panels(
                inventory, candidate.panels, provenance_seed=candidate.seed
            )
            reconstructed_frozen_hash = _canonical_hash(reconstructed)
            if reconstructed_frozen_hash != original_frozen_hash:
                raise AssertionError("deduplicated candidate replay hash changed")
            if reconstructed.solution_hash != candidate.solution_hash:
                raise AssertionError("deduplicated joint geometry hash changed")
            if (
                semantic_candidate_hash(input_hash, reconstructed.solution_hash)
                != semantic_identity
            ):
                raise AssertionError("deduplicated semantic candidate hash changed")
            metrics_by_hash[semantic_identity] = {
                "source_manifest_hash": manifest_hash,
                "problem_hash": input_hash,
                "joint_geometry_hash": candidate.solution_hash,
                "semantic_candidate_hash": semantic_identity,
                "reconstructed_joint_geometry_hash": reconstructed.solution_hash,
                "reconstructed_semantic_candidate_hash": semantic_candidate_hash(
                    input_hash, reconstructed.solution_hash
                ),
                "semantic_layout_hashes": [
                    semantic_layout_hash(input_hash, panel.geometry_hash)
                    for panel in candidate.panels
                ],
                "candidate": asdict(candidate),
                "tail_metrics": asdict(tail),
                "candidate_original_frozen_hash": original_frozen_hash,
                "candidate_reconstructed_frozen_hash": reconstructed_frozen_hash,
            }
            geometric_records.append(
                SelectionRecord(
                    solution_hash=semantic_identity,
                    candidate_index=index,
                    board_count=tail.board_count,
                    tail_fraction=tail.tail_occupied_fraction_gross,
                    remnant_ratio=tail.largest_reusable_remnant_ratio_usable,
                    planning_value_seconds=1.0,
                    feasible=True,
                    joint_geometry_hash=candidate.solution_hash,
                )
            )
            _write_json(staging / "candidate_metrics.json", metrics_by_hash)

        delayed_geometric_front = _joint_material_front(geometric_records)
        for record in delayed_geometric_front:
            delayed_public[record.solution_hash] = _public_candidate_evaluation(
                api,
                candidates_by_hash[record.solution_hash],
                delayed_oracle,
                tool,
                profile,
                expected_schema,
                problem_hash_value=input_hash,
                semantic_candidate_identity=record.solution_hash,
            )
            _write_json(staging / "selection_public_delayed.json", {
                "semantic_material_front_hashes": [
                    item.solution_hash for item in delayed_geometric_front
                ],
                "public_handoff_responses": delayed_public,
            })
        delayed_records = tuple(
            replace(
                record,
                feasible=bool(delayed_public[record.solution_hash]["feasible"]),
                planning_value_seconds=delayed_public[record.solution_hash][
                    "planning_value_seconds"
                ],
            )
            for record in delayed_geometric_front
        )
        delayed_selection = select_delayed(delayed_records)

        for record in geometric_records:
            independent_public[record.solution_hash] = _public_candidate_evaluation(
                api,
                candidates_by_hash[record.solution_hash],
                independent_oracle,
                independent_tool,
                independent_profile,
                expected_schema,
                problem_hash_value=input_hash,
                semantic_candidate_identity=record.solution_hash,
            )
            _write_json(staging / "selection_public_independent.json", {
                "public_handoff_responses": independent_public,
            })
        independent_records = tuple(
            replace(
                record,
                feasible=bool(independent_public[record.solution_hash]["feasible"]),
                planning_value_seconds=independent_public[record.solution_hash][
                    "planning_value_seconds"
                ],
            )
            for record in geometric_records
        )
        independent_selection = select_independent(independent_records)
        if not independent_selection.valid_hashes:
            raise RuntimeError("no feasible candidates remain after public CutEval")
        identity_sets_match = set(delayed_selection.selected_hashes) == set(
            independent_selection.selected_hashes
        )
        if not identity_sets_match:
            raise AssertionError("delayed and independent selected identity sets differ")

        _write_json(staging / "selection_public_delayed.json", {
            "semantic_material_front_hashes": [
                record.solution_hash for record in delayed_geometric_front
            ],
            "public_handoff_responses": delayed_public,
            "selection": asdict(delayed_selection),
            "classification": _selection_classification(
                delayed_records, delayed_selection
            ),
        })
        _write_json(staging / "selection_public_independent.json", {
            "public_handoff_responses": independent_public,
            "selection": asdict(independent_selection),
            "classification": _selection_classification(
                independent_records, independent_selection
            ),
        })
        result = {
            "status": "completed",
            "runner_version": RUNNER_VERSION,
            "seeds": list(SEEDS),
            "source_manifest_hash": manifest_hash,
            "problem_hash": input_hash,
            "thesis_input_problem_hash": input_hash,
            "unique_candidate_count": len(candidate_items),
            "selection": {
                "delayed_selected_hashes": list(delayed_selection.selected_hashes),
                "independent_selected_hashes": list(independent_selection.selected_hashes),
                "identity_sets_match": identity_sets_match,
            },
        }
        _write_json(staging / "run_summary.json", result)
        _write_json(staging / "manifest.json", _file_manifest(staging))
        staging.replace(final_output)
        return result
    except Exception as error:
        if bundle is not None:
            _write_json(staging / "normalized_input.json", bundle)
        if input_hash is not None:
            _write_json(staging / "seed_runs.json", seed_runs)
        if audit_records:
            _write_json(staging / "audit_records.json", audit_records)
        if metrics_by_hash:
            _write_json(staging / "candidate_metrics.json", metrics_by_hash)
        if delayed_public:
            _write_json(staging / "selection_public_delayed.json", {
                "public_handoff_responses": delayed_public,
            })
        if independent_public:
            _write_json(staging / "selection_public_independent.json", {
                "public_handoff_responses": independent_public,
            })
        _write_json(staging / "failure.json", {
            "status": "failed",
            "runner_version": RUNNER_VERSION,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        _write_json(staging / "manifest.json", _file_manifest(staging))
        staging.replace(final_output)
        raise
