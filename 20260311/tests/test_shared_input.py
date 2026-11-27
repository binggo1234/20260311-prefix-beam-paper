from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from inrp.handoff import PackingHandoff, PackingPlacement
from thesis_input.adapters import to_nominal_layouts, to_packing_request
from thesis_input.contract import problem_hash, source_manifest_hash, validate_bundle


FIXTURE = Path(__file__).parent / "fixtures" / "shared_six_parts.json"


@pytest.fixture
def fixture_bundle() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _valid_handoff(bundle: dict, *, panel_id: int = 7) -> PackingHandoff:
    request = to_packing_request(bundle, seed=17)
    rows = (("p01__copy_001", 8, 8, 120, 60, False), ("p02__copy_001", 134, 8, 120, 60, False), ("p03__copy_001", 8, 74, 80, 60, False), ("p04__copy_001", 94, 74, 80, 60, False), ("p05__copy_001", 180, 74, 40, 40, False), ("p06__copy_001", 226, 74, 40, 40, False))
    placements = tuple(PackingPlacement(panel_id, part_id, part_id.split("__", 1)[0], x, y, width, height, rotated) for part_id, x, y, width, height, rotated in rows)
    return PackingHandoff("inrp-packing-handoff-v1", True, 1, placements, tuple(item.part_id for item in placements), (1.0,), 0.01, {"request": asdict(request), "thesis_input_problem_hash": problem_hash(bundle)})


def test_declared_fixture_is_not_verified_production(fixture_bundle: dict) -> None:
    with pytest.raises(ValueError, match="verified"):
        validate_bundle(fixture_bundle, require_verified=True)


def test_synthetic_facts_cannot_be_elevated_to_verified_or_packed(fixture_bundle: dict) -> None:
    for fact in _all_facts(fixture_bundle):
        fact["status"] = "verified"

    with pytest.raises(ValueError, match="synthetic.*model_declared"):
        validate_bundle(fixture_bundle)
    with pytest.raises(ValueError, match="synthetic.*model_declared"):
        to_packing_request(fixture_bundle, seed=17)


def test_validation_normalizes_without_mutating_and_expands_quantity(fixture_bundle: dict) -> None:
    fixture_bundle["parts"][0]["quantity"] = 2
    fixture_bundle["parts"][0]["length_mm"]["value"] = "120.00"
    original = deepcopy(fixture_bundle)

    normalized = validate_bundle(fixture_bundle)

    assert fixture_bundle == original
    assert normalized["stock"]["length_mm"]["value"] == "320"
    assert normalized["parts"][0]["length_mm"]["value"] == "120"
    assert [part["instance_id"] for part in normalized["instances"][:2]] == ["p01__copy_001", "p01__copy_002"]
    assert normalized["summary"]["source_quantity_total"] == 7
    assert normalized["summary"]["expanded_quantity_total"] == 7
    assert normalized["parts"][0]["dimension_basis"]["value"] == "synthetic_defined"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "0"])
def test_validation_rejects_nonfinite_or_nonpositive_dimensions(fixture_bundle: dict, value: str) -> None:
    fixture_bundle["stock"]["length_mm"]["value"] = value

    with pytest.raises(ValueError, match="positive finite"):
        validate_bundle(fixture_bundle)


def test_validation_rejects_missing_or_duplicate_ids_and_asymmetric_margins(fixture_bundle: dict) -> None:
    fixture_bundle["parts"][1]["part_id"] = "p01"
    with pytest.raises(ValueError, match="unique"):
        validate_bundle(fixture_bundle)

    fixture_bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture_bundle["stock"]["margin_mm"]["right_mm"]["value"] = "6"
    with pytest.raises(ValueError, match="symmetric"):
        validate_bundle(fixture_bundle)

    fixture_bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del fixture_bundle["parts"][0]["part_id"]
    with pytest.raises(ValueError, match="missing fields"):
        validate_bundle(fixture_bundle)


def test_validation_rejects_mixed_stock_properties_bad_evidence_and_grain_conflict(fixture_bundle: dict) -> None:
    fixture_bundle["parts"][0]["material"]["value"] = "other"
    with pytest.raises(ValueError, match="material"):
        validate_bundle(fixture_bundle)

    fixture_bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture_bundle["parts"][0]["allowed_rotations_deg"]["value"] = [90]
    with pytest.raises(ValueError, match="grain"):
        validate_bundle(fixture_bundle)

    fixture_bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture_bundle["tool"]["diameter_mm"]["source"] = "untracked"
    with pytest.raises(ValueError, match="evidence"):
        validate_bundle(fixture_bundle)


def test_validation_rejects_unresolved_or_unknown_schema_structure(fixture_bundle: dict) -> None:
    fixture_bundle["tool"]["diameter_mm"]["status"] = "unresolved"
    with pytest.raises(ValueError, match="unresolved"):
        validate_bundle(fixture_bundle)

    fixture_bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture_bundle["unexpected"] = "silently ignored before strict validation"
    with pytest.raises(ValueError, match="unknown fields"):
        validate_bundle(fixture_bundle)

    fixture_bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture_bundle["parts"][0]["unexpected"] = "silently ignored before strict validation"
    with pytest.raises(ValueError, match="unknown fields"):
        validate_bundle(fixture_bundle)


@pytest.mark.parametrize("field, value", [("Graining", []), ("Orientation", None)])
def test_validation_requires_raw_graining_and_orientation_scalars(
    fixture_bundle: dict, field: str, value: object
) -> None:
    fixture_bundle["parts"][0]["original_fields"][field] = value
    with pytest.raises(ValueError, match="original_fields"):
        validate_bundle(fixture_bundle)

    fixture_bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del fixture_bundle["parts"][0]["original_fields"][field]
    with pytest.raises(ValueError, match="original_fields"):
        validate_bundle(fixture_bundle)


def test_hashes_normalize_numeric_spelling(fixture_bundle: dict) -> None:
    normalized = validate_bundle(fixture_bundle)
    spelling_variant = deepcopy(fixture_bundle)
    spelling_variant["stock"]["length_mm"]["value"] = "320.00"
    assert problem_hash(normalized) == problem_hash(validate_bundle(spelling_variant))


def test_raw_labels_change_source_manifest_but_not_problem_hash(
    fixture_bundle: dict,
) -> None:
    normalized = validate_bundle(fixture_bundle)
    changed = deepcopy(fixture_bundle)
    changed["parts"][0]["original_fields"]["Graining"] = "raw-graining-relabel"
    changed["parts"][0]["original_fields"]["Orientation"] = "raw-orientation-relabel"

    assert problem_hash(normalized) == problem_hash(changed)
    assert source_manifest_hash(normalized) != source_manifest_hash(changed)


def test_evidence_status_and_locator_change_only_source_manifest_hash(
    fixture_bundle: dict,
) -> None:
    verified = _file_backed_bundle(fixture_bundle, "a" * 64)
    status_changed = deepcopy(verified)
    status_changed["tool"]["diameter_mm"]["status"] = "model_declared"

    locator_changed = _file_backed_bundle(fixture_bundle, "b" * 64)

    assert problem_hash(verified) == problem_hash(status_changed)
    assert problem_hash(verified) == problem_hash(locator_changed)
    assert source_manifest_hash(verified) != source_manifest_hash(status_changed)
    assert source_manifest_hash(verified) != source_manifest_hash(locator_changed)


def test_allowed_rotations_change_problem_hash(fixture_bundle: dict) -> None:
    changed = deepcopy(fixture_bundle)
    changed["parts"][2]["allowed_rotations_deg"]["value"] = [0]

    assert problem_hash(fixture_bundle) != problem_hash(changed)


def test_packing_request_keeps_nominal_dimensions_permissions_and_synthetic_budget(fixture_bundle: dict) -> None:
    request = to_packing_request(validate_bundle(fixture_bundle), seed=17)

    assert request.board_width_mm == 320.0
    assert request.board_height_mm == 240.0
    assert request.trim_mm == 5.0
    assert request.tool_diameter_mm == 6.0
    assert request.time_limit_s == 0.2
    assert request.parts[0].nominal_width_mm == 120.0
    assert request.parts[0].allowed_rotations_deg == (0,)
    assert request.parts[2].allowed_rotations_deg == (0, 90)


def test_nominal_adapter_accepts_matching_handoff_and_stable_identity(fixture_bundle: dict) -> None:
    bundle = validate_bundle(fixture_bundle)
    first = to_nominal_layouts(bundle, _valid_handoff(bundle, panel_id=7))
    second = to_nominal_layouts(bundle, _valid_handoff(bundle, panel_id=99))

    assert len(first) == 1
    assert first[0].layout_id == second[0].layout_id
    assert first[0].panel_id == second[0].panel_id
    assert first[0].source_hash == second[0].source_hash
    assert first[0].parts[0].x_mm == 8.0
    assert first[0].geometry_semantics.value == "nominal_with_tool_clearance"


def test_nominal_layout_identities_ignore_multiple_panel_renumbering(fixture_bundle: dict) -> None:
    bundle = validate_bundle(fixture_bundle)
    handoff = _valid_handoff(bundle)
    first_placements = tuple(
        PackingPlacement(7 if index < 3 else 42, placement.part_id, placement.source_part_id, placement.x_mm, placement.y_mm, placement.nominal_width_mm, placement.nominal_height_mm, placement.rotated)
        for index, placement in enumerate(handoff.placements)
    )
    second_placements = tuple(
        PackingPlacement(42 if placement.panel_id == 7 else 7, placement.part_id, placement.source_part_id, placement.x_mm, placement.y_mm, placement.nominal_width_mm, placement.nominal_height_mm, placement.rotated)
        for placement in first_placements
    )
    first = PackingHandoff(handoff.schema_version, True, 2, first_placements, handoff.sequence, handoff.score_global, handoff.runtime_s, handoff.provenance)
    second = PackingHandoff(handoff.schema_version, True, 2, second_placements, handoff.sequence, handoff.score_global, handoff.runtime_s, handoff.provenance)

    assert [(layout.panel_id, layout.layout_id, layout.source_hash) for layout in to_nominal_layouts(bundle, first)] == [(layout.panel_id, layout.layout_id, layout.source_hash) for layout in to_nominal_layouts(bundle, second)]


@pytest.mark.parametrize("change", ["wrong_rotation", "duplicate", "negative", "out_of_bounds", "overlap", "request_mismatch"])
def test_nominal_adapter_rejects_invalid_handoff(fixture_bundle: dict, change: str) -> None:
    bundle = validate_bundle(fixture_bundle)
    handoff = _valid_handoff(bundle)
    placements = list(handoff.placements)
    provenance = dict(handoff.provenance)
    if change == "wrong_rotation":
        placements[0] = PackingPlacement(7, "p01__copy_001", "p01", 8, 8, 60, 120, True)
    elif change == "duplicate":
        placements[-1] = PackingPlacement(7, "p05__copy_001", "p05", 226, 74, 40, 40, False)
    elif change == "negative":
        placements[0] = PackingPlacement(7, "p01__copy_001", "p01", -1, 8, 120, 60, False)
    elif change == "out_of_bounds":
        placements[0] = PackingPlacement(7, "p01__copy_001", "p01", 4, 8, 120, 60, False)
    elif change == "overlap":
        placements[1] = PackingPlacement(7, "p02__copy_001", "p02", 10, 10, 120, 60, False)
    else:
        provenance["request"] = {**provenance["request"], "board_width_mm": 321.0}
    invalid = PackingHandoff(handoff.schema_version, handoff.feasible, handoff.panel_count, tuple(placements), handoff.sequence, handoff.score_global, handoff.runtime_s, provenance)

    with pytest.raises(ValueError):
        to_nominal_layouts(bundle, invalid)


def test_nominal_adapter_rejects_empty_handoff_output(fixture_bundle: dict) -> None:
    bundle = validate_bundle(fixture_bundle)
    handoff = _valid_handoff(bundle)
    empty = PackingHandoff(
        handoff.schema_version,
        True,
        0,
        (),
        (),
        handoff.score_global,
        handoff.runtime_s,
        handoff.provenance,
    )

    with pytest.raises(ValueError, match="nonempty"):
        to_nominal_layouts(bundle, empty)


def test_nominal_adapter_rejects_semantic_hash_mismatch(fixture_bundle: dict) -> None:
    bundle = validate_bundle(fixture_bundle)
    handoff = _valid_handoff(bundle)
    invalid = PackingHandoff(
        handoff.schema_version,
        handoff.feasible,
        handoff.panel_count,
        handoff.placements,
        handoff.sequence,
        handoff.score_global,
        handoff.runtime_s,
        {**handoff.provenance, "thesis_input_problem_hash": "0" * 64},
    )

    with pytest.raises(ValueError, match="semantic hash"):
        to_nominal_layouts(bundle, invalid)


def _all_facts(bundle: dict):
    for value in bundle["stock"].values():
        if isinstance(value, dict) and "value" in value:
            yield value
        elif isinstance(value, dict):
            yield from value.values()
    yield from bundle["tool"].values()
    for part in bundle["parts"]:
        for key in ("dimension_basis", "length_mm", "width_mm", "thickness_mm", "material", "grain_axis", "allowed_rotations_deg"):
            yield part[key]


def _file_backed_bundle(bundle: dict, digest: str) -> dict:
    """Build a strictly valid file-backed hash fixture; do not weaken validation."""

    changed = deepcopy(bundle)
    changed["source_kind"] = "file"
    changed["source"] = {"file_sha256": digest, "encoding": "utf-8"}
    evidence = f"sha256:{digest}"
    for fact in _all_facts(changed):
        fact["status"] = "verified"
        fact["source"] = evidence
    for part in changed["parts"]:
        part["dimension_basis"]["value"] = "original_dimensions"
        part["original_fields"]["length_mm"] = part["length_mm"]["value"]
        part["original_fields"]["width_mm"] = part["width_mm"]["value"]
    return changed
