"""Pure adapters from the common thesis input to packing and CutEval contracts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from typing import Any, Mapping

from cut_eval import GeometrySemantics, NominalLayout, NominalPart
from inrp.handoff import PACKING_HANDOFF_SCHEMA_VERSION, PackingHandoff, PackingPart, PackingRequest

from .contract import problem_hash, validate_bundle


SYNTHETIC_DIAGNOSTIC_TIME_LIMIT_S = 0.2
_TOLERANCE = 1e-9


def to_packing_request(bundle: dict, seed: int) -> PackingRequest:
    """Create the only supported synthetic diagnostic packing request.

    The 0.2 second limit is a fixture-integration diagnostic, not a production
    manufacturing budget and must not be used for comparative conclusions.
    """

    normalized = validate_bundle(bundle)
    if normalized["source_kind"] != "synthetic":
        raise ValueError("no packing runtime budget is declared for file-backed inputs")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    stock = normalized["stock"]
    request = PackingRequest(
        case_id=normalized["task_id"],
        parts=tuple(
            PackingPart(
                part_id=instance["instance_id"],
                source_part_id=instance["source_part_id"],
                nominal_width_mm=float(instance["length_mm"]["value"]),
                nominal_height_mm=float(instance["width_mm"]["value"]),
                allowed_rotations_deg=tuple(instance["allowed_rotations_deg"]["value"]),
            )
            for instance in normalized["instances"]
        ),
        board_width_mm=float(stock["length_mm"]["value"]),
        board_height_mm=float(stock["width_mm"]["value"]),
        trim_mm=float(stock["margin_mm"]["left_mm"]["value"]),
        tool_diameter_mm=float(normalized["tool"]["diameter_mm"]["value"]),
        allow_rotation=True,
        seed=seed,
        time_limit_s=SYNTHETIC_DIAGNOSTIC_TIME_LIMIT_S,
        geometry_semantics=normalized["tool"]["geometry_rule"]["value"],
    )
    return request


def to_nominal_layouts(bundle: dict, handoff: PackingHandoff) -> tuple[NominalLayout, ...]:
    """Translate a verified packing handoff without performing another packing pass."""

    normalized = validate_bundle(bundle)
    if not isinstance(handoff, PackingHandoff) or handoff.schema_version != PACKING_HANDOFF_SCHEMA_VERSION:
        raise ValueError("packing handoff schema is unsupported")
    if not handoff.feasible or handoff.panel_count <= 0 or not handoff.placements:
        raise ValueError("packing handoff must contain a feasible nonempty layout")
    request_data = handoff.provenance.get("request") if isinstance(handoff.provenance, Mapping) else None
    if not isinstance(request_data, Mapping):
        raise ValueError("packing handoff lacks request provenance")
    seed = request_data.get("seed")
    if type(seed) is not int:
        raise ValueError("packing handoff request provenance has no integer seed")
    expected_request = to_packing_request(normalized, seed)
    if dict(request_data) != asdict(expected_request):
        raise ValueError("packing handoff request provenance does not match input")
    if handoff.provenance.get("thesis_input_problem_hash") != problem_hash(normalized):
        raise ValueError("packing handoff semantic hash does not match input")

    expected = {part.part_id: part for part in expected_request.parts}
    placements = tuple(handoff.placements)
    observed = [placement.part_id for placement in placements]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError("packing handoff lost or duplicated input parts")
    if handoff.panel_count != len({placement.panel_id for placement in placements}):
        raise ValueError("packing handoff panel count does not match placements")

    grouped: dict[int, list] = {}
    for placement in placements:
        _validate_placement(placement, expected[placement.part_id], expected_request)
        grouped.setdefault(placement.panel_id, []).append(placement)
    for panel in grouped.values():
        _validate_envelopes(panel, expected_request)

    bundle_hash = problem_hash(normalized)
    layouts = []
    for panel in grouped.values():
        joint_panel_geometry_hash = _joint_panel_geometry_hash(panel)
        identity = semantic_layout_hash(bundle_hash, joint_panel_geometry_hash)
        parts = tuple(
            NominalPart(
                part_id=placement.part_id,
                x_mm=float(placement.x_mm),
                y_mm=float(placement.y_mm),
                width_mm=float(placement.nominal_width_mm),
                height_mm=float(placement.nominal_height_mm),
                rotated=bool(placement.rotated),
            )
            for placement in sorted(panel, key=lambda item: item.part_id)
        )
        layouts.append(
            NominalLayout(
                order_id=normalized["task_id"],
                panel_id=f"panel-{identity[:16]}",
                layout_id=f"{normalized['task_id']}-{bundle_hash[:16]}-{identity[:16]}",
                board_width_mm=expected_request.board_width_mm,
                board_height_mm=expected_request.board_height_mm,
                trim_mm=expected_request.trim_mm,
                parts=parts,
                geometry_semantics=GeometrySemantics.NOMINAL_WITH_TOOL_CLEARANCE,
                source_method="thesis_input:packing_handoff_adapter_v1",
                source_seed=str(seed),
                source_hash=identity,
            )
        )
    return tuple(sorted(layouts, key=lambda layout: layout.layout_id))


def _validate_placement(placement: Any, part: PackingPart, request: PackingRequest) -> None:
    if type(placement.panel_id) is not int or placement.panel_id <= 0:
        raise ValueError("packing handoff panel_id must be a positive integer")
    if placement.source_part_id != part.source_part_id:
        raise ValueError("packing handoff source part identity does not match input")
    if type(placement.rotated) is not bool:
        raise ValueError("packing handoff rotation flag is malformed")
    angle = 90 if placement.rotated else 0
    if angle not in part.allowed_rotations_deg:
        raise ValueError("packing handoff rotation violates input permission")
    expected_width = part.nominal_height_mm if placement.rotated else part.nominal_width_mm
    expected_height = part.nominal_width_mm if placement.rotated else part.nominal_height_mm
    if abs(float(placement.nominal_width_mm) - expected_width) > _TOLERANCE or abs(float(placement.nominal_height_mm) - expected_height) > _TOLERANCE:
        raise ValueError("packing handoff nominal dimensions do not match input")
    if not all(_finite(float(value)) for value in (placement.x_mm, placement.y_mm, placement.nominal_width_mm, placement.nominal_height_mm)):
        raise ValueError("packing handoff coordinates must be finite")
    radius = request.tool_diameter_mm / 2.0
    envelope_x = float(placement.x_mm) - radius
    envelope_y = float(placement.y_mm) - radius
    envelope_right = envelope_x + expected_width + request.tool_diameter_mm
    envelope_top = envelope_y + expected_height + request.tool_diameter_mm
    if envelope_x < request.trim_mm - _TOLERANCE or envelope_y < request.trim_mm - _TOLERANCE or envelope_right > request.board_width_mm - request.trim_mm + _TOLERANCE or envelope_top > request.board_height_mm - request.trim_mm + _TOLERANCE:
        raise ValueError("packing handoff envelope is outside stock bounds")


def _validate_envelopes(panel: list, request: PackingRequest) -> None:
    radius = request.tool_diameter_mm / 2.0
    for index, first in enumerate(panel):
        first_box = (float(first.x_mm) - radius, float(first.y_mm) - radius, float(first.nominal_width_mm) + request.tool_diameter_mm, float(first.nominal_height_mm) + request.tool_diameter_mm)
        for second in panel[index + 1 :]:
            second_box = (float(second.x_mm) - radius, float(second.y_mm) - radius, float(second.nominal_width_mm) + request.tool_diameter_mm, float(second.nominal_height_mm) + request.tool_diameter_mm)
            overlap_x = min(first_box[0] + first_box[2], second_box[0] + second_box[2]) - max(first_box[0], second_box[0])
            overlap_y = min(first_box[1] + first_box[3], second_box[1] + second_box[3]) - max(first_box[1], second_box[1])
            if overlap_x > _TOLERANCE and overlap_y > _TOLERANCE:
                raise ValueError("packing handoff envelopes overlap")


def _joint_panel_geometry_hash(panel: list) -> str:
    """Match the joint package's canonical CandidatePanel geometry hash."""

    return _sha256(
        [
            {
                "part_id": placement.part_id,
                "source_part_id": placement.source_part_id,
                "x_mm": round(float(placement.x_mm), 6),
                "y_mm": round(float(placement.y_mm), 6),
                "nominal_width_mm": round(float(placement.nominal_width_mm), 6),
                "nominal_height_mm": round(float(placement.nominal_height_mm), 6),
                "rotated": placement.rotated,
            }
            for placement in sorted(panel, key=lambda item: item.part_id)
        ]
    )


def semantic_layout_hash(
    problem_hash_value: str, joint_panel_geometry_hash: str
) -> str:
    """Bind label-invariant canonical panel geometry to problem semantics."""

    return _sha256(
        {
            "problem_hash": problem_hash_value,
            "joint_panel_geometry_hash": joint_panel_geometry_hash,
        }
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")).hexdigest()


def _finite(value: float) -> bool:
    return math.isfinite(value)
