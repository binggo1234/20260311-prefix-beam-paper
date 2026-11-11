"""Area relaxations for the nominal-with-tool-clearance handoff convention."""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .handoff import PackingRequest


def packing_area_bounds(request: PackingRequest) -> Dict[str, Any]:
    """Return a model-specific relaxation bound, never a claim of attainability."""
    def dimension(value, *, zero_allowed=False):
        result = Decimal(str(value))
        if not result.is_finite() or result < 0 or (result == 0 and not zero_allowed):
            raise ValueError("geometry dimensions must be finite and positive (reserves may be zero)")
        return result

    if request.geometry_semantics != "nominal_with_tool_clearance" or not request.parts:
        raise ValueError("bounds require a nonempty nominal-with-tool-clearance order")
    width, height = dimension(request.board_width_mm), dimension(request.board_height_mm)
    trim = dimension(request.trim_mm, zero_allowed=True)
    tool = dimension(request.tool_diameter_mm, zero_allowed=True)
    usable_width, usable_height = width - 2 * trim, height - 2 * trim
    if usable_width <= 0 or usable_height <= 0:
        raise ValueError("trim leaves no usable board area")
    nominal_area = envelope_area = Decimal(0)
    for part in request.parts:
        w, h = dimension(part.nominal_width_mm), dimension(part.nominal_height_mm)
        ew, eh = w + tool, h + tool
        rotations = part.allowed_rotations_deg
        if not rotations or any(r not in (0, 90) for r in rotations):
            raise ValueError("unsupported part rotations")
        fits = any(
            (ew <= usable_width and eh <= usable_height) if rotation == 0 else
            (request.allow_rotation and eh <= usable_width and ew <= usable_height)
            for rotation in rotations
        )
        if not fits:
            raise ValueError("part envelope cannot fit any permitted board orientation")
        nominal_area += w * h
        envelope_area += ew * eh
    board_area, usable_area = width * height, usable_width * usable_height
    nominal_lb = int((nominal_area / board_area).to_integral_value(rounding=ROUND_CEILING))
    envelope_lb = int((envelope_area / usable_area).to_integral_value(rounding=ROUND_CEILING))
    lower_bound = max(nominal_lb, envelope_lb)
    return {
        "nominal_area_mm2": float(nominal_area), "envelope_area_mm2": float(envelope_area),
        "board_area_mm2": float(board_area), "usable_area_mm2": float(usable_area),
        "nominal_area_lower_bound": nominal_lb, "envelope_area_lower_bound": envelope_lb,
        "panel_lower_bound": lower_bound,
        "net_utilization_ceiling": float(nominal_area / (lower_bound * board_area)),
        "bound_kind": "area_relaxation_current_envelope_model",
    }
