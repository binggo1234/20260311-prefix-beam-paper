from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    return max(0.0, out)


@dataclass(frozen=True)
class GeometryPolicy:
    trim: float
    part_inflate_gap: float
    safe_gap: float
    tool_d: float
    binary_spacing: bool
    spacing_mode: str


def resolve_geometry_policy(cfg: Any) -> GeometryPolicy:
    trim = _nonnegative(getattr(cfg, "TRIM", 0.0))
    gap = _nonnegative(getattr(cfg, "GAP", 0.0))
    tool_d = _nonnegative(getattr(cfg, "TOOL_D", 0.0))
    binary_spacing = bool(getattr(cfg, "BINARY_SPACING", False))
    safe_gap_raw = getattr(cfg, "SAFE_GAP", None)
    safe_gap = _nonnegative(tool_d if safe_gap_raw is None else safe_gap_raw)

    if binary_spacing:
        return GeometryPolicy(
            trim=trim,
            part_inflate_gap=0.0,
            safe_gap=safe_gap,
            tool_d=tool_d,
            binary_spacing=True,
            spacing_mode="binary",
        )

    if gap > 0.0:
        return GeometryPolicy(
            trim=trim,
            part_inflate_gap=gap,
            safe_gap=0.0,
            tool_d=tool_d,
            binary_spacing=False,
            spacing_mode="inflate",
        )

    return GeometryPolicy(
        trim=trim,
        part_inflate_gap=0.0,
        safe_gap=0.0,
        tool_d=tool_d,
        binary_spacing=False,
        spacing_mode="none",
    )


def effective_trim(cfg: Any) -> float:
    return resolve_geometry_policy(cfg).trim


def effective_safe_gap(cfg: Any) -> float:
    return resolve_geometry_policy(cfg).safe_gap


def effective_part_inflate_gap(cfg: Any) -> float:
    return resolve_geometry_policy(cfg).part_inflate_gap
