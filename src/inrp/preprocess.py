from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Sequence, Tuple

from .dataio import Part
from .geometry import effective_trim


def _nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    return max(0.0, out)


def _round_to_step(value: float, step: float) -> float:
    step_f = _nonnegative_float(step)
    if step_f <= 0.0:
        return float(value)
    return round(float(value) / step_f) * step_f


@dataclass(frozen=True)
class PreprocessSummary:
    enabled: bool
    n_parts: int
    n_distinct_shapes: int
    duplicate_parts: int
    max_dup_count: int
    large_part_count: int
    strip_part_count: int
    quantized_parts: int
    dim_step: float
    board_area_effective: float
    large_area_ratio: float
    large_edge_ratio: float
    strip_aspect: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _shape_key(part: Part, *, allow_rot: bool) -> Tuple[float, float]:
    dims = (float(part.w0), float(part.h0))
    if allow_rot:
        return tuple(sorted(dims))
    return dims


def preprocess_parts(parts: Sequence[Part], cfg: Any) -> Tuple[List[Part], PreprocessSummary]:
    enabled = bool(getattr(cfg, "PREPROCESS_ENABLE", False))
    allow_rot = bool(getattr(cfg, "ALLOW_ROT", True))
    dim_step = _nonnegative_float(getattr(cfg, "PREPROCESS_DIM_STEP", 0.0))
    strip_aspect = max(1.0, _nonnegative_float(getattr(cfg, "PREPROCESS_STRIP_ASPECT", 8.0), 8.0))
    large_area_ratio = _nonnegative_float(getattr(cfg, "PREPROCESS_LARGE_AREA_RATIO", 0.08), 0.08)
    large_edge_ratio = _nonnegative_float(getattr(cfg, "PREPROCESS_LARGE_EDGE_RATIO", 0.75), 0.75)

    trim = effective_trim(cfg)
    board_w = _nonnegative_float(getattr(cfg, "BOARD_W", 0.0))
    board_h = _nonnegative_float(getattr(cfg, "BOARD_H", 0.0))
    board_area_effective = max(0.0, board_w - 2.0 * trim) * max(0.0, board_h - 2.0 * trim)
    board_long_edge = max(board_w - 2.0 * trim, board_h - 2.0 * trim, 0.0)

    processed: List[Part] = []
    quantized_parts = 0
    groups: Dict[Tuple[float, float], List[int]] = {}

    for idx, part in enumerate(parts):
        w0 = float(part.w0)
        h0 = float(part.h0)
        if enabled and dim_step > 0.0:
            q_w0 = _round_to_step(w0, dim_step)
            q_h0 = _round_to_step(h0, dim_step)
            inflate_w = max(0.0, float(part.w) - float(part.w0))
            inflate_h = max(0.0, float(part.h) - float(part.h0))
            q_w = _round_to_step(q_w0 + inflate_w, dim_step)
            q_h = _round_to_step(q_h0 + inflate_h, dim_step)
            if abs(q_w0 - w0) > 1e-9 or abs(q_h0 - h0) > 1e-9:
                quantized_parts += 1
        else:
            q_w0 = w0
            q_h0 = h0
            q_w = float(part.w)
            q_h = float(part.h)

        long_edge = max(q_w0, q_h0)
        short_edge = max(1e-9, min(q_w0, q_h0))
        area = q_w0 * q_h0
        aspect = long_edge / short_edge
        area_ratio = area / max(1e-9, board_area_effective) if board_area_effective > 0.0 else 0.0
        long_edge_ratio = long_edge / max(1e-9, board_long_edge) if board_long_edge > 0.0 else 0.0
        is_large = area_ratio >= large_area_ratio or long_edge_ratio >= large_edge_ratio
        is_strip = aspect >= strip_aspect

        item = replace(
            part,
            w=q_w,
            h=q_h,
            w0=q_w0,
            h0=q_h0,
            long_edge=long_edge,
            short_edge=short_edge,
            area=area,
            aspect=aspect,
            area_ratio=area_ratio,
            long_edge_ratio=long_edge_ratio,
            is_large=is_large,
            is_strip=is_strip,
        )
        processed.append(item)
        groups.setdefault(_shape_key(item, allow_rot=allow_rot), []).append(idx)

    for group_idx, (_, indices) in enumerate(sorted(groups.items(), key=lambda item: (item[0], len(item[1]))), start=1):
        dup_count = len(indices)
        for rank, idx in enumerate(indices, start=1):
            processed[idx] = replace(
                processed[idx],
                dup_group=f"shape_{group_idx}",
                dup_count=dup_count,
                dup_rank=rank,
            )

    summary = PreprocessSummary(
        enabled=enabled,
        n_parts=len(processed),
        n_distinct_shapes=len(groups),
        duplicate_parts=sum(max(0, len(indices) - 1) for indices in groups.values()),
        max_dup_count=max((len(indices) for indices in groups.values()), default=0),
        large_part_count=sum(1 for part in processed if bool(part.is_large)),
        strip_part_count=sum(1 for part in processed if bool(part.is_strip)),
        quantized_parts=int(quantized_parts),
        dim_step=float(dim_step),
        board_area_effective=float(board_area_effective),
        large_area_ratio=float(large_area_ratio),
        large_edge_ratio=float(large_edge_ratio),
        strip_aspect=float(strip_aspect),
    )
    return processed, summary
