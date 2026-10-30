"""Versioned packing handoff for downstream manufacturing evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Mapping, Tuple

from .budget import TotalBudget
from .cfg import CFG
from .dataio import Part
from .mcts import SearchResult, solve_with_receding_horizon_mcts
from .preprocess import preprocess_parts
from .runner import _case_stats, _solve_baseline_greedy, _variant_cfg
from .validate import validate_solution


PACKING_HANDOFF_SCHEMA_VERSION = "inrp-packing-handoff-v1"
SUPPORTED_GEOMETRY_SEMANTICS = "nominal_with_tool_clearance"


@dataclass(frozen=True)
class PackingPart:
    part_id: str
    source_part_id: str
    nominal_width_mm: float
    nominal_height_mm: float
    allowed_rotations_deg: Tuple[int, ...] = (0, 90)


@dataclass(frozen=True)
class PackingRequest:
    case_id: str
    parts: Tuple[PackingPart, ...]
    board_width_mm: float = 2440.0
    board_height_mm: float = 1220.0
    trim_mm: float = 5.0
    tool_diameter_mm: float = 6.0
    allow_rotation: bool = True
    seed: int = 1000
    time_limit_s: float = 900.0
    solver_variant: str = "rh_mcts_prefix_beam"
    geometry_semantics: str = SUPPORTED_GEOMETRY_SEMANTICS


@dataclass(frozen=True)
class PackingPlacement:
    panel_id: int
    part_id: str
    source_part_id: str
    x_mm: float
    y_mm: float
    nominal_width_mm: float
    nominal_height_mm: float
    rotated: bool


@dataclass(frozen=True)
class PackingHandoff:
    schema_version: str
    feasible: bool
    panel_count: int
    placements: Tuple[PackingPlacement, ...]
    sequence: Tuple[str, ...]
    score_global: Tuple[float, ...]
    runtime_s: float
    provenance: Mapping[str, Any]


def _validate_request(request: PackingRequest) -> None:
    if request.geometry_semantics != SUPPORTED_GEOMETRY_SEMANTICS:
        raise ValueError(f"unsupported geometry semantics: {request.geometry_semantics}")
    if not request.parts:
        raise ValueError("packing request must contain at least one part")
    if request.board_width_mm <= 0.0 or request.board_height_mm <= 0.0:
        raise ValueError("board dimensions must be positive")
    if request.trim_mm < 0.0 or request.tool_diameter_mm < 0.0:
        raise ValueError("trim and tool diameter must be nonnegative")
    ids = [part.part_id for part in request.parts]
    if len(ids) != len(set(ids)):
        raise ValueError("part_id must be unique within a packing request")
    for part in request.parts:
        if part.nominal_width_mm <= 0.0 or part.nominal_height_mm <= 0.0:
            raise ValueError(f"part dimensions must be positive: {part.part_id}")
        _effective_allowed_rotations(part, request.allow_rotation)


def _effective_allowed_rotations(part: PackingPart, allow_rotation: bool) -> Tuple[int, ...]:
    allowed = part.allowed_rotations_deg
    if not isinstance(allowed, tuple) or not allowed:
        raise ValueError(f"allowed_rotations_deg must be a nonempty tuple for part: {part.part_id}")
    if any(type(angle) is not int for angle in allowed):
        raise ValueError(f"allowed_rotations_deg must contain integers for part: {part.part_id}")
    if len(set(allowed)) != len(allowed):
        raise ValueError(f"allowed_rotations_deg must not contain duplicates for part: {part.part_id}")
    if any(angle not in (0, 90) for angle in allowed):
        raise ValueError(f"allowed_rotations_deg contains an unsupported angle for part: {part.part_id}")
    effective = tuple(angle for angle in allowed if angle == 0 or bool(allow_rotation))
    if not effective:
        raise ValueError(f"allowed_rotations_deg has no permitted orientation under allow_rotation for part: {part.part_id}")
    return effective


def _build_solver_input(request: PackingRequest) -> tuple[list[Part], dict[int, PackingPart]]:
    allowance = float(request.tool_diameter_mm)
    by_uid: dict[int, PackingPart] = {}
    solver_parts: list[Part] = []
    for uid, source in enumerate(request.parts, 1):
        by_uid[uid] = source
        solver_parts.append(
            Part(
                uid=uid,
                pid_raw=source.part_id,
                w=float(source.nominal_width_mm) + allowance,
                h=float(source.nominal_height_mm) + allowance,
                w0=float(source.nominal_width_mm),
                h0=float(source.nominal_height_mm),
                allowed_rotations_deg=source.allowed_rotations_deg,
            )
        )
    return solver_parts, by_uid


def _build_solver_config(request: PackingRequest, parts: list[Part]):
    cfg = CFG()
    cfg.CASE_NAME = request.case_id
    cfg.BOARD_W = float(request.board_width_mm)
    cfg.BOARD_H = float(request.board_height_mm)
    cfg.TRIM = float(request.trim_mm)
    cfg.TOOL_D = float(request.tool_diameter_mm)
    cfg.GAP = float(request.tool_diameter_mm)
    cfg.BINARY_SPACING = False
    cfg.SAFE_GAP = 0.0
    cfg.ALLOW_ROT = bool(request.allow_rotation)
    cfg.PREPROCESS_ENABLE = False
    cfg.SOLVER_TIME_LIMIT_S = float(request.time_limit_s)
    processed, _ = preprocess_parts(parts, cfg)
    return processed, _variant_cfg(cfg, request.solver_variant, _case_stats(processed, cfg))


def _solve_with_config(parts: list[Part], cfg, seed: int) -> SearchResult:
    solver_kind = str(getattr(cfg, "SOLVER_KIND", "rh_mcts"))
    if solver_kind == "baseline_greedy":
        return _solve_baseline_greedy(parts, cfg, seed)
    if solver_kind != "rh_mcts":
        raise ValueError(f"unsupported handoff solver kind: {solver_kind}")
    return solve_with_receding_horizon_mcts(parts, cfg, seed)


def _convert_result(
    request: PackingRequest,
    result: SearchResult,
    by_uid: Mapping[int, PackingPart],
    runtime_s: float,
) -> PackingHandoff:
    radius = float(request.tool_diameter_mm) / 2.0
    placements: list[PackingPlacement] = []
    for board in result.boards:
        for placed in board.placed:
            source = by_uid[int(placed.uid)]
            rotation_deg = 90 if bool(placed.rotated) else 0
            if rotation_deg not in _effective_allowed_rotations(source, request.allow_rotation):
                raise RuntimeError(
                    f"Returned placement rotation is not permitted: part_id={source.part_id} rotation={rotation_deg}"
                )
            width = source.nominal_height_mm if placed.rotated else source.nominal_width_mm
            height = source.nominal_width_mm if placed.rotated else source.nominal_height_mm
            placements.append(
                PackingPlacement(
                    panel_id=int(board.bid),
                    part_id=source.part_id,
                    source_part_id=source.source_part_id,
                    x_mm=float(placed.rect.x) + radius,
                    y_mm=float(placed.rect.y) + radius,
                    nominal_width_mm=float(width),
                    nominal_height_mm=float(height),
                    rotated=bool(placed.rotated),
                )
            )
    placements.sort(key=lambda item: (item.panel_id, item.part_id))
    observed = {item.part_id for item in placements}
    expected = {item.part_id for item in request.parts}
    if len(placements) != len(request.parts) or observed != expected:
        raise RuntimeError("packing handoff lost or duplicated a requested part")

    sequence = tuple(by_uid[int(uid)].part_id for uid in result.sequence)
    return PackingHandoff(
        schema_version=PACKING_HANDOFF_SCHEMA_VERSION,
        feasible=True,
        panel_count=len(result.boards),
        placements=tuple(placements),
        sequence=sequence,
        score_global=tuple(float(value) for value in result.score_global),
        runtime_s=float(runtime_s),
        provenance={
            "case_id": request.case_id,
            "solver_variant": request.solver_variant,
            "seed": int(request.seed),
            "geometry_semantics": request.geometry_semantics,
            "tool_diameter_mm": float(request.tool_diameter_mm),
            "request": asdict(request),
            "solver_meta": dict(result.meta),
        },
    )


def solve_packing_handoff(request: PackingRequest) -> PackingHandoff:
    """Run the published nesting solver and return nominal, kerf-valid placements."""

    _validate_request(request)
    parts, by_uid = _build_solver_input(request)
    processed, cfg = _build_solver_config(request, parts)
    started = perf_counter()
    result = _solve_with_config(processed, cfg, int(request.seed))
    runtime_s = perf_counter() - started
    validate_solution(
        processed,
        result.boards,
        cfg.BOARD_W,
        cfg.BOARD_H,
        trim=cfg.TRIM,
        safe_gap=0.0,
        touch_tol=cfg.TOUCH_TOL,
    )
    return _convert_result(request, result, by_uid, runtime_s)


def solve_packing_handoff_with_budget(
    request: PackingRequest,
    budget: TotalBudget,
) -> PackingHandoff:
    """Run the handoff path under one caller-created absolute total budget.

    The caller must create ``budget`` before reading or constructing the formal
    input. This function is the calibration entry point; formal batch execution
    must additionally write its minimal result with
    :func:`inrp.budget.atomic_write_json_with_budget`.
    """

    budget.require_open("request_validation")
    _validate_request(request)
    budget.mark("request_validated")
    parts, by_uid = _build_solver_input(request)
    processed, cfg = _build_solver_config(request, parts)
    cfg.FORMAL_TOTAL_DEADLINE = float(budget.deadline)
    cfg.FORMAL_CLOSING_RESERVE_S = float(budget.closing_reserve_s)
    budget.mark("solver_input_built")
    result = _solve_with_config(processed, cfg, int(request.seed))
    budget.require_open("result_validation")
    budget.mark("solver_returned")
    validate_solution(
        processed,
        result.boards,
        cfg.BOARD_W,
        cfg.BOARD_H,
        trim=cfg.TRIM,
        safe_gap=0.0,
        touch_tol=cfg.TOUCH_TOL,
    )
    handoff = _convert_result(request, result, by_uid, budget.elapsed_s())
    budget.require_open("result_conversion")
    budget.mark("usable_handoff_ready")
    return handoff
