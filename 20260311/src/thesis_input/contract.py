"""Validation and canonicalization for thesis-manufacturing-input-v1."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "thesis-manufacturing-input-v1"
_FACT_STATUSES = {"verified", "model_declared", "unresolved"}
_GRAIN_AXES = {"length", "width", "none"}
_MARGIN_KEYS = ("left_mm", "right_mm", "top_mm", "bottom_mm")


def load_bundle(path: Path, *, require_verified: bool = False) -> dict:
    """Load a JSON bundle and return independently-owned normalized data."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load manufacturing input bundle: {path}") from error
    return validate_bundle(payload, require_verified=require_verified)


def validate_bundle(payload: dict, *, require_verified: bool = False) -> dict:
    """Fail closed unless every manufacturing semantic is explicit and coherent."""

    if not isinstance(payload, Mapping):
        raise ValueError("manufacturing input must be a JSON object")
    _require_keys(
        payload,
        {"schema_version", "task_id", "source_kind", "source", "stock", "tool", "parts"},
        allowed_derived={"instances", "summary"},
        name="manufacturing input",
    )
    _require_exact_value(payload, "schema_version", SCHEMA_VERSION)
    task_id = _nonempty_text(payload.get("task_id"), "task_id")
    source_kind = payload.get("source_kind")
    if source_kind not in {"synthetic", "file"}:
        raise ValueError("source_kind must be synthetic or file")
    source = _normalize_source(payload.get("source"), source_kind)
    evidence = _evidence_locator(source_kind, source)

    stock_raw = _mapping(payload.get("stock"), "stock")
    _require_keys(stock_raw, {"length_mm", "width_mm", "thickness_mm", "material", "grain_axis", "margin_mm"}, name="stock")
    stock = {
        "length_mm": _fact(stock_raw.get("length_mm"), "stock.length_mm", evidence, require_verified, numeric=True),
        "width_mm": _fact(stock_raw.get("width_mm"), "stock.width_mm", evidence, require_verified, numeric=True),
        "thickness_mm": _fact(stock_raw.get("thickness_mm"), "stock.thickness_mm", evidence, require_verified, numeric=True),
        "material": _fact(stock_raw.get("material"), "stock.material", evidence, require_verified, text=True),
        "grain_axis": _fact(stock_raw.get("grain_axis"), "stock.grain_axis", evidence, require_verified, choice=_GRAIN_AXES),
        "margin_mm": _normalize_margins(stock_raw.get("margin_mm"), evidence, require_verified),
    }
    margin_values = [Decimal(stock["margin_mm"][key]["value"]) for key in _MARGIN_KEYS]
    if any(value != margin_values[0] for value in margin_values[1:]):
        raise ValueError("only symmetric stock margins are supported")
    if margin_values[0] * 2 >= min(Decimal(stock["length_mm"]["value"]), Decimal(stock["width_mm"]["value"])):
        raise ValueError("stock margin leaves no usable area")

    tool_raw = _mapping(payload.get("tool"), "tool")
    _require_keys(tool_raw, {"diameter_mm", "geometry_rule"}, name="tool")
    tool = {
        "diameter_mm": _fact(tool_raw.get("diameter_mm"), "tool.diameter_mm", evidence, require_verified, numeric=True),
        "geometry_rule": _fact(tool_raw.get("geometry_rule"), "tool.geometry_rule", evidence, require_verified, choice={"nominal_with_tool_clearance"}),
    }
    parts_raw = payload.get("parts")
    if not isinstance(parts_raw, list) or not parts_raw:
        raise ValueError("parts must be a nonempty list")

    parts: list[dict] = []
    seen_ids: set[str] = set()
    seen_rows: set[int] = set()
    for index, value in enumerate(parts_raw, 1):
        part_raw = _mapping(value, f"parts[{index}]")
        _require_keys(
            part_raw,
            {
                "part_id",
                "source_row",
                "quantity",
                "original_fields",
                "dimension_basis",
                "length_mm",
                "width_mm",
                "thickness_mm",
                "material",
                "grain_axis",
                "allowed_rotations_deg",
            },
            name=f"parts[{index}]",
        )
        part_id = _nonempty_text(part_raw.get("part_id"), f"parts[{index}].part_id")
        if part_id in seen_ids:
            raise ValueError("part_id values must be unique")
        seen_ids.add(part_id)
        source_row = _positive_int(part_raw.get("source_row"), f"parts[{index}].source_row")
        if source_row in seen_rows:
            raise ValueError("source_row values must be unique")
        seen_rows.add(source_row)
        quantity = _positive_int(part_raw.get("quantity"), f"parts[{index}].quantity")
        original_fields = _mapping(part_raw.get("original_fields"), f"parts[{index}].original_fields")
        _validate_original_fields(original_fields)
        part = {
            "part_id": part_id,
            "source_row": source_row,
            "quantity": quantity,
            "original_fields": deepcopy(dict(original_fields)),
            "dimension_basis": _fact(part_raw.get("dimension_basis"), f"parts[{index}].dimension_basis", evidence, require_verified, choice={"synthetic_defined", "original_dimensions"}),
            "length_mm": _fact(part_raw.get("length_mm"), f"parts[{index}].length_mm", evidence, require_verified, numeric=True),
            "width_mm": _fact(part_raw.get("width_mm"), f"parts[{index}].width_mm", evidence, require_verified, numeric=True),
            "thickness_mm": _fact(part_raw.get("thickness_mm"), f"parts[{index}].thickness_mm", evidence, require_verified, numeric=True),
            "material": _fact(part_raw.get("material"), f"parts[{index}].material", evidence, require_verified, text=True),
            "grain_axis": _fact(part_raw.get("grain_axis"), f"parts[{index}].grain_axis", evidence, require_verified, choice=_GRAIN_AXES),
            "allowed_rotations_deg": _fact(part_raw.get("allowed_rotations_deg"), f"parts[{index}].allowed_rotations_deg", evidence, require_verified, rotations=True),
        }
        _validate_dimension_basis(part, source_kind)
        if part["thickness_mm"]["value"] != stock["thickness_mm"]["value"]:
            raise ValueError("part thickness must match homogeneous stock")
        if part["material"]["value"] != stock["material"]["value"]:
            raise ValueError("part material must match homogeneous stock")
        _validate_grain_permissions(part, stock)
        parts.append(part)

    instances = []
    for part in parts:
        for copy_index in range(1, part["quantity"] + 1):
            instances.append(
                {
                    "instance_id": f"{part['part_id']}__copy_{copy_index:03d}",
                    "source_part_id": part["part_id"],
                    "source_row": part["source_row"],
                    "copy_index": copy_index,
                    "length_mm": part["length_mm"],
                    "width_mm": part["width_mm"],
                    "allowed_rotations_deg": part["allowed_rotations_deg"],
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "source_kind": source_kind,
        "source": source,
        "stock": stock,
        "tool": tool,
        "parts": parts,
        "instances": instances,
        "summary": {
            "source_quantity_total": sum(part["quantity"] for part in parts),
            "expanded_quantity_total": len(instances),
        },
    }


def problem_hash(bundle: dict) -> str:
    """Hash only manufacturing semantics used by packing and evaluation.

    Evidence status/locators, source rows, and uninterpreted raw labels are
    intentionally excluded.  Those belong to :func:`source_manifest_hash`.
    """

    normalized = validate_bundle(bundle)
    return _projection_hash(_problem_projection(normalized))


def source_manifest_hash(bundle: dict) -> str:
    """Hash source locators, row/copy provenance, raw fields, and evidence."""

    normalized = validate_bundle(bundle)
    return _projection_hash(_source_manifest_projection(normalized))


def _problem_projection(normalized: Mapping[str, Any]) -> dict[str, Any]:
    stock = normalized["stock"]
    tool = normalized["tool"]
    parts = []
    instances_by_source: dict[str, list[str]] = {}
    for instance in normalized["instances"]:
        instances_by_source.setdefault(instance["source_part_id"], []).append(
            instance["instance_id"]
        )
    for part in sorted(normalized["parts"], key=lambda item: item["part_id"]):
        parts.append(
            {
                "part_id": part["part_id"],
                "instance_ids": sorted(instances_by_source[part["part_id"]]),
                "quantity": part["quantity"],
                "length_mm": part["length_mm"]["value"],
                "width_mm": part["width_mm"]["value"],
                "thickness_mm": part["thickness_mm"]["value"],
                "material": part["material"]["value"],
                "grain_axis": part["grain_axis"]["value"],
                "allowed_rotations_deg": part["allowed_rotations_deg"]["value"],
            }
        )
    return {
        "schema_version": normalized["schema_version"],
        "task_id": normalized["task_id"],
        "stock": {
            "length_mm": stock["length_mm"]["value"],
            "width_mm": stock["width_mm"]["value"],
            "thickness_mm": stock["thickness_mm"]["value"],
            "material": stock["material"]["value"],
            "grain_axis": stock["grain_axis"]["value"],
            "margin_mm": {
                key: stock["margin_mm"][key]["value"] for key in _MARGIN_KEYS
            },
        },
        "tool": {
            "diameter_mm": tool["diameter_mm"]["value"],
            "geometry_rule": tool["geometry_rule"]["value"],
        },
        "parts": parts,
    }


def _source_manifest_projection(normalized: Mapping[str, Any]) -> dict[str, Any]:
    stock = normalized["stock"]
    tool = normalized["tool"]
    parts = []
    for part in sorted(normalized["parts"], key=lambda item: item["part_id"]):
        parts.append(
            {
                "part_id": part["part_id"],
                "source_row": part["source_row"],
                "quantity": part["quantity"],
                "original_fields": part["original_fields"],
                "dimension_basis": part["dimension_basis"],
                "fact_evidence": {
                    key: _fact_evidence(part[key])
                    for key in (
                        "length_mm",
                        "width_mm",
                        "thickness_mm",
                        "material",
                        "grain_axis",
                        "allowed_rotations_deg",
                    )
                },
            }
        )
    return {
        "schema_version": normalized["schema_version"],
        "task_id": normalized["task_id"],
        "source_kind": normalized["source_kind"],
        "source": normalized["source"],
        "stock_evidence": {
            "length_mm": _fact_evidence(stock["length_mm"]),
            "width_mm": _fact_evidence(stock["width_mm"]),
            "thickness_mm": _fact_evidence(stock["thickness_mm"]),
            "material": _fact_evidence(stock["material"]),
            "grain_axis": _fact_evidence(stock["grain_axis"]),
            "margin_mm": {
                key: _fact_evidence(stock["margin_mm"][key])
                for key in _MARGIN_KEYS
            },
        },
        "tool_evidence": {
            "diameter_mm": _fact_evidence(tool["diameter_mm"]),
            "geometry_rule": _fact_evidence(tool["geometry_rule"]),
        },
        "parts": parts,
        "instances": [
            {
                "instance_id": instance["instance_id"],
                "source_part_id": instance["source_part_id"],
                "source_row": instance["source_row"],
                "copy_index": instance["copy_index"],
            }
            for instance in sorted(
                normalized["instances"], key=lambda item: item["instance_id"]
            )
        ],
    }


def _fact_evidence(fact: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": fact["status"], "source": fact["source"]}


def _projection_hash(projection: object) -> str:
    encoded = json.dumps(
        projection,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_source(value: Any, source_kind: str) -> dict:
    source = _mapping(value, "source")
    if source_kind == "synthetic":
        _require_keys(source, {"synthetic_id"}, name="synthetic source")
        return {"synthetic_id": _nonempty_text(source.get("synthetic_id"), "source.synthetic_id")}
    _require_keys(source, {"file_sha256", "encoding"}, name="file source")
    digest = _nonempty_text(source.get("file_sha256"), "source.file_sha256").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("source.file_sha256 must be a SHA-256 hex digest")
    return {"file_sha256": digest, "encoding": _nonempty_text(source.get("encoding"), "source.encoding")}


def _evidence_locator(source_kind: str, source: Mapping[str, str]) -> str:
    return f"synthetic:{source['synthetic_id']}" if source_kind == "synthetic" else f"sha256:{source['file_sha256']}"


def _normalize_margins(value: Any, evidence: str, require_verified: bool) -> dict:
    raw = _mapping(value, "stock.margin_mm")
    if set(raw) != set(_MARGIN_KEYS):
        raise ValueError("stock.margin_mm requires four explicit sides")
    return {key: _fact(raw[key], f"stock.margin_mm.{key}", evidence, require_verified, numeric=True, nonnegative=True) for key in _MARGIN_KEYS}


def _fact(value: Any, name: str, evidence: str, require_verified: bool, *, numeric: bool = False, nonnegative: bool = False, text: bool = False, choice: set[str] | None = None, rotations: bool = False) -> dict:
    raw = _mapping(value, name)
    if set(raw) != {"value", "status", "source"}:
        raise ValueError(f"{name} must contain value, status, and source")
    status = raw["status"]
    if status not in _FACT_STATUSES:
        raise ValueError(f"{name}.status is unknown")
    if status == "unresolved":
        raise ValueError(f"{name} is unresolved and cannot run")
    if evidence.startswith("synthetic:") and status != "model_declared":
        raise ValueError(f"synthetic facts must be model_declared and cannot be verified: {name}")
    if require_verified and status != "verified":
        raise ValueError(f"{name} is not verified")
    if raw["source"] != evidence:
        raise ValueError(f"{name} has malformed or unknown evidence")
    if numeric:
        normalized_value = _decimal_text(raw["value"], name, nonnegative=nonnegative)
    elif text:
        normalized_value = _nonempty_text(raw["value"], f"{name}.value")
    elif choice is not None:
        normalized_value = raw["value"]
        if normalized_value not in choice:
            raise ValueError(f"{name}.value has unknown manufacturing semantics")
    elif rotations:
        normalized_value = _rotations(raw["value"], name)
    else:
        normalized_value = raw["value"]
    return {"value": normalized_value, "status": status, "source": evidence}


def _decimal_text(value: Any, name: str, *, nonnegative: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{name} must be a positive finite decimal")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite decimal") from error
    if not decimal.is_finite() or decimal < 0 or (not nonnegative and decimal <= 0):
        raise ValueError(f"{name} must be a positive finite decimal")
    return format(decimal.normalize(), "f")


def _rotations(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value or any(type(angle) is not int for angle in value):
        raise ValueError(f"{name}.value must be a nonempty angle list")
    if len(value) != len(set(value)) or any(angle not in {0, 90} for angle in value):
        raise ValueError(f"{name}.value has unsupported angles")
    return sorted(value)


def _validate_dimension_basis(part: Mapping[str, Any], source_kind: str) -> None:
    basis = part["dimension_basis"]["value"]
    if basis == "synthetic_defined":
        if source_kind != "synthetic":
            raise ValueError("synthetic_defined dimensions require synthetic source_kind")
        return
    original = part["original_fields"]
    if not isinstance(original.get("length_mm"), (str, int, float, Decimal)) or not isinstance(original.get("width_mm"), (str, int, float, Decimal)):
        raise ValueError("original_dimensions basis requires explicit original dimensions")
    if _decimal_text(original["length_mm"], "original length") != part["length_mm"]["value"] or _decimal_text(original["width_mm"], "original width") != part["width_mm"]["value"]:
        raise ValueError("dimension_basis does not agree with original dimensions")


def _validate_original_fields(original_fields: Mapping[str, Any]) -> None:
    for name in ("Graining", "Orientation"):
        if type(original_fields.get(name)) is not str:
            raise ValueError(f"original_fields.{name} must preserve a raw string")


def _validate_grain_permissions(part: Mapping[str, Any], stock: Mapping[str, Any]) -> None:
    part_axis = part["grain_axis"]["value"]
    stock_axis = stock["grain_axis"]["value"]
    allowed = set(part["allowed_rotations_deg"]["value"])
    if part_axis == "none" or stock_axis == "none":
        return
    compatible = {0} if part_axis == stock_axis else {90}
    if not allowed <= compatible:
        raise ValueError("allowed rotations contradict declared grain axes")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _require_exact_value(payload: Mapping[str, Any], name: str, expected: str) -> None:
    if payload.get(name) != expected:
        raise ValueError(f"{name} must be {expected}")


def _require_keys(
    value: Mapping[str, Any],
    required: set[str],
    *,
    name: str,
    allowed_derived: set[str] | None = None,
) -> None:
    allowed = required | (allowed_derived or set())
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
