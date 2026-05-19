from __future__ import annotations

from typing import Any


def _variables_by_name(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not payload:
        return {}
    return {item["name"]: dict(item) for item in payload.get("variables", []) if item.get("name")}


def _to_space(variable: dict[str, Any]) -> dict[str, Any]:
    focus = variable.get("focus_range") or variable.get("focus") or variable.get("bounds")
    bounds = variable.get("bounds")
    if not bounds or len(bounds) != 2:
        raise ValueError(f"invalid bounds for variable {variable.get('name')}")
    if not focus or len(focus) != 2:
        raise ValueError(f"invalid focus_range for variable {variable.get('name')}")
    lower, upper = float(bounds[0]), float(bounds[1])
    focus_lower, focus_upper = float(focus[0]), float(focus[1])
    if lower >= upper:
        raise ValueError(f"bounds lower must be < upper for {variable.get('name')}")
    if focus_lower > focus_upper:
        raise ValueError(f"focus_range lower must be <= upper for {variable.get('name')}")
    if focus_lower < lower or focus_upper > upper:
        raise ValueError(f"focus_range must stay within bounds for {variable.get('name')}")
    return {
        "bounds": [lower, upper],
        "focus": [focus_lower, focus_upper],
        "unit": variable.get("unit", ""),
    }


def merge_space(
    defaults: dict[str, Any],
    knowledge_rules: dict[str, Any] | None,
    researcher_config: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    merged = _variables_by_name(defaults)
    source = {name: "defaults" for name in merged}

    for name, variable in _variables_by_name(knowledge_rules).items():
        merged[name] = variable
        source[name] = "literature"

    for name, variable in _variables_by_name(researcher_config).items():
        merged[name] = variable
        source[name] = "researcher"

    if not merged:
        raise ValueError("parameter space cannot be empty")

    space = {name: _to_space({"name": name, **variable}) for name, variable in merged.items()}
    return space, source
