from __future__ import annotations

import operator
from typing import Any, Callable

SUPPORTED_OPS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}


def _check_single(constraint: dict[str, Any], params: dict[str, float]) -> bool:
    var = constraint.get("var")
    op = constraint.get("op")
    value = constraint.get("value")
    if var not in params:
        return False
    if op not in SUPPORTED_OPS:
        raise ValueError(f"unsupported constraint op: {op}")
    return SUPPORTED_OPS[op](float(params[var]), float(value))


def _check_compound(constraint: dict[str, Any], params: dict[str, float]) -> bool:
    conditions = constraint.get("conditions", [])
    logic = constraint.get("logic", "and")
    results = [_check_single(item, params) for item in conditions]
    if logic == "and":
        return all(results)
    if logic == "or":
        return any(results)
    raise ValueError(f"unsupported constraint logic: {logic}")


def is_valid(params: dict[str, float], hard_constraints: list[dict[str, Any]] | None) -> bool:
    for constraint in hard_constraints or []:
        hit = _check_compound(constraint, params) if "conditions" in constraint else _check_single(constraint, params)
        if hit:
            return False
    return True
