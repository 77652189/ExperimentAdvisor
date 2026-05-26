from __future__ import annotations

from typing import Mapping

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (28.0, 37.0),
    "ph": (6.5, 7.5),
    "feed_amount": (50.0, 300.0),
    "feed_time": (6.0, 24.0),
    "induction_time": (12.0, 30.0),
    "inducer_dose": (0.1, 1.0),
}


def build_search_space(bounds: Mapping[str, tuple[float, float]] | None = None):
    """
    构建 Ax SearchSpace。

    默认包含六个连续工艺参数，并添加线性约束：
    feed_time - induction_time <= -2，即 induction_time >= feed_time + 2。
    """

    try:
        from ax.core.parameter import ParameterType, RangeParameter
        from ax.core.parameter_constraint import ParameterConstraint
        from ax.core.search_space import SearchSpace
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError("Ax is required. Install dependencies with: pip install -r requirements.txt") from exc

    effective_bounds = dict(DEFAULT_BOUNDS)
    if bounds is not None:
        effective_bounds.update({name: (float(low), float(high)) for name, (low, high) in bounds.items()})

    parameters = []
    for name, (lower, upper) in effective_bounds.items():
        if lower >= upper:
            raise ValueError(f"lower bound must be < upper bound for {name}")
        parameters.append(
            RangeParameter(
                name=name,
                parameter_type=ParameterType.FLOAT,
                lower=float(lower),
                upper=float(upper),
            )
        )

    try:
        constraint = ParameterConstraint(
            constraint_dict={"feed_time": 1.0, "induction_time": -1.0},
            bound=-2.0,
        )
    except TypeError:
        constraint = ParameterConstraint("feed_time - induction_time <= -2")
    return SearchSpace(parameters=parameters, parameter_constraints=[constraint])
