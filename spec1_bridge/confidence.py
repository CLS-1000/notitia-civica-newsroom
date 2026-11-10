"""ICD 203 analytic confidence wording.

Intelligence Community Directive 203 fixes the words an analyst may use for a
probability. This maps a numeric score onto those words so copy never says
"likely" for a 30% estimate.
"""

from __future__ import annotations

import math
from typing import Final

# (lower bound, upper bound, term) in percent. Each band includes its lower bound
# and excludes its upper bound, except the last, which runs to 100%. Scores under
# 1% fall in the first band, because ICD 203 has no word below "almost no chance".
ICD203_BANDS: Final = (
    (1, 5, "almost no chance"),
    (5, 20, "very unlikely"),
    (20, 45, "unlikely"),
    (45, 55, "roughly even chance"),
    (55, 80, "likely"),
    (80, 95, "very likely"),
    (95, 99, "almost certain"),
)
# Rounding before comparison keeps 0.55 * 100 == 55.000000000000007 in its band.
_PRECISION: Final = 9


def _to_percent(score: float, is_percentage: bool) -> float:
    """Return score in percent. Out-of-range scores raise; nothing is clamped."""
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise TypeError(f"score must be a number, not {type(score).__name__}")
    ceiling = 100.0 if is_percentage else 1.0
    if math.isnan(score) or not 0.0 <= score <= ceiling:
        unit = "percentage" if is_percentage else "probability"
        hint = "" if is_percentage else "; pass is_percentage=True for percentages"
        raise ValueError(
            f"score must be a {unit} in [0.0, {ceiling}], got {score}{hint}"
        )
    percent = float(score) if is_percentage else score * 100.0
    return round(percent, _PRECISION)


def format_icd203_confidence(
    score: float, is_percentage: bool = False, include_range: bool = False
) -> str:
    """Return the ICD 203 term for score, optionally with its band.

    score is a probability in [0.0, 1.0] unless is_percentage is True, in which
    case it is a percentage in [0.0, 100.0]. A score outside its unit's range
    raises ValueError. With include_range, 0.72 gives "likely (55–80%)".
    """
    percent = _to_percent(score, is_percentage)
    lower, upper, term = next(
        (band for band in ICD203_BANDS if percent < band[1]), ICD203_BANDS[-1]
    )
    if not include_range:
        return term
    return f"{term} ({lower:02d}–{upper:02d}%)"
