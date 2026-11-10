"""Five-Mask Standards: ICD 203 confidence wording and the uncertainty gate."""

from __future__ import annotations

import math

import pytest

from newsroom.dors_venabili.gatekeeper import verify_uncertainty_disclosures
from spec1_bridge.confidence import format_icd203_confidence

# --- ICD 203 confidence bands -------------------------------------------------


@pytest.mark.parametrize(
    ("score", "term"),
    [
        (0.01, "almost no chance"),
        (0.04, "almost no chance"),
        (0.05, "very unlikely"),
        (0.19, "very unlikely"),
        (0.20, "unlikely"),
        (0.44, "unlikely"),
        (0.45, "roughly even chance"),
        (0.50, "roughly even chance"),
        (0.55, "likely"),
        (0.79, "likely"),
        (0.80, "very likely"),
        (0.94, "very likely"),
        (0.95, "almost certain"),
        (0.99, "almost certain"),
    ],
)
def test_every_band_from_a_probability(score: float, term: str) -> None:
    assert format_icd203_confidence(score) == term


@pytest.mark.parametrize(
    ("score", "term"),
    [
        (3.0, "almost no chance"),
        (12.5, "very unlikely"),
        (30.0, "unlikely"),
        (50.0, "roughly even chance"),
        (72.0, "likely"),
        (90.0, "very likely"),
        (97.0, "almost certain"),
    ],
)
def test_every_band_from_a_percentage(score: float, term: str) -> None:
    assert format_icd203_confidence(score, is_percentage=True) == term


@pytest.mark.parametrize(
    ("score", "term"),
    [
        pytest.param(0.0, "almost no chance", id="zero"),
        pytest.param(0.001, "almost no chance", id="below-one-percent"),
        pytest.param(1.0, "almost certain", id="one"),
        pytest.param(0.995, "almost certain", id="above-ninety-nine"),
        pytest.param(0, "almost no chance", id="integer-zero"),
        pytest.param(1, "almost certain", id="integer-one"),
    ],
)
def test_edge_probabilities(score: float, term: str) -> None:
    assert format_icd203_confidence(score) == term


@pytest.mark.parametrize(
    ("score", "term"),
    [
        pytest.param(0.0, "almost no chance", id="zero"),
        pytest.param(1.0, "almost no chance", id="one-percent"),
        pytest.param(99.5, "almost certain", id="above-ninety-nine"),
        pytest.param(100.0, "almost certain", id="hundred"),
        pytest.param(100, "almost certain", id="integer"),
    ],
)
def test_edge_percentages(score: float, term: str) -> None:
    assert format_icd203_confidence(score, is_percentage=True) == term


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "almost no chance (01–05%)"),
        (0.1, "very unlikely (05–20%)"),
        (0.3, "unlikely (20–45%)"),
        (0.5, "roughly even chance (45–55%)"),
        (0.72, "likely (55–80%)"),
        (0.85, "very likely (80–95%)"),
        (1.0, "almost certain (95–99%)"),
    ],
)
def test_formatted_ranges(score: float, expected: str) -> None:
    assert format_icd203_confidence(score, include_range=True) == expected


def test_percentage_range_is_formatted() -> None:
    assert (
        format_icd203_confidence(72.0, is_percentage=True, include_range=True)
        == "likely (55–80%)"
    )


def test_probability_and_percentage_agree() -> None:
    assert format_icd203_confidence(0.72) == format_icd203_confidence(
        72.0, is_percentage=True
    )


@pytest.mark.parametrize("score", [72.0, 1.01, 100.0])
def test_percentage_without_the_flag_is_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="is_percentage=True"):
        format_icd203_confidence(score)


@pytest.mark.parametrize("score", [-0.01, math.inf, -math.inf, math.nan])
def test_out_of_range_probabilities_are_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        format_icd203_confidence(score)


@pytest.mark.parametrize("score", [-0.01, 100.01, math.inf, math.nan])
def test_out_of_range_percentages_are_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="percentage"):
        format_icd203_confidence(score, is_percentage=True)


@pytest.mark.parametrize("score", [True, "0.5", None])
def test_non_numbers_are_rejected(score: object) -> None:
    with pytest.raises(TypeError):
        format_icd203_confidence(score)  # type: ignore[arg-type]


# --- Uncertainty disclosure gate ----------------------------------------------


@pytest.mark.parametrize(
    "disclosures",
    [None, [], [""], ["   ", "\n\t"]],
    ids=["none", "empty", "empty-string", "whitespace"],
)
def test_missing_disclosures_are_blocked(disclosures: list[str] | None) -> None:
    issues = verify_uncertainty_disclosures(disclosures)
    assert len(issues) == 1
    assert "no uncertainty disclosures" in issues[0]


@pytest.mark.parametrize(
    "disclosures",
    [
        ["The audit period is not stated in the filing."],
        ["", "The vendor has not confirmed the contract value."],
    ],
    ids=["one", "blank-beside-real"],
)
def test_real_disclosures_pass(disclosures: list[str]) -> None:
    assert verify_uncertainty_disclosures(disclosures) == []


@pytest.mark.parametrize("disclosures", [None, [], ["  "]])
def test_affirmative_override_passes(disclosures: list[str] | None) -> None:
    assert (
        verify_uncertainty_disclosures(disclosures, no_open_questions_confirmed=True)
        == []
    )


def test_override_is_off_by_default() -> None:
    assert verify_uncertainty_disclosures([]) != []
