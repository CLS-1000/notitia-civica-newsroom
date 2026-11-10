"""Five-Mask Standards: ICD 203 confidence wording and the uncertainty gate."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import Conn
from fastapi.testclient import TestClient
from psycopg import errors
from test_phase2_workstation import (
    BASE_URL,
    FakeAnthropic,
    article_row,
    drafter,
    form,
    seed,
    sign,
)
from test_phase3_export import export

from newsroom.dors_venabili.gatekeeper import verify_uncertainty_disclosures
from newsroom.harlan_eternity.signoff import (
    CorrectionError,
    SignOffError,
    canonical_proof_hash,
    record_correction,
)
from newsroom.penfield_desk.app import create_app
from spec1_bridge.confidence import format_icd203_confidence

SLUG = "budget-shortfall"
ORIGINAL = "County records list 14 closed sites."
CORRECTED = "County records list 15 closed sites."

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


# --- Sign-off repeats the uncertainty gate --------------------------------------


def test_sign_off_refuses_an_article_with_no_disclosures(db: Conn) -> None:
    seed(db, disclosures="[]")
    with pytest.raises(SignOffError) as raised:
        sign(db)
    assert any(issue.startswith("uncertainty:") for issue in raised.value.issues)
    assert article_row(db) == ("drafting", "rule_based", False, None)


def test_sign_off_refuses_blank_disclosures_passed_in(db: Conn) -> None:
    seed(db)
    with pytest.raises(SignOffError, match="uncertainty"):
        sign(db, uncertainty_disclosures=["", "  "])


def test_stored_confirmation_does_not_count_at_sign_off(db: Conn) -> None:
    seed(db, disclosures="[]")
    db.execute("UPDATE articles SET no_open_questions_confirmed = TRUE")
    with pytest.raises(SignOffError, match="uncertainty"):
        sign(db)


def test_sign_off_saves_the_confirmation(db: Conn) -> None:
    seed(db, disclosures="[]")
    result = sign(db, no_open_questions_confirmed=True)
    assert result["status"] == "published"
    row = db.execute(
        "SELECT uncertainty_disclosures, no_open_questions_confirmed FROM articles"
    ).fetchone()
    assert row == ([], True)


def test_sign_off_saves_the_disclosures_it_checked(db: Conn) -> None:
    seed(db, disclosures="[]")
    sign(db, uncertainty_disclosures=["  The vendor has not responded.  ", ""])
    row = db.execute(
        "SELECT uncertainty_disclosures, no_open_questions_confirmed FROM articles"
    ).fetchone()
    assert row == (["The vendor has not responded."], False)


def test_sign_off_defaults_to_the_stored_disclosures(db: Conn) -> None:
    seed(db)
    assert sign(db)["uncertainty_disclosures"] == ["The audit period is not stated."]


# --- Corrections are append-only ------------------------------------------------


def published(db: Conn) -> dict[str, Any]:
    seed(db)
    return sign(db)


def correct(db: Conn, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "correction_type": "factual_correction",
        "explanation": "The county revised its count on Oct. 20.",
        "editor_signature": "Desk Editor",
        "original_text": ORIGINAL,
        "corrected_text": CORRECTED,
    } | overrides
    return record_correction(SLUG, conn=db, **values)


def test_correction_leaves_the_signed_article_untouched(db: Conn) -> None:
    signed = published(db)
    before = db.execute("SELECT * FROM articles").fetchone()

    result = correct(db)

    assert db.execute("SELECT * FROM articles").fetchone() == before
    assert result["correction_type"] == "factual_correction"
    row = db.execute(
        "SELECT body_markdown, source_ids, editor_name, signed_at, "
        "editorial_proof_hash FROM articles"
    ).fetchone()
    assert row is not None
    assert canonical_proof_hash(*row[:4]) == row[4] == signed["editorial_proof_hash"]


def test_corrections_accumulate(db: Conn) -> None:
    published(db)
    correct(db)
    correct(
        db,
        correction_type="methodology_update",
        explanation="We now count sites from the state register.",
        original_text=None,
        corrected_text=None,
    )
    assert db.execute("SELECT count(*) FROM article_corrections").fetchone() == (2,)


@pytest.mark.parametrize(
    ("statement", "operation"),
    [
        ("UPDATE article_corrections SET explanation = 'reworded'", "UPDATE"),
        ("DELETE FROM article_corrections", "DELETE"),
        ("TRUNCATE article_corrections", "TRUNCATE"),
    ],
)
def test_corrections_cannot_be_changed_or_removed(
    db: Conn, statement: str, operation: str
) -> None:
    published(db)
    correct(db)
    with pytest.raises(errors.RestrictViolation, match=operation):
        db.execute(statement)
    assert db.execute("SELECT count(*) FROM article_corrections").fetchone() == (1,)


def test_corrected_article_cannot_be_deleted(db: Conn) -> None:
    published(db)
    correct(db)
    with pytest.raises(errors.RestrictViolation, match="article_corrections"):
        db.execute("DELETE FROM articles")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"correction_type": "retcon"}, "unknown correction type", id="type"
        ),
        pytest.param(
            {"explanation": "  "}, "explanation is required", id="explanation"
        ),
        pytest.param({"editor_signature": ""}, "signature is required", id="signature"),
        pytest.param({"corrected_text": ""}, "original and corrected", id="factual"),
        pytest.param(
            {"original_text": "County records list 99 closed sites."},
            "does not appear in the signed article",
            id="misquoted",
        ),
    ],
)
def test_bad_corrections_are_refused(
    db: Conn, overrides: dict[str, Any], message: str
) -> None:
    published(db)
    with pytest.raises(CorrectionError, match=message):
        correct(db, **overrides)
    assert db.execute("SELECT count(*) FROM article_corrections").fetchone() == (0,)


def test_database_requires_a_factual_correction_to_show_the_change(db: Conn) -> None:
    published(db)
    with pytest.raises(errors.CheckViolation):
        db.execute(
            """
            INSERT INTO article_corrections
                (article_id, correction_type, explanation, editor_signature)
            SELECT article_id, 'factual_correction', 'x', 'Desk Editor' FROM articles
            """
        )


def test_draft_cannot_be_corrected(db: Conn) -> None:
    seed(db)
    with pytest.raises(CorrectionError, match="only published"):
        correct(db)


def test_unknown_article_cannot_be_corrected(db: Conn) -> None:
    with pytest.raises(CorrectionError, match="no article"):
        correct(db)


def test_clarification_needs_no_text_change(db: Conn) -> None:
    published(db)
    result = correct(
        db,
        correction_type="clarification",
        original_text="  ",
        corrected_text=None,
    )
    assert (result["original_text"], result["corrected_text"]) == (None, None)


# --- Desk -----------------------------------------------------------------------


@pytest.fixture
def desk(ledger_dsn: str, db: Conn) -> Iterator[TestClient]:
    app = create_app(ledger_dsn, lambda: drafter(FakeAnthropic()))
    with TestClient(app, base_url=BASE_URL) as client:
        yield client


def test_desk_disclosures_typed_in_the_editor_pass_the_gate(
    desk: TestClient, db: Conn
) -> None:
    seed(db, disclosures="[]")
    data = form(uncertainty_disclosures="The vendor has not responded.\n\n")
    assert desk.post(f"/articles/{SLUG}/sign", data=data).status_code == 200
    row = db.execute("SELECT uncertainty_disclosures FROM articles").fetchone()
    assert row == (["The vendor has not responded."],)


def test_desk_clearing_the_disclosures_blocks_sign_off(
    desk: TestClient, db: Conn
) -> None:
    seed(db)
    data = form(uncertainty_disclosures="  \n")
    assert desk.post(f"/articles/{SLUG}/sign", data=data).status_code == 422


def test_desk_save_keeps_disclosures_and_confirmation(
    desk: TestClient, db: Conn
) -> None:
    seed(db)
    data = form(uncertainty_disclosures="One.\nTwo.", no_open_questions_confirmed="on")
    assert desk.post(f"/articles/{SLUG}/save", data=data).status_code == 204
    row = db.execute(
        "SELECT uncertainty_disclosures, no_open_questions_confirmed FROM articles"
    ).fetchone()
    assert row == (["One.", "Two."], True)
    page = desk.get(f"/?article={SLUG}").text
    assert 'name="no_open_questions_confirmed" checked' in page


def test_desk_records_a_correction(desk: TestClient, db: Conn) -> None:
    published(db)
    page = desk.get(f"/?article={SLUG}").text
    assert f'hx-post="/articles/{SLUG}/corrections"' in page

    data = {
        "correction_type": "factual_correction",
        "original_text": ORIGINAL,
        "corrected_text": CORRECTED,
        "explanation": "The county revised its count.",
        "editor_signature": "Desk Editor",
    }
    response = desk.post(f"/articles/{SLUG}/corrections", data=data)
    assert response.status_code == 200
    assert CORRECTED in response.text
    assert db.execute("SELECT count(*) FROM article_corrections").fetchone() == (1,)


def test_desk_refuses_a_bad_correction(desk: TestClient, db: Conn) -> None:
    published(db)
    data = {
        "correction_type": "clarification",
        "explanation": "",
        "editor_signature": "",
    }
    response = desk.post(f"/articles/{SLUG}/corrections", data=data)
    assert response.status_code == 422
    assert "Correction refused" in response.text


def test_drafts_show_no_correction_form(desk: TestClient, db: Conn) -> None:
    seed(db)
    assert "/corrections" not in desk.get(f"/?article={SLUG}").text


# --- Export ---------------------------------------------------------------------


def test_export_writes_corrections_for_readers(
    ledger_dsn: str, db: Conn, tmp_path: Path
) -> None:
    published(db)
    correct(db)
    assert export.main(["--dsn", ledger_dsn, "--out", str(tmp_path)]) == 0

    corrections = json.loads((tmp_path / "corrections.json").read_text())
    assert len(corrections) == 1
    item = corrections[0]
    assert item["slug"] == SLUG
    assert item["correction_type"] == "factual_correction"
    assert (item["before"], item["after"]) == (ORIGINAL, CORRECTED)
    assert item["reason"] == "The county revised its count on Oct. 20."
    assert item["editor_name"] == "Desk Editor"
    assert set(item) == {
        "correction_id",
        "slug",
        "headline",
        "correction_type",
        "corrected_at",
        "reason",
        "before",
        "after",
        "editor_name",
    }
    # The article still exports, because its proof hash still matches.
    articles = json.loads((tmp_path / "articles.json").read_text())
    assert [article["slug"] for article in articles] == [SLUG]


def test_export_writes_an_empty_corrections_list(
    ledger_dsn: str, db: Conn, tmp_path: Path
) -> None:
    published(db)
    assert export.main(["--dsn", ledger_dsn, "--out", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "corrections.json").read_text()) == []


def test_corrections_to_withheld_articles_are_withheld(db: Conn) -> None:
    published(db)
    correct(db)
    db.execute("UPDATE articles SET body_markdown = body_markdown || ' edited'")

    articles, withheld = export.collect_articles(db)
    assert articles == [] and len(withheld) == 1
    assert export.collect_corrections(db, [item["slug"] for item in articles]) == []
