"""Five-Mask Standards: ICD 203 confidence wording and the uncertainty gate."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import Conn
from fastapi.testclient import TestClient
from psycopg import errors
from test_phase2_workstation import (
    BASE_URL,
    NOW,
    FakeAnthropic,
    article_row,
    drafter,
    form,
    seed,
    sign,
)
from test_phase3_export import export

from newsroom.dors_venabili.gatekeeper import (
    verify_right_of_reply_log,
    verify_uncertainty_disclosures,
)
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


# --- Right of reply: gate -------------------------------------------------------

DAY = timedelta(days=1)


def reply(
    name: str = "Example Agency",
    status: str = "responded",
    deadline: datetime = NOW + DAY,
) -> dict[str, Any]:
    return {"subject_name": name, "response_status": status, "deadline_at": deadline}


def reply_gate(
    records: list[dict[str, Any]],
    subjects: list[str] | None = None,
    **options: Any,
) -> list[str]:
    return verify_right_of_reply_log(records, subjects or [], now=NOW, **options)


def test_no_reply_record_blocks_without_the_attestation() -> None:
    issues = reply_gate([])
    assert len(issues) == 1
    assert "no right-of-reply request is logged" in issues[0]


def test_attestation_allows_an_article_with_no_reply_record() -> None:
    assert reply_gate([], no_allegations_attested=True) == []


def test_a_reply_record_needs_no_attestation() -> None:
    assert reply_gate([reply()]) == []


@pytest.mark.parametrize("badge", ["INVESTIGATION", "[INVESTIGATION]", "investigation"])
def test_investigations_cannot_use_the_attestation(badge: str) -> None:
    issues = reply_gate([], no_allegations_attested=True, badge_type=badge)
    assert any("not available here (badge INVESTIGATION)" in item for item in issues)
    assert any("no right-of-reply request is logged" in item for item in issues)


def test_investigation_passes_with_a_reply_record() -> None:
    assert reply_gate([reply()], badge_type="INVESTIGATION") == []


@pytest.mark.parametrize(
    "body",
    [
        "The treasurer embezzled $40,000. [S-1]",
        "The audit found misconduct by the board. [S-1]",
        "Records show a conflict of interest. [S-1]",
        'A resident said "the mayor is corrupt" at the meeting. [S-1]',
    ],
    ids=["allegation", "misconduct", "phrase", "inside-quote"],
)
def test_adverse_terms_disable_the_attestation(body: str) -> None:
    issues = reply_gate([], no_allegations_attested=True, body=body)
    assert any("not available here (adverse terms" in item for item in issues)


def test_adverse_terms_match_whole_words() -> None:
    body = "The fraudster-proofing pilot was praised; costs fell. [S-1]"
    assert reply_gate([], no_allegations_attested=True, body="Costs fell. [S-1]") == []
    assert reply_gate([], no_allegations_attested=True, body=body) == []


def test_attestation_cannot_coexist_with_named_subjects() -> None:
    issues = reply_gate(
        [reply("Mayor Ruiz")], ["Mayor Ruiz"], no_allegations_attested=True
    )
    assert issues == [
        "the no-allegations attestation contradicts the named subjects: Mayor Ruiz"
    ]


def test_blank_subject_list_cannot_bypass_the_log() -> None:
    assert reply_gate([], ["", "  "]) != []


def test_every_named_subject_needs_a_record() -> None:
    issues = reply_gate([reply("Mayor Ruiz")], ["Mayor Ruiz", "Acme Paving"])
    assert issues == ["no right-of-reply request logged for Acme Paving"]


def test_subject_names_match_ignoring_case_and_spacing() -> None:
    assert reply_gate([reply("acme  paving")], ["Acme Paving"]) == []


def test_named_subjects_without_any_record_are_listed() -> None:
    issues = reply_gate([], ["Mayor Ruiz", "Acme Paving"])
    assert issues == [
        (
            "allegations name Acme Paving, Mayor Ruiz, but no right-of-reply request "
            "is logged"
        )
    ]


def test_pending_reply_inside_its_window_blocks() -> None:
    issues = reply_gate([reply(status="pending", deadline=NOW + DAY)])
    assert issues == ["reply from Example Agency is pending until 2026-10-16 12:00 UTC"]


def test_pending_reply_after_its_deadline_counts_as_no_response() -> None:
    assert reply_gate([reply(status="pending", deadline=NOW - DAY)]) == []


def test_pending_reply_from_an_unnamed_subject_still_blocks() -> None:
    records = [reply("Mayor Ruiz"), reply("Acme Paving", "pending", NOW + DAY)]
    assert reply_gate(records, ["Mayor Ruiz"]) != []


@pytest.mark.parametrize("status", ["no_response_by_deadline", "declined_to_comment"])
def test_closed_requests_pass(status: str) -> None:
    assert reply_gate([reply(status=status)], ["Example Agency"]) == []


# --- Right of reply: ledger and sign-off ------------------------------------------


def log_reply(
    db: Conn,
    name: str = "Example Agency",
    status: str = "responded",
    deadline: datetime = NOW + DAY,
    summary: str | None = "a spokesperson said the figures are under review.",
) -> None:
    db.execute(
        """
        INSERT INTO article_reply_records
            (article_id, subject_name, subject_type, contact_method, requested_at,
             deadline_at, response_status, response_summary)
        SELECT article_id, %s, 'institution', 'email', %s, %s, %s, %s FROM articles
        """,
        (name, NOW - 2 * DAY, deadline, status, summary),
    )


def test_sign_off_blocks_without_a_reply_or_attestation(db: Conn) -> None:
    seed(db)
    with pytest.raises(SignOffError) as raised:
        sign(db, no_allegations_attested=False)
    assert any(item.startswith("right_of_reply:") for item in raised.value.issues)
    assert article_row(db)[0] == "drafting"


def test_stored_attestation_does_not_count_at_sign_off(db: Conn) -> None:
    seed(db)
    db.execute("UPDATE articles SET no_allegations_attested = TRUE")
    with pytest.raises(SignOffError, match="right_of_reply"):
        sign(db, no_allegations_attested=False)


def test_sign_off_saves_subjects_and_attestation(db: Conn) -> None:
    seed(db)
    log_reply(db, "Mayor Ruiz")
    result = sign(
        db, no_allegations_attested=False, allegation_subjects=[" Mayor Ruiz ", ""]
    )
    assert result["allegation_subjects"] == ["Mayor Ruiz"]
    row = db.execute(
        "SELECT allegation_subjects, no_allegations_attested FROM articles"
    ).fetchone()
    assert row == (["Mayor Ruiz"], False)


def test_sign_off_blocks_an_investigation_attested_without_a_reply(db: Conn) -> None:
    seed(db)
    db.execute("UPDATE articles SET badge_type = 'INVESTIGATION'")
    with pytest.raises(SignOffError, match="not available here"):
        sign(db)


def test_sign_off_blocks_adverse_terms_attested_without_a_reply(db: Conn) -> None:
    seed(db)
    body = "County records list 14 closed sites after the fraud. [S-2]"
    with pytest.raises(SignOffError, match="adverse terms 'fraud'"):
        sign(db, body_markdown=body, sorted_source_ids=["S-2"])


def test_sign_off_waits_for_a_pending_reply(db: Conn) -> None:
    seed(db)
    log_reply(db, status="pending", summary=None, deadline=NOW + DAY)
    with pytest.raises(SignOffError, match="pending until"):
        sign(db, no_allegations_attested=False)


def test_sign_off_proceeds_once_the_deadline_passes(db: Conn) -> None:
    seed(db)
    log_reply(db, status="pending", summary=None, deadline=NOW - timedelta(minutes=1))
    assert sign(db, no_allegations_attested=False)["status"] == "published"


@pytest.mark.parametrize(
    "statement",
    [
        (
            "INSERT INTO article_reply_records (article_id, subject_name, subject_type,"
            " contact_method, requested_at, deadline_at, response_status)"
            " SELECT article_id, 'Late Co', 'institution', 'email', now(),"
            " now() + interval '1 day', 'pending' FROM articles"
        ),
        "UPDATE article_reply_records SET response_summary = 'rewritten'",
        "DELETE FROM article_reply_records",
    ],
    ids=["insert", "update", "delete"],
)
def test_reply_log_freezes_at_publication(db: Conn, statement: str) -> None:
    seed(db)
    log_reply(db)
    sign(db, no_allegations_attested=False)
    with pytest.raises(errors.RestrictViolation, match="frozen after publication"):
        db.execute(statement.encode())


@pytest.mark.parametrize(
    ("overrides", "constraint"),
    [
        ({"deadline": NOW - 3 * DAY}, "reply_deadline_follows_request"),
        ({"summary": " "}, "reply_response_is_summarized"),
    ],
)
def test_reply_log_rejects_impossible_records(
    db: Conn, overrides: dict[str, Any], constraint: str
) -> None:
    seed(db)
    with pytest.raises(errors.CheckViolation, match=constraint):
        log_reply(db, **overrides)


# --- Right of reply: desk ---------------------------------------------------------


def reply_form(**overrides: str) -> dict[str, str]:
    return {
        "subject_name": "Acme  Paving",
        "subject_type": "institution",
        "contact_method": "formal_letter",
        "requested_at": "2026-10-10T09:00",
        "deadline_at": "2026-10-12T17:00",
        "response_status": "pending",
        "response_summary": "",
    } | overrides


def test_desk_logs_a_reply_request(desk: TestClient, db: Conn) -> None:
    seed(db)
    response = desk.post(f"/articles/{SLUG}/replies", data=reply_form())
    assert response.status_code == 200
    assert "Acme Paving" in response.text
    row = db.execute(
        "SELECT subject_name, contact_method, deadline_at FROM article_reply_records"
    ).fetchone()
    assert row == (
        "Acme Paving",
        "formal_letter",
        datetime(2026, 10, 12, 17, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"subject_name": " "}, "subject name is required"),
        ({"contact_method": "carrier_pigeon"}, "unknown contact method"),
        ({"deadline_at": "2026-10-09T09:00"}, "deadline must come after the request"),
        ({"requested_at": "next tuesday"}, "is not a date and time"),
        ({"response_status": "responded"}, "summarize the response"),
    ],
)
def test_desk_refuses_bad_reply_records(
    desk: TestClient, db: Conn, overrides: dict[str, str], message: str
) -> None:
    seed(db)
    response = desk.post(f"/articles/{SLUG}/replies", data=reply_form(**overrides))
    assert response.status_code == 422
    assert "Reply record refused" in response.text and message in response.text
    assert db.execute("SELECT count(*) FROM article_reply_records").fetchone() == (0,)


def test_desk_records_the_response(desk: TestClient, db: Conn) -> None:
    seed(db)
    desk.post(f"/articles/{SLUG}/replies", data=reply_form())
    reply_id = db.execute("SELECT reply_id FROM article_reply_records").fetchone()
    assert reply_id is not None
    data = {"response_status": "declined_to_comment", "response_summary": ""}
    response = desk.post(f"/articles/{SLUG}/replies/{reply_id[0]}", data=data)
    assert response.status_code == 200
    row = db.execute("SELECT response_status FROM article_reply_records").fetchone()
    assert row == ("declined_to_comment",)


def test_desk_gate_shows_the_reply_requirement(desk: TestClient, db: Conn) -> None:
    seed(db)
    data = form(no_allegations_attested=None)
    html = desk.post(f"/articles/{SLUG}/validate", data=data).text
    assert 'class="gate" data-gate="right_of_reply"' in html
    assert desk.post(f"/articles/{SLUG}/sign", data=data).status_code == 422


def test_desk_named_subject_needs_its_own_record(desk: TestClient, db: Conn) -> None:
    seed(db)
    desk.post(
        f"/articles/{SLUG}/replies",
        data=reply_form(response_status="declined_to_comment"),
    )
    data = form(
        no_allegations_attested=None, allegation_subjects="Acme Paving\nMayor Ruiz"
    )
    html = desk.post(f"/articles/{SLUG}/validate", data=data).text
    assert "no right-of-reply request logged for Mayor Ruiz" in html
    assert desk.post(f"/articles/{SLUG}/sign", data=data).status_code == 422

    data["allegation_subjects"] = "Acme Paving"
    assert desk.post(f"/articles/{SLUG}/sign", data=data).status_code == 200
    row = db.execute("SELECT allegation_subjects FROM articles").fetchone()
    assert row == (["Acme Paving"],)


def test_desk_freezes_the_reply_log_after_publication(
    desk: TestClient, db: Conn
) -> None:
    published(db)
    page = desk.get(f"/?article={SLUG}").text
    assert f'hx-post="/articles/{SLUG}/replies"' not in page
    response = desk.post(f"/articles/{SLUG}/replies", data=reply_form())
    assert response.status_code == 422 and "frozen" in response.text


def test_desk_swaps_in_refusals(desk: TestClient, db: Conn) -> None:
    seed(db)
    page = desk.get(f"/?article={SLUG}").text
    assert '{"code":"422","swap":true}' in page


# --- Right of reply: export ------------------------------------------------------


def test_export_includes_reply_statements(db: Conn) -> None:
    seed(db)
    log_reply(
        db, "Example Agency", summary="a spokesperson said the figures are under review"
    )
    log_reply(db, "Acme Paving", status="no_response_by_deadline", summary=None)
    log_reply(db, "Mayor Ruiz", status="declined_to_comment", summary=None)
    sign(db, no_allegations_attested=False)

    article = export.collect_articles(db)[0][0]
    statements = [item["statement"] for item in article["replies"]]
    assert statements == [
        (
            "We contacted Example Agency on October 13, 2026 with specific findings; "
            "a spokesperson said the figures are under review."
        ),
        (
            "We contacted Acme Paving on October 13, 2026 with specific findings; "
            "they did not respond by the publication deadline."
        ),
        (
            "We contacted Mayor Ruiz on October 13, 2026 with specific findings; "
            "they declined to comment."
        ),
    ]
    assert set(article["replies"][0]) == {
        "subject_name",
        "subject_type",
        "contact_method",
        "requested_at",
        "deadline_at",
        "response_status",
        "response_summary",
        "statement",
    }


def test_expired_pending_reply_reads_as_no_response() -> None:
    record = {
        "subject_name": "Acme Paving",
        "requested_at": datetime(2026, 10, 1, 23, 30, tzinfo=UTC),
        "response_status": "pending",
    }
    assert export.reply_statement(record) == (
        "We contacted Acme Paving on October 1, 2026 with specific findings; "
        "they did not respond by the publication deadline."
    )


def test_article_without_replies_exports_an_empty_list(db: Conn) -> None:
    published(db)
    assert export.collect_articles(db)[0][0]["replies"] == []
