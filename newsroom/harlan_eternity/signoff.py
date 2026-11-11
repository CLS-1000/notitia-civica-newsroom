"""Human sign-off: the only code path that publishes an article.

Everything is checked again here, inside the transaction that publishes, so no
route or UI state can bypass the gates. The database trigger is the final barrier.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Final

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from newsroom.dors_venabili.gatekeeper import (
    extract_citations,
    has_reply_record,
    normalize_text,
    run_all_gates,
    verify_right_of_reply_log,
    verify_uncertainty_disclosures,
)

DEFAULT_DSN: Final = "dbname=radiant_ledger"

REQUIRED_CHECKS: Final = {
    "sources_read": "I read every cited primary source",
    "quotes_verified": "I checked every quote against its source",
    "fairness_reviewed": "Subjects were treated fairly and given context",
    "legal_reviewed": "PII and defamation risk were reviewed",
}

CORRECTION_TYPES: Final = ("clarification", "factual_correction", "methodology_update")

Connection = psycopg.Connection[Any]


class SignOffError(RuntimeError):
    """Publication was refused. issues lists every reason."""

    def __init__(self, issues: list[str]) -> None:
        super().__init__("; ".join(issues))
        self.issues = issues


class CorrectionError(RuntimeError):
    """A correction was refused. issues lists every reason."""

    def __init__(self, issues: list[str]) -> None:
        super().__init__("; ".join(issues))
        self.issues = issues


def canonical_timestamp(moment: datetime) -> str:
    """Render the ISO 8601 UTC timestamp that the proof hash covers."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_proof_hash(
    body: str, source_ids: list[str], editor_signature: str, timestamp: str
) -> str:
    """Return the canonical proof hash. Source order does not affect it."""
    payload = "\n".join(
        [body.strip(), "\n".join(sorted(source_ids)), editor_signature, timestamp]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def missing_checks(checks: dict[str, bool]) -> list[str]:
    return [name for name in REQUIRED_CHECKS if checks.get(name) is not True]


def load_sources(
    conn: Connection, source_ids: list[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ledger text and tier by source ID. No text form maps to ''."""
    with conn.cursor(row_factory=dict_row) as cursor:
        rows = cursor.execute(
            """
            SELECT source_id, tier, verbatim_text FROM source_registry
            WHERE source_id = ANY(%s)
            """,
            (source_ids,),
        ).fetchall()
    texts = {row["source_id"]: row["verbatim_text"] or "" for row in rows}
    return texts, {row["source_id"]: row["tier"] for row in rows}


def load_reply_log(conn: Connection, article_id: Any) -> list[dict[str, Any]]:
    """Return an article's reply records, oldest request first."""
    with conn.cursor(row_factory=dict_row) as cursor:
        return cursor.execute(
            """
            SELECT reply_id, subject_name, subject_type, contact_method,
                   requested_at, deadline_at, response_status, response_summary
            FROM article_reply_records WHERE article_id = %s
            ORDER BY requested_at, created_at, reply_id
            """,
            (article_id,),
        ).fetchall()


def load_tier_review_flags(conn: Connection, source_ids: list[str]) -> dict[str, str]:
    """Return the review reason for each of source_ids flagged for a tier review.

    Flags are set by database triggers (migrations/007) when a factual correction
    is recorded or an article is retracted, whichever code path does it.
    """
    with conn.cursor(row_factory=dict_row) as cursor:
        rows = cursor.execute(
            """
            SELECT source_id, tier_review_reason FROM source_registry
            WHERE source_id = ANY(%s) AND tier_review_flagged
            """,
            (source_ids,),
        ).fetchall()
    return {row["source_id"]: row["tier_review_reason"] for row in rows}


def clean_lines(items: list[str]) -> list[str]:
    return [item.strip() for item in items if item.strip()]


def _collect_issues(
    conn: Connection,
    article: dict[str, Any],
    body_markdown: str,
    source_ids: list[str],
    editor_name: str,
    checks: dict[str, bool],
    disclosures: list[str],
    no_open_questions_confirmed: bool,
    allegation_subjects: list[str],
    no_allegations_attested: bool,
    now: datetime,
) -> list[str]:
    issues = [f"human check not ticked: {name}" for name in missing_checks(checks)]
    if not editor_name.strip():
        issues.append("editor name is required")
    if article["status"] in {"published", "retracted"}:
        issues.append(f"article is already {article['status']}")

    outside = sorted(set(source_ids) - set(article["source_ids"]))
    issues.extend(f"{item} is not a source of this article" for item in outside)

    texts, tiers = load_sources(conn, sorted(set(source_ids) - set(outside)))
    if set(extract_citations(body_markdown)) != set(source_ids):
        issues.append("source list does not match the citations in the body")

    replies = load_reply_log(conn, article["article_id"])
    gates = run_all_gates(
        headline=article["headline"],
        body=body_markdown,
        source_texts=texts,
        roles=article["source_roles"],
        is_private_person=article["is_private_person"],
        right_of_reply_logged=has_reply_record(replies),
        badge_type=article["badge_type"],
        source_tiers=tiers,
    )
    issues.extend(f"{gate.name}: {issue}" for gate in gates for issue in gate.issues)
    issues.extend(
        f"uncertainty: {issue}"
        for issue in verify_uncertainty_disclosures(
            disclosures, no_open_questions_confirmed
        )
    )
    issues.extend(
        f"right_of_reply: {issue}"
        for issue in verify_right_of_reply_log(
            replies,
            allegation_subjects,
            no_allegations_attested=no_allegations_attested,
            body=body_markdown,
            badge_type=article["badge_type"],
            now=now,
        )
    )
    return issues


def _sign(
    conn: Connection,
    slug: str,
    body_markdown: str,
    source_ids: list[str],
    editor_name: str,
    checks: dict[str, bool],
    clock: Callable[[], datetime],
    disclosures: list[str] | None,
    no_open_questions_confirmed: bool,
    allegation_subjects: list[str] | None,
    no_allegations_attested: bool,
) -> dict[str, Any]:
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cursor:
        article = cursor.execute(
            "SELECT * FROM articles WHERE slug = %s FOR UPDATE", (slug,)
        ).fetchone()
        if article is None:
            raise SignOffError([f"no article with slug {slug!r}"])

        if disclosures is None:
            disclosures = list(article["uncertainty_disclosures"])
        disclosures = clean_lines(disclosures)
        if allegation_subjects is None:
            allegation_subjects = list(article["allegation_subjects"])
        allegation_subjects = clean_lines(allegation_subjects)
        # One moment for both the reply deadlines and the signed timestamp.
        moment = clock()
        issues = _collect_issues(
            conn,
            article,
            body_markdown,
            source_ids,
            editor_name,
            checks,
            disclosures,
            no_open_questions_confirmed,
            allegation_subjects,
            no_allegations_attested,
            moment,
        )
        if issues:
            raise SignOffError(issues)

        signed_at = canonical_timestamp(moment)
        proof_hash = canonical_proof_hash(
            body_markdown, source_ids, editor_name, signed_at
        )
        row = cursor.execute(
            """
            UPDATE articles
            SET body_markdown = %s,
                source_ids = %s,
                analysis_tier = 'llm_verified',
                human_approved = TRUE,
                editor_name = %s,
                editorial_proof_hash = %s,
                signed_at = %s,
                sign_off_checks = %s,
                uncertainty_disclosures = %s,
                no_open_questions_confirmed = %s,
                allegation_subjects = %s,
                no_allegations_attested = %s,
                right_of_reply_logged = EXISTS (
                    SELECT 1 FROM article_reply_records
                    WHERE article_id = articles.article_id
                ),
                status = 'published'
            WHERE article_id = %s
            RETURNING article_id, status, analysis_tier, human_approved, published_at
            """,
            (
                body_markdown,
                source_ids,
                editor_name,
                proof_hash,
                signed_at,
                Jsonb(checks),
                Jsonb(disclosures),
                no_open_questions_confirmed,
                Jsonb(allegation_subjects),
                no_allegations_attested,
                article["article_id"],
            ),
        ).fetchone()
    assert row is not None
    return {
        "article_id": str(row["article_id"]),
        "slug": slug,
        "status": row["status"],
        "analysis_tier": row["analysis_tier"],
        "human_approved": row["human_approved"],
        "editor_name": editor_name,
        "editorial_proof_hash": proof_hash,
        "signed_at": signed_at,
        "published_at": row["published_at"],
        "source_ids": source_ids,
        "uncertainty_disclosures": disclosures,
        "no_open_questions_confirmed": no_open_questions_confirmed,
        "allegation_subjects": allegation_subjects,
        "no_allegations_attested": no_allegations_attested,
    }


def sign_and_promote_article(
    slug: str,
    body_markdown: str,
    sorted_source_ids: list[str],
    editor_name: str,
    checks: dict[str, bool],
    *,
    conn: Connection | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    uncertainty_disclosures: list[str] | None = None,
    no_open_questions_confirmed: bool = False,
    allegation_subjects: list[str] | None = None,
    no_allegations_attested: bool = False,
) -> dict[str, Any]:
    """Verify, hash, and publish an article, or raise SignOffError and change nothing.

    uncertainty_disclosures and allegation_subjects default to the lists stored
    on the article, and whatever lists are checked are saved with the sign-off.
    So are the editor's attestations that there are no open questions and no
    allegations, which are never read back from the row: they have to be given
    at the moment of signing.
    The reply log is checked against clock(), the same moment that is signed.
    """
    args = (
        slug,
        body_markdown,
        sorted(sorted_source_ids),
        editor_name,
        checks,
        clock,
        uncertainty_disclosures,
        no_open_questions_confirmed,
        allegation_subjects,
        no_allegations_attested,
    )
    if conn is not None:
        return _sign(conn, *args)
    dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
    with psycopg.connect(dsn) as owned:
        return _sign(owned, *args)


def _blank_to_none(text: str | None) -> str | None:
    return text.strip() if text and text.strip() else None


def _correction_issues(
    article: dict[str, Any] | None,
    slug: str,
    correction_type: str,
    explanation: str,
    editor_signature: str,
    original_text: str | None,
    corrected_text: str | None,
) -> list[str]:
    issues: list[str] = []
    if correction_type not in CORRECTION_TYPES:
        issues.append(f"unknown correction type {correction_type!r}")
    if not explanation.strip():
        issues.append("explanation is required")
    if not editor_signature.strip():
        issues.append("editor signature is required")
    if correction_type == "factual_correction" and not (
        original_text and corrected_text
    ):
        issues.append("a factual correction needs the original and corrected text")
    if article is None:
        return [f"no article with slug {slug!r}", *issues]
    if article["status"] != "published":
        issues.append(
            f"article is {article['status']}, only published articles take corrections"
        )
    if original_text and normalize_text(original_text) not in normalize_text(
        article["body_markdown"]
    ):
        issues.append("original text does not appear in the signed article")
    return issues


def _record(
    conn: Connection,
    slug: str,
    correction_type: str,
    explanation: str,
    editor_signature: str,
    original_text: str | None,
    corrected_text: str | None,
) -> dict[str, Any]:
    original = _blank_to_none(original_text)
    corrected = _blank_to_none(corrected_text)
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cursor:
        # FOR SHARE keeps a retraction or edit from landing mid-correction.
        article = cursor.execute(
            """
            SELECT article_id, status, body_markdown, source_ids FROM articles
            WHERE slug = %s FOR SHARE
            """,
            (slug,),
        ).fetchone()
        issues = _correction_issues(
            article,
            slug,
            correction_type,
            explanation,
            editor_signature,
            original,
            corrected,
        )
        if issues:
            raise CorrectionError(issues)
        assert article is not None
        row = cursor.execute(
            """
            INSERT INTO article_corrections
                (article_id, correction_type, original_text, corrected_text,
                 explanation, editor_signature)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING correction_id, corrected_at
            """,
            (
                article["article_id"],
                correction_type,
                original,
                corrected,
                explanation.strip(),
                editor_signature.strip(),
            ),
        ).fetchone()
    assert row is not None
    return {
        "correction_id": str(row["correction_id"]),
        "slug": slug,
        "correction_type": correction_type,
        "original_text": original,
        "corrected_text": corrected,
        "explanation": explanation.strip(),
        "editor_signature": editor_signature.strip(),
        "corrected_at": row["corrected_at"],
        "flagged_sources": sorted(
            load_tier_review_flags(conn, list(article["source_ids"]))
        )
        if correction_type == "factual_correction"
        else [],
    }


def record_correction(
    slug: str,
    correction_type: str,
    explanation: str,
    editor_signature: str,
    *,
    original_text: str | None = None,
    corrected_text: str | None = None,
    conn: Connection | None = None,
) -> dict[str, Any]:
    """Append a correction to a published article, or raise CorrectionError.

    The article row is never written, so its body and editorial proof hash still
    match what the editor signed; readers see the correction beside it. Any
    original_text must quote the signed article. A factual correction flags every
    source the article cites for a tier review (a database trigger does it, in the
    same transaction), and the result lists them as flagged_sources.
    """
    args = (
        slug,
        correction_type,
        explanation,
        editor_signature,
        original_text,
        corrected_text,
    )
    if conn is not None:
        return _record(conn, *args)
    dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
    with psycopg.connect(dsn) as owned:
        return _record(owned, *args)
