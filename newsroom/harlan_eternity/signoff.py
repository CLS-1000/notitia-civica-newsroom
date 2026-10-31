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

from newsroom.dors_venabili.gatekeeper import extract_citations, run_all_gates

DEFAULT_DSN: Final = "dbname=radiant_ledger"

REQUIRED_CHECKS: Final = {
    "sources_read": "I read every cited primary source",
    "quotes_verified": "I checked every quote against its source",
    "fairness_reviewed": "Subjects were treated fairly and given context",
    "legal_reviewed": "PII and defamation risk were reviewed",
}

Connection = psycopg.Connection[Any]


class SignOffError(RuntimeError):
    """Publication was refused. issues lists every reason."""

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


def _collect_issues(
    conn: Connection,
    article: dict[str, Any],
    body_markdown: str,
    source_ids: list[str],
    editor_name: str,
    checks: dict[str, bool],
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

    gates = run_all_gates(
        headline=article["headline"],
        body=body_markdown,
        source_texts=texts,
        roles=article["source_roles"],
        is_private_person=article["is_private_person"],
        right_of_reply_logged=article["right_of_reply_logged"],
        badge_type=article["badge_type"],
        source_tiers=tiers,
    )
    issues.extend(f"{gate.name}: {issue}" for gate in gates for issue in gate.issues)
    return issues


def _sign(
    conn: Connection,
    slug: str,
    body_markdown: str,
    source_ids: list[str],
    editor_name: str,
    checks: dict[str, bool],
    clock: Callable[[], datetime],
) -> dict[str, Any]:
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cursor:
        article = cursor.execute(
            "SELECT * FROM articles WHERE slug = %s FOR UPDATE", (slug,)
        ).fetchone()
        if article is None:
            raise SignOffError([f"no article with slug {slug!r}"])

        issues = _collect_issues(
            conn, article, body_markdown, source_ids, editor_name, checks
        )
        if issues:
            raise SignOffError(issues)

        signed_at = canonical_timestamp(clock())
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
) -> dict[str, Any]:
    """Verify, hash, and publish an article, or raise SignOffError and change nothing."""
    source_ids = sorted(sorted_source_ids)
    if conn is not None:
        return _sign(conn, slug, body_markdown, source_ids, editor_name, checks, clock)
    dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
    with psycopg.connect(dsn) as owned:
        return _sign(owned, slug, body_markdown, source_ids, editor_name, checks, clock)
