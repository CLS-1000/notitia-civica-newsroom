"""The public export must publish only verified records, and refuse altered ones."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from conftest import REPO_ROOT, Conn
from psycopg import errors
from test_phase2_workstation import BODY, SLUG, seed, sign

from newsroom.harlan_eternity.signoff import canonical_proof_hash

_SCRIPT = REPO_ROOT / "web_public" / "scripts" / "export_verified.py"
_spec = importlib.util.spec_from_file_location("export_verified", _SCRIPT)
assert _spec is not None and _spec.loader is not None
export = importlib.util.module_from_spec(_spec)
sys.modules["export_verified"] = export
_spec.loader.exec_module(export)


REASON = "The cited record was superseded."


def published(conn: Conn) -> dict[str, Any]:
    seed(conn)
    return sign(conn)


def retract(conn: Conn, reason: str | None = REASON) -> None:
    conn.execute(
        """
        UPDATE articles
        SET status = 'retracted', retraction_reason = %(reason)s,
            retracted_at = CASE WHEN %(reason)s::text IS NULL THEN NULL ELSE now() END
        """,
        {"reason": reason},
    )


def test_hash_matches_the_newsroom_standard() -> None:
    args = (BODY, ["S-2", "S-1"], "Desk Editor", "2026-10-15T12:00:05Z")
    assert export.proof_hash(*args) == canonical_proof_hash(*args)


def test_exports_a_published_article(db: Conn) -> None:
    result = published(db)

    articles, withheld = export.collect_articles(db)

    assert withheld == []
    article = articles[0]
    assert article["slug"] == SLUG
    assert article["editorial_proof_hash"] == result["editorial_proof_hash"]
    assert article["evidence_bar"] == 100
    assert [item["source_id"] for item in article["sources"]] == ["S-1", "S-2"]
    json.dumps(articles)


def test_drafts_are_never_exported(db: Conn) -> None:
    seed(db)
    assert export.collect_articles(db) == ([], [])


@pytest.mark.parametrize("status", ["approved", "compliance_hold"])
def test_unpublished_statuses_are_never_exported(db: Conn, status: str) -> None:
    published(db)
    db.execute("UPDATE articles SET status = %s", (status,))
    assert export.collect_articles(db) == ([], [])


def test_published_articles_carry_no_retraction(db: Conn) -> None:
    published(db)
    article = export.collect_articles(db)[0][0]
    assert article["status"] == "published"
    assert article["retraction"] is None


def test_retracted_article_stays_public_with_its_explanation(db: Conn) -> None:
    result = published(db)
    retract(db)

    articles, withheld = export.collect_articles(db)

    assert withheld == []
    article = articles[0]
    assert article["status"] == "retracted"
    assert article["retraction"]["reason"] == REASON
    assert article["retraction"]["retracted_at"]
    assert article["editorial_proof_hash"] == result["editorial_proof_hash"]


def test_retraction_without_explanation_is_not_silent(db: Conn) -> None:
    published(db)
    retract(db, reason=None)

    articles, withheld = export.collect_articles(db)

    assert articles == []
    assert "no recorded explanation" in withheld[0]


def test_retraction_that_lost_approval_is_not_silent(db: Conn) -> None:
    published(db)
    retract(db)
    db.execute("UPDATE articles SET human_approved = FALSE")

    articles, withheld = export.collect_articles(db)

    assert articles == []
    assert "no longer verified and approved" in withheld[0]


def test_never_published_retraction_stays_private(db: Conn) -> None:
    seed(db)
    db.execute("UPDATE articles SET status = 'retracted'")
    assert export.collect_articles(db) == ([], [])


def test_text_edited_after_retraction_is_withheld(db: Conn) -> None:
    published(db)
    retract(db)
    db.execute("UPDATE articles SET body_markdown = 'Rewritten.'")

    articles, withheld = export.collect_articles(db)

    assert articles == []
    assert "proof hash does not match" in withheld[0]


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE articles SET retraction_reason = 'Reworded.'",
        "UPDATE articles SET retracted_at = '2020-01-01T00:00:00Z'",
        "UPDATE articles SET status = 'approved'",
        "UPDATE articles SET retraction_reason = NULL, retracted_at = NULL",
        "DELETE FROM articles",
    ],
)
def test_recorded_retraction_is_permanent(db: Conn, statement: str) -> None:
    published(db)
    retract(db)
    with pytest.raises(errors.RestrictViolation, match="permanent"):
        db.execute(statement.encode())


@pytest.mark.parametrize("reason", ["", "   "])
def test_retraction_needs_a_real_explanation(db: Conn, reason: str) -> None:
    published(db)
    with pytest.raises(errors.CheckViolation, match="retraction_is_complete"):
        retract(db, reason=reason)


def test_live_article_cannot_carry_a_retraction(db: Conn) -> None:
    published(db)
    with pytest.raises(errors.CheckViolation, match="needs_retracted_status"):
        db.execute("UPDATE articles SET retraction_reason = 'x', retracted_at = now()")


def test_private_fields_are_not_exported(db: Conn) -> None:
    published(db)
    article = export.collect_articles(db)[0][0]

    for field in ("sign_off_checks", "source_roles", "is_private_person"):
        assert field not in article
    assert "verbatim_text" not in article["sources"][0]
    assert "archive_path" not in article["sources"][0]


def test_text_edited_after_sign_off_is_withheld(db: Conn) -> None:
    published(db)
    db.execute("UPDATE articles SET body_markdown = body_markdown || ' Altered.'")

    articles, withheld = export.collect_articles(db)

    assert articles == []
    assert "proof hash does not match" in withheld[0]


@pytest.mark.parametrize(
    "change",
    ["editor_name = 'Someone Else'", "signed_at = '2020-01-01T00:00:00Z'"],
)
def test_altered_signature_is_withheld(db: Conn, change: str) -> None:
    published(db)
    db.execute(f"UPDATE articles SET {change}".encode())
    assert export.collect_articles(db)[0] == []


def test_archive_links_use_the_public_base(db: Conn) -> None:
    published(db)
    article = export.collect_articles(db, "https://cdn.example.org/warc/")[0][0]
    assert article["sources"][0]["archive_url"] == (
        "https://cdn.example.org/warc/x.blob"
    )


def restrict(conn: Conn, source_id: str) -> None:
    conn.execute(
        "INSERT INTO restricted_sources (source_id, reason) VALUES (%s, %s)",
        (source_id, "Editor Eyes Only: protected location"),
    )


def test_restricted_source_is_exported_without_its_location(db: Conn) -> None:
    published(db)
    restrict(db, "S-1")

    article = export.collect_articles(db, "https://cdn.example.org/warc/")[0][0]
    hidden, shown = article["sources"]
    assert hidden["source_id"] == "S-1"
    assert (hidden["url"], hidden["archive_url"]) == (None, None)
    assert (hidden["domain"], hidden["tier"]) == ("example.gov", "tier_1_primary")
    assert hidden["raw_sha256"]
    assert shown["url"] == "https://example.gov/x" and shown["archive_url"]
    assert article["evidence_bar"] == 100


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE restricted_sources SET reason = 'lifted'",
        "DELETE FROM restricted_sources",
        "TRUNCATE restricted_sources",
    ],
)
def test_a_restriction_cannot_be_lifted(db: Conn, statement: str) -> None:
    published(db)
    restrict(db, "S-1")
    with pytest.raises(errors.RaiseException, match="append-only"):
        db.execute(statement)


def test_a_restriction_needs_a_reason(db: Conn) -> None:
    published(db)
    with pytest.raises(errors.CheckViolation):
        db.execute(
            "INSERT INTO restricted_sources (source_id, reason) VALUES ('S-1', ' ')"
        )


def test_archive_links_are_absent_without_a_base(db: Conn) -> None:
    published(db)
    assert export.collect_articles(db)[0][0]["sources"][0]["archive_url"] is None


def test_main_writes_json(ledger_dsn: str, db: Conn, tmp_path: Path) -> None:
    published(db)

    code = export.main(["--dsn", ledger_dsn, "--out", str(tmp_path)])

    assert code == 0
    written = json.loads((tmp_path / "articles.json").read_text(encoding="utf-8"))
    assert written[0]["slug"] == SLUG


def test_main_exits_nonzero_when_it_withholds(
    ledger_dsn: str, db: Conn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    published(db)
    db.execute("UPDATE articles SET body_markdown = 'Altered.'")

    assert export.main(["--dsn", ledger_dsn, "--out", str(tmp_path)]) == 1
    assert "withheld" in capsys.readouterr().err
    assert json.loads((tmp_path / "articles.json").read_text(encoding="utf-8")) == []
