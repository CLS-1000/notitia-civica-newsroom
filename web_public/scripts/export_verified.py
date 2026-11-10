#!/usr/bin/env python3
"""Export verified articles from radiant_ledger to JSON for the public site.

This is the only bridge from the ledger to the public. It selects only rows where
status IN ('published', 'retracted') AND analysis_tier = 'llm_verified' AND
human_approved = TRUE, and recomputes each proof hash, so an article edited after
sign-off is withheld and the run exits non-zero. A retracted article stays public
with its explanation. One that cannot be exported also fails the run, so a
retraction never disappears silently. Corrections to exported articles go to
corrections.json; a correction to a withheld article is withheld with it. It
reads no other schema, and exports neither source text nor sign-off internals.

Standalone by design: it needs only psycopg, so it can run away from the newsroom
code. The hash below must stay identical to newsroom/harlan_eternity/signoff.py,
which tests/test_phase3_export.py enforces.

Usage:
    python web_public/scripts/export_verified.py --out web_public/src/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import psycopg
from psycopg.rows import dict_row

DEFAULT_DSN: Final = "dbname=radiant_ledger"
DEFAULT_OUT: Final = Path(__file__).resolve().parent.parent / "src" / "data"
PRIMARY_TIER: Final = "tier_1_primary"

_ARTICLES: Final = """
    SELECT slug, section, badge_type, headline, deck, body_markdown, editor_name,
           editorial_proof_hash, signed_at, published_at, source_ids,
           verified_claims, uncertainty_disclosures, status,
           retraction_reason, retracted_at
    FROM articles
    WHERE status IN ('published', 'retracted')
      AND analysis_tier = 'llm_verified'
      AND human_approved = TRUE
    ORDER BY published_at DESC
"""

# Retracted after going public, but failing the export filter above.
_BLOCKED_RETRACTIONS: Final = """
    SELECT slug
    FROM articles
    WHERE status = 'retracted'
      AND published_at IS NOT NULL
      AND NOT (analysis_tier = 'llm_verified' AND human_approved = TRUE)
    ORDER BY slug
"""

_CORRECTIONS: Final = """
    SELECT c.correction_id, a.slug, a.headline, c.correction_type, c.original_text,
           c.corrected_text, c.explanation, c.editor_signature, c.corrected_at
    FROM article_corrections c
    JOIN articles a USING (article_id)
    WHERE a.slug = ANY(%s)
    ORDER BY c.corrected_at DESC, c.correction_id
"""

_SOURCES: Final = """
    SELECT source_id, url, domain, tier, raw_sha256, archive_path, retrieved_at
    FROM source_registry
    WHERE source_id = ANY(%s)
"""


def proof_hash(
    body: str, source_ids: list[str], editor_signature: str, timestamp: str
) -> str:
    """Recompute the canonical hash defined by the newsroom sign-off."""
    payload = "\n".join(
        [body.strip(), "\n".join(sorted(source_ids)), editor_signature, timestamp]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _public_source(row: dict[str, Any], archive_base: str | None) -> dict[str, Any]:
    archive_url = None
    if archive_base:
        name = Path(str(row["archive_path"])).name
        archive_url = f"{archive_base.rstrip('/')}/{name}"
    return {
        "source_id": row["source_id"],
        "url": row["url"],
        "domain": row["domain"],
        "tier": row["tier"],
        "raw_sha256": row["raw_sha256"],
        "archive_url": archive_url,
        "retrieved_at": _iso(row["retrieved_at"]),
    }


def _evidence_bar(sources: list[dict[str, Any]]) -> int:
    """Return the percentage of an article's sources that are Tier 1 primary."""
    primary = sum(1 for source in sources if source["tier"] == PRIMARY_TIER)
    return round(100 * primary / len(sources)) if sources else 0


def _problem_with(article: dict[str, Any], found: set[str]) -> str | None:
    if article["status"] == "retracted" and article["retraction_reason"] is None:
        return "retracted with no recorded explanation"

    required = ("editorial_proof_hash", "signed_at", "editor_name", "published_at")
    missing = [name for name in required if article[name] is None]
    if missing:
        return f"missing {', '.join(missing)}"

    absent = sorted(set(article["source_ids"]) - found)
    if absent:
        return f"sources not in the ledger: {', '.join(absent)}"

    expected = proof_hash(
        article["body_markdown"],
        article["source_ids"],
        article["editor_name"],
        article["signed_at"],
    )
    if expected != article["editorial_proof_hash"].strip():
        return "proof hash does not match the stored article"
    return None


def _retraction(article: dict[str, Any]) -> dict[str, str] | None:
    if article["status"] != "retracted":
        return None
    return {
        "reason": article["retraction_reason"],
        "retracted_at": _iso(article["retracted_at"]),
    }


def collect_articles(
    conn: psycopg.Connection[Any], archive_base: str | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return exportable articles, and a reason for each article withheld."""
    exported: list[dict[str, Any]] = []
    withheld: list[str] = []
    with conn.cursor(row_factory=dict_row) as cursor:
        for row in cursor.execute(_BLOCKED_RETRACTIONS).fetchall():
            withheld.append(
                f"{row['slug']}: retracted but no longer verified and approved"
            )

        for article in cursor.execute(_ARTICLES).fetchall():
            rows = cursor.execute(_SOURCES, (article["source_ids"],)).fetchall()
            problem = _problem_with(article, {row["source_id"] for row in rows})
            if problem:
                withheld.append(f"{article['slug']}: {problem}")
                continue

            by_id = {row["source_id"]: row for row in rows}
            sources = [
                _public_source(by_id[source_id], archive_base)
                for source_id in sorted(article["source_ids"])
            ]
            exported.append(
                {
                    "slug": article["slug"],
                    "status": article["status"],
                    "retraction": _retraction(article),
                    "section": article["section"],
                    "badge_type": article["badge_type"],
                    "headline": article["headline"],
                    "deck": article["deck"],
                    "body_markdown": article["body_markdown"],
                    "editor_name": article["editor_name"],
                    "editorial_proof_hash": article["editorial_proof_hash"].strip(),
                    "signed_at": article["signed_at"],
                    "published_at": _iso(article["published_at"]),
                    "verified_claims": article["verified_claims"],
                    "uncertainty_disclosures": article["uncertainty_disclosures"],
                    "evidence_bar": _evidence_bar(sources),
                    "sources": sources,
                }
            )
    return exported, withheld


def collect_corrections(
    conn: psycopg.Connection[Any], slugs: list[str]
) -> list[dict[str, Any]]:
    """Return the public corrections for the given exported articles, newest first.

    The keys match the Correction type in web_public/src/lib/data.ts.
    """
    with conn.cursor(row_factory=dict_row) as cursor:
        rows = cursor.execute(_CORRECTIONS, (slugs,)).fetchall()
    return [
        {
            "correction_id": str(row["correction_id"]),
            "slug": row["slug"],
            "headline": row["headline"],
            "correction_type": row["correction_type"],
            "corrected_at": _iso(row["corrected_at"]),
            "reason": row["explanation"],
            "before": row["original_text"],
            "after": row["corrected_text"],
            "editor_name": row["editor_signature"],
        }
        for row in rows
    ]


def write_json(path: Path, payload: Any) -> None:
    """Replace path atomically, so a failed run never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temp.chmod(0o644)
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", DEFAULT_DSN),
        help="PostgreSQL DSN (default: $DATABASE_URL, else radiant_ledger)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--archive-base-url",
        default=os.environ.get("NC_ARCHIVE_BASE_URL"),
        help="public URL where sealed blobs are hosted; omit to show no links",
    )
    args = parser.parse_args(argv)

    with psycopg.connect(args.dsn) as conn:
        articles, withheld = collect_articles(conn, args.archive_base_url)
        corrections = collect_corrections(conn, [item["slug"] for item in articles])

    write_json(args.out / "articles.json", articles)
    write_json(args.out / "corrections.json", corrections)
    print(
        f"exported={len(articles)} corrections={len(corrections)} "
        f"withheld={len(withheld)}"
    )
    for reason in withheld:
        print(f"withheld {reason}", file=sys.stderr)
    return 1 if withheld else 0


if __name__ == "__main__":
    raise SystemExit(main())
