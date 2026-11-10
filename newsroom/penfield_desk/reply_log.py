"""The desk's right-of-reply log: who was asked to respond, how, and what they said.

Every write is refused once the article is published; the database trigger in
migrations/005_reply_log.sql enforces the same rule for any other writer.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Final

import psycopg

from newsroom.dors_venabili.gatekeeper import (
    REPLY_CONTACT_METHODS,
    REPLY_STATUSES,
    REPLY_SUBJECT_TYPES,
)

FROZEN_STATUSES: Final = frozenset({"published", "retracted"})


class ReplyLogError(ValueError):
    """A reply record was refused. issues lists every reason."""

    def __init__(self, issues: list[str]) -> None:
        super().__init__("; ".join(issues))
        self.issues = issues


def parse_utc(value: object, label: str, issues: list[str]) -> datetime | None:
    """Read an HTML datetime-local value as UTC. Offsets that are given are kept."""
    text = str(value or "").strip()
    if not text:
        issues.append(f"{label} is required")
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        issues.append(f"{label} is not a date and time: {text!r}")
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _status_issues(status: str, summary: str | None) -> list[str]:
    issues = []
    if status not in REPLY_STATUSES:
        issues.append(f"unknown response status {status!r}")
    if status == "responded" and not summary:
        issues.append("summarize the response readers will see")
    return issues


def _summary(form: dict[str, Any]) -> str | None:
    return str(form.get("response_summary") or "").strip() or None


def _refuse_if_frozen(article: dict[str, Any]) -> None:
    if article["status"] in FROZEN_STATUSES:
        raise ReplyLogError(
            [f"article is {article['status']}; the reply log is frozen"]
        )


def add_reply(
    conn: psycopg.Connection[Any], article: dict[str, Any], form: dict[str, Any]
) -> None:
    """Log a reply request from the desk form, or raise ReplyLogError."""
    _refuse_if_frozen(article)
    issues: list[str] = []
    name = " ".join(str(form.get("subject_name") or "").split())
    subject_type = str(form.get("subject_type") or "")
    contact = str(form.get("contact_method") or "")
    status = str(form.get("response_status") or "pending")
    summary = _summary(form)

    if not name:
        issues.append("subject name is required")
    if subject_type not in REPLY_SUBJECT_TYPES:
        issues.append(f"unknown subject type {subject_type!r}")
    if contact not in REPLY_CONTACT_METHODS:
        issues.append(f"unknown contact method {contact!r}")
    requested = parse_utc(form.get("requested_at"), "request time", issues)
    deadline = parse_utc(form.get("deadline_at"), "deadline", issues)
    if requested and deadline and deadline <= requested:
        issues.append("deadline must come after the request")
    issues.extend(_status_issues(status, summary))
    if issues:
        raise ReplyLogError(issues)

    conn.execute(
        """
        INSERT INTO article_reply_records
            (article_id, subject_name, subject_type, contact_method,
             requested_at, deadline_at, response_status, response_summary)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            article["article_id"],
            name,
            subject_type,
            contact,
            requested,
            deadline,
            status,
            summary,
        ),
    )


def update_reply(
    conn: psycopg.Connection[Any],
    article: dict[str, Any],
    reply_id: uuid.UUID,
    form: dict[str, Any],
) -> None:
    """Record the outcome of a request, or raise ReplyLogError."""
    _refuse_if_frozen(article)
    status = str(form.get("response_status") or "")
    summary = _summary(form)
    issues = _status_issues(status, summary)
    if issues:
        raise ReplyLogError(issues)
    updated = conn.execute(
        """
        UPDATE article_reply_records
        SET response_status = %s, response_summary = %s
        WHERE reply_id = %s AND article_id = %s
        """,
        (status, summary, reply_id, article["article_id"]),
    )
    if updated.rowcount != 1:
        raise ReplyLogError(["no such reply record for this article"])
