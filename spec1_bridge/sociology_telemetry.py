"""Shared plumbing for the Phase 4 detectors: receipts, quarantine, and streams.

Everything a detector observes about what people say is unverified narrative. It
is written only to sociology_telemetry.narrative_observations, never to the
evidence tables, so it cannot be cited as proof of what physically happened.

Stream parsers keep what was said and when, and drop who said it. A detector
studies narratives, not the private people repeating them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Protocol

import httpx
from psycopg import Connection
from psycopg.types.json import Jsonb

from newsroom.harlan_eternity.custody import USER_AGENT

DOMAIN_IRRATIONAL: Final = 5  # UAP reports, prophecy, and fringe belief
DOMAIN_FAKE_NEWS: Final = 6  # rumor virality and pink-slime networks
EVIDENTIARY_STATUS: Final = "unverified_narrative"

JETSTREAM_URL: Final = (
    "wss://jetstream2.us-east.bsky.network/subscribe"
    "?wantedCollections=app.bsky.feed.post"
)
WIKIMEDIA_URL: Final = "https://stream.wikimedia.org/v2/stream/recentchange"
MAX_POST_CHARS: Final = 1000


@dataclass(frozen=True, slots=True)
class DecompositionReceipt:
    """A plain-English account of why a signal tripped and what it cannot show."""

    detector: str
    finding: str
    reasons: tuple[str, ...]
    caveat: str
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str | None = None

    def __post_init__(self) -> None:
        if not self.finding.strip() or not self.reasons or not self.caveat.strip():
            raise ValueError("a receipt needs a finding, reasons, and a caveat")

    def as_dict(self) -> dict[str, Any]:
        receipt = {
            "detector": self.detector,
            "finding": self.finding,
            "reasons": list(self.reasons),
            "caveat": self.caveat,
            "metrics": self.metrics,
        }
        return receipt if self.status is None else {**receipt, "status": self.status}

    def plain_english(self) -> str:
        lines = [self.finding, *(f"- {reason}" for reason in self.reasons)]
        if self.status is not None:
            lines.insert(0, f"Status: {self.status}")
        lines.append(f"Limits: {self.caveat}")
        return "\n".join(lines)


def write_observation(
    conn: Connection[Any],
    *,
    domain: int,
    narrative_label: str,
    observed_text: str,
    receipt: DecompositionReceipt,
    observed_at: datetime,
    source_ids: Iterable[str] = (),
) -> str:
    """Quarantine one observation and return its id."""
    metrics = {
        "evidentiary_status": EVIDENTIARY_STATUS,
        "receipt": receipt.as_dict(),
    }
    row = conn.execute(
        """
        INSERT INTO sociology_telemetry.narrative_observations
            (domain, narrative_label, observed_text, source_ids, metrics, observed_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING observation_id
        """,
        (
            domain,
            narrative_label,
            observed_text,
            list(source_ids),
            Jsonb(metrics),
            observed_at,
        ),
    ).fetchone()
    if row is None:
        raise RuntimeError("the observation insert returned no row")
    return str(row[0])


@dataclass(frozen=True, slots=True)
class StreamPost:
    """One public utterance, with no author attached."""

    platform: str
    text: str
    at: datetime


def parse_bluesky_event(event: dict[str, Any]) -> StreamPost | None:
    """Return the post in a Jetstream event, or None for anything else."""
    commit = event.get("commit")
    if event.get("kind") != "commit" or not isinstance(commit, dict):
        return None
    if commit.get("operation") != "create":
        return None
    if commit.get("collection") != "app.bsky.feed.post":
        return None
    record = commit.get("record")
    text = record.get("text") if isinstance(record, dict) else None
    time_us = event.get("time_us")
    if not isinstance(text, str) or not text.strip() or not isinstance(time_us, int):
        return None
    # time_us is the relay's clock. A post's createdAt is whatever the client said.
    at = datetime.fromtimestamp(time_us / 1_000_000, UTC)
    return StreamPost("bluesky", text[:MAX_POST_CHARS], at)


def parse_wikimedia_event(event: dict[str, Any]) -> StreamPost | None:
    """Return a human edit from EventStreams recentchange, or None."""
    if event.get("bot") is True or event.get("type") not in {"edit", "new"}:
        return None
    meta = event.get("meta")
    stamp = meta.get("dt") if isinstance(meta, dict) else None
    title = event.get("title")
    if not isinstance(stamp, str) or not isinstance(title, str):
        return None
    comment = event.get("comment")
    text = f"{title}: {comment}" if isinstance(comment, str) and comment else title
    at = datetime.fromisoformat(stamp)
    return StreamPost("wikimedia", text[:MAX_POST_CHARS], at)


class MessageSocket(Protocol):
    """The part of a websocket connection that Jetstream needs."""

    def recv(self) -> str | bytes: ...

    def close(self) -> None: ...


def _connect_websocket(url: str) -> MessageSocket:
    from websockets.sync.client import connect

    return connect(url, user_agent_header=USER_AGENT)


def iter_jetstream(
    url: str = JETSTREAM_URL,
    connect: Callable[[str], MessageSocket] = _connect_websocket,
) -> Iterator[StreamPost]:
    """Yield posts until the connection closes."""
    socket = connect(url)
    try:
        while True:
            try:
                raw = socket.recv()
            except (EOFError, OSError):
                return
            except Exception as error:
                if type(error).__name__.startswith("ConnectionClosed"):
                    return
                raise
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if isinstance(event, dict) and (post := parse_bluesky_event(event)):
                yield post
    finally:
        socket.close()


def iter_eventstream(
    url: str = WIKIMEDIA_URL,
    transport: httpx.BaseTransport | None = None,
) -> Iterator[StreamPost]:
    """Yield human edits from a Wikimedia server-sent event stream."""
    headers = {"User-Agent": USER_AGENT, "Accept": "text/event-stream"}
    timeout = httpx.Timeout(30.0, read=None)
    with (
        httpx.Client(headers=headers, timeout=timeout, transport=transport) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line.removeprefix("data:").strip())
            except ValueError:
                continue
            if isinstance(event, dict) and (post := parse_wikimedia_event(event)):
                yield post
