"""Chain of custody: fetch a source once and seal its exact bytes to disk."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from typing import Final, TypedDict
from urllib.parse import urlsplit

import httpx

USER_AGENT: Final = (
    "NotitiaCivica-SPEC1/1.0 (+https://notitiacivica.com; contact@evastararcana.com)"
)
DEFAULT_ARCHIVE_DIR: Final = Path("archive/warc")
DEFAULT_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_MAX_BYTES: Final = 256 * 1024 * 1024

# Mirrors source_tier_enum in migrations/001_init_ledger.sql.
SOURCE_TIERS: Final = frozenset(
    {
        "tier_1_primary",
        "tier_2_institutional",
        "tier_3_secondary",
        "tier_4_unverified",
    }
)

# source_id becomes part of a filename, so this check also blocks path traversal.
_SOURCE_ID_PATTERN: Final = re.compile(r"S-[0-9]+")
_MAX_SOURCE_ID_LENGTH: Final = 32
_TEXT_SUBTYPES: Final = frozenset({"json", "xml", "xhtml+xml", "javascript", "csv"})


class CustodyError(RuntimeError):
    """A source could not be fetched and sealed."""


class CustodyRecord(TypedDict):
    """Custody receipt. Every value is JSON-serializable, so one receipt is one
    line of harvest_manifest.jsonl."""

    source_id: str
    url: str
    final_url: str
    domain: str
    tier: str
    raw_sha256: str
    archive_path: str
    byte_length: int
    http_status: int
    content_type: str | None
    verbatim_text: str | None
    retrieved_at: str


def validate_source_id(source_id: str) -> None:
    """Raise ValueError unless source_id has the ledger's S-<digits> form."""
    if (
        len(source_id) > _MAX_SOURCE_ID_LENGTH
        or _SOURCE_ID_PATTERN.fullmatch(source_id) is None
    ):
        raise ValueError(f"invalid source_id: {source_id!r}")


def validate_tier(tier: str) -> None:
    """Raise ValueError unless tier is a member of source_tier_enum."""
    if tier not in SOURCE_TIERS:
        raise ValueError(f"invalid source tier: {tier!r}")


def extract_domain(url: str) -> str:
    """Return the lowercase host of an http(s) URL, or raise ValueError."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"not an absolute http(s) URL: {url!r}")
    return parts.hostname


def decode_verbatim_text(raw: bytes, content_type: str | None) -> str | None:
    """Decode a payload without altering it, or return None if it is not text.

    The declared charset must decode the whole payload strictly. Lossy decoding
    would put words in the ledger that the source never contained, so undecodable
    payloads stay byte-only, as do those PostgreSQL TEXT cannot hold (NUL bytes).
    """
    message = Message()
    message["Content-Type"] = content_type or "text/plain"
    if message.get_content_maintype() != "text" and (
        message.get_content_subtype() not in _TEXT_SUBTYPES
    ):
        return None

    charset = message.get_content_charset() or "utf-8"
    try:
        text = raw.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return None
    return None if "\x00" in text else text


def _download(
    client: httpx.Client, url: str, destination: Path, max_bytes: int
) -> tuple[str, int, httpx.Response]:
    """Stream the response body to destination, hashing the bytes as written."""
    digest = hashlib.sha256()
    total = 0
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise CustodyError(f"{url} exceeds the {max_bytes} byte limit")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    return digest.hexdigest(), total, response


def fetch_and_seal_source(
    url: str,
    source_id: str,
    tier: str,
    *,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    transport: httpx.BaseTransport | None = None,
) -> CustodyRecord:
    """Fetch url, seal the raw payload under archive_dir, and return its receipt.

    The payload lands at {source_id}_{timestamp}_{sha256[:12]}.blob. It appears
    under that name only once fully written, and is then read-only, so a reader
    never sees a partial blob. Raises ValueError for bad arguments, httpx.HTTPError
    for failed fetches, and CustodyError for oversized payloads.
    """
    validate_source_id(source_id)
    validate_tier(tier)
    domain = extract_domain(url)

    archive_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=archive_dir, suffix=".partial")
    os.close(descriptor)
    temp_path = Path(temp_name)

    try:
        with httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        ) as client:
            raw_sha256, byte_length, response = _download(
                client, url, temp_path, max_bytes
            )

        retrieved_at = datetime.now(UTC)
        stamp = retrieved_at.strftime("%Y%m%dT%H%M%S%fZ")
        archive_path = archive_dir / f"{source_id}_{stamp}_{raw_sha256[:12]}.blob"
        temp_path.chmod(0o444)
        temp_path.replace(archive_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    content_type = response.headers.get("content-type")
    return CustodyRecord(
        source_id=source_id,
        url=url,
        final_url=str(response.url),
        domain=domain,
        tier=tier,
        raw_sha256=raw_sha256,
        archive_path=str(archive_path),
        byte_length=byte_length,
        http_status=response.status_code,
        content_type=content_type,
        verbatim_text=decode_verbatim_text(archive_path.read_bytes(), content_type),
        retrieved_at=retrieved_at.isoformat(),
    )
