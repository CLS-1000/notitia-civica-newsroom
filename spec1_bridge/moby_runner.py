"""Register a SPEC-1 harvest in source_registry after verifying it against disk.

harvest_manifest.jsonl holds one JSON object per line, in either of two shapes:

* run_moby_dork_batch.sh output:
  {"domain", "retrieved_at_utc", "analysis_tier", "sha256", "file"}
* a receipt returned by fetch_and_seal_source()

Moby records name no source, URL, or tier. They are registered with a sequence-
allocated source_id, a NULL url rather than a guessed one, and tier_4_unverified,
and a re-run matches them on domain and digest. analysis_tier is checked but not
stored, because source_registry has no such column.

A record is registered only if its blob exists inside the base directory and
hashes to the digest the manifest claims. verbatim_text is always re-derived from
the verified blob, so the ledger cannot carry text that differs from the sealed
bytes. Rejected records are reported, and the run exits non-zero, but they do not
stop valid records from being registered.

Usage:
    python -m spec1_bridge.moby_runner harvest_manifest.jsonl --dsn postgresql://...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import psycopg
from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from newsroom.harlan_eternity.custody import (
    decode_verbatim_text,
    extract_domain,
    validate_source_id,
    validate_tier,
)

MOBY_SOURCE_TIER = "tier_4_unverified"

_DOMAIN_PATTERN = re.compile(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?")

INSERT_SOURCE = """
    INSERT INTO source_registry
        (source_id, url, domain, tier, raw_sha256, archive_path,
         verbatim_text, retrieved_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (source_id) DO NOTHING
"""


Connection = psycopg.Connection[tuple[object, ...]]


class ManifestError(ValueError):
    """A manifest record failed validation."""


def _checked_source_id(value: str) -> str:
    validate_source_id(value)
    return value


def _checked_tier(value: str) -> str:
    validate_tier(value)
    return value


def _checked_url(value: str) -> str:
    extract_domain(value)
    return value


def _checked_domain(value: str) -> str:
    domain = value.lower()
    if _DOMAIN_PATTERN.fullmatch(domain) is None:
        raise ValueError(f"invalid domain: {value!r}")
    return domain


def _assume_utc(value: datetime) -> datetime:
    """retrieved_at_utc is UTC by contract, even when it omits an offset."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class SourceEntry:
    """A manifest record normalized for the ledger. source_id is None until
    allocated, which is the case for every Moby record."""

    source_id: str | None
    url: str | None
    domain: str
    tier: str
    raw_sha256: str
    archive_path: str
    retrieved_at: datetime
    content_type: str | None

    @property
    def label(self) -> str:
        return self.source_id or self.archive_path


class ManifestRecord(BaseModel):
    """A custody receipt. Fields the ledger does not use are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    source_id: Annotated[str, AfterValidator(_checked_source_id)]
    url: Annotated[str, AfterValidator(_checked_url)]
    tier: Annotated[str, AfterValidator(_checked_tier)]
    raw_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    archive_path: Annotated[str, Field(min_length=1)]
    retrieved_at: AwareDatetime
    content_type: str | None = None

    def to_entry(self) -> SourceEntry:
        return SourceEntry(
            source_id=self.source_id,
            url=self.url,
            domain=extract_domain(self.url),
            tier=self.tier,
            raw_sha256=self.raw_sha256,
            archive_path=self.archive_path,
            retrieved_at=self.retrieved_at,
            content_type=self.content_type,
        )


class MobyRecord(BaseModel):
    """One line emitted by run_moby_dork_batch.sh."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    domain: Annotated[str, AfterValidator(_checked_domain)]
    retrieved_at_utc: Annotated[datetime, AfterValidator(_assume_utc)]
    analysis_tier: Literal["rule_based"]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    file: Annotated[str, Field(min_length=1)]

    def to_entry(self) -> SourceEntry:
        return SourceEntry(
            source_id=None,
            url=None,
            domain=self.domain,
            tier=MOBY_SOURCE_TIER,
            raw_sha256=self.sha256,
            archive_path=self.file,
            retrieved_at=self.retrieved_at_utc,
            content_type=None,
        )


@dataclass(frozen=True, slots=True)
class Rejection:
    """A manifest line that was not registered."""

    line_number: int
    reason: str


@dataclass(slots=True)
class RunReport:
    """Outcome of one manifest run."""

    registered: list[str] = field(default_factory=list)
    already_registered: list[str] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)


def _resolve_blob(entry: SourceEntry, base_dir: Path) -> Path:
    """Locate the blob, refusing any path that leaves base_dir."""
    root = base_dir.resolve()
    blob = (root / entry.archive_path).resolve()
    if not blob.is_relative_to(root):
        raise ManifestError(
            f"archive_path escapes base directory: {entry.archive_path}"
        )
    if not blob.is_file():
        raise ManifestError(f"blob not found: {entry.archive_path}")
    return blob


def _parse_line(line: str) -> SourceEntry:
    try:
        payload = json.loads(line)
    except ValueError as error:
        raise ManifestError(f"invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError("record is not a JSON object")

    model = MobyRecord if "file" in payload else ManifestRecord
    try:
        return model.model_validate(payload).to_entry()
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'record'}: {item['msg']}"
            for item in error.errors()
        )
        raise ManifestError(details) from error


def allocate_source_id(conn: Connection) -> str:
    row = conn.execute("SELECT nextval('source_registry_harvest_id_seq')").fetchone()
    assert row is not None
    return f"S-{row[0]}"


def _find_harvested_source(conn: Connection, entry: SourceEntry) -> str | None:
    row = conn.execute(
        """
        SELECT source_id FROM source_registry
        WHERE domain = %s AND raw_sha256 = %s
        ORDER BY retrieved_at
        LIMIT 1
        """,
        (entry.domain, entry.raw_sha256),
    ).fetchone()
    return None if row is None else str(row[0])


def _register(
    conn: Connection, entry: SourceEntry, verbatim_text: str | None
) -> tuple[Literal["registered", "already_registered"], str]:
    """Insert a verified entry. A sealed source is never overwritten."""
    source_id = entry.source_id
    if source_id is None:
        existing = _find_harvested_source(conn, entry)
        if existing is not None:
            return "already_registered", existing
        source_id = allocate_source_id(conn)

    cursor = conn.execute(
        INSERT_SOURCE,
        (
            source_id,
            entry.url,
            entry.domain,
            entry.tier,
            entry.raw_sha256,
            entry.archive_path,
            verbatim_text,
            entry.retrieved_at,
        ),
    )
    if cursor.rowcount == 1:
        return "registered", source_id

    row = conn.execute(
        "SELECT raw_sha256 FROM source_registry WHERE source_id = %s",
        (source_id,),
    ).fetchone()
    if row is None or row[0] != entry.raw_sha256:
        raise ManifestError(
            f"{source_id} is already registered with a different digest"
        )
    return "already_registered", source_id


def run_manifest(
    manifest_path: Path,
    conn: Connection,
    base_dir: Path | None = None,
) -> RunReport:
    """Verify every record in the manifest and commit the valid ones as one unit.

    Relative blob paths resolve against base_dir, which defaults to the
    current directory because custody receipts are written relative to it.
    """
    root = base_dir if base_dir is not None else Path.cwd()
    report = RunReport()

    with conn.transaction(), manifest_path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                entry = _parse_line(line)
                blob = _resolve_blob(entry, root)
                # One read, so the text comes from exactly the bytes that were hashed.
                raw = blob.read_bytes()
                actual = hashlib.sha256(raw).hexdigest()
                if actual != entry.raw_sha256:
                    raise ManifestError(
                        f"SHA-256 mismatch for {entry.label}: "
                        f"manifest {entry.raw_sha256}, disk {actual}"
                    )
                text = decode_verbatim_text(raw, entry.content_type)
                outcome, source_id = _register(conn, entry, text)
            except (ManifestError, OSError) as error:
                report.rejected.append(Rejection(line_number, str(error)))
                continue

            if outcome == "registered":
                report.registered.append(source_id)
            else:
                report.already_registered.append(source_id)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path, help="path to harvest_manifest.jsonl")
    parser.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL DSN (default: $DATABASE_URL)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="directory archive_path is relative to (default: current directory)",
    )
    args = parser.parse_args(argv)

    if not args.dsn:
        parser.error("no database configured: pass --dsn or set DATABASE_URL")

    with psycopg.connect(args.dsn) as conn:
        report = run_manifest(args.manifest, conn, args.base_dir)

    print(
        f"registered={len(report.registered)} "
        f"already_registered={len(report.already_registered)} "
        f"rejected={len(report.rejected)}"
    )
    for rejection in report.rejected:
        print(f"line {rejection.line_number}: {rejection.reason}", file=sys.stderr)
    return 1 if report.rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
