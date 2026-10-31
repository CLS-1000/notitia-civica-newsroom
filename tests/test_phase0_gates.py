"""Phase 0 gates, exercised against a real PostgreSQL 18 with pgvector.

Set NOTITIA_TEST_DSN to use an existing server; throwaway databases are created
there and dropped afterwards. Otherwise a private cluster is started for the run,
using a system pgvector or the copy from scripts/bootstrap_test_pgvector.sh.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from psycopg import errors, sql
from psycopg.conninfo import make_conninfo

from newsroom.harlan_eternity.custody import USER_AGENT, fetch_and_seal_source
from spec1_bridge.moby_runner import main, run_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION = REPO_ROOT / "migrations" / "001_init_ledger.sql"
LOCAL_PGVECTOR = REPO_ROOT / ".pgdev" / "pgvector"
PG_BIN = Path("/usr/lib/postgresql/18/bin")

PROOF_HASH = "a" * 64
PUBLISHABLE: dict[str, Any] = {
    "analysis_tier": "llm_verified",
    "human_approved": True,
    "editorial_proof_hash": PROOF_HASH,
}

Conn = psycopg.Connection[tuple[Any, ...]]


def _pg_tool(name: str) -> str:
    return shutil.which(name) or str(PG_BIN / name)


def _pgvector_settings() -> list[str]:
    """Return server settings that make pgvector loadable, or fail with the fix."""
    sharedir = subprocess.run(
        [_pg_tool("pg_config"), "--sharedir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (Path(sharedir) / "extension" / "vector.control").exists():
        return []
    if (LOCAL_PGVECTOR / "share" / "extension" / "vector.control").exists():
        return [
            f"extension_control_path={LOCAL_PGVECTOR / 'share'}:$system",
            f"dynamic_library_path={LOCAL_PGVECTOR / 'lib'}:$libdir",
        ]
    pytest.fail(
        "pgvector is required: run scripts/bootstrap_test_pgvector.sh, "
        "install postgresql-18-pgvector, or set NOTITIA_TEST_DSN"
    )


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="session")
def server_dsn() -> Iterator[str]:
    """Yield a superuser DSN: the configured server, or a private cluster."""
    configured = os.environ.get("NOTITIA_TEST_DSN")
    if configured:
        yield configured
        return

    settings = _pgvector_settings()
    password = secrets.token_hex(16)
    port = _free_port()
    with tempfile.TemporaryDirectory(prefix="notitia-pg-") as workdir:
        data_dir = Path(workdir) / "data"
        password_file = Path(workdir) / "password"
        password_file.write_text(password)
        subprocess.run(
            [
                _pg_tool("initdb"),
                "--pgdata",
                str(data_dir),
                "--username",
                "postgres",
                "--auth",
                "scram-sha-256",
                "--pwfile",
                str(password_file),
                "--encoding",
                "UTF8",
                "--no-sync",
            ],
            check=True,
            capture_output=True,
        )
        options = [
            "-c listen_addresses=127.0.0.1",
            f"-c port={port}",
            "-c unix_socket_directories=''",
            "-c fsync=off",
            *(f"-c '{setting}'" for setting in settings),
        ]
        log_file = Path(workdir) / "server.log"
        pg_ctl = _pg_tool("pg_ctl")
        started = subprocess.run(
            [
                pg_ctl,
                "start",
                "--wait",
                "--pgdata",
                str(data_dir),
                "--log",
                str(log_file),
                "--options",
                " ".join(options),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if started.returncode != 0:
            pytest.fail(f"test cluster failed to start:\n{log_file.read_text()}")
        try:
            yield make_conninfo(
                host="127.0.0.1",
                port=port,
                user="postgres",
                password=password,
                dbname="postgres",
            )
        finally:
            subprocess.run(
                [pg_ctl, "stop", "--pgdata", str(data_dir), "--mode", "immediate"],
                check=True,
                capture_output=True,
            )


@pytest.fixture(scope="session")
def template_database(server_dsn: str) -> Iterator[str]:
    """Yield the name of a throwaway database with the migration applied."""
    name = f"notitia_template_{secrets.token_hex(6)}"
    with psycopg.connect(server_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        dsn = make_conninfo(server_dsn, dbname=name)
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(MIGRATION.read_text(encoding="utf-8").encode())
        yield name
    finally:
        with psycopg.connect(server_dsn, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.fixture
def ledger_dsn(server_dsn: str, template_database: str) -> Iterator[str]:
    """Yield a DSN for a fresh clone of the migrated database.

    Each test gets its own clone because the ledger is append-only by design and
    cannot be emptied in place, and because the CLI commits on its own connection,
    which rules out rolling back a shared transaction.
    """
    name = f"notitia_test_{secrets.token_hex(6)}"
    with psycopg.connect(server_dsn, autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                sql.Identifier(name), sql.Identifier(template_database)
            )
        )
    try:
        yield make_conninfo(server_dsn, dbname=name)
    finally:
        with psycopg.connect(server_dsn, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.fixture
def db(ledger_dsn: str) -> Iterator[Conn]:
    with psycopg.connect(ledger_dsn, autocommit=True) as conn:
        yield conn


def insert_article(conn: Conn, **overrides: Any) -> str:
    """Insert an article, starting from the column defaults, and return its id."""
    values: dict[str, Any] = {
        "slug": f"story-{secrets.token_hex(4)}",
        "section": "ledger",
        "badge_type": "investigation",
        "headline": "Headline",
        "body_markdown": "Body",
        **overrides,
    }
    query = sql.SQL(
        "INSERT INTO articles ({}) VALUES ({}) RETURNING article_id"
    ).format(
        sql.SQL(", ").join(map(sql.Identifier, values)),
        sql.SQL(", ").join(sql.Placeholder() * len(values)),
    )
    row = conn.execute(query, list(values.values())).fetchone()
    assert row is not None
    return str(row[0])


def update_article(conn: Conn, article_id: str, **changes: Any) -> None:
    query = sql.SQL("UPDATE articles SET {} WHERE article_id = %s").format(
        sql.SQL(", ").join(
            sql.SQL("{} = %s").format(sql.Identifier(column)) for column in changes
        )
    )
    conn.execute(query, [*changes.values(), article_id])


def article_field(conn: Conn, article_id: str, column: str) -> Any:
    query = sql.SQL("SELECT {} FROM articles WHERE article_id = %s").format(
        sql.Identifier(column)
    )
    row = conn.execute(query, (article_id,)).fetchone()
    assert row is not None
    return row[0]


# --- Schema ---------------------------------------------------------------


def test_migration_installs_extensions(db: Conn) -> None:
    rows = db.execute(
        "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pgcrypto')"
    ).fetchall()
    assert {row[0] for row in rows} == {"vector", "pgcrypto"}


@pytest.mark.parametrize(
    ("enum_name", "labels"),
    [
        ("analysis_tier_enum", ["rule_based", "llm_verified"]),
        (
            "source_tier_enum",
            [
                "tier_1_primary",
                "tier_2_institutional",
                "tier_3_secondary",
                "tier_4_unverified",
            ],
        ),
        (
            "pub_status_enum",
            [
                "harvested",
                "triaged",
                "drafting",
                "compliance_hold",
                "approved",
                "published",
                "retracted",
            ],
        ),
    ],
)
def test_enum_labels(db: Conn, enum_name: str, labels: list[str]) -> None:
    rows = db.execute(
        """
        SELECT e.enumlabel
        FROM pg_enum e JOIN pg_type t ON t.oid = e.enumtypid
        WHERE t.typname = %s
        ORDER BY e.enumsortorder
        """,
        (enum_name,),
    ).fetchall()
    assert [row[0] for row in rows] == labels


def test_article_defaults_are_unpublishable(db: Conn) -> None:
    article_id = insert_article(db)
    assert article_field(db, article_id, "status") == "drafting"
    assert article_field(db, article_id, "analysis_tier") == "rule_based"
    assert article_field(db, article_id, "human_approved") is False


def test_triage_record_defaults_to_rule_based(db: Conn) -> None:
    row = db.execute(
        """
        INSERT INTO triage_records
            (adapter_name, subject_domain, anomaly_score, paste_ready_prompt, source_ids)
        VALUES ('moby', 'procurement', 0.9, 'prompt', ARRAY['S-1042'])
        RETURNING analysis_tier, source_ids
        """
    ).fetchone()
    assert row == ("rule_based", ["S-1042"])


# --- Publication gates: blocked ---------------------------------------------


BLOCKED_CASES = [
    pytest.param({"analysis_tier": "rule_based"}, "analysis_tier", id="rule-based"),
    pytest.param({"human_approved": False}, "human_approved", id="not-approved"),
    pytest.param({"editorial_proof_hash": None}, "editorial_proof_hash", id="no-proof"),
]


@pytest.mark.parametrize(("flaw", "gate"), BLOCKED_CASES)
def test_insert_as_published_blocked_by_each_gate(
    db: Conn, flaw: dict[str, Any], gate: str
) -> None:
    with pytest.raises(errors.CheckViolation, match=gate) as caught:
        insert_article(db, **{**PUBLISHABLE, **flaw}, status="published")
    assert caught.value.diag.constraint_name == "enforce_publication_gates"
    row = db.execute("SELECT count(*) FROM articles").fetchone()
    assert row == (0,)


@pytest.mark.parametrize(("flaw", "gate"), BLOCKED_CASES)
def test_promotion_blocked_by_each_gate(
    db: Conn, flaw: dict[str, Any], gate: str
) -> None:
    article_id = insert_article(db, **{**PUBLISHABLE, **flaw}, status="approved")
    with pytest.raises(errors.CheckViolation, match=gate):
        update_article(db, article_id, status="published")
    assert article_field(db, article_id, "status") == "approved"
    assert article_field(db, article_id, "published_at") is None


def test_default_article_reports_every_failed_gate(db: Conn) -> None:
    article_id = insert_article(db)
    with pytest.raises(errors.CheckViolation) as caught:
        update_article(db, article_id, status="published")
    message = str(caught.value)
    assert "analysis_tier" in message
    assert "human_approved" in message
    assert "editorial_proof_hash" in message


@pytest.mark.parametrize(
    "regression",
    [
        pytest.param({"analysis_tier": "rule_based"}, id="downgrade-tier"),
        pytest.param({"human_approved": False}, id="revoke-approval"),
        pytest.param({"editorial_proof_hash": None}, id="clear-proof"),
    ],
)
def test_published_article_cannot_lose_its_gates(
    db: Conn, regression: dict[str, Any]
) -> None:
    article_id = insert_article(db, **PUBLISHABLE, status="published")
    with pytest.raises(errors.CheckViolation):
        update_article(db, article_id, **regression)
    assert article_field(db, article_id, "status") == "published"


@pytest.mark.parametrize("bad_hash", ["", "   ", "not-a-digest", "A" * 64])
def test_malformed_proof_hash_cannot_satisfy_gate(db: Conn, bad_hash: str) -> None:
    with pytest.raises(errors.CheckViolation):
        insert_article(
            db, **{**PUBLISHABLE, "editorial_proof_hash": bad_hash}, status="published"
        )


# --- Publication gates: allowed ---------------------------------------------


def test_full_workflow_promotes_to_published(db: Conn) -> None:
    article_id = insert_article(db, status="drafting")
    update_article(db, article_id, status="compliance_hold")
    update_article(db, article_id, analysis_tier="llm_verified")
    update_article(
        db,
        article_id,
        status="approved",
        human_approved=True,
        editor_name="Desk Editor",
        editorial_proof_hash=PROOF_HASH,
    )
    assert article_field(db, article_id, "published_at") is None

    update_article(db, article_id, status="published")

    assert article_field(db, article_id, "status") == "published"
    assert article_field(db, article_id, "published_at") is not None


def test_direct_insert_as_published_passes_with_all_gates(db: Conn) -> None:
    article_id = insert_article(db, **PUBLISHABLE, status="published")
    assert article_field(db, article_id, "published_at") is not None


def test_republishing_edit_keeps_original_published_at(db: Conn) -> None:
    article_id = insert_article(db, **PUBLISHABLE, status="published")
    first = article_field(db, article_id, "published_at")
    update_article(db, article_id, headline="Corrected headline")
    assert article_field(db, article_id, "published_at") == first


def test_published_article_can_be_retracted(db: Conn) -> None:
    article_id = insert_article(db, **PUBLISHABLE, status="published")
    update_article(db, article_id, status="retracted")
    assert article_field(db, article_id, "status") == "retracted"


def test_retracted_article_cannot_return_without_gates(db: Conn) -> None:
    article_id = insert_article(db, **PUBLISHABLE, status="published")
    update_article(db, article_id, status="retracted", human_approved=False)
    with pytest.raises(errors.CheckViolation, match="human_approved"):
        update_article(db, article_id, status="published")


@pytest.mark.parametrize(
    "status", ["harvested", "triaged", "drafting", "compliance_hold", "approved"]
)
def test_unpublished_statuses_are_not_gated(db: Conn, status: str) -> None:
    article_id = insert_article(db, status=status)
    assert article_field(db, article_id, "status") == status


# --- Append-only ledger -----------------------------------------------------


def insert_source(conn: Conn, source_id: str = "S-1") -> None:
    conn.execute(
        """
        INSERT INTO source_registry
            (source_id, url, domain, tier, raw_sha256, archive_path, retrieved_at)
        VALUES (%s, 'https://example.gov/', 'example.gov', 'tier_1_primary', %s,
                'archive/warc/x.blob', now())
        """,
        (source_id, "b" * 64),
    )


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param(
            "UPDATE source_registry SET tier = 'tier_4_unverified'", id="tier"
        ),
        pytest.param(
            f"UPDATE source_registry SET raw_sha256 = '{'c' * 64}'", id="hash"
        ),
        pytest.param("DELETE FROM source_registry", id="delete"),
        pytest.param("TRUNCATE source_registry", id="truncate"),
        pytest.param("TRUNCATE articles, source_registry", id="truncate-multi"),
    ],
)
def test_source_registry_refuses_mutation(db: Conn, statement: str) -> None:
    insert_source(db)
    before = db.execute("SELECT * FROM source_registry").fetchall()

    with pytest.raises(errors.RaiseException, match="CRITICAL_AUDIT_VIOLATION"):
        db.execute(statement.encode())

    assert db.execute("SELECT * FROM source_registry").fetchall() == before


def test_source_registry_still_accepts_new_rows(db: Conn) -> None:
    insert_source(db, "S-1")
    insert_source(db, "S-2")
    assert registered_ids(db) == {"S-1", "S-2"}


# --- Quarantine -------------------------------------------------------------


def test_narrative_observations_accept_domains_five_and_six(db: Conn) -> None:
    for domain in (5, 6):
        db.execute(
            """
            INSERT INTO sociology_telemetry.narrative_observations
                (domain, narrative_label, observed_text, embedding)
            VALUES (%s, 'label', 'text', '[0.1, 0.2, 0.3]')
            """,
            (domain,),
        )
    row = db.execute(
        "SELECT count(*) FROM sociology_telemetry.narrative_observations"
    ).fetchone()
    assert row == (2,)


@pytest.mark.parametrize("domain", [1, 4, 7])
def test_narrative_observations_reject_other_domains(db: Conn, domain: int) -> None:
    with pytest.raises(errors.CheckViolation):
        db.execute(
            """
            INSERT INTO sociology_telemetry.narrative_observations
                (domain, narrative_label, observed_text)
            VALUES (%s, 'label', 'text')
            """,
            (domain,),
        )


def test_quarantine_is_closed_to_ordinary_roles(db: Conn) -> None:
    role = f"notitia_probe_{secrets.token_hex(4)}"
    with db.transaction():
        db.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(role)))
        row = db.execute(
            "SELECT has_schema_privilege(%s, 'sociology_telemetry', 'USAGE')", (role,)
        ).fetchone()
        assert row == (False,)
        raise psycopg.Rollback()


def test_quarantine_has_no_foreign_keys_across_schemas(db: Conn) -> None:
    row = db.execute(
        """
        SELECT count(*)
        FROM pg_constraint c
        JOIN pg_class local_table ON local_table.oid = c.conrelid
        JOIN pg_class other_table ON other_table.oid = c.confrelid
        WHERE c.contype = 'f'
          AND local_table.relnamespace <> other_table.relnamespace
        """
    ).fetchone()
    assert row == (0,)


# --- Custody ----------------------------------------------------------------


def transport_returning(
    body: bytes,
    content_type: str = "text/plain; charset=utf-8",
    status: int = 200,
    seen: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(
            status, content=body, headers={"content-type": content_type}
        )

    return httpx.MockTransport(handler)


def seal(
    tmp_path: Path, body: bytes, source_id: str = "S-1042", **response: Any
) -> dict[str, Any]:
    return dict(
        fetch_and_seal_source(
            "https://www.Example.gov/report",
            source_id,
            "tier_1_primary",
            archive_dir=tmp_path / "archive" / "warc",
            transport=transport_returning(body, **response),
        )
    )


def test_seal_digest_matches_bytes_on_disk(tmp_path: Path) -> None:
    body = "Résumé of findings\n".encode()
    custody = seal(tmp_path, body)

    sealed = Path(custody["archive_path"]).read_bytes()
    assert sealed == body
    assert custody["raw_sha256"] == hashlib.sha256(body).hexdigest()
    assert hashlib.sha256(sealed).hexdigest() == custody["raw_sha256"]


def test_seal_filename_follows_custody_format(tmp_path: Path) -> None:
    custody = seal(tmp_path, b"payload")
    path = Path(custody["archive_path"])

    source_id, timestamp, digest_prefix = path.stem.split("_")
    assert source_id == "S-1042"
    assert timestamp.endswith("Z")
    assert digest_prefix == custody["raw_sha256"][:12]
    assert path.suffix == ".blob"
    assert path.parent == tmp_path / "archive" / "warc"


def test_seal_returns_complete_receipt(tmp_path: Path) -> None:
    custody = seal(tmp_path, b"payload")
    assert custody["source_id"] == "S-1042"
    assert custody["tier"] == "tier_1_primary"
    assert custody["domain"] == "www.example.gov"
    assert custody["verbatim_text"] == "payload"
    assert custody["byte_length"] == 7
    assert custody["http_status"] == 200
    assert custody["retrieved_at"].endswith("+00:00")
    json.dumps(custody)


def test_seal_sends_required_user_agent(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    fetch_and_seal_source(
        "https://example.gov/",
        "S-1",
        "tier_3_secondary",
        archive_dir=tmp_path,
        transport=transport_returning(b"x", seen=seen),
    )
    assert seen[0].headers["user-agent"] == (
        "NotitiaCivica-SPEC1/1.0 (+https://notitiacivica.com; contact@evastararcana.com)"
    )
    assert USER_AGENT == seen[0].headers["user-agent"]


def test_seal_keeps_binary_payload_without_text(tmp_path: Path) -> None:
    body = b"%PDF-1.7\x00\xff\xfe"
    custody = seal(tmp_path, body, content_type="application/pdf")
    assert custody["verbatim_text"] is None
    assert Path(custody["archive_path"]).read_bytes() == body


def test_seal_refuses_lossy_text_decoding(tmp_path: Path) -> None:
    custody = seal(tmp_path, b"caf\xe9", content_type="text/plain; charset=utf-8")
    assert custody["verbatim_text"] is None


def test_seal_honours_declared_charset(tmp_path: Path) -> None:
    custody = seal(tmp_path, b"caf\xe9", content_type="text/plain; charset=latin-1")
    assert custody["verbatim_text"] == "café"


def test_sealed_blob_is_read_only(tmp_path: Path) -> None:
    custody = seal(tmp_path, b"payload")
    assert Path(custody["archive_path"]).stat().st_mode & 0o222 == 0


def test_seal_leaves_nothing_behind_on_http_error(tmp_path: Path) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        seal(tmp_path, b"gone", status=404)
    assert list((tmp_path / "archive" / "warc").iterdir()) == []


def test_seal_enforces_size_limit(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="limit"):
        fetch_and_seal_source(
            "https://example.gov/",
            "S-1",
            "tier_3_secondary",
            archive_dir=tmp_path,
            max_bytes=4,
            transport=transport_returning(b"too large"),
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "source_id", ["../../etc/passwd", "S-1/../x", "S-", "1042", ""]
)
def test_seal_rejects_unsafe_source_id(tmp_path: Path, source_id: str) -> None:
    with pytest.raises(ValueError, match="source_id"):
        seal(tmp_path, b"x", source_id=source_id)
    assert not (tmp_path / "archive").exists()


def test_seal_rejects_unknown_tier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tier"):
        fetch_and_seal_source(
            "https://example.gov/", "S-1", "tier_9", archive_dir=tmp_path
        )


@pytest.mark.parametrize("url", ["ftp://example.gov/x", "file:///etc/passwd", "nope"])
def test_seal_rejects_non_http_urls(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="URL"):
        fetch_and_seal_source(url, "S-1", "tier_1_primary", archive_dir=tmp_path)


# --- Bridge -----------------------------------------------------------------


def harvest(
    base_dir: Path, *records: dict[str, Any], name: str = "harvest_manifest.jsonl"
) -> Path:
    manifest = base_dir / name
    lines = [json.dumps(record) for record in records]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def sealed_record(base_dir: Path, source_id: str, body: bytes) -> dict[str, Any]:
    """Seal a payload and return its receipt with a base_dir-relative path."""
    record = seal(base_dir, body, source_id=source_id)
    record["archive_path"] = str(Path(record["archive_path"]).relative_to(base_dir))
    return record


def registered_ids(conn: Conn) -> set[str]:
    rows = conn.execute("SELECT source_id FROM source_registry").fetchall()
    return {row[0] for row in rows}


def test_runner_registers_verified_records(db: Conn, tmp_path: Path) -> None:
    first = sealed_record(tmp_path, "S-1", b"first payload")
    second = sealed_record(tmp_path, "S-2", b"second payload")

    report = run_manifest(harvest(tmp_path, first, second), db, tmp_path)

    assert report.registered == ["S-1", "S-2"]
    assert report.rejected == []
    row = db.execute(
        """
        SELECT url, domain, tier, raw_sha256, archive_path, verbatim_text
        FROM source_registry WHERE source_id = 'S-1'
        """
    ).fetchone()
    assert row == (
        first["url"],
        "www.example.gov",
        "tier_1_primary",
        hashlib.sha256(b"first payload").hexdigest(),
        first["archive_path"],
        "first payload",
    )


def test_runner_rejects_tampered_blob_and_keeps_valid_records(
    db: Conn, tmp_path: Path
) -> None:
    tampered = sealed_record(tmp_path, "S-1", b"original")
    valid = sealed_record(tmp_path, "S-2", b"untouched")
    blob = tmp_path / tampered["archive_path"]
    blob.chmod(0o644)
    blob.write_bytes(b"altered")

    report = run_manifest(harvest(tmp_path, tampered, valid), db, tmp_path)

    assert report.registered == ["S-2"]
    assert [item.line_number for item in report.rejected] == [1]
    assert "SHA-256 mismatch" in report.rejected[0].reason
    assert registered_ids(db) == {"S-2"}


def test_runner_rejects_missing_blob(db: Conn, tmp_path: Path) -> None:
    record = sealed_record(tmp_path, "S-1", b"payload")
    (tmp_path / record["archive_path"]).unlink()

    report = run_manifest(harvest(tmp_path, record), db, tmp_path)

    assert "blob not found" in report.rejected[0].reason
    assert registered_ids(db) == set()


def test_runner_refuses_paths_outside_base_dir(db: Conn, tmp_path: Path) -> None:
    base_dir = tmp_path / "harvest"
    base_dir.mkdir()
    outside = tmp_path / "secret.blob"
    outside.write_bytes(b"secret")
    record = sealed_record(base_dir, "S-1", b"payload")
    record["archive_path"] = "../secret.blob"
    record["raw_sha256"] = hashlib.sha256(b"secret").hexdigest()

    report = run_manifest(harvest(base_dir, record), db, base_dir)

    assert "escapes base directory" in report.rejected[0].reason
    assert registered_ids(db) == set()


def test_runner_never_trusts_manifest_text(db: Conn, tmp_path: Path) -> None:
    record = sealed_record(tmp_path, "S-1", b"what the source said")
    record["verbatim_text"] = "what the source never said"

    run_manifest(harvest(tmp_path, record), db, tmp_path)

    row = db.execute("SELECT verbatim_text FROM source_registry").fetchone()
    assert row == ("what the source said",)


def test_runner_is_idempotent(db: Conn, tmp_path: Path) -> None:
    manifest = harvest(tmp_path, sealed_record(tmp_path, "S-1", b"payload"))

    run_manifest(manifest, db, tmp_path)
    report = run_manifest(manifest, db, tmp_path)

    assert report.registered == []
    assert report.already_registered == ["S-1"]
    assert report.rejected == []


def test_runner_refuses_to_overwrite_a_sealed_source(db: Conn, tmp_path: Path) -> None:
    run_manifest(
        harvest(tmp_path, sealed_record(tmp_path, "S-1", b"original")), db, tmp_path
    )

    report = run_manifest(
        harvest(tmp_path, sealed_record(tmp_path, "S-1", b"replacement")), db, tmp_path
    )

    assert "different digest" in report.rejected[0].reason
    row = db.execute("SELECT raw_sha256 FROM source_registry").fetchone()
    assert row == (hashlib.sha256(b"original").hexdigest(),)


@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param({"tier": "tier_9"}, id="bad-tier"),
        pytest.param({"source_id": "../evil"}, id="bad-source-id"),
        pytest.param({"url": "ftp://example.gov/x"}, id="bad-url"),
        pytest.param({"raw_sha256": "abc"}, id="bad-digest"),
        pytest.param({"retrieved_at": "2026-10-31T12:00:00"}, id="naive-timestamp"),
    ],
)
def test_runner_rejects_invalid_records(
    db: Conn, tmp_path: Path, corruption: dict[str, str]
) -> None:
    record = sealed_record(tmp_path, "S-1", b"payload") | corruption

    report = run_manifest(harvest(tmp_path, record), db, tmp_path)

    assert len(report.rejected) == 1
    assert registered_ids(db) == set()


def test_runner_survives_malformed_lines(db: Conn, tmp_path: Path) -> None:
    valid = sealed_record(tmp_path, "S-1", b"payload")
    manifest = tmp_path / "harvest_manifest.jsonl"
    manifest.write_text(f"{{not json\n\n{json.dumps(valid)}\n", encoding="utf-8")

    report = run_manifest(manifest, db, tmp_path)

    assert report.registered == ["S-1"]
    assert [item.line_number for item in report.rejected] == [1]


def moby_line(base_dir: Path, body: bytes, **overrides: Any) -> dict[str, Any]:
    """Write a blob and return its run_moby_dork_batch.sh manifest line."""
    digest = hashlib.sha256(body).hexdigest()
    blob = base_dir / "dorks" / f"{digest[:12]}.html"
    blob.parent.mkdir(exist_ok=True)
    blob.write_bytes(body)
    return {
        "domain": "agency.example.gov",
        "retrieved_at_utc": "2026-10-31T12:00:00Z",
        "analysis_tier": "rule_based",
        "sha256": digest,
        "file": str(blob.relative_to(base_dir)),
    } | overrides


def test_moby_record_is_registered_conservatively(db: Conn, tmp_path: Path) -> None:
    line = moby_line(tmp_path, b"harvested page")

    report = run_manifest(harvest(tmp_path, line), db, tmp_path)

    assert report.rejected == []
    assert len(report.registered) == 1
    assert report.registered[0].startswith("S-")
    row = db.execute(
        """
        SELECT source_id, url, domain, tier, raw_sha256, archive_path, verbatim_text,
               retrieved_at = '2026-10-31T12:00:00Z'::timestamptz
        FROM source_registry
        """
    ).fetchone()
    assert row == (
        report.registered[0],
        None,
        "agency.example.gov",
        "tier_4_unverified",
        line["sha256"],
        line["file"],
        "harvested page",
        True,
    )


def test_moby_naive_timestamp_is_read_as_utc(db: Conn, tmp_path: Path) -> None:
    line = moby_line(tmp_path, b"page", retrieved_at_utc="2026-10-31T12:00:00")

    report = run_manifest(harvest(tmp_path, line), db, tmp_path)

    assert report.rejected == []
    row = db.execute(
        "SELECT retrieved_at = '2026-10-31T12:00:00Z'::timestamptz FROM source_registry"
    ).fetchone()
    assert row == (True,)


def test_moby_rerun_does_not_duplicate(db: Conn, tmp_path: Path) -> None:
    manifest = harvest(tmp_path, moby_line(tmp_path, b"page"))

    first = run_manifest(manifest, db, tmp_path)
    second = run_manifest(manifest, db, tmp_path)

    assert second.registered == []
    assert second.already_registered == first.registered
    assert second.rejected == []
    assert len(registered_ids(db)) == 1


def test_moby_same_bytes_on_two_domains_are_two_sources(
    db: Conn, tmp_path: Path
) -> None:
    lines = [
        moby_line(tmp_path, b"page"),
        moby_line(tmp_path, b"page", domain="mirror.example.org"),
    ]

    report = run_manifest(harvest(tmp_path, *lines), db, tmp_path)

    assert len(set(report.registered)) == 2


def test_moby_tampered_blob_is_rejected(db: Conn, tmp_path: Path) -> None:
    line = moby_line(tmp_path, b"original")
    (tmp_path / line["file"]).write_bytes(b"altered")

    report = run_manifest(harvest(tmp_path, line), db, tmp_path)

    assert "SHA-256 mismatch" in report.rejected[0].reason
    assert registered_ids(db) == set()


def test_moby_path_cannot_leave_base_dir(db: Conn, tmp_path: Path) -> None:
    base_dir = tmp_path / "harvest"
    base_dir.mkdir()
    (tmp_path / "secret").write_bytes(b"secret")
    line = moby_line(
        base_dir,
        b"page",
        file="../secret",
        sha256=hashlib.sha256(b"secret").hexdigest(),
    )

    report = run_manifest(harvest(base_dir, line), db, base_dir)

    assert "escapes base directory" in report.rejected[0].reason


@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param({"analysis_tier": "llm_verified"}, id="claims-verification"),
        pytest.param({"domain": "bad domain/x"}, id="bad-domain"),
        pytest.param({"sha256": "abc"}, id="bad-digest"),
        pytest.param({"retrieved_at_utc": "yesterday"}, id="bad-timestamp"),
    ],
)
def test_moby_rejects_invalid_records(
    db: Conn, tmp_path: Path, corruption: dict[str, str]
) -> None:
    line = moby_line(tmp_path, b"page") | corruption

    report = run_manifest(harvest(tmp_path, line), db, tmp_path)

    assert len(report.rejected) == 1
    assert registered_ids(db) == set()


def test_manifest_can_mix_moby_lines_and_receipts(db: Conn, tmp_path: Path) -> None:
    receipt = sealed_record(tmp_path, "S-1", b"sealed")
    line = moby_line(tmp_path, b"harvested")

    report = run_manifest(harvest(tmp_path, receipt, line), db, tmp_path)

    assert report.rejected == []
    assert len(report.registered) == 2


def test_cli_exit_codes_reflect_rejections(
    ledger_dsn: str, db: Conn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    valid = sealed_record(tmp_path, "S-1", b"payload")
    clean = harvest(tmp_path, valid, name="clean.jsonl")
    dirty = harvest(tmp_path, valid | {"tier": "tier_9"}, name="dirty.jsonl")
    common = ["--dsn", ledger_dsn, "--base-dir", str(tmp_path)]

    assert main([str(clean), *common]) == 0
    assert "registered=1" in capsys.readouterr().out
    assert main([str(dirty), *common]) == 1
    assert "line 1" in capsys.readouterr().err
    assert registered_ids(db) == {"S-1"}
