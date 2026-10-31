"""Real PostgreSQL fixtures shared by every test suite.

Set NOTITIA_TEST_DSN to use an existing server; throwaway databases are created
there and dropped afterwards. Otherwise a private cluster is started for the run,
using a system pgvector or the copy from scripts/bootstrap_test_pgvector.sh.
"""

from __future__ import annotations

import os
import secrets
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION = REPO_ROOT / "migrations" / "001_init_ledger.sql"
LOCAL_PGVECTOR = REPO_ROOT / ".pgdev" / "pgvector"
PG_BIN = Path("/usr/lib/postgresql/18/bin")

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
