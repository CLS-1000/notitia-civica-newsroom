"""Master Control Cockpit: the tabs around the 3-pane Editorial Desk.

Fleet Control, Triage Queue, Evidence Vault, and Publish & Deploy, plus the
status badge in the top bar. Everything here reads the ledger except two
actions: starting a harvest and starting a deploy, both of which run on
background threads and report through HTMX polling.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from pathlib import Path
from typing import Any, Final

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from psycopg.rows import dict_row

from newsroom.penfield_desk.deploy import Deployer
from newsroom.penfield_desk.fleet import (
    ENGINE_BY_KEY,
    ENGINES,
    FAMILIES,
    Fleet,
    card,
    domains_in,
    family_of,
    ledger_counts,
)

PAGE_SIZE: Final = 50
RAW_PREVIEW_BYTES: Final = 512 * 1024
HEX_PREVIEW_BYTES: Final = 4096
TIERS: Final = {
    "tier_1_primary": "Tier 1 · Primary",
    "tier_2_secondary": "Tier 2 · Secondary",
    "tier_3_tertiary": "Tier 3 · Tertiary",
    "tier_4_unverified": "Tier 4 · Unverified",
}


def status_badge(dsn: str) -> dict[str, Any]:
    """The top-bar badge. A down database is shown, never raised."""
    try:
        with (
            psycopg.connect(dsn, autocommit=True, connect_timeout=3) as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            row = cursor.execute(
                """
                SELECT (SELECT count(*) FROM articles) AS articles,
                       (SELECT count(*) FROM articles
                        WHERE status = 'published') AS published,
                       (SELECT max(created_at) FROM triage_records) AS last_harvest
                """
            ).fetchone()
    except psycopg.Error as error:
        return {"db_ok": False, "db_error": str(error).splitlines()[0][:160]}
    assert row is not None
    last = row["last_harvest"]
    return {
        "db_ok": True,
        "articles": row["articles"],
        "published": row["published"],
        "last_harvest": last.isoformat(timespec="minutes") if last else None,
    }


def _like(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def resolve_blob(archive_root: Path, archive_path: str) -> Path:
    """Locate a sealed blob, refusing any path that leaves the archive root."""
    root = archive_root.resolve()
    blob = (root / archive_path).resolve()
    if not blob.is_relative_to(root):
        raise HTTPException(403, "archive_path escapes the archive root")
    return blob


def raw_view(blob: Path, expected_sha256: str) -> dict[str, Any]:
    """Read a blob from disk and re-check its seal."""
    if not blob.is_file():
        return {"found": False, "path": str(blob)}
    digest = hashlib.sha256()
    size = 0
    head = b""
    with blob.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            if len(head) < RAW_PREVIEW_BYTES:
                head += chunk[: RAW_PREVIEW_BYTES - len(head)]
            digest.update(chunk)
            size += len(chunk)
    actual = digest.hexdigest()
    try:
        text: str | None = head.decode("utf-8")
    except UnicodeDecodeError as error:
        # A multibyte character cut at the preview edge is still text.
        text = (
            head[: error.start].decode("utf-8")
            if size > RAW_PREVIEW_BYTES and error.start > len(head) - 4
            else None
        )
    hex_lines = []
    if text is None:
        sample = head[:HEX_PREVIEW_BYTES]
        for offset in range(0, len(sample), 16):
            row = sample[offset : offset + 16]
            printable = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
            hex_lines.append(f"{offset:08x}  {row.hex(' '):<47}  {printable}")
    return {
        "found": True,
        "path": str(blob),
        "size": size,
        "sha256": actual,
        "seal_ok": actual == expected_sha256,
        "text": text,
        "hex_lines": hex_lines,
        "truncated": size
        > (RAW_PREVIEW_BYTES if text is not None else HEX_PREVIEW_BYTES),
    }


def register(
    app: FastAPI,
    templates: Jinja2Templates,
    *,
    fleet: Fleet,
    deployer: Deployer,
    archive_root: Path,
) -> None:
    app.state.fleet = fleet
    app.state.deployer = deployer
    templates.env.globals["tier_label"] = lambda tier: TIERS.get(tier, tier)
    templates.env.globals["family_of"] = family_of

    def connect() -> psycopg.Connection[Any]:
        return psycopg.connect(app.state.dsn, autocommit=True)

    def page(request: Request, name: str, tab: str, **context: Any) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            name,
            {"tab": tab, "status": status_badge(app.state.dsn), **context},
        )

    @app.get("/status", response_class=HTMLResponse)
    def status(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "_status.html", {"status": status_badge(app.state.dsn)}
        )

    # --- Fleet ----------------------------------------------------------------

    def cards() -> list[dict[str, Any]]:
        with connect() as conn:
            counts = ledger_counts(conn)
        return [card(engine, fleet, counts) for engine in ENGINES]

    @app.get("/fleet", response_class=HTMLResponse)
    def fleet_page(request: Request) -> HTMLResponse:
        return page(request, "fleet.html", "fleet", cards=cards())

    @app.get("/fleet/{key}", response_class=HTMLResponse)
    def engine_card(request: Request, key: str) -> HTMLResponse:
        engine = ENGINE_BY_KEY.get(key)
        if engine is None:
            raise HTTPException(404, "no such engine")
        with connect() as conn:
            item = card(engine, fleet, ledger_counts(conn))
        return templates.TemplateResponse(request, "_engine_card.html", {"c": item})

    @app.post("/fleet/{key}/run", response_class=HTMLResponse)
    def run_engine(request: Request, key: str) -> HTMLResponse:
        engine = ENGINE_BY_KEY.get(key)
        if engine is None:
            raise HTTPException(404, "no such engine")
        blocker = fleet.blocker(engine)
        if blocker is not None:
            raise HTTPException(409, blocker)
        started = fleet.start(engine)
        with connect() as conn:
            item = card(engine, fleet, ledger_counts(conn))
        item["notice"] = None if started else "A run is already in progress."
        return templates.TemplateResponse(request, "_engine_card.html", {"c": item})

    # --- Triage ---------------------------------------------------------------

    def triage_rows(family: str, min_score: float) -> list[dict[str, Any]]:
        clauses = ["t.anomaly_score >= %s"]
        params: list[Any] = [min_score]
        if family:
            clauses.append("t.subject_domain = ANY(%s)")
            params.append(domains_in(family) or [family.lower()])
        with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
            return cursor.execute(
                f"""
                SELECT t.record_id, t.adapter_name, t.subject_domain,
                       t.anomaly_score, t.created_at, t.source_ids,
                       t.public_question, a.slug, a.status AS article_status
                FROM triage_records t
                LEFT JOIN articles a ON a.triage_record_id = t.record_id
                WHERE {" AND ".join(clauses)}
                ORDER BY t.anomaly_score DESC, t.created_at DESC
                LIMIT 500
                """,  # clauses are fixed strings; values are bound
                params,
            ).fetchall()

    def triage_filters(request: Request) -> tuple[str, float]:
        family = request.query_params.get("family", "")
        if family and family not in FAMILIES:
            family = ""
        try:
            score = float(request.query_params.get("min_score", "0"))
        except ValueError:
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        return family, max(0.0, min(100.0, score))

    @app.get("/triage", response_class=HTMLResponse)
    def triage_page(request: Request) -> HTMLResponse:
        family, score = triage_filters(request)
        rows = triage_rows(family, score)
        context = {
            "rows": rows,
            "family": family,
            "min_score": score,
            "families": FAMILIES,
        }
        if (
            request.headers.get("HX-Request")
            and request.headers.get("HX-Target") == "triage-rows"
        ):
            return templates.TemplateResponse(request, "_triage_rows.html", context)
        return page(request, "triage.html", "triage", **context)

    @app.get("/triage/{record_id}", response_class=HTMLResponse)
    def triage_detail(request: Request, record_id: uuid.UUID) -> HTMLResponse:
        with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
            lead = cursor.execute(
                """
                SELECT t.*, a.slug FROM triage_records t
                LEFT JOIN articles a ON a.triage_record_id = t.record_id
                WHERE t.record_id = %s
                """,
                (record_id,),
            ).fetchone()
        if lead is None:
            raise HTTPException(404, "no such lead")
        return templates.TemplateResponse(request, "_lead_detail.html", {"lead": lead})

    # --- Evidence Vault -------------------------------------------------------

    @app.get("/vault", response_class=HTMLResponse)
    def vault_page(request: Request) -> HTMLResponse:
        query = request.query_params.get("q", "").strip()
        flagged_only = request.query_params.get("flagged") == "on"
        try:
            page_no = max(1, int(request.query_params.get("page", "1")))
        except ValueError:
            page_no = 1
        clauses = ["TRUE"]
        params: list[Any] = []
        if query:
            like = _like(query)
            clauses.append(
                "(s.source_id ILIKE %s OR s.domain ILIKE %s OR s.url ILIKE %s "
                "OR s.raw_sha256 ILIKE %s)"
            )
            params += [like, like, like, like]
        if flagged_only:
            clauses.append("s.tier_review_flagged")
        where = " AND ".join(clauses)
        with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
            total_row = cursor.execute(
                f"SELECT count(*) AS n FROM source_registry s WHERE {where}",
                params,
            ).fetchone()
            rows = cursor.execute(
                f"""
                SELECT s.source_id, s.url, s.domain, s.tier, s.raw_sha256,
                       s.retrieved_at, s.tier_review_flagged, s.tier_review_reason,
                       r.reason AS restricted_reason
                FROM source_registry s
                LEFT JOIN restricted_sources r USING (source_id)
                WHERE {where}
                ORDER BY s.tier_review_flagged DESC, s.retrieved_at DESC,
                         s.source_id
                LIMIT %s OFFSET %s
                """,  # clauses are fixed strings; values are bound
                [*params, PAGE_SIZE, (page_no - 1) * PAGE_SIZE],
            ).fetchall()
        total = total_row["n"] if total_row else 0
        context = {
            "rows": rows,
            "q": query,
            "flagged": flagged_only,
            "page": page_no,
            "pages": max(1, math.ceil(total / PAGE_SIZE)),
            "total": total,
        }
        if request.headers.get("HX-Target") == "vault-results":
            return templates.TemplateResponse(request, "_vault_rows.html", context)
        return page(request, "vault.html", "vault", **context)

    @app.get("/vault/{source_id}/raw", response_class=HTMLResponse)
    def vault_raw(request: Request, source_id: str) -> HTMLResponse:
        with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
            source = cursor.execute(
                """
                SELECT s.*, r.reason AS restricted_reason
                FROM source_registry s
                LEFT JOIN restricted_sources r USING (source_id)
                WHERE s.source_id = %s
                """,
                (source_id,),
            ).fetchone()
        if source is None:
            raise HTTPException(404, "no such source")
        blob = resolve_blob(archive_root, source["archive_path"])
        return templates.TemplateResponse(
            request,
            "_source_raw.html",
            {"source": source, "raw": raw_view(blob, source["raw_sha256"])},
        )

    # --- Publish & Deploy -----------------------------------------------------

    def signed_articles() -> list[dict[str, Any]]:
        with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
            return cursor.execute(
                """
                SELECT slug, headline, section, badge_type, editor_name,
                       editorial_proof_hash, published_at
                FROM articles
                WHERE status = 'published' AND analysis_tier = 'llm_verified'
                ORDER BY published_at DESC NULLS LAST
                """
            ).fetchall()

    def deploy_context() -> dict[str, Any]:
        return {"run": deployer.current, "config": deployer.config}

    @app.get("/publish", response_class=HTMLResponse)
    def publish_page(request: Request) -> HTMLResponse:
        return page(
            request,
            "publish.html",
            "publish",
            articles=signed_articles(),
            **deploy_context(),
        )

    @app.post("/publish/deploy", response_class=HTMLResponse)
    def start_deploy(request: Request) -> HTMLResponse:
        started = deployer.start()
        context = {**deploy_context(), "busy": not started}
        return templates.TemplateResponse(request, "_deploy_log.html", context)

    @app.get("/publish/deploy/log", response_class=HTMLResponse)
    def deploy_log(request: Request) -> Response:
        return templates.TemplateResponse(request, "_deploy_log.html", deploy_context())
