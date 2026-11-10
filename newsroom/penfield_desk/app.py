"""Penfield Desk: the local editorial workstation on 127.0.0.1:8000.

There is no login, because the desk is meant for one editor on one machine. Host
and Origin checks stop other websites reaching it through the editor's browser.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final

import psycopg
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from newsroom.dors_venabili.gatekeeper import (
    GateResult,
    extract_citations,
    normalize_text,
    run_all_gates,
    split_paragraphs,
    verify_grounding,
    verify_uncertainty_disclosures,
)
from newsroom.harlan_eternity.signoff import (
    CORRECTION_TYPES,
    DEFAULT_DSN,
    REQUIRED_CHECKS,
    CorrectionError,
    SignOffError,
    missing_checks,
    record_correction,
    sign_and_promote_article,
)
from newsroom.penfield_desk.claude_drafter import (
    ClaudeDrafter,
    DraftError,
    DraftSource,
)

HOST: Final = "127.0.0.1"
PORT: Final = 8000
ALLOWED_HOSTS: Final = [HOST, "localhost"]
ALLOWED_ORIGINS: Final = frozenset(
    {f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"}
)
SOURCE_ROLES: Final = (
    "government_record",
    "official",
    "independent_expert",
    "affected_party",
    "industry",
    "advocacy",
)
BADGES: Final = ("ANALYSIS", "FACT BRIEF", "DEEP EXPLAINER", "INVESTIGATION")
EDITABLE_STATUSES: Final = (
    "harvested",
    "triaged",
    "drafting",
    "compliance_hold",
    "approved",
)

_HERE: Final = Path(__file__).parent
_SENTENCE: Final = re.compile(r"\S.*?(?:[.!?](?=\s|$)|$)", re.DOTALL)
_TAG_SPLIT: Final = re.compile(r"(\[S-\d+\])")
_QUOTED: Final = re.compile(r'"([^"]+)"')
_WORD: Final = re.compile(r"[a-z0-9$][a-z0-9,.$]*")

DrafterFactory = Callable[[], ClaudeDrafter]


def connection(request: Request) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(request.app.state.dsn, autocommit=True) as conn:
        yield conn


DbConn = Annotated[psycopg.Connection[Any], Depends(connection)]


class OriginGuardMiddleware:
    """Refuse state-changing requests sent from any other website."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] not in {"GET", "HEAD"}:
            headers = {key: value for key, value in scope["headers"]}
            origin = headers.get(b"origin", b"").decode("latin-1")
            fetch_site = headers.get(b"sec-fetch-site", b"").decode("latin-1")
            if (origin and origin not in ALLOWED_ORIGINS) or fetch_site == "cross-site":
                await Response("cross-origin request refused", 403)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


@dataclass(frozen=True, slots=True)
class Sentence:
    index: int
    text: str
    normalized: str


def split_sentences(text: str) -> list[Sentence]:
    return [
        Sentence(index, match.group(), normalize_text(match.group()))
        for index, match in enumerate(_SENTENCE.finditer(text))
    ]


def locate_sentence(claim: str, sentences: list[Sentence]) -> int | None:
    """Return the sentence holding a quote from claim, else the closest in wording."""
    if not sentences:
        return None
    normalized = normalize_text(claim)
    for quote in _QUOTED.findall(normalized):
        for sentence in sentences:
            if quote in sentence.normalized:
                return sentence.index

    words = set(_WORD.findall(normalized.lower()))
    best = max(
        sentences,
        key=lambda item: len(words & set(_WORD.findall(item.normalized.lower()))),
    )
    overlap = words & set(_WORD.findall(best.normalized.lower()))
    return best.index if overlap else None


def _preview_paragraphs(
    body: str, texts: dict[str, str], sentences: dict[str, list[Sentence]]
) -> list[dict[str, Any]]:
    grounding = verify_grounding(body, texts)
    results: list[dict[str, Any]] = []
    for paragraph, status in zip(
        split_paragraphs(body), grounding["paragraphs"], strict=True
    ):
        segments: list[dict[str, Any]] = []
        claim = ""
        for piece in _TAG_SPLIT.split(paragraph):
            if not piece:
                continue
            source_id = piece[1:-1] if _TAG_SPLIT.fullmatch(piece) else None
            if source_id is None:
                claim = piece
                segments.append({"text": piece})
                continue
            known = source_id in texts
            segments.append(
                {
                    "text": piece,
                    "source_id": source_id,
                    "known": known,
                    "sentence": (
                        locate_sentence(claim, sentences.get(source_id, []))
                        if known
                        else None
                    ),
                }
            )
        results.append({"segments": segments, **status})
    return results


def roles_from_form(form: dict[str, Any]) -> dict[str, str]:
    prefix = "role__"
    return {
        key.removeprefix(prefix): str(value)
        for key, value in form.items()
        if key.startswith(prefix) and value in SOURCE_ROLES
    }


def badge_from_form(form: dict[str, Any], article: dict[str, Any]) -> str:
    badge = str(form.get("badge_type", ""))
    return badge if badge in BADGES else str(article["badge_type"])


def disclosures_from_form(form: dict[str, Any], article: dict[str, Any]) -> list[str]:
    """Return one disclosure per non-blank line, or the stored list if absent."""
    if "uncertainty_disclosures" not in form:
        return [str(item) for item in article["uncertainty_disclosures"]]
    lines = str(form["uncertainty_disclosures"]).splitlines()
    return [line.strip() for line in lines if line.strip()]


def confirmed_from_form(form: dict[str, Any]) -> bool:
    return form.get("no_open_questions_confirmed") == "on"


def checks_from_form(form: dict[str, Any]) -> dict[str, bool]:
    return {name: form.get(f"check__{name}") == "on" for name in REQUIRED_CHECKS}


def _slug_for(headline: str, record_id: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", headline.lower()).strip("-")[:60].strip("-")
    digest = hashlib.sha256(record_id.encode()).hexdigest()[:8]
    return f"{stem or 'draft'}-{digest}"


def create_app(
    dsn: str | None = None,
    drafter_factory: DrafterFactory = ClaudeDrafter,
) -> FastAPI:
    database = dsn or os.environ.get("DATABASE_URL", DEFAULT_DSN)
    app = FastAPI(
        title="Penfield Desk", docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.dsn = database
    app.add_middleware(OriginGuardMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
    templates = Jinja2Templates(directory=_HERE / "templates")

    def fetch_article(conn: psycopg.Connection[Any], slug: str) -> dict[str, Any]:
        with conn.cursor(row_factory=dict_row) as cursor:
            article = cursor.execute(
                "SELECT * FROM articles WHERE slug = %s", (slug,)
            ).fetchone()
        if article is None:
            raise HTTPException(404, "no such article")
        return article

    def fetch_sources(
        conn: psycopg.Connection[Any], source_ids: list[str]
    ) -> list[dict[str, Any]]:
        with conn.cursor(row_factory=dict_row) as cursor:
            return cursor.execute(
                """
                SELECT source_id, url, domain, tier, raw_sha256,
                       COALESCE(verbatim_text, '') AS text
                FROM source_registry WHERE source_id = ANY(%s)
                ORDER BY source_id
                """,
                (source_ids,),
            ).fetchall()

    def evaluate(
        conn: psycopg.Connection[Any],
        article: dict[str, Any],
        form: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the gates on the form as it stands, for display only."""
        body = str(form.get("body_markdown", ""))
        sources = fetch_sources(conn, article["source_ids"])
        texts = {item["source_id"]: item["text"] for item in sources}
        tiers = {item["source_id"]: item["tier"] for item in sources}
        gates: list[GateResult] = run_all_gates(
            headline=str(form.get("headline", "")),
            body=body,
            source_texts=texts,
            roles=roles_from_form(form),
            is_private_person=form.get("is_private_person") == "on",
            right_of_reply_logged=form.get("right_of_reply_logged") == "on",
            badge_type=badge_from_form(form, article),
            source_tiers=tiers,
        )
        gates.append(
            GateResult(
                "uncertainty",
                "Uncertainty disclosures",
                verify_uncertainty_disclosures(
                    disclosures_from_form(form, article), confirmed_from_form(form)
                ),
            )
        )
        checks = checks_from_form(form)
        unticked = missing_checks(checks)
        sentences = {key: split_sentences(value) for key, value in texts.items()}
        editable = article["status"] in EDITABLE_STATUSES
        return {
            "article": article,
            "gates": gates,
            "gates_passed": all(gate.passed for gate in gates),
            "unticked": len(unticked),
            "paragraphs": _preview_paragraphs(body, texts, sentences),
            "can_sign": (
                editable and not unticked and all(gate.passed for gate in gates)
            ),
        }

    def save_form(
        conn: psycopg.Connection[Any], article: dict[str, Any], form: dict[str, Any]
    ) -> None:
        conn.execute(
            """
            UPDATE articles
            SET headline = %s, deck = %s, body_markdown = %s, source_roles = %s,
                badge_type = %s, is_private_person = %s, right_of_reply_logged = %s,
                uncertainty_disclosures = %s, no_open_questions_confirmed = %s
            WHERE article_id = %s AND status = ANY(%s::pub_status_enum[])
            """,
            (
                form.get("headline", ""),
                form.get("deck", ""),
                form.get("body_markdown", ""),
                Jsonb(roles_from_form(form)),
                badge_from_form(form, article),
                form.get("is_private_person") == "on",
                form.get("right_of_reply_logged") == "on",
                Jsonb(disclosures_from_form(form, article)),
                confirmed_from_form(form),
                article["article_id"],
                list(EDITABLE_STATUSES),
            ),
        )

    def fetch_corrections(
        conn: psycopg.Connection[Any], article_id: uuid.UUID
    ) -> list[dict[str, Any]]:
        with conn.cursor(row_factory=dict_row) as cursor:
            return cursor.execute(
                """
                SELECT correction_type, original_text, corrected_text, explanation,
                       editor_signature, corrected_at
                FROM article_corrections WHERE article_id = %s
                ORDER BY corrected_at, correction_id
                """,
                (article_id,),
            ).fetchall()

    @app.get("/", response_class=HTMLResponse)
    def desk(
        request: Request,
        conn: DbConn,
        article: str | None = None,
    ) -> HTMLResponse:
        with conn.cursor(row_factory=dict_row) as cursor:
            leads = cursor.execute(
                """
                SELECT t.record_id, t.adapter_name, t.subject_domain, t.anomaly_score,
                       t.decomposition_receipt, t.source_ids, t.created_at, a.slug
                FROM triage_records t
                LEFT JOIN articles a ON a.triage_record_id = t.record_id
                ORDER BY t.created_at DESC
                LIMIT 100
                """
            ).fetchall()

        context: dict[str, Any] = {
            "leads": leads,
            "required_checks": REQUIRED_CHECKS,
            "source_roles": SOURCE_ROLES,
            "badges": BADGES,
            "correction_types": CORRECTION_TYPES,
            "article": None,
            "sources": [],
        }
        if article is not None:
            row = fetch_article(conn, article)
            sources = fetch_sources(conn, row["source_ids"])
            for source in sources:
                source["sentences"] = split_sentences(source["text"])
            form = {
                "headline": row["headline"],
                "body_markdown": row["body_markdown"],
                "badge_type": row["badge_type"],
                "is_private_person": "on" if row["is_private_person"] else "",
                "right_of_reply_logged": "on" if row["right_of_reply_logged"] else "",
                "no_open_questions_confirmed": (
                    "on" if row["no_open_questions_confirmed"] else ""
                ),
                **{f"role__{key}": value for key, value in row["source_roles"].items()},
            }
            context.update(evaluate(conn, row, form))
            context["sources"] = sources
            context["corrections"] = fetch_corrections(conn, row["article_id"])
            context["lead"] = next(
                (item for item in leads if item["slug"] == article), None
            )
        return templates.TemplateResponse(request, "desk.html", context)

    @app.post("/leads/{record_id}/draft")
    def draft_lead(
        record_id: uuid.UUID,
        conn: DbConn,
    ) -> Response:
        with conn.cursor(row_factory=dict_row) as cursor:
            lead = cursor.execute(
                "SELECT * FROM triage_records WHERE record_id = %s", (record_id,)
            ).fetchone()
            if lead is None:
                raise HTTPException(404, "no such lead")
            existing = cursor.execute(
                "SELECT slug FROM articles WHERE triage_record_id = %s", (record_id,)
            ).fetchone()
        if existing is not None:
            return _go_to(existing["slug"])

        sources = [
            DraftSource(item["source_id"], item["tier"], item["url"], item["text"])
            for item in fetch_sources(conn, lead["source_ids"])
        ]
        try:
            draft = drafter_factory().draft(lead["paste_ready_prompt"], sources)
        except DraftError as error:
            raise HTTPException(502, str(error)) from error

        slug = _slug_for(draft.headline, str(record_id))
        conn.execute(
            """
            INSERT INTO articles
                (slug, section, badge_type, headline, deck, body_markdown,
                 triage_record_id, source_ids, verified_claims,
                 uncertainty_disclosures)
            VALUES (%s, %s, 'ANALYSIS', %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                slug,
                lead["subject_domain"],
                draft.headline,
                draft.deck,
                draft.body_markdown,
                record_id,
                [source.source_id for source in sources],
                Jsonb([claim.model_dump() for claim in draft.verified_claims]),
                Jsonb(draft.uncertainty_disclosures),
            ),
        )
        return _go_to(slug)

    def _go_to(slug: str) -> Response:
        return RedirectResponse(
            f"/?article={slug}",
            status_code=303,
            headers={"HX-Redirect": f"/?article={slug}"},
        )

    @app.post("/articles/{slug}/preview", response_class=HTMLResponse)
    async def preview(
        request: Request,
        slug: str,
        conn: DbConn,
    ) -> HTMLResponse:
        form = dict(await request.form())
        context = evaluate(conn, fetch_article(conn, slug), form)
        return templates.TemplateResponse(request, "_preview.html", context)

    @app.post("/articles/{slug}/validate", response_class=HTMLResponse)
    async def validate(
        request: Request,
        slug: str,
        conn: DbConn,
    ) -> HTMLResponse:
        form = dict(await request.form())
        context = evaluate(conn, fetch_article(conn, slug), form)
        context["live_update"] = True
        return templates.TemplateResponse(request, "_live.html", context)

    @app.post("/articles/{slug}/save", status_code=204)
    async def save(
        request: Request,
        slug: str,
        conn: DbConn,
    ) -> Response:
        article = fetch_article(conn, slug)
        if article["status"] not in EDITABLE_STATUSES:
            raise HTTPException(409, f"article is {article['status']}")
        save_form(conn, article, dict(await request.form()))
        return Response(status_code=204)

    @app.post("/articles/{slug}/sign", response_class=HTMLResponse)
    async def sign(
        request: Request,
        slug: str,
        conn: DbConn,
    ) -> HTMLResponse:
        form = dict(await request.form())
        article = fetch_article(conn, slug)
        if article["status"] not in EDITABLE_STATUSES:
            raise HTTPException(409, f"article is {article['status']}")

        state = evaluate(conn, article, form)
        if not state["can_sign"]:
            context = {**state, "refused": True}
            return templates.TemplateResponse(
                request, "_live.html", context, status_code=422
            )

        save_form(conn, article, form)
        body = str(form.get("body_markdown", ""))
        try:
            result = sign_and_promote_article(
                slug,
                body,
                sorted(set(extract_citations(body))),
                str(form.get("editor_name", "")),
                checks_from_form(form),
                conn=conn,
                uncertainty_disclosures=disclosures_from_form(form, article),
                no_open_questions_confirmed=confirmed_from_form(form),
            )
        except SignOffError as error:
            context = {**state, "refused": True, "sign_issues": error.issues}
            return templates.TemplateResponse(
                request, "_live.html", context, status_code=422
            )
        return templates.TemplateResponse(request, "_signed.html", {"result": result})

    @app.post("/articles/{slug}/corrections", response_class=HTMLResponse)
    async def correct(
        request: Request,
        slug: str,
        conn: DbConn,
    ) -> HTMLResponse:
        form = dict(await request.form())
        article = fetch_article(conn, slug)
        issues: list[str] = []
        try:
            record_correction(
                slug,
                str(form.get("correction_type", "")),
                str(form.get("explanation", "")),
                str(form.get("editor_signature", "")),
                original_text=str(form.get("original_text", "")),
                corrected_text=str(form.get("corrected_text", "")),
                conn=conn,
            )
        except CorrectionError as error:
            issues = error.issues
        context = {
            "article": article,
            "correction_types": CORRECTION_TYPES,
            "corrections": fetch_corrections(conn, article["article_id"]),
            "correction_issues": issues,
        }
        return templates.TemplateResponse(
            request, "_corrections.html", context, status_code=422 if issues else 200
        )

    return app


def main() -> None:
    uvicorn.run(create_app(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
