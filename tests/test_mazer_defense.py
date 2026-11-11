"""Mazer Defense (Engine 6), tested offline against mocked USAspending records,
GAO decisions, and DoD IG reports.

Every adapter request goes through test_phase1_adapters.Router, which fails the
test on any URL it was not given, and the autouse guard below fails any request
or socket connection that bypasses the mocks.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_phase1_adapters import Router, counter, minimal_pdf, options, spool_rows

from spec1_bridge.adapters.base_lazlong import PollError, PollRequest
from spec1_bridge.adapters.mazer_defense import (
    KIND_AWARD,
    KIND_DODIG,
    KIND_GAO,
    MAX_SEARCH_PAGES,
    SEARCH_REQUEST,
    SOLE_SOURCE_THRESHOLD,
    USASPENDING_AWARD,
    USASPENDING_SEARCH,
    Docket,
    DocketRejected,
    MazerDefenseAdapter,
    award_facts,
    award_url,
    format_amount,
    load_dockets_jsonl,
    sealed_docket_holds,
)

AWARD = "CONT_AWD_N0001926C0042_9700_-NONE-_-NONE-"
OTHER = "CONT_AWD_W58RGZ26C0099_9700_-NONE-_-NONE-"
RECIPIENT = "EXAMPLE AEROSPACE SYSTEMS INC"
GAO_URL = "https://www.gao.gov/products/b-423456.2"
IG_URL = "https://media.defense.gov/2026/Sep/30/2003000000/-1/-1/1/DODIG-2026-045.PDF"
FINDING = "the agency's evaluation of technical proposals was unreasonable and unequal"
AWARDEE_AS_NAMED = "Example Aerospace Systems, Inc."
IG_FINDING = "contracting officers did not adequately justify the sole-source award"


# --- Network guard --------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any real HTTP request or socket connection."""
    attempts: list[str] = []

    def refuse_http(_self: Any, request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        raise AssertionError(f"real network request: {request.url}")

    def refuse_socket(_self: Any, address: Any) -> None:
        attempts.append(str(address))
        raise AssertionError(f"real socket connection: {address}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse_http)
    monkeypatch.setattr(socket.socket, "connect", refuse_socket)
    yield
    assert attempts == []


# --- Fixtures shaped like live responses ------------------------------------------


def award_detail(
    award_id: str = AWARD,
    *,
    amount: Any = 412_500_000.0,
    extent: str = "C",
    agency: tuple[str, str] = ("Department of Defense", "097"),
    recipient: str = RECIPIENT,
    signed: str = "2026-10-02",
    category: str = "contract",
) -> dict[str, Any]:
    """Field names as api.usaspending.gov/api/v2/awards/{id}/ returns them."""
    return {
        "category": category,
        "generated_unique_award_id": award_id,
        "piid": "N0001926C0042",
        "total_obligation": amount,
        "date_signed": signed,
        "awarding_agency": {
            "toptier_agency": {
                "name": agency[0],
                "code": agency[1],
                "abbreviation": "DOD",
            },
            "subtier_agency": {"name": "Department of the Navy", "code": "1700"},
        },
        "recipient": {"recipient_name": recipient},
        "latest_transaction_contract_data": {
            "extent_competed": extent,
            "extent_competed_description": {
                "A": "FULL AND OPEN COMPETITION",
                "C": "NOT COMPETED",
            }.get(extent, "OTHER"),
            "type_of_contract_pricing": "U",
            "type_of_contract_pricing_description": "COST PLUS FIXED FEE",
        },
    }


def gao_docket(**overrides: Any) -> dict[str, Any]:
    return {
        "kind": "gao_protest",
        "docket": "B-423456.2",
        "url": GAO_URL,
        "award_id": AWARD,
        "issued": "2026-10-09",
        "outcome": "sustained",
        "finding": FINDING,
        "awardee_as_named": AWARDEE_AS_NAMED,
    } | overrides


def ig_docket(**overrides: Any) -> dict[str, Any]:
    return {
        "kind": "dodig_audit",
        "docket": "DODIG-2026-045",
        "url": IG_URL,
        "award_id": AWARD,
        "issued": "2026-09-30",
        "finding": IG_FINDING,
    } | overrides


GAO_HTML = (
    "<html><body><h1>Matter of: Rival Defense Corp.</h1>"
    "<p>File: B-423456.2</p><p>Date: October 9, 2026</p>"
    "<p>Rival Defense Corp. protests the award of a contract to "
    f"{AWARDEE_AS_NAMED} by the Department of the Navy.</p>"
    f"<p>We conclude that {FINDING}.</p>"
    "<p>We sustain the protest.</p></body></html>"
)
DENIED_HTML = GAO_HTML.replace("We sustain the protest.", "We deny the protest.")
IG_TEXT = (
    "Report No. DODIG-2026-045\n"
    "Audit of Navy Sole-Source Aircraft Sustainment Contracts\n"
    f"We found that {IG_FINDING}."
)


def search_page(*award_ids: str, has_next: bool = False) -> dict[str, Any]:
    return {
        "limit": 100,
        "results": [
            {
                "internal_id": number,
                "Award ID": "N0001926C0042",
                "Recipient Name": RECIPIENT,
                "Award Amount": 412_500_000.0,
                "Awarding Agency": "Department of Defense",
                "generated_internal_id": award_id,
            }
            for number, award_id in enumerate(award_ids)
        ],
        "page_metadata": {"page": 1, "hasNext": has_next},
    }


def router(
    *,
    hits: tuple[str, ...] = (AWARD,),
    awards: dict[str, Any] | None = None,
    gao: Any = GAO_HTML,
    ig: Any = None,
    search: Any = None,
) -> Router:
    details = awards if awards is not None else {AWARD: award_detail()}

    def search_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(200, json=search_page(*hits))

    routes: dict[str, Any] = {
        USASPENDING_SEARCH: search if search is not None else search_handler,
        USASPENDING_AWARD: httpx.Response(404),
        GAO_URL: (
            gao
            if isinstance(gao, httpx.Response)
            else httpx.Response(200, text=gao, headers={"content-type": "text/html"})
        ),
        IG_URL: (
            ig
            if isinstance(ig, httpx.Response)
            else httpx.Response(
                200,
                content=minimal_pdf(ig or IG_TEXT),
                headers={"content-type": "application/pdf"},
            )
        ),
    }
    for award_id, body in details.items():
        routes[award_url(award_id)] = body
    return Router(routes)


def adapter(
    tmp_path: Path, mock: Router, dockets: list[dict[str, Any]] | None = None
) -> MazerDefenseAdapter:
    parsed = [Docket.from_dict(row) for row in dockets or []]
    return MazerDefenseAdapter(parsed, **options(tmp_path, mock))


def harvest(
    tmp_path: Path, mock: Router, dockets: list[dict[str, Any]] | None = None
) -> Any:
    return adapter(tmp_path, mock, dockets).harvest(counter())


def sealed_files(tmp_path: Path) -> list[Path]:
    archive = tmp_path / "archive"
    return sorted(archive.glob("*.blob")) if archive.exists() else []


def enriched(
    tmp_path: Path, mock: Router, dockets: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    klein = adapter(tmp_path, mock, dockets)
    [record] = [
        item
        for item in klein.assemble_records({SEARCH_REQUEST: [{"award_id": AWARD}]})
        if item["award_id"] == AWARD
    ]
    with klein._session():
        return klein.enrich_record(record)


# --- Dockets ---------------------------------------------------------------------


def test_gao_and_dodig_dockets_load() -> None:
    gao = Docket.from_dict(gao_docket())
    assert (gao.kind, gao.number, gao.sustained, gao.counts) == (
        KIND_GAO,
        "B-423456.2",
        True,
        True,
    )
    assert gao.issued == date(2026, 10, 9)
    ig = Docket.from_dict(ig_docket())
    assert (ig.kind, ig.sustained, ig.counts) == (KIND_DODIG, False, True)


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (
            gao_docket(url="https://defense-insider-blog.example/navy-protest"),
            "not an official gao.gov",
        ),
        (
            gao_docket(url="http://www.gao.gov/products/b-423456.2"),
            "not an official gao.gov",
        ),
        (
            gao_docket(url="https://gao.gov.leaks.example/b-423456"),
            "not an official gao.gov",
        ),
        (
            ig_docket(url="https://www.reddit.com/r/defense/comments/x"),
            "not an official DoD IG",
        ),
        (gao_docket(docket="423456"), "not a GAO B-number"),
        (ig_docket(docket="DOD-IG-45"), "not a DoD IG report number"),
        (gao_docket(award_id=""), "names no USAspending award"),
        (gao_docket(finding="  "), "no verbatim finding"),
        (gao_docket(issued="October 2026"), "no issue date"),
        (gao_docket(outcome="rumored"), "unknown outcome"),
        (gao_docket(awardee_as_named=""), "does not say how it names the awardee"),
        (gao_docket(kind="tip_line"), "unknown docket kind"),
        ("B-423456.2", "not an object"),
    ],
)
def test_unofficial_or_malformed_dockets_are_rejected(row: Any, reason: str) -> None:
    with pytest.raises(DocketRejected, match=reason):
        Docket.from_dict(row)


def test_docket_file_keeps_official_lines_and_reports_the_rest(tmp_path: Path) -> None:
    path = tmp_path / "dockets.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(gao_docket()),
                "",
                json.dumps(gao_docket(url="https://defense-chatter.example/post/1")),
                "{not json",
                json.dumps(ig_docket()),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dockets, rejected = load_dockets_jsonl(path)
    assert [item.number for item in dockets] == ["B-423456.2", "DODIG-2026-045"]
    assert len(rejected) == 2
    assert rejected[0].startswith("line 3: B-423456.2: not an official gao.gov URL")
    assert rejected[1].startswith("line 4:")


# --- Award records ---------------------------------------------------------------


def test_award_record_is_read_as_published() -> None:
    facts = award_facts(award_detail(), AWARD)
    assert facts is not None
    assert facts.is_dod and facts.sole_source and facts.major_sole_source
    assert facts.recipient == RECIPIENT
    assert facts.total_obligation == 412_500_000.0
    assert facts.subagency == "Department of the Navy"
    assert facts.pricing == "COST PLUS FIXED FEE"
    assert facts.date_signed == date(2026, 10, 2)


@pytest.mark.parametrize(
    ("amount", "major"),
    [
        (SOLE_SOURCE_THRESHOLD, True),
        (SOLE_SOURCE_THRESHOLD - 0.01, False),
        (49_000_000, False),
    ],
)
def test_sole_source_threshold_is_fifty_million(amount: float, major: bool) -> None:
    facts = award_facts(award_detail(amount=amount))
    assert facts is not None and facts.major_sole_source is major


@pytest.mark.parametrize("extent", ["B", "C", "G", "NDO"])
def test_uncompeted_codes_are_sole_source(extent: str) -> None:
    facts = award_facts(award_detail(extent=extent))
    assert facts is not None and facts.major_sole_source


@pytest.mark.parametrize("extent", ["A", "D", "E", "F", "CDO", ""])
def test_competed_or_unreported_awards_are_not_sole_source(extent: str) -> None:
    facts = award_facts(award_detail(extent=extent))
    assert facts is not None and not facts.sole_source and not facts.major_sole_source


@pytest.mark.parametrize(
    "agency",
    [("Department of Energy", "089"), ("Department of Defense", "089"), ("DOD", "097")],
)
def test_dod_needs_both_name_and_toptier_code(agency: tuple[str, str]) -> None:
    facts = award_facts(award_detail(agency=agency))
    assert facts is not None and not facts.is_dod and not facts.major_sole_source


@pytest.mark.parametrize(
    "detail",
    [
        None,
        [],
        "award",
        award_detail(amount="412500000"),
        award_detail(amount=True),
        award_detail(recipient=""),
        award_detail(category="grant"),
        {**award_detail(), "generated_unique_award_id": ""},
    ],
)
def test_malformed_award_records_are_not_read(detail: Any) -> None:
    assert award_facts(detail) is None


def test_award_record_for_another_award_is_not_read() -> None:
    assert award_facts(award_detail(OTHER), AWARD) is None


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (412_500_000, "412.5 million"),
        (50_000_000, "50 million"),
        (1_250_000_000, "1.25 billion"),
        (2_000_000_000, "2 billion"),
        (61_241.34, "61,241"),
    ],
)
def test_format_amount(value: float, text: str) -> None:
    assert format_amount(value) == text


# --- Polling ---------------------------------------------------------------------


def test_search_body_asks_for_major_uncompeted_dod_contracts(tmp_path: Path) -> None:
    [request] = adapter(tmp_path, router()).poll_requests()
    body = json.loads(request.params["body"])
    filters = body["filters"]
    assert filters["award_type_codes"] == ["A", "B", "C", "D"]
    assert filters["agencies"] == [
        {"type": "awarding", "tier": "toptier", "name": "Department of Defense"}
    ]
    assert filters["award_amounts"] == [{"lower_bound": 50_000_000.0}]
    assert filters["extent_competed_type_codes"] == ["B", "C", "G", "NDO"]
    assert filters["time_period"] == [
        {"start_date": "2026-09-15", "end_date": "2026-10-15"}
    ]
    assert "generated_internal_id" in body["fields"]


def test_search_is_posted_and_paged(tmp_path: Path) -> None:
    pages: list[dict[str, Any]] = []

    def search(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        body = json.loads(request.content)
        pages.append(body)
        award_id = f"CONT_AWD_P{body['page']}"
        return httpx.Response(200, json=search_page(award_id, has_next=True))

    klein = adapter(tmp_path, router(search=search, awards={}))
    records = klein.poll_sources()
    assert [item["page"] for item in pages] == list(range(1, MAX_SEARCH_PAGES + 1))
    assert [item["award_id"] for item in records] == [
        f"CONT_AWD_P{page}" for page in range(1, MAX_SEARCH_PAGES + 1)
    ]


def test_failed_search_is_spooled_for_replay(tmp_path: Path) -> None:
    mock = router(search=httpx.Response(503))
    assert adapter(tmp_path, mock).poll_sources() == []
    [(name_adapter, name, failures, error)] = spool_rows(tmp_path)
    assert (name_adapter, name, failures, error) == (
        "mazer_defense",
        SEARCH_REQUEST,
        1,
        "HTTP 503",
    )


def test_rejected_search_is_not_spooled(tmp_path: Path) -> None:
    adapter(tmp_path, router(search=httpx.Response(422))).poll_sources()
    assert spool_rows(tmp_path) == []


def test_unreadable_spooled_body_is_dropped(tmp_path: Path) -> None:
    klein = adapter(tmp_path, router())
    with klein._session(), pytest.raises(PollError, match="unreadable search body"):
        klein._fetch_all_pages(
            PollRequest(SEARCH_REQUEST, USASPENDING_SEARCH, {"body": "{"})
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": "x"},
        {
            "results": [
                None,
                {"generated_internal_id": 5},
                {"generated_internal_id": " "},
            ]
        },
    ],
)
def test_malformed_search_results_give_no_items(tmp_path: Path, payload: Any) -> None:
    klein = adapter(tmp_path, router())
    [request] = klein.poll_requests()
    assert klein.extract_items(request, payload) == []


def test_search_error_in_body_is_retried(tmp_path: Path) -> None:
    mock = router(search={"error": {"message": "upstream timeout"}})
    adapter(tmp_path, mock).poll_sources()
    assert spool_rows(tmp_path)[0][3] == "upstream timeout"


def test_assembly_joins_search_hits_and_docketed_awards(tmp_path: Path) -> None:
    klein = adapter(tmp_path, router(), [gao_docket(award_id=OTHER), ig_docket()])
    records = klein.assemble_records(
        {SEARCH_REQUEST: [{"award_id": AWARD}, {"award_id": AWARD}]}
    )
    assert [item["award_id"] for item in records] == [AWARD, OTHER]
    assert [item.number for item in records[0]["dockets"]] == ["DODIG-2026-045"]
    assert [item.number for item in records[1]["dockets"]] == ["B-423456.2"]


# --- Keys and timestamps --------------------------------------------------------------


def test_record_key_changes_when_a_sustained_protest_is_linked(tmp_path: Path) -> None:
    plain = adapter(tmp_path, router())
    with_protest = adapter(tmp_path, router(), [gao_docket(), ig_docket()])
    denied = adapter(tmp_path, router(), [gao_docket(outcome="denied")])
    records = {AWARD: [{"award_id": AWARD}]}
    key = plain.record_key(plain.assemble_records({SEARCH_REQUEST: records[AWARD]})[0])
    assert key == f"usaspending:{AWARD}"
    assert (
        with_protest.record_key(
            with_protest.assemble_records({SEARCH_REQUEST: records[AWARD]})[0]
        )
        == f"usaspending:{AWARD}:B-423456.2+DODIG-2026-045"
    )
    assert (
        denied.record_key(denied.assemble_records({SEARCH_REQUEST: records[AWARD]})[0])
        == f"usaspending:{AWARD}"
    )


def test_record_key_rejects_unknown_kinds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown record kind"):
        adapter(tmp_path, router()).record_key({"kind": "rumor"})


def test_timestamp_is_the_latest_event(tmp_path: Path) -> None:
    klein = adapter(tmp_path, router())
    assert klein.record_timestamp(enriched(tmp_path, router())) == datetime(
        2026, 10, 2, tzinfo=UTC
    )
    record = enriched(tmp_path, router(), [gao_docket()])
    assert klein.record_timestamp(record) == datetime(2026, 10, 9, tzinfo=UTC)


def test_unread_or_undated_award_has_no_timestamp(tmp_path: Path) -> None:
    klein = adapter(tmp_path, router())
    with pytest.raises(ValueError, match="not been read"):
        klein.record_timestamp({"kind": "award", "award_id": AWARD, "dockets": []})
    record = enriched(tmp_path, router(awards={AWARD: award_detail(signed="")}))
    with pytest.raises(ValueError, match="no signed date"):
        klein.record_timestamp(record)


# --- Harvest: sole-source detection --------------------------------------------------


def test_major_sole_source_award_becomes_a_lead(tmp_path: Path) -> None:
    result = harvest(tmp_path, router())
    assert result.errors == [] and result.evidence_rejections == []
    [packet] = result.packets
    assert packet.adapter_name == "mazer_defense"
    assert packet.anomaly_score == 60.0
    assert packet.timestamp == datetime(2026, 10, 2, tzinfo=UTC)
    assert packet.public_question == (
        "What do federal procurement and audit records show regarding the "
        f"$412.5 million DoD award to {RECIPIENT}?"
    )
    assert packet.public_interest_reason == (
        "USAspending dockets confirm a $412.5 million sole-source obligation; "
        "no sustained GAO bid protest has been linked to it."
    )
    [artifact] = packet.source_artifacts
    assert artifact.metadata["kind"] == KIND_AWARD
    assert artifact.origin_url == award_url(AWARD)
    assert artifact.tier == "tier_1_primary"
    assert packet.verbatim_excerpts == {artifact.source_id: RECIPIENT}
    assert packet.receipt_details["extent_competed"] == "C (NOT COMPETED)"
    assert "ceiling" in packet.receipt_details["not_checked"]


def test_billion_dollar_award_scores_higher(tmp_path: Path) -> None:
    mock = router(awards={AWARD: award_detail(amount=1_250_000_000.0)})
    [packet] = harvest(tmp_path, mock).packets
    assert packet.anomaly_score == 70.0
    assert "$1.25 billion DoD award" in packet.public_question


@pytest.mark.parametrize(
    "detail",
    [
        award_detail(amount=49_999_999.99),
        award_detail(extent="A"),
        award_detail(agency=("Department of Energy", "089")),
        award_detail(category="idv"),
    ],
)
def test_below_threshold_competed_or_non_dod_awards_make_no_lead(
    tmp_path: Path, detail: dict[str, Any]
) -> None:
    result = harvest(tmp_path, router(awards={AWARD: detail}))
    assert result.packets == [] and result.errors == []
    assert sealed_files(tmp_path) == []


def test_search_row_alone_is_never_evidence(tmp_path: Path) -> None:
    """The search says $412.5M uncompeted; the award's own record says otherwise."""
    result = harvest(tmp_path, router(awards={AWARD: award_detail(amount=1_000_000.0)}))
    assert result.packets == []


def test_award_missing_from_usaspending_is_an_error(tmp_path: Path) -> None:
    result = harvest(tmp_path, router(awards={}))
    assert result.packets == []
    assert result.errors == [f"usaspending:{AWARD}: HTTP 404"]


# --- Harvest: sustained GAO protest correlation --------------------------------------


def test_sole_source_award_with_sustained_protest(tmp_path: Path) -> None:
    result = harvest(tmp_path, router(), [gao_docket()])
    assert result.errors == [] and result.evidence_rejections == []
    [packet] = result.packets
    assert packet.anomaly_score == 85.0
    assert packet.timestamp == datetime(2026, 10, 9, tzinfo=UTC)
    assert packet.public_interest_reason == (
        "USAspending dockets confirm a $412.5 million sole-source obligation; "
        "GAO bid protest dockets document a sustained protest "
        f'(B-423456.2: "{FINDING}").'
    )
    kinds = {item.metadata["kind"]: item for item in packet.source_artifacts}
    assert set(kinds) == {KIND_AWARD, KIND_GAO}
    assert kinds[KIND_GAO].origin_url == GAO_URL
    assert packet.verbatim_excerpts[kinds[KIND_GAO].source_id] == FINDING
    [docket] = packet.receipt_details["dockets"]
    assert docket["linked_by"] == "editor" and docket["sealed"] is True


def test_sustained_protest_on_a_competed_award_is_a_lead(tmp_path: Path) -> None:
    mock = router(hits=(), awards={AWARD: award_detail(extent="A", amount=8_200_000.0)})
    [packet] = harvest(tmp_path, mock, [gao_docket()]).packets
    assert packet.anomaly_score == 65.0
    assert "sole-source" not in packet.public_interest_reason
    assert packet.public_interest_reason.startswith(
        "USAspending dockets confirm a $8.2 million DoD obligation; GAO bid protest"
    )


@pytest.mark.parametrize("outcome", ["denied", "dismissed"])
def test_unsustained_protest_corroborates_nothing(tmp_path: Path, outcome: str) -> None:
    mock = router(hits=(), awards={AWARD: award_detail(extent="A")})
    result = harvest(tmp_path, mock, [gao_docket(outcome=outcome)])
    assert result.packets == []
    assert sealed_files(tmp_path) == []
    assert not any(str(item.url) == GAO_URL for item in mock.requests)


def test_protest_against_a_non_dod_award_makes_no_lead(tmp_path: Path) -> None:
    mock = router(
        hits=(), awards={AWARD: award_detail(agency=("Department of Energy", "089"))}
    )
    assert harvest(tmp_path, mock, [gao_docket()]).packets == []


@pytest.mark.parametrize(
    "page",
    [
        DENIED_HTML,
        GAO_HTML.replace(FINDING, "the evaluation was reasonable"),
        GAO_HTML.replace(AWARDEE_AS_NAMED, "Another Contractor LLC"),
        GAO_HTML.replace("B-423456.2", "B-999999"),
        "<html><body>Access Denied</body></html>",
    ],
)
def test_protest_whose_sealed_decision_does_not_hold_is_rejected(
    tmp_path: Path, page: str
) -> None:
    result = harvest(tmp_path, router(gao=page), [gao_docket()])
    assert result.packets == []
    assert result.evidence_rejections == [f"usaspending:{AWARD}:B-423456.2"]


def test_gao_refusing_the_fetch_is_an_error_not_a_lead(tmp_path: Path) -> None:
    result = harvest(tmp_path, router(gao=httpx.Response(403)), [gao_docket()])
    assert result.packets == []
    assert len(result.errors) == 1
    assert result.errors[0].startswith(f"usaspending:{AWARD}:B-423456.2:")


def test_dodig_report_corroborates_from_its_pdf(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(), [ig_docket()]).packets
    assert packet.anomaly_score == 65.0
    kinds = {item.metadata["kind"] for item in packet.source_artifacts}
    assert kinds == {KIND_AWARD, KIND_DODIG}
    assert "DODIG-2026-045" in packet.paste_ready_prompt
    assert "GAO" not in packet.public_interest_reason.split(";")[0]


def test_dodig_report_alone_does_not_make_a_competed_award_a_lead(
    tmp_path: Path,
) -> None:
    mock = router(hits=(), awards={AWARD: award_detail(extent="A")})
    assert harvest(tmp_path, mock, [ig_docket()]).packets == []


def test_dodig_report_that_lacks_the_finding_fails_the_lead(tmp_path: Path) -> None:
    mock = router(ig="Report No. DODIG-2026-045\nNo findings on this contract.")
    result = harvest(tmp_path, mock, [ig_docket()])
    assert result.packets == []
    assert result.evidence_rejections == [f"usaspending:{AWARD}:DODIG-2026-045"]


# --- Evidence bar ------------------------------------------------------------------


def sealed(tmp_path: Path, dockets: list[dict[str, Any]] | None = None) -> Any:
    klein = adapter(tmp_path, router(), dockets)
    [packet] = klein.harvest(counter()).packets
    return klein, packet


def test_evidence_bar_holds_on_sealed_records(tmp_path: Path) -> None:
    klein, packet = sealed(tmp_path, [gao_docket()])
    assert klein.verify_evidence_bar(packet.source_artifacts)


def test_evidence_bar_rejects_a_blog_source(tmp_path: Path) -> None:
    klein, packet = sealed(tmp_path, [gao_docket()])
    gao = next(
        item for item in packet.source_artifacts if item.metadata["kind"] == KIND_GAO
    )
    blog = replace(gao, origin_url="https://defense-insider-blog.example/b-423456")
    award = [
        item for item in packet.source_artifacts if item.metadata["kind"] == KIND_AWARD
    ]
    assert not klein.verify_evidence_bar([*award, blog])
    assert not sealed_docket_holds(blog)
    rumor = replace(gao, metadata={**gao.metadata, "kind": "tip"})
    assert not klein.verify_evidence_bar([*award, rumor])


def test_evidence_bar_rejects_changed_award_figures(tmp_path: Path) -> None:
    klein, packet = sealed(tmp_path)
    [award] = packet.source_artifacts
    inflated = replace(award, metadata={**award.metadata, "total_obligation": 9e9})
    assert not klein.verify_evidence_bar([inflated])
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(award_detail(amount=10_000_000.0)))
    assert not klein.verify_evidence_bar([replace(award, raw_payload_path=str(forged))])


def test_evidence_bar_needs_exactly_one_official_award(tmp_path: Path) -> None:
    klein, packet = sealed(tmp_path)
    [award] = packet.source_artifacts
    assert not klein.verify_evidence_bar([])
    assert not klein.verify_evidence_bar([award, award])
    assert not klein.verify_evidence_bar(
        [replace(award, origin_url="https://usaspending.example/awards/x")]
    )
    assert not klein.verify_evidence_bar(
        [replace(award, raw_payload_path="/nonexistent")]
    )


def test_evidence_bar_rejects_a_docket_for_another_award(tmp_path: Path) -> None:
    klein, packet = sealed(tmp_path, [gao_docket()])
    moved = [
        replace(item, metadata={**item.metadata, "award_id": OTHER})
        if item.metadata["kind"] == KIND_GAO
        else item
        for item in packet.source_artifacts
    ]
    assert not klein.verify_evidence_bar(moved)


# --- Public text ------------------------------------------------------------------------


def test_public_text_needs_a_qualifying_award(tmp_path: Path) -> None:
    klein = adapter(tmp_path, router())
    record = enriched(tmp_path, router(awards={AWARD: award_detail(extent="A")}))
    with pytest.raises(ValueError, match="neither a major sole-source award"):
        klein.public_interest_for(record)
    with pytest.raises(ValueError, match="not a DoD award"):
        klein.public_interest_for(
            {"kind": "award", "award_id": AWARD, "dockets": [], "facts": None}
        )


def test_record_cannot_carry_its_own_public_text(tmp_path: Path) -> None:
    klein = adapter(tmp_path, router())
    record = enriched(tmp_path, router()) | {
        "public_question": "Did the Navy rig this contract?",
        "public_interest_reason": "Insiders say the award was improper.",
    }
    question, reason = klein.public_interest_for(record)
    assert "rig" not in question and "Insiders" not in reason
    assert reason.startswith("USAspending dockets confirm a $412.5 million sole-source")


def test_two_sustained_protests_are_both_cited(tmp_path: Path) -> None:
    second = gao_docket(docket="B-423456.3")
    record = enriched(tmp_path, router(), [gao_docket(), second])
    reason = adapter(tmp_path, router()).default_public_interest_reason(record)
    assert "B-423456.2" in reason and "B-423456.3" in reason


def test_known_lead_is_skipped_before_reading_the_award(tmp_path: Path) -> None:
    mock = router()
    result = adapter(tmp_path, mock).harvest(counter(), is_known=lambda _lead: True)
    assert result.packets == []
    assert not any(
        str(item.url).startswith(USASPENDING_AWARD) for item in mock.requests
    )
