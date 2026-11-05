"""The 2026-11-01 pdx-1i run, replayed against the pages it actually sealed.

LEDGER AUDITOR found that seven of the eight sources sealed that day were
hollow: two ORESTAR search forms, an OLIS "Cannot display measure" notice, two
OGEC "Page not found" pages and two copies of the WA PDC landing page. Only
HB 4092's OLIS overview is a real record page. tests/fixtures/pdx1_hollow holds
byte-for-byte copies of those sealed blobs (their SHA-256 values are the ones
in source_registry), served here through an httpx.MockTransport.

The records below carry the patterns those leads were built from, as pdx-1i
wrote them. Nothing here reads or writes radiant_ledger.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from newsroom.dors_venabili.gatekeeper import verify_grounding
from spec1_bridge import mycroft_triage, pdx1_importer
from spec1_bridge.adapters.base_lazlong import excerpt_is_verbatim, read_artifact_text
from spec1_bridge.page_content import (
    error_notice,
    hollow_page_reason,
    is_search_form_page,
    read_main_content,
)
from spec1_bridge.pdx1_importer import (
    hollow_source_reason,
    import_pdx1_signals,
    parse_record,
    record_identifiers,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pdx1_hollow"
HTML = "text/html; charset=utf-8"

ORESTAR = "https://secure.sos.state.or.us/orestar/gotoPublicTransactionDetail.do?tranId="
OLIS = "https://olis.oregonlegislature.gov/liz/2026R1/Measures/Overview/"
SEI_LANDING = "https://www.oregon.gov/ogec/Pages/sei-filings.aspx"
PDC_LANDING = "https://www.pdc.wa.gov/browse/campaign-explorer"

# record_id: (source_id, source_type, pattern, url, fixture, sealed sha256)
AUDITED: dict[str, tuple[str, str, str, str, str, str]] = {
    "rec_933cef894bc2c523": (
        "S-1000003",
        "ORESTAR",
        (
            "ORESTAR transaction ORE-2026-4417392 filed by committee Friends of Metro "
            "Transit Access (PAC-19442) records a contribution of $12,500.00 of type Cash "
            "Contribution received from Regional Utility Workers PAC of Portland, OR, employer or..."
        ),
        ORESTAR + "4417392",
        "orestar_search_form_4417392.html",
        "9e265f954f32721bc357835315a7e9e507dae956860294b85e003658c6e70370",
    ),
    "rec_f867ff57de2bb6d6": (
        "S-1000004",
        "ORESTAR",
        (
            "ORESTAR transaction ORE-2026-4417511 filed by committee Clackamas Forward "
            "Committee (PAC-20118) records a contribution of $4,800.00 of type In-Kind "
            "Contribution received from NW Natural Employees Fund of Portland, OR, employer or..."
        ),
        ORESTAR + "4417511",
        "orestar_search_form_4417511.html",
        "b4a0f33a5457245e9b56faf1aff9f9e7a656244decf19d0a66ae8d9f45d68a77",
    ),
    "rec_691a7d884e21d3e8": (
        "S-1000005",
        "OLIS",
        (
            "OLIS record for SB 1147, Relating to energy facility siting review timelines, "
            "in the 2026 Regular session. The measure is sponsored by Senate Committee on "
            "Energy and Environment and referred to the Senate Committee on Energy and..."
        ),
        OLIS + "SB1147",
        "olis_sb1147_not_released.html",
        "a05ba528598d9868590fb52d07629d38b2e735ddbaaf82d4fdcf62ec39693633",
    ),
    "rec_788528a934cc251a": (
        "S-1000006",
        "OLIS",
        (
            "OLIS record for HB 4092, Relating to regional transportation district "
            "governance, in the 2026 Regular session. The measure is sponsored by House "
            "Interim Committee on Transportation and referred to the House Interim Committee on..."
        ),
        OLIS + "HB4092",
        "olis_hb4092_overview.html",
        "266fa8457a1e7fb93626e6118ffbdd34dc48a84ec26bf52ee840c3a01af00240",
    ),
    "rec_f2dc38279dd20536": (
        "S-1000007",
        "SEI",
        (
            "Statement of economic interest SEI-2026-00871 for the seat Metro Councilor - "
            "District 4 on Metro Council, covering calendar year 2025, filed "
            "2026-05-26T15:20:00+00:00 as a amendment filing. Declared interests: Position held -- Unpaid..."
        ),
        SEI_LANDING,
        "ogec_sei_page_not_found.html",
        "5dd3e62d7cf4899fe07dfa050a09294cbbecaf4a96bbf9412f72b73eaf023b4d",
    ),
    "rec_6d7a68894dfe499e": (
        "S-1000008",
        "SEI",
        (
            "Statement of economic interest SEI-2026-00902 for the seat County Commissioner - "
            "Position 2 on Washington County, covering calendar year 2025, filed "
            "2026-05-27T14:05:00+00:00 as a original filing. Declared interests: Business investment --..."
        ),
        SEI_LANDING,
        "ogec_sei_page_not_found.html",
        "5dd3e62d7cf4899fe07dfa050a09294cbbecaf4a96bbf9412f72b73eaf023b4d",
    ),
    "rec_16a2e13c4601b8e5": (
        "S-1000009",
        "WA_PDC",
        (
            "Washington PDC receipt WA-2026-778341 reports a contribution of $6,000.00 to "
            "Clark County Council District 2 Committee (Candidate committee) in Clark County, "
            "Washington, for election cycle 2026. The contributor is Cascade Energy Advocacy..."
        ),
        PDC_LANDING,
        "pdc_campaign_explorer_landing.html",
        "0e41cb75d8be5f0887a8f5379b21df4fc62be6bb9ab8575ad231badc3dd66b64",
    ),
    "rec_57e5434062d3e34a": (
        "S-1000010",
        "WA_PDC",
        (
            "Washington PDC receipt WA-2026-778402 reports a contribution of $3,500.00 to "
            "Vancouver City Council Position 4 Committee (Candidate committee) in City of "
            "Vancouver, Washington, for election cycle 2026. The contributor is Columbia Crossing..."
        ),
        PDC_LANDING,
        "pdc_campaign_explorer_landing.html",
        "0e41cb75d8be5f0887a8f5379b21df4fc62be6bb9ab8575ad231badc3dd66b64",
    ),
}
SUBSTANTIVE = "rec_788528a934cc251a"
HOLLOW = sorted(set(AUDITED) - {SUBSTANTIVE})


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _text(name: str) -> str:
    return _fixture(name).decode("utf-8")


def _record(record_id: str, url: str | None = None, **overrides: Any) -> dict[str, Any]:
    _source_id, source_type, pattern, real_url, _fixture_name, _sha = AUDITED[record_id]
    published = (datetime.now(UTC) - timedelta(days=5)).replace(microsecond=0)
    record: dict[str, Any] = {
        "record_id": record_id,
        "source_type": source_type,
        "pattern": pattern,
        "outcome": "CORROBORATED",
        "priority": "ELEVATED",
        "tier": "HARD_RECORD",
        "anomaly": None,
        "entity_ids": ["metro"],
        "url": url or real_url,
        "published_at": published.isoformat(),
        "origin": "live",
    }
    record.update(overrides)
    return record


def _store(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    return path


def _allocator() -> Callable[[], str]:
    counter = itertools.count(1)
    return lambda: f"S-{next(counter)}"


def _transport(pages: dict[str, bytes], requests: list[str] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if requests is not None:
            requests.append(url)
        if url not in pages:
            return httpx.Response(404, headers={"content-type": "text/plain"}, content=b"no")
        return httpx.Response(200, headers={"content-type": HTML}, content=pages[url])

    return httpx.MockTransport(handler)


def _content_only_url(record_id: str) -> str:
    """A URL that no URL rule knows, so only the page's content can reject it."""
    return f"https://records.example.org/{record_id}"


def test_fixtures_are_the_sealed_blobs() -> None:
    for _source_id, _type, _pattern, _url, name, sealed_sha in AUDITED.values():
        assert hashlib.sha256(_fixture(name)).hexdigest() == sealed_sha


@pytest.mark.parametrize("record_id", HOLLOW)
def test_each_audited_hollow_page_is_rejected_by_its_content(record_id: str) -> None:
    """No URL rule or host marker applies at a neutral URL: the content decides."""
    name = AUDITED[record_id][4]
    record = parse_record(json.dumps(_record(record_id)))
    page = read_main_content(_text(name))
    reason = hollow_source_reason(record, _content_only_url(record_id), _text(name), page)
    assert reason is not None


@pytest.mark.parametrize(
    "name, expected",
    [
        ("orestar_search_form_4417392.html", "is a search form"),
        ("orestar_search_form_4417511.html", "is a search form"),
        ("olis_sb1147_not_released.html", "shows an error notice"),
        ("ogec_sei_page_not_found.html", "shows an error notice"),
    ],
)
def test_page_level_reasons(name: str, expected: str) -> None:
    reason = hollow_page_reason(read_main_content(_text(name)))
    assert reason is not None and reason.startswith(expected)


def test_pdc_landing_page_is_rejected_for_not_stating_the_receipt() -> None:
    page = read_main_content(_text("pdc_campaign_explorer_landing.html"))
    for record_id in ("rec_16a2e13c4601b8e5", "rec_57e5434062d3e34a"):
        record = parse_record(json.dumps(_record(record_id)))
        reason = hollow_source_reason(record, _content_only_url(record_id), page.text, page)
        assert reason == "does not state the record's identifier"


def test_substantive_hb4092_page_passes() -> None:
    page = read_main_content(_text("olis_hb4092_overview.html"))
    assert hollow_page_reason(page) is None
    record = parse_record(json.dumps(_record(SUBSTANTIVE)))
    assert hollow_source_reason(record, OLIS + "HB4092", _text("olis_hb4092_overview.html"), page) is None
    assert hollow_source_reason(
        record, _content_only_url(SUBSTANTIVE), _text("olis_hb4092_overview.html"), page
    ) is None


def _audited_pages(url_for: Callable[[str], str]) -> dict[str, bytes]:
    return {url_for(record_id): _fixture(AUDITED[record_id][4]) for record_id in AUDITED}


@pytest.mark.parametrize("by_content", [False, True], ids=["real_urls", "content_only_urls"])
def test_audited_run_replay_keeps_only_hb4092(tmp_path: Path, by_content: bool) -> None:
    def url_for(record_id: str) -> str:
        return _content_only_url(record_id) if by_content else AUDITED[record_id][3]

    store = _store(tmp_path / "s.jsonl", [_record(rid, url=url_for(rid)) for rid in AUDITED])
    requests: list[str] = []
    result = import_pdx1_signals(
        store,
        _allocator(),
        archive_dir=tmp_path / "archive",
        transport=_transport(_audited_pages(url_for), requests),
    )

    kept = [p.source_artifacts[0].metadata["pdx1_record_id"] for p in result.packets]
    assert kept == [SUBSTANTIVE]
    assert result.skipped == {"hollow_source": 7}
    for record_id in HOLLOW:
        assert any(error.startswith(record_id) for error in result.errors)
    if not by_content:
        # The known landing URLs are refused before anything is fetched.
        assert SEI_LANDING not in requests and PDC_LANDING not in requests


def test_head_script_style_and_nav_text_is_not_content() -> None:
    page = read_main_content(
        "<html><head><title>Records</title><meta name='description' content='Big filing'>"
        "<script>var filing = 'ORE-2026-1';</script><style>p{}</style></head>"
        "<body><nav><a href='/'>Home</a> ORE-2026-1</nav>"
        "<header>Site banner</header><p>Only this paragraph.</p>"
        "<footer>Footer ORE-2026-1</footer></body></html>"
    )
    assert page.text == "Only this paragraph."
    assert page.title == "Records"


def test_main_is_used_only_when_it_holds_the_content() -> None:
    wrapped = read_main_content(
        "<body><div>Banner words here</div><main><p>"
        + "Record text. " * 20
        + "</p></main></body>"
    )
    assert wrapped.text.startswith("Record text.")
    title_only = read_main_content(
        "<body><main><h1>HB 1</h1></main><div><p>" + "Record text. " * 20 + "</p></div></body>"
    )
    assert "Record text." in title_only.text


def test_a_long_record_that_mentions_does_not_exist_is_not_a_notice() -> None:
    body = "The committee said the account does not exist in that name. " + "Filed text. " * 150
    page = read_main_content(f"<html><body><h1>Transaction 4417392</h1><p>{body}</p></body></html>")
    assert error_notice(page) is None
    assert hollow_page_reason(page) is None


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><h1>404</h1><p>Sorry.</p></body></html>",
        "<html><head><title>Page Not Found</title></head><body><p>Try again later please.</p></body></html>",
        "<html><body><p>No records found for your search terms.</p></body></html>",
        "<html><body><script>document.write('a b c d e f')</script></body></html>",
        "<html><body>" + "<a href='/x'>Candidates and committees</a> " * 20 + "</body></html>",
    ],
    ids=["404_heading", "not_found_title", "no_records", "script_only", "link_list"],
)
def test_generic_hollow_pages(html: str) -> None:
    assert hollow_page_reason(read_main_content(html)) is not None


def test_search_form_needs_controls_not_just_a_form_tag() -> None:
    aspnet = read_main_content(
        "<html><body><form id='aspnetForm'><input type='hidden' name='v' value='1'>"
        "<h1>Statement of Economic Interest</h1><p>Filed 2026 by the seat holder.</p>"
        "</form></body></html>"
    )
    assert not is_search_form_page(aspnet)
    assert hollow_page_reason(aspnet) is None


def test_record_identifiers() -> None:
    def matches(pattern: str, text: str) -> bool:
        return any(item.search(text) for item in record_identifiers(pattern))

    assert matches("transaction ORE-2026-4417392 filed", "Transaction Id: 4417392")
    assert matches("receipt WA-2026-778341", "WA-2026-778341")
    assert not matches("receipt WA-2026-778341", "Total 1778341.00")
    assert matches("OLIS record for HB 4092", "House Bill 4092 Enrolled")
    assert matches("OLIS record for HB 4092", "HB4092")
    assert not matches("OLIS record for HB 4092", "HB 40921")
    assert record_identifiers("Contribution $25,000 to Committee X") == []


def test_json_evidence_rows_skip_the_page_checks() -> None:
    record = parse_record(json.dumps(_record("rec_16a2e13c4601b8e5")))
    row = '[{"id":"21807083","amount":"96.62"}]'
    assert hollow_source_reason(record, "https://data.wa.gov/resource/kv7h-kjye.json", row, None) is None
    assert pdx1_importer.is_empty_answer("[]")


# Fix 2: the excerpt comes from the page's body, and stays groundable. -------


def _hb4092_lead(tmp_path: Path) -> tuple[Any, str]:
    store = _store(tmp_path / "s.jsonl", [_record(SUBSTANTIVE)])
    result = import_pdx1_signals(
        store,
        _allocator(),
        archive_dir=tmp_path / "archive",
        transport=_transport({OLIS + "HB4092": _fixture("olis_hb4092_overview.html")}),
    )
    (packet,) = result.packets
    artifact = packet.source_artifacts[0]
    raw = Path(artifact.raw_payload_path).read_bytes()
    text = read_artifact_text(raw, artifact.metadata["content_type"])
    assert text is not None
    return packet, text


def test_hb4092_excerpt_is_body_text_not_head_boilerplate(tmp_path: Path) -> None:
    packet, text = _hb4092_lead(tmp_path)
    (excerpt,) = packet.verbatim_excerpts.values()

    assert packet.evidence_bar_passed
    assert "<" not in excerpt and "DOCTYPE" not in excerpt
    assert excerpt.startswith("Prohibits a retailer from knowingly selling")
    assert len(excerpt) <= pdx1_importer.EXCERPT_CHARS
    assert excerpt in read_main_content(text).text
    assert excerpt_is_verbatim(excerpt, text)


def test_hb4092_excerpt_passes_the_grounding_gate_as_it_reads_sources(tmp_path: Path) -> None:
    """verify_grounding compares quotes against source_registry.verbatim_text,
    which _register_artifact fills from read_artifact_text of the sealed blob."""
    packet, text = _hb4092_lead(tmp_path)
    ((source_id, excerpt),) = packet.verbatim_excerpts.items()
    body = f'The measure\'s summary reads "{excerpt}" [{source_id}].'
    report = verify_grounding(body, {source_id: text.replace("\x00", "")})
    assert report["passed"], report["issues"]
    assert mycroft_triage._excerpt_problem(packet) is None


def test_excerpt_skips_head_script_style_nav_and_links() -> None:
    pattern = "Transaction 4417392 records a contribution of $12,500.00 from a PAC"
    html = (
        "<html><head><title>" + pattern + "</title>"
        "<meta name='description' content='" + pattern + "'>"
        "<script>var note = 'this script text is long enough to be a sentence';</script>"
        "<style>.x { content: 'this style text is long enough to be a sentence too' }</style>"
        "</head><body><nav><p>Home and every menu item a long sentence of navigation text</p></nav>"
        "<p><a href='/a'>A link whose text is long enough to read as a sentence</a></p>"
        "<p>Short label</p>"
        "<p>Contribution of $12,500.00 received on 2026-05-27, transaction 4417392, cash.</p>"
        "</body></html>"
    )
    record = parse_record(json.dumps(_record("rec_933cef894bc2c523", pattern=pattern)))
    page = read_main_content(html)
    excerpt = pdx1_importer.pick_excerpt(record, html, page)
    assert excerpt == "Contribution of $12,500.00 received on 2026-05-27, transaction 4417392, cash."
    assert excerpt_is_verbatim(excerpt, html)


def test_excerpt_prefers_the_pattern_only_when_the_body_states_it() -> None:
    pattern = "Contribution $25,000 to Committee X"
    record = parse_record(json.dumps(_record("rec_933cef894bc2c523", pattern=pattern)))
    in_body = f"<html><body><p>{pattern}</p></body></html>"
    assert pdx1_importer.pick_excerpt(record, in_body, read_main_content(in_body)) == pattern
    head_only = (
        f"<html><head><meta name='d' content='{pattern}'></head>"
        "<body><p>The committee filed its report late in the cycle this year.</p></body></html>"
    )
    excerpt = pdx1_importer.pick_excerpt(record, head_only, read_main_content(head_only))
    assert excerpt == "The committee filed its report late in the cycle this year."


def test_excerpt_skips_character_references_and_stays_verbatim() -> None:
    html = (
        "<html><body><p>The commissioner&#39;s statement was filed on time and in full.</p>"
        "<p>Filed amount of $9,000 for the committee in the 2026 cycle.</p></body></html>"
    )
    record = parse_record(json.dumps(_record("rec_933cef894bc2c523", pattern="x")))
    excerpt = pdx1_importer.pick_excerpt(record, html, read_main_content(html))
    assert "&#39;" not in excerpt
    assert excerpt_is_verbatim(excerpt, html)


def test_long_runs_are_clipped_at_a_word_boundary() -> None:
    html = "<html><body><p>" + "word " * 200 + "</p></body></html>"
    record = parse_record(json.dumps(_record("rec_933cef894bc2c523", pattern="x")))
    excerpt = pdx1_importer.pick_excerpt(record, html, read_main_content(html))
    assert 0 < len(excerpt) <= pdx1_importer.EXCERPT_CHARS
    assert excerpt.endswith("word")
    assert excerpt_is_verbatim(excerpt, html)


def test_non_html_excerpt_is_unchanged() -> None:
    record = parse_record(json.dumps(_record("rec_16a2e13c4601b8e5", pattern="x")))
    row = '[{"id":"21807083","amount":"96.62"}]'
    assert pdx1_importer.pick_excerpt(record, row, None) == row
