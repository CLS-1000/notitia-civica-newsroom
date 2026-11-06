"""OLIS sources must be about the record's measure subject and session.

The 2026-11-01 pdx-1i run sealed HB 4092's OLIS overview for a record that
says HB 4092 is "Relating to regional transportation district governance". The
sealed page (tests/fixtures/pdx1_hollow/olis_hb4092_overview.html) says HB 4092
is "Relating to child safety systems." in the 2026 Regular Session. The page
names the bill, so the identifier check passes, but it does not state the
record. These tests pin the subject and session checks that reject it, and the
WA PDC subject domain.
"""

from __future__ import annotations

import itertools
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest

from newsroom.harlan_eternity.custody import fetch_and_seal_source
from spec1_bridge import mycroft_triage, pdx1_importer
from spec1_bridge.contracts import Spec1LeadPacket
from spec1_bridge.mycroft_triage import (
    Enrichment,
    OllamaEnricher,
    packet_subject_domain,
)
from spec1_bridge.page_content import read_main_content
from spec1_bridge.pdx1_importer import (
    hollow_source_reason,
    import_pdx1_signals,
    olis_mismatch_reason,
    parse_record,
    record_session,
    record_subject,
    record_title,
    subject_domain_for,
    subjects_match,
    url_session,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "pdx1_hollow" / "olis_hb4092_overview.html"
)
HTML = "text/html; charset=utf-8"
OLIS_2026 = "https://olis.oregonlegislature.gov/liz/2026R1/Measures/Overview/HB4092"
OLIS_2025 = "https://olis.oregonlegislature.gov/liz/2025R1/Measures/Overview/HB4092"

# The pattern pdx-1i wrote for rec_788528a934cc251a on 2026-11-01.
AUDITED_PATTERN = (
    "OLIS record for HB 4092, Relating to regional transportation district "
    "governance, in the 2026 Regular session. The measure is sponsored by House "
    "Interim Committee on Transportation and referred to the House Interim Committee on..."
)
# The same record as it should read, stating what the page states.
MATCHING_PATTERN = (
    "OLIS record for HB 4092, Relating to child safety systems, in the 2026 "
    "Regular session. The measure is sponsored by House Interim Committee on..."
)


class _OfflineEnricher(OllamaEnricher):
    def enrich(self, packet: Any) -> Enrichment:
        return Enrichment("unavailable", "test", error="offline")


def _text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _record(pattern: str, url: str = OLIS_2026, **overrides: Any) -> Any:
    published = (datetime.now(UTC) - timedelta(days=5)).replace(microsecond=0)
    raw: dict[str, Any] = {
        "record_id": "rec_788528a934cc251a",
        "source_type": "OLIS",
        "pattern": pattern,
        "outcome": "CORROBORATED",
        "priority": "ELEVATED",
        "tier": "HARD_RECORD",
        "anomaly": None,
        "entity_ids": ["metro"],
        "url": url,
        "published_at": published.isoformat(),
        "origin": "live",
    }
    raw.update(overrides)
    return raw


def _parsed(pattern: str, url: str = OLIS_2026, **overrides: Any) -> Any:
    return parse_record(json.dumps(_record(pattern, url, **overrides)))


def _reason(pattern: str, url: str = OLIS_2026) -> str | None:
    text = _text()
    return hollow_source_reason(
        _parsed(pattern, url), url, text, read_main_content(text)
    )


# 1a. Subject ----------------------------------------------------------------


def test_subject_mismatch_on_the_audited_hb4092_record_is_rejected() -> None:
    reason = _reason(AUDITED_PATTERN)
    assert reason is not None
    assert "subject" in reason


def test_matching_subject_and_session_is_accepted() -> None:
    assert _reason(MATCHING_PATTERN) is None


def test_session_mismatch_is_rejected() -> None:
    """The record is the 2025 Regular session; the page and URL are 2026R1."""
    pattern = MATCHING_PATTERN.replace("2026 Regular", "2025 Regular")
    reason = _reason(pattern)
    assert reason is not None and "session" in reason


def test_url_session_mismatch_is_rejected() -> None:
    """The record says 2026 Regular; the URL fetched is a 2025R1 page."""
    reason = _reason(MATCHING_PATTERN, OLIS_2025)
    assert reason is not None and "session" in reason


def test_record_without_a_session_is_rejected_for_olis() -> None:
    assert (
        _reason("OLIS record for HB 4092, Relating to child safety systems.")
        is not None
    )


def test_record_without_a_subject_or_title_is_rejected() -> None:
    reason = _reason("OLIS record for HB 4092, in the 2026 Regular session.")
    assert reason is not None and "subject" in reason
    reason = _reason("OLIS record for HB 4092, in the 2025 Regular session.")
    assert reason is not None and "session" in reason


# Live pdx-1i OLIS records title the measure from its CatchLine first, and name
# the session by its SessionKey. HB 4092 2026R1: CatchLine "Prohibits a
# retailer ...", RelatingTo "Relating to child safety systems." A CatchLine is a
# plain sentence with no "Relating to" subject, so the record's title must
# appear in the page's visible content ("Catchline/Summary: ...") instead.
HB4092_CATCHLINE = (
    "Prohibits a retailer from knowingly selling or offering for sale, and a "
    "retail platform operator from knowingly permitting a retailer to advertise "
    "or offer for sale, a child safety system that does not comply with federal "
    "standards or standards the Department of Transportation adopts by rule."
)
CATCHLINE_PATTERN = (
    f"OLIS record for HB 4092, {HB4092_CATCHLINE.rstrip('.')}, in the 2026R1 session."
)
OTHER_CATCHLINE_PATTERN = (
    "OLIS record for HB 4092, Establishes a regional transportation district "
    "governance task force, in the 2026R1 session."
)


def test_record_title_reads_the_catchline() -> None:
    assert record_subject(CATCHLINE_PATTERN) is None
    assert record_title(CATCHLINE_PATTERN) == HB4092_CATCHLINE.rstrip(".")
    assert record_title("OLIS record for HB 4092, in the 2026R1 session.") is None


def test_session_key_form_is_read_from_the_record() -> None:
    assert record_session(CATCHLINE_PATTERN) == "2026R1"
    assert record_session("OLIS record for HB 4092, in the 2025S1 session.") == (
        "2025S1"
    )


def test_hb4092_catchline_passes_on_its_own_page() -> None:
    assert _reason(CATCHLINE_PATTERN) is None


def test_catchline_match_ignores_whitespace_and_case() -> None:
    shouty = CATCHLINE_PATTERN.replace(
        "Prohibits a retailer", "PROHIBITS  a\n retailer"
    )
    assert _reason(shouty) is None


def test_different_catchline_is_rejected_as_a_subject_mismatch() -> None:
    reason = _reason(OTHER_CATCHLINE_PATTERN)
    assert reason is not None and "subject mismatch" in reason


def test_catchline_title_still_gets_the_session_check() -> None:
    reason = _reason(CATCHLINE_PATTERN.replace("2026R1", "2025R1"))
    assert reason is not None and "session" in reason
    reason = _reason(CATCHLINE_PATTERN, OLIS_2025)
    assert reason is not None and "session" in reason


def test_check_applies_only_to_olis_hosts() -> None:
    text = _text()
    record = _parsed(AUDITED_PATTERN, "https://records.example.org/hb4092")
    assert (
        olis_mismatch_reason(
            record, "https://records.example.org/hb4092", text, read_main_content(text)
        )
        is None
    )


def test_api_json_relating_to_is_checked() -> None:
    url = (
        "https://api.oregonlegislature.gov/odata/odataservice.svc/"
        "LegislativeSessions('2026R1')/Measures?$filter=MeasureNumber eq 4092"
    )
    row = json.dumps(
        {
            "value": [
                {
                    "MeasurePrefix": "HB",
                    "MeasureNumber": 4092,
                    "RelatingTo": "Relating to child safety systems.",
                }
            ]
        }
    )
    assert hollow_source_reason(_parsed(MATCHING_PATTERN, url), url, row, None) is None
    reason = hollow_source_reason(_parsed(AUDITED_PATTERN, url), url, row, None)
    assert reason is not None and "subject" in reason


def test_audited_record_is_skipped_end_to_end(tmp_path: Path) -> None:
    store = tmp_path / "s.jsonl"
    store.write_text(json.dumps(_record(AUDITED_PATTERN)) + "\n", encoding="utf-8")
    counter = itertools.count(1)
    allocate: Callable[[], str] = lambda: f"S-{next(counter)}"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": HTML}, content=FIXTURE.read_bytes()
        )

    result = import_pdx1_signals(
        store,
        allocate,
        archive_dir=tmp_path / "archive",
        transport=httpx.MockTransport(handler),
    )
    assert result.packets == []
    assert result.skipped == {"hollow_source": 1}
    assert "subject" in result.errors[0]


def test_subject_and_session_parsing() -> None:
    assert (
        record_subject(AUDITED_PATTERN) == "regional transportation district governance"
    )
    assert record_subject(
        "Bill Title: Relating to child safety systems. Catchline"
    ) == ("child safety systems")
    assert record_session(AUDITED_PATTERN) == "2026R1"
    assert record_session("in the 2025 First Special Session.") == "2025S1"
    assert record_session("in the 2024 Second Special session") == "2024S2"
    assert record_session("no session here") is None
    assert url_session(OLIS_2025) == "2025R1"
    assert url_session(
        "https://api.oregonlegislature.gov/x/LegislativeSessions('2026R1')/M"
    ) == ("2026R1")
    assert url_session("https://olis.oregonlegislature.gov/liz/") is None


@pytest.mark.parametrize(
    "ours, theirs, expected",
    [
        ("child safety systems", "child safety systems", True),
        ("child safety system", "Child Safety Systems", True),
        ("taxes", "taxes; prescribing an effective date", True),
        ("regional transportation district governance", "child safety systems", False),
        ("transportation", "child safety systems", False),
        ("", "child safety systems", False),
        ("the", "child safety systems", False),
    ],
)
def test_subjects_match(ours: str, theirs: str, expected: bool) -> None:
    assert subjects_match(ours, theirs) is expected


# 2. Washington leads ---------------------------------------------------------


def test_subject_domain_for_washington_pdc_leads() -> None:
    pdc = _parsed(
        "Washington PDC receipt WA-2026-778341",
        "https://www.pdc.wa.gov/browse/campaign-explorer",
        source_type="WA_PDC",
    )
    assert subject_domain_for(pdc) == "washington_civic"
    my_pdc = _parsed(
        "x",
        "https://my.pdc.wa.gov/public/registrations/campaign-finance-report/110401493",
    )
    assert subject_domain_for(my_pdc) == "washington_civic"
    with_evidence = _parsed(
        "x",
        "https://www.pdc.wa.gov/browse/campaign-explorer",
        evidence_url="https://data.wa.gov/resource/kv7h-kjye.json?id=21807083",
    )
    assert subject_domain_for(with_evidence) == "washington_civic"
    assert subject_domain_for(_parsed(MATCHING_PATTERN)) == "oregon_civic"
    assert subject_domain_for(_parsed("x", "https://notpdc.wa.gov.example.com/")) == (
        "oregon_civic"
    )
    assert pdx1_importer.SUBJECT_DOMAIN == "oregon_civic"


def test_washington_domain_is_carried_on_the_artifact(tmp_path: Path) -> None:
    url = "https://my.pdc.wa.gov/public/registrations/campaign-finance-report/110401493"
    body = (
        b"<html><body><h1>Receipt WA-2026-778341</h1><p>Washington PDC receipt "
        b"WA-2026-778341 reports a contribution of $6,000.00 to the committee.</p>"
        b"</body></html>"
    )
    store = tmp_path / "s.jsonl"
    store.write_text(
        json.dumps(
            _record("Washington PDC receipt WA-2026-778341", url, source_type="WA_PDC")
        )
        + "\n"
        + json.dumps(_record(MATCHING_PATTERN, record_id="rec_or"))
        + "\n",
        encoding="utf-8",
    )
    pages = {url: body, OLIS_2026: FIXTURE.read_bytes()}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": HTML}, content=pages[str(request.url)]
        )

    counter = itertools.count(1)
    result = import_pdx1_signals(
        store,
        lambda: f"S-{next(counter)}",
        archive_dir=tmp_path / "archive",
        transport=httpx.MockTransport(handler),
    )
    domains = {
        p.source_artifacts[0].metadata["pdx1_record_id"]: p.source_artifacts[
            0
        ].metadata["subject_domain"]
        for p in result.packets
    }
    assert domains == {
        "rec_788528a934cc251a": "washington_civic",
        "rec_or": "oregon_civic",
    }


def test_washington_domain_reaches_triage_records(
    tmp_path: Path,
    db: psycopg.Connection[tuple[Any, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_pdx1 writes each lead with its own subject_domain, not the batch's."""
    monkeypatch.chdir(tmp_path)
    url = "https://my.pdc.wa.gov/public/registrations/campaign-finance-report/110401493"
    body = (
        b"<html><body><h1>Receipt WA-2026-778341</h1><p>Washington PDC receipt "
        b"WA-2026-778341 reports a contribution of $6,000.00 to the committee.</p>"
        b"</body></html>"
    )
    store = tmp_path / "s.jsonl"
    store.write_text(
        json.dumps(
            _record(
                "Washington PDC receipt WA-2026-778341",
                url,
                source_type="WA_PDC",
                record_id="rec_wa",
            )
        )
        + "\n"
        + json.dumps(_record(MATCHING_PATTERN, record_id="rec_or"))
        + "\n",
        encoding="utf-8",
    )
    pages = {url: body, OLIS_2026: FIXTURE.read_bytes()}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": HTML}, content=pages[str(request.url)]
        )

    def offline_seal(*args: Any, **kwargs: Any) -> Any:
        return fetch_and_seal_source(
            *args, **{**kwargs, "transport": httpx.MockTransport(handler)}
        )

    monkeypatch.setattr(pdx1_importer, "fetch_and_seal_source", offline_seal)
    report = mycroft_triage.run_pdx1(str(store), "ELEVATED", db, _OfflineEnricher())

    assert len(report.written) == 2, report.rejected
    rows = db.execute(
        "SELECT record_id::text, subject_domain FROM triage_records"
    ).fetchall()
    assert dict(rows) == {
        str(uuid.uuid5(pdx1_importer.PDX1_NAMESPACE, "rec_wa")): "washington_civic",
        str(uuid.uuid5(pdx1_importer.PDX1_NAMESPACE, "rec_or")): "oregon_civic",
    }


def test_packet_subject_domain_falls_back_to_the_default() -> None:
    packet = Spec1LeadPacket(
        lead_id="x",
        timestamp=datetime.now(UTC),
        adapter_name="pdx1_importer",
        anomaly_score=0.0,
        evidence_bar_passed=False,
        source_artifacts=[],
        verbatim_excerpts={},
        paste_ready_prompt="",
    )
    assert packet_subject_domain(packet, "oregon_civic") == "oregon_civic"
