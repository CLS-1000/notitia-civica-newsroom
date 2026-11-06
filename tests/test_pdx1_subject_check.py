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
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from spec1_bridge import pdx1_importer
from spec1_bridge.page_content import read_main_content
from spec1_bridge.pdx1_importer import (
    hollow_source_reason,
    import_pdx1_signals,
    olis_mismatch_reason,
    parse_record,
    record_session,
    record_subject,
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


def test_record_without_a_subject_or_session_is_rejected_for_olis() -> None:
    assert _reason("OLIS record for HB 4092, in the 2026 Regular session.") is not None
    assert (
        _reason("OLIS record for HB 4092, Relating to child safety systems.")
        is not None
    )


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
