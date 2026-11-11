"""Ubik Spectral (Engine 9), tested offline against mocked Chronicling America
search results and page full text.

The pages' wording follows the style of the period's wire reports. Every request
goes through test_phase1_adapters.Router, which fails the test on any URL it was
not given.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import Conn
from test_claim_matcher import FAKE_ARTIFACT
from test_phase1_adapters import NOW, Router, counter, ollama, options

from spec1_bridge.adapters.living_folklore import (
    ARCHIVE_TIER,
    CHRONAM_SEARCH,
    INFRASOUND_NOTE,
    LORE_SCORE,
    SIGNAL,
    LivingFolkloreAdapter,
    LoreReport,
    fatal_disaster,
    fulltext_url,
    is_promotional,
    lccn_of,
    load_reports_jsonl,
    sighting_patterns,
)
from spec1_bridge.contracts import public_interest
from spec1_bridge.mycroft_triage import (
    independent_publishers,
    run_adapter,
    triage_packets,
)

SITE = "Black Diamond Mine"
LAT, LON = 47.3087, -122.0032

FIRE_TEXT = (
    "SEVENTEEN DIE IN MINE\n"
    "Fire Sweeps the Black Diamond Mine and Miners\n"
    "Are Trapped Below\n"
    "BLACK DIAMOND, Wash., Feb. 9.-Seventeen miners perished\n"
    "today when fire broke out in the lower levels of the Black Diamond Mine."
)
SECOND_PAPER_TEXT = (
    "MINE HORROR AT BLACK DIAMOND\n"
    "The fire in the Black Diamond Mine has cost seventeen lives; the bodies\n"
    "of the men were brought to the surface last night."
)
SAFE_TEXT = (
    "A small fire at the Black Diamond Mine was put out this morning. No one\n"
    "was killed and work resumed by noon."
)
SOCIAL_TEXT = "The Black Diamond Mine band gave a concert at the hall on Saturday."


def account(number: int, **overrides: Any) -> LoreReport:
    values: dict[str, Any] = {
        "report_id": f"R-{number}",
        "text": (
            "We heard voices calling from the old portal after dark and felt a "
            "heavy pressure in our chests."
        ),
        "site": SITE,
        "locality": "Black Diamond",
        "state": "Washington",
        "latitude": LAT + number * 0.0005,
        "longitude": LON,
        "reported_at": NOW - timedelta(days=10 * number),
    }
    return LoreReport(**(values | overrides))


TOUR = account(
    9,
    text="Join our Black Diamond ghost tour! Tickets $25, book now for October.",
)


def segment(lccn: str, page: str) -> str:
    return (
        f"/service/ndnp/wa/batch_wa_elm_ver01/data/{lccn}/00211108890/"
        f"1907020901/{page}.xml"
    )


def result(
    lccn: str, page: str, text: str, published: str = "1907-02-09"
) -> dict[str, Any]:
    return {
        "id": f"http://www.loc.gov/resource/{lccn}/{published}/ed-1/?sp=1",
        "date": published,
        "partof_title": [f"newspaper {lccn}"],
        "title": f"Image 1 of newspaper {lccn}",
        "description": [" ".join(text.split())],
        "word_coordinates_url": (
            "https://tile.loc.gov/text-services/word-coordinates-service?"
            f"format=alto_xml&segment={segment(lccn, page).replace('/', '%2F')}"
            "&q=black+diamond+mine&relevant_snippet=1"
        ),
    }


def fulltext(lccn: str, page: str, text: str) -> dict[str, Any]:
    return {segment(lccn, page): {"full_text": text, "height": 1, "width": 1}}


def router(*pages: tuple[str, str, str, str]) -> Router:
    """pages: (lccn, page, text, date)."""
    routes: dict[str, Any] = {
        CHRONAM_SEARCH: {
            "results": [
                result(lccn, page, text, day) for lccn, page, text, day in pages
            ]
        }
    }
    for lccn, page, text, day in pages:
        url = fulltext_url(result(lccn, page, text, day))
        assert url is not None
        routes[url] = fulltext(lccn, page, text)
    return Router(routes)


TWO_PAPERS = (
    ("sn87093407", "0101", FIRE_TEXT, "1907-02-09"),
    ("sn83045604", "0202", SECOND_PAPER_TEXT, "1907-02-10"),
)


def adapter(
    tmp_path: Path, mock: Router, reports: list[LoreReport] | None = None
) -> LivingFolkloreAdapter:
    return LivingFolkloreAdapter(
        reports if reports is not None else [account(1), account(2)],
        **options(tmp_path, mock),
    )


def harvest(
    tmp_path: Path, mock: Router, reports: list[LoreReport] | None = None
) -> Any:
    return adapter(tmp_path, mock, reports).harvest(counter())


# --- Buried historical record: disaster dockets ----------------------------------


def test_fatal_fire_naming_the_site_is_found() -> None:
    found = fatal_disaster(SITE, FIRE_TEXT)
    assert found is not None
    event, excerpt = found
    assert event == "fire"
    assert "Seventeen miners perished" in excerpt
    assert "\n" not in excerpt


@pytest.mark.parametrize(
    ("text", "event"),
    [
        (
            "Twelve men were killed in a cave-in at the Black Diamond Mine.",
            "mine collapse",
        ),
        ("An explosion at the Black Diamond Mine killed four miners.", "explosion"),
        ("The steamer Black Diamond Mine foundered and nine drowned.", "shipwreck"),
    ],
)
def test_each_kind_of_disaster_is_named(text: str, event: str) -> None:
    found = fatal_disaster(SITE, text)
    assert found is not None and found[0] == event


@pytest.mark.parametrize(
    "text",
    [SAFE_TEXT, SOCIAL_TEXT, "Fire killed six at the Franklin mine.", ""],
    ids=["no-deaths", "no-disaster", "other-site", "empty"],
)
def test_pages_without_a_fatal_disaster_at_the_site_do_not_count(text: str) -> None:
    assert fatal_disaster(SITE, text) is None


def test_site_name_survives_ocr_line_breaks_and_case() -> None:
    text = "seventeen miners perished in the fire at the BLACK\nDIAMOND   MINE today."
    assert fatal_disaster(SITE, text) is not None


def test_fulltext_url_and_lccn_come_from_the_page_segment() -> None:
    url = fulltext_url(result("sn87093407", "0101", FIRE_TEXT))
    assert url is not None and url.startswith(
        "https://tile.loc.gov/text-services/word-coordinates-service?segment="
    )
    assert "full_text=1" in url
    assert lccn_of(url) == "sn87093407"
    assert fulltext_url({"word_coordinates_url": ""}) is None


def test_two_newspapers_in_the_archive_are_two_publishers() -> None:
    first = fulltext_url(result("sn87093407", "0101", FIRE_TEXT))
    second = fulltext_url(result("sn83045604", "0202", SECOND_PAPER_TEXT))
    same = fulltext_url(result("sn87093407", "0303", SECOND_PAPER_TEXT))
    assert first and second and same
    at = [replace(FAKE_ARTIFACT, origin_url=url) for url in (first, second, same)]
    assert independent_publishers(at) == {"loc.gov/sn87093407", "loc.gov/sn83045604"}


# --- Witness dignity: promotion is filtered out ----------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Join our ghost tour every Friday! Tickets on sale.",
        "Haunted attraction opens nightly in October, family fun for all.",
        "Book now at www.example-hauntings.com and use code SPOOKY.",
        "Admission $15 for the midnight lantern walk.",
    ],
)
def test_commercial_promotion_is_set_aside(text: str) -> None:
    assert is_promotional(account(1, text=text))


def test_ticketing_sites_are_set_aside() -> None:
    report = account(1, source_url="https://www.eventbrite.com/e/black-diamond-123")
    assert is_promotional(report)


def test_an_honest_account_is_kept() -> None:
    assert not is_promotional(account(1))


def test_promotion_does_not_make_a_pattern(tmp_path: Path) -> None:
    patterns = sighting_patterns([account(1), TOUR])
    assert patterns == []
    mock = router(*TWO_PAPERS)
    assert harvest(tmp_path, mock, [account(1), TOUR]).packets == []
    assert mock.requests == []


def test_pattern_counts_the_promotion_it_set_aside() -> None:
    [pattern] = sighting_patterns([account(1), account(2), TOUR])
    assert len(pattern.reports) == 2 and pattern.set_aside == 1


def test_far_apart_accounts_are_not_one_pattern() -> None:
    far = account(2, latitude=LAT + 0.2)
    assert sighting_patterns([account(1), far]) == []


def test_repeated_account_counts_once() -> None:
    assert sighting_patterns([account(1), account(1)]) == []


def test_accounts_load_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "accounts.jsonl"
    path.write_text(
        '{"report_id": "R-1", "text": "Cold spot.", "site": " Black  Diamond Mine ",'
        ' "state": "Washington", "latitude": 47.3, "longitude": -122.0,'
        ' "reported_at": "2026-10-01T20:00:00"}\n\n'
    )
    [report] = load_reports_jsonl(path)
    assert report.site == SITE and report.reported_at.tzinfo is not None


# --- Environmental triggers: the infrasound mechanism ------------------------------


def test_infrasound_is_cited_as_a_proposed_mechanism_not_a_measurement() -> None:
    assert "18.9 Hz" in INFRASOUND_NOTE
    assert "proposed" in INFRASOUND_NOTE
    for claim in ("survey", "measured", "detected", "established"):
        assert claim not in INFRASOUND_NOTE


# --- Lead generation and Five-Mask formatting ------------------------------------


def test_documented_tragedy_makes_a_lead(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(*TWO_PAPERS)).packets

    assert packet.adapter_name == "ubik_spectral"
    assert packet.anomaly_score == LORE_SCORE == 75.0
    assert packet.public_question == (
        "What historical event or environmental factor is tied to the spectral "
        "accounts reported in Black Diamond Mine in Black Diamond, Washington?"
    )
    assert packet.public_interest_reason == (
        "Archival records document a fatal fire at this site in 1907; acoustic "
        "science literature has proposed low-frequency infrasound (around 18.9 Hz) "
        "from industrial machinery, dams, or ventilation as a physical explanation "
        "for spatial dread and visual anomalies."
    )


def test_lead_meets_the_five_mask_standard(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(*TWO_PAPERS)).packets
    question, reason = packet.public_question, packet.public_interest_reason
    assert question.endswith("?") and reason.endswith(".")
    assert public_interest(None, None, question, reason) == (question, reason)
    for text in (question, reason):
        assert text == " ".join(text.split())
        # Accounts are never quoted, and witnesses are never called names.
        assert "voices" not in text
        for word in ("hoax", "liar", "fake", "delusion", "proof", "proves"):
            assert word not in text.lower()


def test_lead_seals_each_newspaper_once_with_verbatim_excerpts(tmp_path: Path) -> None:
    second_page = ("sn87093407", "0105", FIRE_TEXT, "1907-02-11")
    [packet] = harvest(tmp_path, router(*TWO_PAPERS, second_page)).packets

    lccns = [item.metadata["lccn"] for item in packet.source_artifacts]
    assert lccns == ["sn87093407", "sn83045604"]
    assert {item.tier for item in packet.source_artifacts} == {ARCHIVE_TIER}
    assert all(
        item.origin_url.startswith("https://tile.loc.gov/")
        for item in packet.source_artifacts
    )
    assert set(packet.verbatim_excerpts) == {
        item.source_id for item in packet.source_artifacts
    }


def test_receipt_keeps_accounts_unverified(tmp_path: Path) -> None:
    [packet] = harvest(
        tmp_path, router(*TWO_PAPERS), [account(1), account(2), TOUR]
    ).packets
    details = packet.receipt_details
    assert details["signal"] == SIGNAL
    assert details["evidentiary_status"] == "unverified_narrative"
    assert (details["accounts"], details["accounts_set_aside_as_promotion"]) == (2, 1)
    assert [page["published"] for page in details["archival_pages"]] == [
        "1907-02-09",
        "1907-02-10",
    ]
    assert "not measured" in details["infrasound"]


def test_no_disaster_in_the_archive_is_not_a_lead(tmp_path: Path) -> None:
    mock = router(
        ("sn87093407", "0101", SAFE_TEXT, "1907-02-09"),
        ("sn83045604", "0202", SOCIAL_TEXT, "1907-03-01"),
    )
    assert harvest(tmp_path, mock).packets == []


def test_a_later_different_disaster_is_left_out(tmp_path: Path) -> None:
    later = (
        "sn83045604",
        "0909",
        "An explosion at the Black Diamond Mine killed four miners.",
        "1929-05-01",
    )
    [packet] = harvest(tmp_path, router(TWO_PAPERS[0], later)).packets
    assert [item.metadata["event"] for item in packet.source_artifacts] == ["fire"]


def test_search_asks_for_the_site_in_its_state(tmp_path: Path) -> None:
    mock = router(*TWO_PAPERS)
    harvest(tmp_path, mock)
    [search] = mock.calls_to(CHRONAM_SEARCH)
    assert search.url.params["qs"] == SITE
    assert search.url.params["location_state"] == "washington"
    assert search.url.params["fo"] == "json"


def test_page_text_outage_is_retried_not_flagged(tmp_path: Path) -> None:
    mock = router(*TWO_PAPERS)
    for prefix in list(mock.routes):
        if prefix.startswith("https://tile.loc.gov/"):
            mock.routes[prefix] = httpx.Response(503)
    result_ = harvest(tmp_path, mock)
    assert result_.packets == [] and len(result_.errors) == 1


# --- Triage -----------------------------------------------------------------------


def test_one_newspaper_is_rejected_by_triage(tmp_path: Path, db: Conn) -> None:
    [packet] = harvest(tmp_path, router(TWO_PAPERS[0])).packets
    report = triage_packets([packet], "anomalous_lore", db, ollama()[0], NOW)
    assert [item.gate for item in report.rejected] == ["corroboration"]


def test_two_newspapers_reach_the_ledger(
    tmp_path: Path, db: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    report = run_adapter(adapter(tmp_path, router(*TWO_PAPERS)), db, ollama()[0])
    assert report.rejected == [] and len(report.written) == 1

    row = db.execute(
        """
        SELECT anomaly_score, subject_domain, decomposition_receipt -> 'detector',
               source_ids
        FROM triage_records WHERE adapter_name = 'ubik_spectral'
        """
    ).fetchone()
    assert row is not None
    score, domain, detector, source_ids = row
    assert (score, domain) == (75.0, "anomalous_lore")
    assert detector["signal"] == SIGNAL
    sealed = db.execute(
        "SELECT domain, tier FROM source_registry WHERE source_id = ANY(%s)",
        (source_ids,),
    ).fetchall()
    assert sealed == [("tile.loc.gov", ARCHIVE_TIER)] * 2

    again = run_adapter(adapter(tmp_path, router(*TWO_PAPERS)), db, ollama()[0])
    assert again.written == [] and again.rejected == []


def test_stale_accounts_are_rejected_by_freshness(tmp_path: Path, db: Conn) -> None:
    old = [
        account(1, reported_at=NOW - timedelta(days=400)),
        account(2, reported_at=NOW - timedelta(days=390)),
    ]
    [packet] = harvest(tmp_path, router(*TWO_PAPERS), old).packets
    report = triage_packets([packet], "anomalous_lore", db, ollama()[0], NOW)
    assert [item.gate for item in report.rejected] == ["freshness_anomaly"]


def test_accounts_are_never_sealed_or_fetched(tmp_path: Path) -> None:
    reports = [
        account(1, source_url="https://forum.example.org/thread/1"),
        account(2, source_url="https://forum.example.org/thread/2"),
    ]
    mock = router(*TWO_PAPERS)
    [packet] = harvest(tmp_path, mock, reports).packets
    assert not any(item.url.host == "forum.example.org" for item in mock.requests)
    assert all("forum" not in item.origin_url for item in packet.source_artifacts)
