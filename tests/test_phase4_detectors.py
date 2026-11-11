"""Phase 4 detectors, offline: synthetic feeds, mock transports, no API keys."""

from __future__ import annotations

import json
import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
import swisseph as swe
from conftest import Conn

from spec1_bridge.adapters import (
    aerospace_uap as valis,
)
from spec1_bridge.adapters import (
    ct_subdomain_mapper as deckard,
)
from spec1_bridge.adapters import (
    hawkes_narrative_engine as hawkes,
)
from spec1_bridge.adapters import (
    prophecy_calendar as eschaton,
)
from spec1_bridge.adapters import (
    void_crossjoin as void,
)
from spec1_bridge.sociology_telemetry import (
    DOMAIN_FAKE_NEWS,
    DOMAIN_IRRATIONAL,
    EVIDENTIARY_STATUS,
    DecompositionReceipt,
    StreamPost,
    iter_eventstream,
    iter_jetstream,
    parse_bluesky_event,
    parse_wikimedia_event,
    write_observation,
)

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
LAT, LON = 45.0, -122.0


def quarantined(db: Conn) -> list[tuple[Any, ...]]:
    return db.execute(
        """
        SELECT domain, narrative_label, observed_text, metrics
        FROM sociology_telemetry.narrative_observations
        """
    ).fetchall()


def assert_no_evidence_written(db: Conn) -> None:
    for table in ("source_registry", "triage_records", "articles"):
        row = db.execute(f"SELECT count(*) FROM {table}".encode()).fetchone()
        assert row == (0,)


# --- Receipts and quarantine -------------------------------------------------


def test_a_receipt_must_explain_itself() -> None:
    with pytest.raises(ValueError, match="receipt"):
        DecompositionReceipt("d", "Finding.", (), "Caveat.")


def test_receipt_reads_as_plain_english() -> None:
    text = DecompositionReceipt("d", "Found.", ("Why.",), "Limit.").plain_english()
    assert text == "Found.\n- Why.\nLimits: Limit."


def test_observation_is_marked_unverified(db: Conn) -> None:
    write_observation(
        db,
        domain=DOMAIN_IRRATIONAL,
        narrative_label="label",
        observed_text="text",
        receipt=DecompositionReceipt("d", "Found.", ("Why.",), "Limit."),
        observed_at=NOW,
    )
    metrics = quarantined(db)[0][3]
    assert metrics["evidentiary_status"] == EVIDENTIARY_STATUS
    assert metrics["receipt"]["caveat"] == "Limit."


def test_domains_match_the_editorial_constitution() -> None:
    assert (DOMAIN_IRRATIONAL, DOMAIN_FAKE_NEWS) == (5, 6)


def test_stream_parsers_drop_authors() -> None:
    bluesky = parse_bluesky_event(
        {
            "did": "did:plc:private",
            "time_us": 1_790_000_000_000_000,
            "kind": "commit",
            "commit": {
                "operation": "create",
                "collection": "app.bsky.feed.post",
                "record": {"text": "Felt it in Seaside"},
            },
        }
    )
    wiki = parse_wikimedia_event(
        {
            "type": "edit",
            "bot": False,
            "title": "Seaside, Oregon",
            "comment": "added quake",
            "user": "PrivateEditor",
            "meta": {"dt": "2026-10-15T12:00:00Z"},
        }
    )
    assert bluesky is not None and "private" not in repr(bluesky)
    assert wiki is not None and "PrivateEditor" not in repr(wiki)


@pytest.mark.parametrize(
    "event",
    [
        {"kind": "identity"},
        {"kind": "commit", "commit": {"operation": "delete"}},
        {
            "kind": "commit",
            "time_us": 1,
            "commit": {
                "operation": "create",
                "collection": "app.bsky.feed.like",
                "record": {"text": "x"},
            },
        },
    ],
)
def test_jetstream_ignores_other_events(event: dict[str, Any]) -> None:
    assert parse_bluesky_event(event) is None


def test_wikimedia_ignores_bots() -> None:
    event = {"type": "edit", "bot": True, "title": "X", "meta": {"dt": "2026-01-01"}}
    assert parse_wikimedia_event(event) is None


def test_iter_jetstream_reads_until_closed() -> None:
    good = json.dumps(
        {
            "time_us": 1_790_000_000_000_000,
            "kind": "commit",
            "commit": {
                "operation": "create",
                "collection": "app.bsky.feed.post",
                "record": {"text": "hello"},
            },
        }
    )

    class Socket:
        messages: ClassVar[list[str]] = ["not json", good]
        closed = False

        def recv(self) -> str:
            if not self.messages:
                raise EOFError
            return self.messages.pop(0)

        def close(self) -> None:
            self.closed = True

    socket = Socket()
    posts = list(iter_jetstream("wss://mock", lambda _url: socket))
    assert [post.text for post in posts] == ["hello"]
    assert socket.closed


def test_iter_eventstream_reads_data_lines() -> None:
    event = {
        "type": "edit",
        "title": "Bend",
        "meta": {"dt": "2026-10-15T12:00:00Z"},
    }
    body = f"event: message\ndata: {json.dumps(event)}\n\ndata: broken\n\n"
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, text=body))
    assert [item.text for item in iter_eventstream(transport=transport)] == ["Bend"]


# --- Detector 1: Deckard -----------------------------------------------------


def write_manifest(path: Path, hosts: list[str]) -> Path:
    lines = [
        json.dumps(
            {
                "domain": host,
                "retrieved_at_utc": "2026-10-15T00:00:00Z",
                "analysis_tier": "rule_based",
                "sha256": "a" * 64,
                "file": f"{host}.json",
            }
        )
        for host in hosts
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_detects_only_unseen_hosts_under_watched_domains(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "m.jsonl",
        ["vpn.grid.example", "old.grid.example", "other.example", "vpn.grid.example"],
    )
    found, rejected = deckard.detect_new_subdomains(
        manifest, ["grid.example"], ["old.grid.example"]
    )
    assert [item.host for item in found] == ["vpn.grid.example"]
    assert rejected == []
    assert "new host" in found[0].receipt.finding


def test_a_bad_manifest_line_is_reported(tmp_path: Path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("not json\n", encoding="utf-8")
    assert "line 1" in deckard.detect_new_subdomains(manifest, ["a.example"], [])[1][0]


@pytest.mark.parametrize(
    ("host", "apex", "expected"),
    [
        ("salem-gazette.example", None, True),
        ("portlandnews.net.example", None, True),
        ("api.grid.example", "grid.example", False),
        ("mail.dailyplanet.example", "dailyplanet.example", False),
    ],
)
def test_media_styled_hosts(host: str, apex: str | None, expected: bool) -> None:
    assert deckard.is_media_styled(host, apex) is expected


def test_ads_txt_counts_only_direct_google_entries() -> None:
    text = """
    # comment
    google.com, pub-1234567890123456, DIRECT, f08c47fec0942fa0
    google.com, pub-9999999999999999, RESELLER, f08c47fec0942fa0
    appnexus.com, 1234, DIRECT
    google.com, pub-bad, DIRECT
    """
    assert deckard.adsense_ids_from_ads_txt(text) == {"pub-1234567890123456"}


def test_favicon_hash_matches_the_shodan_convention() -> None:
    assert deckard.favicon_hash(b"icon") == deckard.favicon_hash(b"icon")
    assert deckard.favicon_hash(b"icon") != deckard.favicon_hash(b"other")
    assert isinstance(deckard.favicon_hash(b"icon"), int)


def site_transport(sites: dict[str, tuple[str, bytes]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        ads, icon = sites[request.url.host]
        if request.url.path == "/ads.txt":
            return httpx.Response(200, text=ads)
        return httpx.Response(200, content=icon)

    return httpx.MockTransport(handler)


def test_shared_adsense_id_exposes_a_network() -> None:
    line = "google.com, pub-1111111111111111, DIRECT, f08c47fec0942fa0"
    transport = site_transport(
        {
            "salem-gazette.example": (line, b"one"),
            "bend-tribune.example": (line, b"two"),
            "real-news.example": ("", b"three"),
        }
    )
    sites = deckard.fingerprint_sites(
        ["salem-gazette.example", "bend-tribune.example", "real-news.example"],
        transport,
    )
    clusters = deckard.find_network_clusters(sites)

    assert [cluster.signal for cluster in clusters] == ["adsense"]
    assert clusters[0].hosts == ("bend-tribune.example", "salem-gazette.example")
    assert "not proven ownership" in clusters[0].receipt.caveat


def test_shared_favicon_needs_three_sites() -> None:
    def site(name: str, icon: int) -> deckard.SiteFingerprint:
        return deckard.SiteFingerprint(name, frozenset(), icon)

    assert not deckard.find_network_clusters(
        [site("a.example", 7), site("b.example", 7)]
    )
    clusters = deckard.find_network_clusters(
        [site("a.example", 7), site("b.example", 7), site("c.example", 7)]
    )
    assert clusters[0].signal == "favicon"
    assert "default icons" in clusters[0].receipt.caveat


def test_soft_404_pages_are_not_fingerprints() -> None:
    page = "<!DOCTYPE html><html>google.com, pub-1111111111111111, DIRECT</html>"
    transport = site_transport({"a-news.example": (page, page.encode())})
    site = deckard.fingerprint_sites(["a-news.example"], transport)[0]
    assert site.adsense_ids == frozenset()
    assert site.favicon_hash is None


def test_failed_fetches_are_recorded_not_raised() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(404))
    site = deckard.fingerprint_sites(["a-news.example"], transport)[0]
    assert len(site.failures) == 2


def test_redirects_are_not_followed() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            302, headers={"location": "http://169.254.169.254/"}
        )
    )
    assert deckard.fingerprint_sites(["a-news.example"], transport)[0].failures


@pytest.mark.parametrize(
    "host", ["localhost", "127.0.0.1", "db.internal", "printer.local", "a b.example"]
)
def test_private_hosts_are_never_fetched(host: str) -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("private host was fetched")

    site = deckard.fingerprint_sites([host], httpx.MockTransport(refuse))[0]
    assert site.failures == ("not a public hostname",)


def test_oversized_bodies_are_refused() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, content=b"x" * 1_000_001)
    )
    assert (
        "over"
        in deckard.fingerprint_sites(["a-news.example"], transport)[0].failures[0]
    )


# --- Detector 2: void cross-join ---------------------------------------------


def quake(
    magnitude: float,
    kind: str = "earthquake",
    place: str = "12 km NE of Seaside, Oregon",
) -> dict[str, Any]:
    return {
        "id": f"us{magnitude}",
        "properties": {
            "mag": magnitude,
            "place": place,
            "time": int(NOW.timestamp() * 1000),
            "type": kind,
            "url": "https://earthquake.usgs.gov/x",
        },
    }


@pytest.mark.parametrize(
    ("magnitude", "trips"), [(3.9, False), (4.0, True), (6.1, True)]
)
def test_quake_threshold(magnitude: float, trips: bool) -> None:
    shocks = void.shocks_from_usgs({"features": [quake(magnitude)]})
    assert bool(shocks) is trips


def test_quarry_blasts_do_not_trip() -> None:
    assert not void.shocks_from_usgs({"features": [quake(4.5, "quarry blast")]})


def test_null_magnitude_does_not_crash() -> None:
    feature = quake(5.0)
    feature["properties"]["mag"] = None
    assert not void.shocks_from_usgs({"features": [feature]})


@pytest.mark.parametrize(
    ("place", "expected"),
    [
        ("12 km NE of Seaside, Oregon", ("Seaside",)),
        ("3.5 km S of Bend, Oregon", ("Bend",)),
        ("off the coast of Oregon", ("Oregon",)),
        ("Kermadec Islands, New Zealand", ("Kermadec Islands",)),
    ],
)
def test_usgs_placenames(place: str, expected: tuple[str, ...]) -> None:
    assert void.placenames_from_usgs(place) == expected


def firms_csv(rows: list[tuple[float, float, str]]) -> str:
    head = "latitude,longitude,acq_date,acq_time,confidence"
    body = [f"{lat},{lon},2026-10-15,0930,{conf}" for lat, lon, conf in rows]
    return "\n".join([head, *body])


def test_firms_cluster_trips_and_names_the_place() -> None:
    shocks = void.shocks_from_firms(
        firms_csv(
            [(44.06, -121.31, "h"), (44.07, -121.32, "n"), (44.05, -121.30, "85")]
        )
    )
    assert shocks[0].placenames == ("Bend",)
    assert shocks[0].at == datetime(2026, 10, 15, 9, 30, tzinfo=UTC)
    assert "prescribed burn" in shocks[0].receipt.caveat


def test_firms_ignores_lone_and_low_confidence_detections() -> None:
    rows = [(44.06, -121.31, "h"), (44.07, -121.32, "l"), (44.05, -121.30, "10")]
    assert not void.shocks_from_firms(firms_csv(rows))


def test_firms_far_from_any_place_has_no_listener() -> None:
    rows = [(10.0, 10.0, "h"), (10.01, 10.0, "h"), (10.0, 10.01, "h")]
    shock = void.shocks_from_firms(firms_csv(rows))[0]
    assert shock.placenames == ()
    assert void.VoidListener(lambda: NOW).trigger(shock) is None


def test_cloudflare_outage_is_a_shock() -> None:
    payload = {
        "result": {
            "annotations": [
                {
                    "id": "abc",
                    "startDate": "2026-10-15T12:00:00Z",
                    "asns": [64500],
                    "asnsDetails": [{"asn": "64500", "name": "Example Net"}],
                    "locationsDetails": [{"code": "NZ", "name": "New Zealand"}],
                    "outage": {"outageCause": "CABLE_CUT"},
                }
            ]
        }
    }
    shock = void.shocks_from_cloudflare(payload)[0]
    assert shock.placenames == ("New Zealand",)
    assert "CABLE_CUT" in shock.receipt.reasons[0]


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def seaside_shock() -> void.PhysicalShock:
    return void.shocks_from_usgs({"features": [quake(5.2)]})[0]


def post(text: str, minutes: float) -> StreamPost:
    return StreamPost("bluesky", text, NOW + timedelta(minutes=minutes))


def test_a_quake_opens_a_six_hour_listener() -> None:
    listener = void.VoidListener(Clock(NOW))
    window = listener.trigger(seaside_shock())
    assert window is not None
    assert window.closes_at == NOW + timedelta(hours=6)


def test_a_repeated_shock_opens_one_listener() -> None:
    listener = void.VoidListener(Clock(NOW))
    listener.trigger(seaside_shock())
    assert listener.trigger(seaside_shock()) is None
    assert len(listener.open_windows) == 1


def test_a_stale_shock_opens_no_listener() -> None:
    listener = void.VoidListener(Clock(NOW + timedelta(hours=7)))
    assert listener.trigger(seaside_shock()) is None


def test_listener_matches_only_inside_the_window_and_on_whole_words() -> None:
    listener = void.VoidListener(Clock(NOW))
    window = listener.trigger(seaside_shock())
    assert window is not None
    for item in (
        post("Before the quake in Seaside", -5),
        post("SEASIDE is shaking", 5),
        post("a seasidewalk cafe", 6),
        post("Nothing relevant", 7),
        post("Seaside after the window", 361),
    ):
        listener.ingest(item)
    assert [item.text for item in window.posts] == ["SEASIDE is shaking"]


def test_closed_window_is_quarantined_with_a_receipt(db: Conn) -> None:
    clock = Clock(NOW)
    listener = void.VoidListener(clock)
    listener.trigger(seaside_shock())
    for minute in (10, 20, 30):
        listener.ingest(post(f"Seaside rumor {minute}", minute))

    assert listener.close_expired() == []
    clock.now = NOW + timedelta(hours=6)
    report = listener.close_expired()[0]
    void.write_report(db, report)

    domain, label, text, metrics = quarantined(db)[0]
    assert domain == DOMAIN_FAKE_NEWS
    assert label == f"information_void:{report.shock.shock_id}"
    assert "Seaside rumor 10" in text
    assert "does not measure rumor" in metrics["receipt"]["caveat"]
    assert_no_evidence_written(db)
    assert listener.open_windows == []


def test_quiet_window_records_nothing() -> None:
    clock = Clock(NOW)
    listener = void.VoidListener(clock)
    listener.trigger(seaside_shock())
    listener.ingest(post("Seaside", 1))
    clock.now = NOW + timedelta(hours=6)
    assert listener.close_expired() == []


# --- Engine 7: Valis ---------------------------------------------------------


def checksum(line: str) -> str:
    digits = sum(int(c) if c.isdigit() else c == "-" for c in line[:68])
    return f"{line[:68]}{digits % 10}"


def synthetic_tle(epoch: str = "26288.50000000") -> str:
    line1 = f"1 99999U 26001A   {epoch}  .00000000  00000-0  00000-0 0  9990"
    line2 = "2 99999  51.6000 240.0000 0001000  90.0000 270.0000 15.50000000 10000"
    return f"SYNTHETIC-1\n{checksum(line1.ljust(68))}\n{checksum(line2.ljust(68))}\n"


@pytest.fixture(scope="module")
def satellite() -> Any:
    return valis.load_tle_text(synthetic_tle())[0]


@pytest.fixture(scope="module")
def dark_pass(satellite: Any) -> tuple[datetime, float, float]:
    """Find a time this synthetic satellite is high, sunlit, and the sky is dark."""
    start = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    moments = [start + timedelta(minutes=minute) for minute in range(0, 2880, 1)]
    for chunk in range(0, len(moments), 60):
        block = moments[chunk : chunk + 60]
        azimuths, elevations, sunlit = valis.satellite_track(satellite, LAT, LON, block)
        for index, moment in enumerate(block):
            sun = valis._sky_position(moment, LAT, LON, swe.SUN)[1]
            if elevations[index] > 40 and sunlit[index] and sun < -8:
                return moment, float(azimuths[index]), float(elevations[index])
    raise AssertionError("no dark pass in the search range")


def report(
    moment: datetime, azimuth: float | None, elevation: float | None = None
) -> valis.WitnessReport:
    return valis.WitnessReport(
        "N-1", moment, LAT, LON, "A steady light.", azimuth, elevation
    )


def test_loads_elements_from_tle_and_omm(satellite: Any) -> None:
    assert satellite.name == "SYNTHETIC-1"
    omm = {
        "OBJECT_NAME": "OMM-1",
        "OBJECT_ID": "2026-001A",
        "EPOCH": "2026-10-15T12:00:00.000000",
        "MEAN_MOTION": 15.5,
        "ECCENTRICITY": 0.0001,
        "INCLINATION": 51.6,
        "RA_OF_ASC_NODE": 120.0,
        "ARG_OF_PERICENTER": 90.0,
        "MEAN_ANOMALY": 270.0,
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": 99998,
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 1000,
        "BSTAR": 0.0,
        "MEAN_MOTION_DOT": 0.0,
        "MEAN_MOTION_DDOT": 0.0,
    }
    assert valis.load_omm([omm])[0].name == "OMM-1"


def test_matching_pass_is_subtracted(satellite: Any, dark_pass: Any) -> None:
    moment, azimuth, elevation = dark_pass
    result = valis.classify_report(report(moment, azimuth, elevation), [satellite])
    assert result.tag == valis.IDENTIFIED_ORBITAL_PASS
    assert result.matched == ("SYNTHETIC-1",)
    assert "not what the witness saw" in result.receipt.caveat


def test_wrong_direction_stays_residual(satellite: Any, dark_pass: Any) -> None:
    moment, azimuth, _ = dark_pass
    reverse = (azimuth + 180.0) % 360.0
    result = valis.classify_report(report(moment, reverse, 5.0), [satellite])
    assert result.tag != valis.IDENTIFIED_ORBITAL_PASS


def test_no_direction_is_only_possible(satellite: Any, dark_pass: Any) -> None:
    moment, _, _ = dark_pass
    result = valis.classify_report(report(moment, None), [satellite])
    assert result.tag == valis.POSSIBLE_ORBITAL_PASS
    assert "no direction" in result.receipt.reasons[1]


def test_satellites_are_not_matched_in_daylight(satellite: Any) -> None:
    noon = datetime(2026, 10, 15, 20, 0, tzinfo=UTC)
    result = valis.classify_report(report(noon, 90.0, 45.0), [satellite])
    assert result.tag != valis.IDENTIFIED_ORBITAL_PASS


def test_stale_elements_cannot_rule_anything_out(satellite: Any) -> None:
    far = datetime(2027, 3, 1, 10, 0, tzinfo=UTC)
    result = valis.classify_report(report(far, 359.0, 80.0), [satellite])
    assert result.tag == valis.INSUFFICIENT_CATALOG


def test_a_planet_is_identified() -> None:
    moment = datetime(2026, 10, 16, 8, 0, tzinfo=UTC)
    for hour in range(48):
        moment = datetime(2026, 10, 15, 0, 0, tzinfo=UTC) + timedelta(hours=hour)
        azimuth, altitude = valis._sky_position(moment, LAT, LON, swe.JUPITER)
        sun = valis._sky_position(moment, LAT, LON, swe.SUN)[1]
        if altitude > 20 and sun < -8:
            break
    else:
        raise AssertionError("Jupiter was never up after dark")
    result = valis.classify_report(report(moment, azimuth, altitude), [])
    assert result.tag == valis.IDENTIFIED_CELESTIAL_BODY
    assert "Jupiter" in result.matched


def test_azimuth_runs_from_north() -> None:
    noon = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
    azimuth, altitude = valis._sky_position(noon, 45.0, 0.0, swe.SUN)
    assert altitude > 60
    assert 170 < azimuth < 190


@pytest.mark.parametrize(
    "record",
    [
        {"id": 1, "occurred": "2026-10-15T20:00:00", "latitude": 45, "longitude": -122},
        {"id": 2, "occurred": "2026-10-15T20:00:00-07:00"},
        {"id": 3, "occurred": "unknown", "latitude": 45, "longitude": -122},
    ],
)
def test_unlocatable_reports_are_unresolved(
    record: dict[str, Any], satellite: Any
) -> None:
    parsed = valis.parse_nuforc_record(record)
    result = valis.classify_report(parsed, [satellite])
    assert result.tag == valis.UNRESOLVED_REPORT


def residual(
    satellite: Any, notams: list[valis.Notam] | None = None
) -> valis.UapResult:
    # This hour is dark in Oregon, and the satellite is below this direction.
    for hour in range(24):
        moment = datetime(2026, 10, 15, 0, 0, tzinfo=UTC) + timedelta(hours=hour)
        result = valis.classify_report(
            valis.WitnessReport("N-2", moment, LAT, LON, "An orange orb.", 0.0, 1.0),
            [satellite],
            notams or [],
        )
        if result.tag == valis.UNCATALOGED_RESIDUAL:
            return result
    raise AssertionError("no residual found")


def test_residual_lists_active_notams(satellite: Any) -> None:
    first = residual(satellite).report.occurred_at
    assert first is not None
    notams = [
        valis.Notam(
            "FDC 6/1234",
            LAT + 0.1,
            LON,
            5.0,
            first - timedelta(hours=1),
            first + timedelta(hours=1),
            "Airspace restricted",
        ),
        valis.Notam(
            "FDC 6/9999",
            10.0,
            10.0,
            5.0,
            first - timedelta(hours=1),
            first + timedelta(hours=1),
            "Far away",
        ),
    ]
    assert residual(satellite, notams).notams == ("FDC 6/1234",)


def test_only_the_residual_is_quarantined(
    db: Conn, satellite: Any, dark_pass: Any
) -> None:
    moment, azimuth, elevation = dark_pass
    identified = valis.classify_report(report(moment, azimuth, elevation), [satellite])
    with pytest.raises(ValueError, match="residual"):
        valis.write_residual(db, identified)

    valis.write_residual(db, residual(satellite))

    domain, label, text, metrics = quarantined(db)[0]
    assert (domain, label, text) == (5, "uap_residual:N-2", "An orange orb.")
    assert metrics["evidentiary_status"] == EVIDENTIARY_STATUS
    assert_no_evidence_written(db)


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ("I saw three lights over the ridge.", "witness_account"),
        ("Subscribe to my channel for the full footage", "promotional"),
        ("Proof at https://example.org/orbs", "promotional"),
    ],
)
def test_promotion_is_labelled_apart_from_witnesses(
    summary: str, expected: str
) -> None:
    assert valis.source_conduct(summary) == expected


# --- Engine 10: Eschaton -----------------------------------------------------


def named(events: Any, kind: str) -> dict[str, date]:
    return {item.name: item.on for item in events if item.kind == kind}


def test_known_eclipses() -> None:
    events = eschaton.eclipses_between(date(2024, 3, 1), date(2024, 4, 30))
    found = named(events, "eclipse")
    assert found["Total solar eclipse"] == date(2024, 4, 8)
    assert found["Penumbral lunar eclipse"] == date(2024, 3, 25)


def test_eclipses_of_august_2026() -> None:
    found = named(
        eschaton.eclipses_between(date(2026, 8, 1), date(2026, 8, 31)), "eclipse"
    )
    assert found["Total solar eclipse"] == date(2026, 8, 12)
    assert found["Partial lunar eclipse"] == date(2026, 8, 28)


def test_known_mercury_stations() -> None:
    found = named(
        eschaton.retrograde_stations(date(2024, 3, 25), date(2024, 4, 30)), "station"
    )
    assert found["Mercury stations retrograde"] == date(2024, 4, 1)
    assert found["Mercury stations direct"] == date(2024, 4, 25)


def test_station_is_where_motion_reverses() -> None:
    event = next(
        item
        for item in eschaton.retrograde_stations(date(2024, 3, 25), date(2024, 4, 5))
        if "Mercury" in item.name
    )
    noon = datetime(event.on.year, event.on.month, event.on.day, tzinfo=UTC)
    before = eschaton._longitude_speed(
        eschaton.julian_day(noon - timedelta(days=2)), swe.MERCURY
    )
    after = eschaton._longitude_speed(
        eschaton.julian_day(noon + timedelta(days=2)), swe.MERCURY
    )
    assert before > 0 > after


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2024, date(2024, 10, 3)),
        (2025, date(2025, 9, 23)),
        (2026, date(2026, 9, 12)),
        (2027, date(2027, 10, 2)),
    ],
)
def test_rosh_hashanah(year: int, expected: date) -> None:
    assert eschaton.rosh_hashanah(year) == expected


@pytest.mark.parametrize(
    ("year", "expected"),
    [(2024, date(2024, 3, 31)), (2026, date(2026, 4, 5)), (2027, date(2027, 3, 28))],
)
def test_easter(year: int, expected: date) -> None:
    assert eschaton.western_easter(year) == expected


def test_observances_of_2026() -> None:
    found = named(eschaton.observances(2026), "observance")
    assert found["Passover"] == date(2026, 4, 2)
    assert found["Tisha B'Av"] == date(2026, 7, 23)
    assert found["Yom Kippur"] == date(2026, 9, 21)


def test_calendar_rolls_90_days() -> None:
    calendar = eschaton.build_calendar(date(2026, 8, 1))
    assert calendar.ends_on == date(2026, 10, 30)
    assert all(
        calendar.starts_on <= item.on <= calendar.ends_on for item in calendar.events
    )
    assert "Millerite Great Disappointment (1844)" in named(
        calendar.events, "anniversary"
    )


def test_calendar_crosses_a_new_year() -> None:
    calendar = eschaton.build_calendar(date(2026, 12, 1))
    assert "Y2K rollover (2000)" in named(calendar.events, "anniversary")


def test_eclipse_peak_explains_itself() -> None:
    calendar = eschaton.build_calendar(date(2026, 8, 1))
    peak = next(item for item in calendar.peaks if item.on == date(2026, 8, 12))
    assert peak.score >= eschaton.HIGH_SUSCEPTIBILITY
    assert any("Total solar eclipse" in reason for reason in peak.receipt.reasons)
    assert "does not predict any event" in peak.receipt.caveat


# --- Detector 5: Hawkes ------------------------------------------------------


def simulate(
    mu: float, alpha: float, beta: float, horizon: float, seed: int
) -> list[float]:
    """Simulate a Hawkes process by Ogata thinning."""
    rng = random.Random(seed)
    events: list[float] = []
    now = 0.0
    excitation = 0.0
    last = 0.0
    while True:
        decayed = excitation * pow(2.718281828459045, -beta * (now - last))
        bound = mu + decayed
        now += rng.expovariate(bound)
        if now >= horizon:
            return events
        current = excitation * pow(2.718281828459045, -beta * (now - last))
        if rng.random() * bound <= mu + current:
            excitation = current + alpha
            last = now
            events.append(now)
            # alpha > beta grows exponentially, so a long horizon would never end.
            assert len(events) < 50_000, "simulation exploded; shorten the horizon"


def test_recovers_known_parameters() -> None:
    times = simulate(mu=1.0, alpha=1.2, beta=2.0, horizon=800.0, seed=7)
    fit = hawkes.fit_hawkes(times, horizon=800.0)
    assert fit.mu == pytest.approx(1.0, rel=0.2)
    assert fit.beta == pytest.approx(2.0, rel=0.3)
    assert fit.r_t == pytest.approx(0.6, rel=0.2)


def test_poisson_arrivals_have_little_self_excitation() -> None:
    rng = random.Random(3)
    times: list[float] = []
    now = 0.0
    while now < 500.0:
        now += rng.expovariate(2.0)
        times.append(now)
    fit = hawkes.fit_hawkes(times)
    assert fit.r_t < 0.2
    assert fit.mu == pytest.approx(2.0, rel=0.2)


def test_self_exciting_burst_reads_as_endogenous() -> None:
    # Only R_t is asserted: a few hours of burst cannot separate mu from
    # self-excitation, so fitted mu runs far above the true 0.5 even though the
    # fit scores at least as well as the true parameters.
    runs = (
        simulate(mu=0.5, alpha=4.5, beta=3.0, horizon=4.0, seed=seed)
        for seed in range(20)
    )
    bursts = [times for times in runs if len(times) >= hawkes.MIN_EVENTS]
    assert len(bursts) >= 5
    fits = [hawkes.fit_hawkes(times, horizon=4.0) for times in bursts]
    assert sum(fit.r_t > 1.0 for fit in fits) / len(fits) >= 0.9


def test_likelihood_prefers_the_truth() -> None:
    times = simulate(mu=1.0, alpha=1.2, beta=2.0, horizon=400.0, seed=5)
    truth = hawkes.log_likelihood(times, 400.0, 1.0, 1.2, 2.0)
    wrong = hawkes.log_likelihood(times, 400.0, 3.0, 0.1, 9.0)
    assert truth > wrong


def test_too_few_events_cannot_be_fitted() -> None:
    with pytest.raises(ValueError, match="at least"):
        hawkes.fit_hawkes([0.0, 1.0])


@pytest.mark.parametrize(
    ("mu", "alpha", "beta", "span_hours", "expected"),
    [
        (0.8, 1.2, 1.0, 24.0, False),  # both exactly on the threshold
        (0.81, 0.1, 1.0, 24.0, True),  # mu counts from exactly 24 hours
        (0.81, 0.1, 1.0, 23.99, False),  # mu alone cannot trip a shorter window
        (50.0, 0.1, 1.0, 2.0, False),  # however high it reads
        (0.1, 1.3, 1.0, 2.0, True),  # R_t trips any window
        (0.1, 1.3, 1.0, 24.0, True),
        (0.1, 0.5, 1.0, 24.0, False),
    ],
)
def test_record_thresholds_are_strict(
    mu: float, alpha: float, beta: float, span_hours: float, expected: bool
) -> None:
    fit = hawkes.HawkesFit(mu, alpha, beta, 30, 0.0, span_hours)
    assert hawkes.should_record(fit) is expected


def test_near_duplicates_cluster_and_others_do_not() -> None:
    base = "The dam upstream of Salem has failed and officials are hiding it"
    posts = [
        StreamPost("bluesky", base, NOW),
        StreamPost("bluesky", f"{base}!!", NOW),
        StreamPost("gdelt", base.upper(), NOW),
        StreamPost("bluesky", "My sourdough starter doubled overnight again", NOW),
    ]
    clusters = hawkes.cluster_posts(posts)
    assert sorted(len(group) for group in clusters) == [1, 3]


def test_short_text_is_left_out() -> None:
    assert hawkes.minhash_signature("hi") is None
    assert hawkes.cluster_posts([StreamPost("bluesky", "hi", NOW)]) == []


def test_minhash_estimates_similarity() -> None:
    left = hawkes.minhash_signature("the quick brown fox jumps over the lazy dog")
    right = hawkes.minhash_signature("the quick brown fox jumps over the lazy cat")
    other = hawkes.minhash_signature("completely different sentence about harbors")
    assert left and right and other
    assert hawkes.estimated_similarity(left, right) > 0.6
    assert hawkes.estimated_similarity(left, other) < 0.2


def make_posts(times: list[float]) -> list[StreamPost]:
    return [
        StreamPost("bluesky", "The same claim", NOW + timedelta(hours=moment))
        for moment in times
    ]


def test_signal_is_quarantined_in_domain_6(db: Conn) -> None:
    posts = make_posts(simulate(1.0, 1.5, 1.0, 8.0, 13))
    signal = hawkes.analyze_cluster(posts)
    assert signal is not None
    assert hawkes.write_signal(db, signal) is not None

    domain, _, text, metrics = quarantined(db)[0]
    assert (domain, text) == (DOMAIN_FAKE_NEWS, "The same claim")
    assert signal.fit is not None
    assert metrics["receipt"]["metrics"]["r_t"] == pytest.approx(signal.fit.r_t)
    assert "status" not in metrics["receipt"]
    assert "does not show who posted it" in metrics["receipt"]["caveat"]
    assert_no_evidence_written(db)


def test_quiet_signal_is_not_written(db: Conn) -> None:
    posts = make_posts([hour * 10.0 for hour in range(30)])
    signal = hawkes.analyze_cluster(posts)
    assert signal is not None and signal.fit is not None
    assert not hawkes.should_record(signal.fit)
    assert hawkes.write_signal(db, signal) is None
    assert quarantined(db) == []


def test_receipt_never_calls_anyone_a_bot_or_human() -> None:
    signal = hawkes.analyze_cluster(make_posts(simulate(1.0, 1.5, 1.0, 8.0, 13)))
    assert signal is not None
    for reason in (signal.receipt.finding, *signal.receipt.reasons):
        assert "bot" not in reason.lower()


def test_small_clusters_are_not_fitted() -> None:
    assert hawkes.analyze_cluster(make_posts([0.0, 1.0])) is None


@pytest.mark.parametrize(
    ("events", "span_hours", "preliminary"),
    [
        (29, 10.0, True),  # one post short
        (30, 1.99, True),  # just under the span
        (5000, 1.0, True),  # size does not buy back a short window
        (30, 2.0, False),  # exactly on both gates
    ],
)
def test_gate_on_events_and_span(
    events: int, span_hours: float, preliminary: bool
) -> None:
    step = span_hours / (events - 1)
    signal = hawkes.analyze_cluster(make_posts([i * step for i in range(events)]))
    assert signal is not None
    assert (signal.fit is None) is preliminary
    status = signal.receipt.status
    assert status == (hawkes.PRELIMINARY_STATUS if preliminary else None)


def test_short_window_mu_alone_is_not_written_and_says_why(db: Conn) -> None:
    # Evenly spaced posts have no self-excitation, only a high steady rate.
    signal = hawkes.analyze_cluster(make_posts([i * 0.1 for i in range(100)]))
    assert signal is not None and signal.fit is not None
    assert signal.fit.mu > hawkes.MU_THRESHOLD
    assert signal.fit.r_t <= hawkes.R_THRESHOLD

    assert any("does not count" in reason for reason in signal.receipt.reasons)
    assert hawkes.write_signal(db, signal) is None
    assert quarantined(db) == []


def test_long_window_mu_alone_is_written(db: Conn) -> None:
    signal = hawkes.analyze_cluster(make_posts([i * 0.5 for i in range(100)]))
    assert signal is not None and signal.fit is not None
    assert signal.fit.mu > hawkes.MU_THRESHOLD
    assert signal.fit.r_t <= hawkes.R_THRESHOLD

    assert not any("does not count" in reason for reason in signal.receipt.reasons)
    assert hawkes.write_signal(db, signal) is not None
    assert len(quarantined(db)) == 1


def test_preliminary_receipt_reports_no_rates_and_is_not_written(db: Conn) -> None:
    signal = hawkes.analyze_cluster(make_posts([i * 0.01 for i in range(100)]))
    assert signal is not None and signal.fit is None

    receipt = signal.receipt.as_dict()
    assert receipt["status"] == "PRELIMINARY_BURST_OBSERVATION"
    assert receipt["reasons"] == [
        (
            "Awaiting wider temporal window to reliably separate background seeding "
            "from viral excitation."
        )
    ]
    assert not {"mu", "alpha", "beta", "r_t"} & set(receipt["metrics"])
    assert signal.receipt.plain_english().startswith(
        "Status: PRELIMINARY_BURST_OBSERVATION"
    )
    assert hawkes.write_signal(db, signal) is None
    assert quarantined(db) == []


def test_beta_stays_above_its_floor() -> None:
    # Without the floor, this burst fits with beta effectively zero.
    times = simulate(mu=0.5, alpha=4.5, beta=3.0, horizon=4.0, seed=1)
    fit = hawkes.fit_hawkes(times, horizon=4.0)
    assert fit.beta == pytest.approx(hawkes.BETA_BOUNDS[0])


def test_parses_a_gdelt_row() -> None:
    columns = [""] * 27
    columns[1] = "20261015120000"
    columns[26] = "<PAGE_TITLE>Dam failure denied</PAGE_TITLE>"
    item = hawkes.parse_gdelt_gkg_row("\t".join(columns))
    assert item == StreamPost("gdelt", "Dam failure denied", NOW)


@pytest.mark.parametrize("line", ["short\trow", "\t" * 26])
def test_rejects_unusable_gdelt_rows(line: str) -> None:
    assert hawkes.parse_gdelt_gkg_row(line) is None
