"""Travel (CBP border waits) and nature (USGS streamflow) adapters, offline.

Every request goes through test_phase1_adapters.Router, which fails the test on
any URL it was not given, so nothing here touches the network.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from test_phase1_adapters import NOW, Router, counter, options

from spec1_bridge.adapters.nature_hydro import (
    CONTINUOUS_URL,
    GAUGES,
    MIN_BASELINE_STEPS,
    SIGMA_THRESHOLD,
    NatureHydroAdapter,
    surge_stats,
)
from spec1_bridge.adapters.travel_border import (
    BWT_URL,
    DIVERGENCE_MINUTES,
    QUEUE_MINUTES,
    TravelBorderAdapter,
    parse_crossing,
    parse_update_time,
)

# --- CBP Border Wait Times ----------------------------------------------------


def lanes(delay: str, update: str = "At 11:00 am PST") -> dict[str, str]:
    status = "delay" if delay not in ("", "0") else "no delay"
    return {
        "update_time": update if delay != "" else "",
        "operational_status": status if delay != "" else "Update Pending",
        "delay_minutes": delay,
        "lanes_open": "2" if delay != "" else "",
    }


def port(
    number: str = "300401",
    *,
    commercial: str = "70",
    passenger: str = "10",
    commercial_lanes: str = "3",
    name: str = "Blaine",
    crossing: str = "Pacific Highway",
    update: str = "At 11:00 am PST",
) -> dict[str, Any]:
    """A crossing shaped like a live bwtnew record."""
    return {
        "port_number": number,
        "border": "Canadian Border",
        "port_name": name,
        "crossing_name": crossing,
        "hours": "24 hrs/day",
        "date": "10/15/2026",
        "time": "11:05:03",
        "port_status": "Open",
        "commercial_vehicle_lanes": {
            "maximum_lanes": commercial_lanes,
            "cv_automation_type": "Manual",
            "standard_lanes": lanes(commercial, update),
            "FAST_lanes": lanes(""),
        },
        "passenger_vehicle_lanes": {
            "maximum_lanes": "7",
            "pv_automation_type": "Manual",
            "standard_lanes": lanes(passenger, update),
            "NEXUS_SENTRI_lanes": lanes(""),
            "ready_lanes": lanes(""),
        },
        "pedestrian_lanes": {"maximum_lanes": "N/A"},
        "construction_notice": "",
    }


def bwt_router(
    ports: list[dict[str, Any]], sealed: dict[str, Any] | None = None
) -> Router:
    routes: dict[str, Any] = {
        BWT_URL: httpx.Response(200, json=ports),
    }
    for row in ports:
        record = sealed if sealed is not None else row
        routes[f"{BWT_URL}/{row['port_number']}"] = httpx.Response(200, json=record)
    return Router(routes)


def travel(tmp_path: Path, mock: Router) -> TravelBorderAdapter:
    return TravelBorderAdapter(**options(tmp_path, mock))


def harvest_travel(tmp_path: Path, mock: Router) -> Any:
    return travel(tmp_path, mock).harvest(counter())


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ("At 11:00 am PST", datetime(2026, 10, 15, 19, 0, tzinfo=UTC)),
        ("At 2:57 pm EST", datetime(2026, 10, 15, 19, 57, tzinfo=UTC)),
        ("At 12:05 am CDT", datetime(2026, 10, 15, 5, 5, tzinfo=UTC)),
        ("At 12:30 pm MST", datetime(2026, 10, 15, 19, 30, tzinfo=UTC)),
    ],
)
def test_update_times_are_read_in_the_ports_zone(
    update: str, expected: datetime
) -> None:
    assert parse_update_time("10/15/2026", update) == expected


@pytest.mark.parametrize(
    ("day", "update"),
    [
        ("10/15/2026", ""),
        ("10/15/2026", "At 11:00 am XYZ"),
        ("10/15/2026", "11:00"),
        ("2026-10-15", "At 11:00 am PST"),
        ("10/15/2026", "At 11:75 am PST"),
    ],
)
def test_unreadable_update_times_are_none(day: str, update: str) -> None:
    assert parse_update_time(day, update) is None


def test_divergence_over_45_minutes_flags_a_major_crossing() -> None:
    reading = parse_crossing(port(commercial="70", passenger="10"))
    assert reading is not None
    assert reading.divergence == 60 and reading.diverges
    assert not reading.queue_exceeded and reading.flagged


def test_divergence_of_exactly_45_minutes_does_not_flag() -> None:
    reading = parse_crossing(port(commercial="55", passenger="10"))
    assert reading is not None and reading.divergence == DIVERGENCE_MINUTES
    assert not reading.flagged


def test_a_queue_over_90_minutes_flags_without_divergence() -> None:
    reading = parse_crossing(port(commercial="100", passenger="95"))
    assert reading is not None
    assert not reading.diverges and reading.queue_exceeded and reading.flagged


def test_commercial_only_crossings_can_flag_on_queue_alone() -> None:
    reading = parse_crossing(port(commercial="120", passenger=""))
    assert reading is not None
    assert reading.divergence is None and reading.longest_queue == 120
    assert reading.flagged


@pytest.mark.parametrize("commercial_lanes", ["1", "2", "N/A", ""])
def test_minor_crossings_are_never_flagged(commercial_lanes: str) -> None:
    reading = parse_crossing(
        port(commercial="200", passenger="5", commercial_lanes=commercial_lanes)
    )
    assert reading is not None and not reading.is_major and not reading.flagged


@pytest.mark.parametrize("row", [None, [], "port", {}, {"port_number": ""}])
def test_unusable_port_records_are_not_read(row: Any) -> None:
    assert parse_crossing(row) is None


def test_divergence_lead_is_sealed_from_the_crossings_own_record(
    tmp_path: Path,
) -> None:
    mock = bwt_router([port(), port("090101", commercial="3", passenger="2")])
    result = harvest_travel(tmp_path, mock)
    assert result.errors == [] and result.evidence_rejections == []
    [packet] = result.packets
    assert packet.adapter_name == "travel_border"
    [artifact] = packet.source_artifacts
    assert artifact.origin_url == f"{BWT_URL}/300401"
    assert artifact.tier == "tier_1_primary"
    assert packet.anomaly_score == 57.5  # 50 + (60 - 45) * 0.5
    assert packet.timestamp == datetime(2026, 10, 15, 19, 0, tzinfo=UTC)
    assert packet.public_question == (
        "What explains the gap between freight and passenger waits at the "
        "Blaine (Pacific Highway) port of entry?"
    )
    assert packet.public_interest_reason == (
        "CBP posted a 70 min commercial wait and a 10 min passenger wait at "
        "Blaine (Pacific Highway) on 10/15/2026; long queues at a major crossing "
        "delay goods, workers, and travelers who depend on it."
    )
    receipt = packet.receipt_details
    assert receipt["divergence_minutes"] == 60 and receipt["diverges"] is True
    assert receipt["thresholds"]["queue_minutes"] == QUEUE_MINUTES
    assert "estimates" in packet.paste_ready_prompt
    sealed = [str(item.url) for item in mock.requests if "/bwtnew/" in str(item.url)]
    assert sealed == [f"{BWT_URL}/300401"]  # the calm crossing is never sealed


def test_both_signals_score_higher(tmp_path: Path) -> None:
    [packet] = harvest_travel(
        tmp_path, bwt_router([port(commercial="150", passenger="20")])
    ).packets
    # max(50 + 85 * 0.5, 50 + 60 * 0.5) + 10
    assert packet.anomaly_score == 100.0
    assert packet.public_question.startswith("What explains the gap")


def test_queue_only_lead_asks_what_is_holding_traffic(tmp_path: Path) -> None:
    row = port("250602", commercial="120", passenger="", name="Otay Mesa")
    row["crossing_name"] = "Commercial"
    [packet] = harvest_travel(tmp_path, bwt_router([row])).packets
    assert packet.anomaly_score == 65.0
    assert packet.public_question == (
        "What is holding up traffic at the Otay Mesa (Commercial) port of entry?"
    )
    assert "a not posted passenger wait" in packet.public_interest_reason


def test_calm_ports_make_no_lead(tmp_path: Path) -> None:
    mock = bwt_router([port(commercial="20", passenger="15")])
    result = harvest_travel(tmp_path, mock)
    assert result.packets == [] and result.errors == []
    assert [str(item.url) for item in mock.requests] == [BWT_URL]


def test_waits_that_changed_before_sealing_are_rejected(tmp_path: Path) -> None:
    changed = port(commercial="75", passenger="10", update="At 11:15 am PST")
    result = harvest_travel(tmp_path, bwt_router([port()], sealed=changed))
    assert result.packets == []
    assert result.evidence_rejections == ["cbp:300401:10/15/2026"]


def test_a_queue_that_cleared_before_sealing_is_rejected(tmp_path: Path) -> None:
    cleared = port(commercial="10", passenger="10")
    result = harvest_travel(tmp_path, bwt_router([port()], sealed=cleared))
    assert result.packets == [] and len(result.evidence_rejections) == 1


def test_sealed_record_for_another_port_is_rejected(tmp_path: Path) -> None:
    result = harvest_travel(tmp_path, bwt_router([port()], sealed=port("380001")))
    assert result.packets == [] and len(result.evidence_rejections) == 1


def test_a_poll_that_is_not_an_array_makes_no_records(tmp_path: Path) -> None:
    mock = Router({BWT_URL: {"error": "maintenance"}})
    result = harvest_travel(tmp_path, mock)
    assert result.packets == [] and result.errors == []


def test_record_cannot_carry_its_own_public_text(tmp_path: Path) -> None:
    adapter = travel(tmp_path, bwt_router([]))
    reading = parse_crossing(port())
    question, reason = adapter.public_interest_for(
        {"reading": reading, "public_question": "Who is to blame?"}
    )
    assert "blame" not in question and reason.startswith("CBP posted")


def test_one_lead_per_crossing_per_day(tmp_path: Path) -> None:
    adapter = travel(tmp_path, bwt_router([]))
    early = {"reading": parse_crossing(port())}
    later = {"reading": parse_crossing(port(update="At 3:00 pm PST", commercial="99"))}
    assert adapter.lead_id_for(early) == adapter.lead_id_for(later)


# --- USGS streamflow ----------------------------------------------------------

START = NOW - timedelta(days=30)


def calm_rows(step_minutes: int = 15, end: datetime = NOW) -> list[dict[str, Any]]:
    """A daily cycle with a deterministic wobble: a steady, noisy baseline."""
    rows: list[dict[str, Any]] = []
    moment = START
    index = 0
    while moment < end:
        hours = (moment - START) / timedelta(hours=1)
        value = 1000 + 40 * math.sin(2 * math.pi * hours / 24) + 7 * ((index * 37) % 11)
        rows.append(
            {
                "time": moment.isoformat(),
                "value": f"{value:.0f}",
                "monitoring_location_id": "USGS-14210000",
                "parameter_code": "00060",
                "unit_of_measure": "ft^3/s",
                "approval_status": "Provisional",
            }
        )
        moment += timedelta(minutes=step_minutes)
        index += 1
    return rows


def surging_rows(rise_per_step: float = 400.0, **kwargs: Any) -> list[dict[str, Any]]:
    """Calm, then four quarter-hours climbing fast at the end."""
    rows = calm_rows(**kwargs)
    quarter = [
        row for row in rows if datetime.fromisoformat(row["time"]).minute % 15 == 0
    ]
    base = float(quarter[-5]["value"])
    for offset, row in enumerate(quarter[-4:], start=1):
        row["value"] = f"{base + rise_per_step * offset:.0f}"
    return rows


def collection(rows: list[dict[str, Any]], location: str) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {**row, "monitoring_location_id": location},
                "geometry": None,
            }
            for row in rows
        ],
        "links": [{"rel": "self", "href": CONTINUOUS_URL}],
    }


Series = Callable[[str, int], list[dict[str, Any]]]


def usgs_router(series: Series) -> Router:
    calls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(urlsplit(str(request.url)).query)
        location = query["monitoring_location_id"][0]
        calls[location] = calls.get(location, 0) + 1
        return httpx.Response(
            200, json=collection(series(location, calls[location]), location)
        )

    return Router({CONTINUOUS_URL: handler})


def surge_at(target: str, **kwargs: Any) -> Series:
    return lambda location, _call: (
        surging_rows(**kwargs) if location == target else calm_rows()
    )


def hydro(tmp_path: Path, mock: Router) -> NatureHydroAdapter:
    return NatureHydroAdapter(**options(tmp_path, mock))


def harvest_hydro(tmp_path: Path, mock: Router) -> Any:
    return hydro(tmp_path, mock).harvest(counter())


def test_calm_series_is_not_a_surge() -> None:
    stats = surge_stats(calm_rows())
    assert stats is not None and not stats.surge
    assert stats.baseline_steps >= MIN_BASELINE_STEPS


def test_fast_rise_is_a_surge_over_three_sigma() -> None:
    stats = surge_stats(surging_rows())
    assert stats is not None and stats.surge
    assert stats.sigma > SIGMA_THRESHOLD
    assert stats.peak_rate > 1000  # ft^3/s per hour
    assert stats.approval_status == "Provisional"


def test_a_fall_is_never_a_surge() -> None:
    stats = surge_stats(surging_rows(rise_per_step=-400.0))
    assert stats is not None and not stats.surge


def test_five_minute_gauges_are_compared_on_the_quarter_hour() -> None:
    quarter_hours = surging_rows()
    between = [
        {
            **row,
            "time": (
                datetime.fromisoformat(row["time"]) + timedelta(minutes=offset)
            ).isoformat(),
            "value": str(int(row["value"]) + 9000),  # off-grid readings are ignored
        }
        for row in quarter_hours
        for offset in (5, 10)
    ]
    fifteen = surge_stats(quarter_hours)
    five = surge_stats([*quarter_hours, *between])
    assert fifteen is not None and five is not None
    assert five.sigma == pytest.approx(fifteen.sigma)
    assert five.peak_at == fifteen.peak_at


def test_a_gap_breaks_the_series_rather_than_bridging_it() -> None:
    cutoff = NOW - timedelta(hours=3)
    kept = [row for row in calm_rows() if datetime.fromisoformat(row["time"]) < cutoff]
    late = {**kept[-1], "time": (NOW - timedelta(minutes=15)).isoformat()}
    kept.append({**late, "value": "5000"})
    # The only recent reading follows a gap, so there is no recent step to score.
    assert surge_stats(kept) is None


def test_thin_or_flat_baselines_give_no_score() -> None:
    assert surge_stats(calm_rows()[-200:]) is None
    flat = [{**row, "value": "900"} for row in calm_rows()]
    assert surge_stats(flat) is None
    assert surge_stats([]) is None


@pytest.mark.parametrize(
    "bad",
    [{"time": "yesterday", "value": "1"}, {"time": NOW.isoformat(), "value": "x"}],
)
def test_unreadable_rows_are_skipped(bad: dict[str, Any]) -> None:
    stats = surge_stats([*calm_rows(), bad])
    assert stats is not None and not stats.surge


def test_surge_lead_is_sealed_from_the_same_usgs_query(tmp_path: Path) -> None:
    mock = usgs_router(surge_at("USGS-14210000"))
    result = harvest_hydro(tmp_path, mock)
    assert result.errors == [] and result.evidence_rejections == []
    [packet] = result.packets
    assert packet.adapter_name == "nature_hydro"
    [artifact] = packet.source_artifacts
    assert artifact.tier == "tier_1_primary"
    query = parse_qs(urlsplit(artifact.origin_url).query)
    assert query["monitoring_location_id"] == ["USGS-14210000"]
    assert query["parameter_code"] == ["00060"]
    assert query["time"] == ["2026-09-15T12:00:00Z/2026-10-15T12:00:00Z"]
    polls = [str(item.url) for item in mock.requests]
    assert len(polls) == len(GAUGES) + 1
    assert artifact.origin_url in polls[:-1]  # the poll's own query, sealed
    receipt = packet.receipt_details
    assert receipt["basin"] == "Clackamas" and receipt["regulated"] is False
    assert receipt["sigma"] > SIGMA_THRESHOLD
    assert packet.public_question == (
        "What drove the sudden rise in flow on the Clackamas River at Estacada, "
        "OR, and who downstream could be affected?"
    )
    assert packet.public_interest_reason.startswith(
        "Provisional USGS readings show discharge on the Clackamas River at "
        "Estacada, OR rising 1,600 cubic feet per second per hour"
    )
    assert packet.timestamp == datetime.fromisoformat(receipt["peak_at"])
    assert 50.0 < packet.anomaly_score <= 100.0


def test_calm_rivers_make_no_lead(tmp_path: Path) -> None:
    mock = usgs_router(lambda _location, _call: calm_rows())
    result = harvest_hydro(tmp_path, mock)
    assert result.packets == [] and result.errors == []
    assert len(mock.requests) == len(GAUGES)  # nothing sealed


def test_regulated_columbia_surge_is_marked_in_the_prompt(tmp_path: Path) -> None:
    [packet] = harvest_hydro(tmp_path, usgs_router(surge_at("USGS-14105700"))).packets
    assert packet.receipt_details["regulated"] is True
    assert "dam-regulated" in packet.paste_ready_prompt


def test_readings_revised_before_sealing_are_rejected(tmp_path: Path) -> None:
    def series(location: str, call: int) -> list[dict[str, Any]]:
        surging = location == "USGS-14191000" and call == 1
        return surging_rows() if surging else calm_rows()

    result = harvest_hydro(tmp_path, usgs_router(series))
    assert result.packets == []
    assert len(result.evidence_rejections) == 1


def test_sealed_series_for_another_gauge_is_rejected(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(urlsplit(str(request.url)).query)
        location = query["monitoring_location_id"][0]
        rows = surging_rows() if location == "USGS-14191000" else calm_rows()
        payload = collection(rows, location)
        if handler.calls.get(location):  # type: ignore[attr-defined]
            payload = collection(rows, "USGS-99999999")
        handler.calls[location] = 1  # type: ignore[attr-defined]
        return httpx.Response(200, json=payload)

    handler.calls = {}  # type: ignore[attr-defined]
    result = harvest_hydro(tmp_path, Router({CONTINUOUS_URL: handler}))
    assert result.packets == [] and len(result.evidence_rejections) == 1


def test_next_page_link_is_followed(tmp_path: Path) -> None:
    rows = surging_rows()
    half = len(rows) // 2
    next_href = f"{CONTINUOUS_URL}?cursor=abc&monitoring_location_id=USGS-14166000"

    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(urlsplit(str(request.url)).query)
        location = query["monitoring_location_id"][0]
        if location != "USGS-14166000":
            return httpx.Response(200, json=collection(calm_rows(), location))
        if "cursor" in query:
            return httpx.Response(200, json=collection(rows[half:], location))
        first = collection(rows[:half], location)
        first["links"].append({"rel": "next", "href": next_href})
        return httpx.Response(200, json=first)

    adapter = hydro(tmp_path, Router({CONTINUOUS_URL: handler}))
    records = adapter.poll_sources()
    assert [record["gauge"].site for record in records] == ["14166000"]


def test_next_link_to_another_host_is_not_followed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(urlsplit(str(request.url)).query)
        payload = collection(calm_rows(), query["monitoring_location_id"][0])
        payload["links"].append({"rel": "next", "href": "https://evil.example/x"})
        return httpx.Response(200, json=payload)

    adapter = hydro(tmp_path, Router({CONTINUOUS_URL: handler}))
    assert adapter.poll_sources() == []


def test_hydro_record_cannot_carry_its_own_public_text(tmp_path: Path) -> None:
    adapter = hydro(tmp_path, usgs_router(lambda _l, _c: calm_rows()))
    stats = surge_stats(surging_rows())
    record = {"gauge": GAUGES[0], "stats": stats, "public_question": "Is it doom?"}
    question, _ = adapter.public_interest_for(record)
    assert "doom" not in question
