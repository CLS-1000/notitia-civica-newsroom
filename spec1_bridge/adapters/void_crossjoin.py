"""Void cross-join: what people say about a place just after a physical shock.

A shock is a USGS earthquake of M 4.0 or more, a cluster of NASA FIRMS thermal
detections, or a Cloudflare Radar network outage. Each shock opens a 6-hour window
that listens for its placenames on Bluesky and Wikimedia, the period before official
information has caught up.

Shocks come from instruments. The posts collected against them are not evidence of
anything: a window records that a place was being discussed, and it does not sort
rumor from accurate first-hand reports.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from psycopg import Connection

from spec1_bridge.sociology_telemetry import (
    DOMAIN_FAKE_NEWS,
    DecompositionReceipt,
    StreamPost,
    write_observation,
)

DETECTOR: Final = "void_crossjoin"
USGS_FEED_URL: Final = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
)
CLOUDFLARE_OUTAGES_URL: Final = (
    "https://api.cloudflare.com/client/v4/radar/annotations/outages"
)
MIN_MAGNITUDE: Final = 4.0
LISTEN_WINDOW: Final = timedelta(hours=6)
MIN_MENTIONS_TO_RECORD: Final = 3
MAX_SAMPLE_POSTS: Final = 5

FIRMS_MIN_DETECTIONS: Final = 3
FIRMS_CLUSTER_KM: Final = 10.0
FIRMS_PLACE_RADIUS_KM: Final = 50.0
_FIRMS_LOW_CONFIDENCE: Final = {"l", "low"}
_MODIS_MIN_CONFIDENCE: Final = 30
_EARTH_RADIUS_KM: Final = 6371.0

_USGS_OFFSET: Final = re.compile(
    r"^\d+(?:\.\d+)?\s*km\s+[NSEW]{1,3}\s+of\s+", re.IGNORECASE
)
_USGS_PREFIX: Final = re.compile(
    r"^(?:off the coast of|near the coast of)\s+", re.IGNORECASE
)

# Replace with a full gazetteer in production. These cover the Oregon beat.
OREGON_PLACES: Final[tuple[tuple[str, float, float], ...]] = (
    ("Portland", 45.52, -122.68),
    ("Salem", 44.94, -123.04),
    ("Eugene", 44.05, -123.09),
    ("Bend", 44.06, -121.31),
    ("Medford", 42.33, -122.87),
    ("Astoria", 46.19, -123.83),
    ("Pendleton", 45.67, -118.79),
    ("Klamath Falls", 42.22, -121.78),
    ("Coos Bay", 43.37, -124.22),
    ("Burns", 43.59, -119.05),
)


@dataclass(frozen=True, slots=True)
class PhysicalShock:
    shock_id: str
    kind: str  # 'earthquake', 'thermal', or 'outage'
    at: datetime
    placenames: tuple[str, ...]
    receipt: DecompositionReceipt


def placenames_from_usgs(place: str) -> tuple[str, ...]:
    """Return the names to listen for in a USGS place string.

    The town comes first. A state or country alone would match far too much, so it
    is used only when USGS gives no town.
    """
    cleaned = _USGS_PREFIX.sub("", _USGS_OFFSET.sub("", place.strip()))
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    return tuple(parts[:1])


def shocks_from_usgs(payload: dict[str, Any]) -> list[PhysicalShock]:
    """Return earthquakes of M 4.0 or more from a USGS GeoJSON feed."""
    shocks: list[PhysicalShock] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        magnitude = properties.get("mag")
        if properties.get("type") != "earthquake" or magnitude is None:
            continue
        if magnitude < MIN_MAGNITUDE:
            continue
        place = str(properties.get("place") or "")
        at = datetime.fromtimestamp(properties["time"] / 1000, UTC)
        shocks.append(
            PhysicalShock(
                f"usgs:{feature['id']}",
                "earthquake",
                at,
                placenames_from_usgs(place),
                DecompositionReceipt(
                    detector=DETECTOR,
                    finding=f"USGS reported an M {magnitude} earthquake: {place}.",
                    reasons=(
                        f"M {magnitude} meets the threshold of M {MIN_MAGNITUDE}.",
                        "USGS classifies the event as an earthquake.",
                    ),
                    caveat=(
                        "Magnitude does not measure damage. Automatic USGS "
                        "solutions can be revised."
                    ),
                    metrics={"magnitude": magnitude, "url": properties.get("url")},
                ),
            )
        )
    return shocks


def distance_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Return the great-circle distance between two points."""
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    half_lat = math.radians(lat_b - lat_a) / 2
    half_lon = math.radians(lon_b - lon_a) / 2
    chord = (
        math.sin(half_lat) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(half_lon) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(chord))


def nearest_place(
    lat: float,
    lon: float,
    gazetteer: Iterable[tuple[str, float, float]] = OREGON_PLACES,
) -> str | None:
    """Return the closest gazetteer name within range, or None."""
    ranked = sorted(
        (distance_km(lat, lon, place_lat, place_lon), name)
        for name, place_lat, place_lon in gazetteer
    )
    if ranked and ranked[0][0] <= FIRMS_PLACE_RADIUS_KM:
        return ranked[0][1]
    return None


def _firms_row_is_usable(row: dict[str, str]) -> bool:
    confidence = row.get("confidence", "").strip().lower()
    if confidence.isdigit():
        return int(confidence) >= _MODIS_MIN_CONFIDENCE
    return confidence not in _FIRMS_LOW_CONFIDENCE


def shocks_from_firms(
    csv_text: str,
    gazetteer: Iterable[tuple[str, float, float]] = OREGON_PLACES,
) -> list[PhysicalShock]:
    """Return clusters of thermal detections from a FIRMS area CSV.

    One detection is often a flare stack or a sun glint, so a shock needs several
    within FIRMS_CLUSTER_KM of each other.
    """
    rows = [
        row
        for row in csv.DictReader(io.StringIO(csv_text))
        if _firms_row_is_usable(row)
    ]
    places = tuple(gazetteer)
    clusters: list[list[dict[str, str]]] = []
    for row in rows:
        lat, lon = float(row["latitude"]), float(row["longitude"])
        for cluster in clusters:
            anchor = cluster[0]
            gap = distance_km(
                lat, lon, float(anchor["latitude"]), float(anchor["longitude"])
            )
            if gap <= FIRMS_CLUSTER_KM:
                cluster.append(row)
                break
        else:
            clusters.append([row])

    shocks: list[PhysicalShock] = []
    for cluster in clusters:
        if len(cluster) < FIRMS_MIN_DETECTIONS:
            continue
        anchor = cluster[0]
        lat, lon = float(anchor["latitude"]), float(anchor["longitude"])
        times = [
            datetime.strptime(
                f"{row['acq_date']} {row['acq_time'].zfill(4)}", "%Y-%m-%d %H%M"
            ).replace(tzinfo=UTC)
            for row in cluster
        ]
        place = nearest_place(lat, lon, places)
        shocks.append(
            PhysicalShock(
                f"firms:{lat:.1f},{lon:.1f}:{anchor['acq_date']}",
                "thermal",
                min(times),
                (place,) if place else (),
                DecompositionReceipt(
                    detector=DETECTOR,
                    finding=f"{len(cluster)} FIRMS detections cluster near "
                    f"{place or f'{lat:.2f}, {lon:.2f}'}.",
                    reasons=(
                        (
                            f"{len(cluster)} detections fall within "
                            f"{FIRMS_CLUSTER_KM:g} km, and {FIRMS_MIN_DETECTIONS} "
                            "are required."
                        ),
                        "Low-confidence detections were left out.",
                    ),
                    caveat=(
                        "A thermal anomaly is not necessarily a wildfire. It can "
                        "be a prescribed burn or an industrial heat source."
                    ),
                    metrics={"detections": len(cluster), "lat": lat, "lon": lon},
                ),
            )
        )
    return shocks


def shocks_from_cloudflare(payload: dict[str, Any]) -> list[PhysicalShock]:
    """Return outages from a Cloudflare Radar annotations response."""
    shocks: list[PhysicalShock] = []
    for item in payload.get("result", {}).get("annotations", []):
        names = tuple(
            str(location["name"])
            for location in item.get("locationsDetails", [])
            if location.get("name")
        )
        networks = ", ".join(
            f"AS{detail['asn']} {detail.get('name', '')}".strip()
            for detail in item.get("asnsDetails", [])
        )
        outage = item.get("outage", {})
        cause = outage.get("outageCause", "unknown cause")
        at = datetime.fromisoformat(str(item["startDate"]))
        shocks.append(
            PhysicalShock(
                f"cloudflare:{item['id']}",
                "outage",
                at,
                names,
                DecompositionReceipt(
                    detector=DETECTOR,
                    finding=f"Cloudflare Radar reported an outage: "
                    f"{networks or 'network unknown'}.",
                    reasons=(
                        f"Radar lists it with cause: {cause}.",
                        f"Affected locations: {', '.join(names) or 'none listed'}.",
                    ),
                    caveat=(
                        "Radar sees traffic to Cloudflare, so an outage there may "
                        "not reach every network or person in that location."
                    ),
                    metrics={"asns": item.get("asns", []), "cause": cause},
                ),
            )
        )
    return shocks


@dataclass(slots=True)
class ListenerWindow:
    shock: PhysicalShock
    closes_at: datetime
    _pattern: re.Pattern[str]
    posts: list[StreamPost] = field(default_factory=list)

    def matches(self, post: StreamPost) -> bool:
        return (
            self.shock.at <= post.at < self.closes_at
            and self._pattern.search(post.text) is not None
        )


@dataclass(frozen=True, slots=True)
class WindowReport:
    shock: PhysicalShock
    posts: tuple[StreamPost, ...]
    receipt: DecompositionReceipt


class VoidListener:
    """Opens one window per shock and collects posts that name its places.

    The clock is injected so that a replayed stream behaves like a live one.
    """

    def __init__(
        self, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        self._clock = clock
        self._windows: dict[str, ListenerWindow] = {}
        self._seen: set[str] = set()

    @property
    def open_windows(self) -> list[ListenerWindow]:
        return list(self._windows.values())

    def trigger(self, shock: PhysicalShock) -> ListenerWindow | None:
        """Open a window, unless the shock is a repeat, stale, or has no place."""
        closes_at = shock.at + LISTEN_WINDOW
        if (
            shock.shock_id in self._seen
            or not shock.placenames
            or closes_at <= self._clock()
        ):
            return None
        self._seen.add(shock.shock_id)
        names = "|".join(re.escape(name) for name in shock.placenames)
        pattern = re.compile(rf"(?<!\w)(?:{names})(?!\w)", re.IGNORECASE)
        window = ListenerWindow(shock, closes_at, pattern)
        self._windows[shock.shock_id] = window
        return window

    def ingest(self, post: StreamPost) -> None:
        for window in self._windows.values():
            if window.matches(post):
                window.posts.append(post)

    def close_expired(self) -> list[WindowReport]:
        """Close windows past their end, and report those with enough mentions."""
        now = self._clock()
        reports: list[WindowReport] = []
        for shock_id, window in list(self._windows.items()):
            if window.closes_at > now:
                continue
            del self._windows[shock_id]
            if len(window.posts) >= MIN_MENTIONS_TO_RECORD:
                reports.append(_report(window))
        return reports


def _report(window: ListenerWindow) -> WindowReport:
    shock = window.shock
    places = ", ".join(shock.placenames)
    platforms = sorted({post.platform for post in window.posts})
    delay = (window.posts[0].at - shock.at).total_seconds() / 60
    return WindowReport(
        shock,
        tuple(window.posts),
        DecompositionReceipt(
            detector=DETECTOR,
            finding=f"{len(window.posts)} public posts named {places} in the 6 hours "
            f"after a {shock.kind}.",
            reasons=(
                shock.receipt.finding,
                f"The first mention came {delay:.0f} minutes after the shock.",
                f"Mentions came from: {', '.join(platforms)}.",
            ),
            caveat=(
                "These are unverified public statements. Some may be accurate "
                "first-hand accounts and some may be unrelated uses of the name, so "
                "this count does not measure rumor. Nothing here is a verified fact."
            ),
            metrics={"mentions": len(window.posts), "shock_id": shock.shock_id},
        ),
    )


def write_report(conn: Connection[Any], report: WindowReport) -> str:
    """Quarantine a closed window."""
    sample = report.posts[:MAX_SAMPLE_POSTS]
    return write_observation(
        conn,
        domain=DOMAIN_FAKE_NEWS,
        narrative_label=f"information_void:{report.shock.shock_id}",
        observed_text="\n".join(post.text for post in sample),
        receipt=report.receipt,
        observed_at=report.posts[0].at,
    )
