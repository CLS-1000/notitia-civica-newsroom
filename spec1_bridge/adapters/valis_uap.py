"""Valis: subtract known objects from UAP sighting reports.

For each NUFORC report this checks what was in the sky at the witness's time and
place: satellites from CelesTrak elements, computed with skyfield, and the Moon and
planets from pyswisseph. What no known object explains is the residual.

A residual means only that this check found no match. It is not evidence of an
unknown craft, and a match shows a known object was there, not what the witness
saw. Witnesses are treated as reporting honestly what they experienced. Reports are
stored in quarantine and are never proof of a physical event.

A satellite is nearly always above any horizon, so 'one was overhead' explains
nothing. A pass needs the satellite sunlit, the witness's sky dark, and the
direction the witness gave. With no direction, the most it can earn is
POSSIBLE_ORBITAL_PASS. Reports without a coordinate or a time zone are not checked,
because a guessed time zone would move every object in the sky.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final

import numpy as np
import swisseph as swe
from numpy.typing import NDArray
from psycopg import Connection
from skyfield.api import EarthSatellite, load, wgs84

from spec1_bridge.adapters.void_crossjoin import distance_km
from spec1_bridge.sociology_telemetry import (
    DOMAIN_IRRATIONAL,
    DecompositionReceipt,
    write_observation,
)

DETECTOR: Final = "valis_uap"
CELESTRAK_GP_URL: Final = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"
)
CELESTRAK_SUPGP_URL: Final = "https://celestrak.org/NORAD/elements/supplemental/"

IDENTIFIED_ORBITAL_PASS: Final = "IDENTIFIED_ORBITAL_PASS"
IDENTIFIED_CELESTIAL_BODY: Final = "IDENTIFIED_CELESTIAL_BODY"
POSSIBLE_ORBITAL_PASS: Final = "POSSIBLE_ORBITAL_PASS"
UNCATALOGED_RESIDUAL: Final = "UNCATALOGED_RESIDUAL"
INSUFFICIENT_CATALOG: Final = "INSUFFICIENT_CATALOG"
UNRESOLVED_REPORT: Final = "UNRESOLVED_REPORT"

TIME_WINDOW: Final = timedelta(minutes=10)
TIME_STEP_SECONDS: Final = 30
ANGLE_TOLERANCE_DEG: Final = 15.0
MIN_SATELLITE_ELEVATION_DEG: Final = 10.0
MAX_SATELLITE_SUN_ALTITUDE_DEG: Final = -4.0
MAX_ELEMENT_AGE: Final = timedelta(days=14)
NOTAM_MARGIN_KM: Final = 25.0
_KM_PER_NM: Final = 1.852
_KM_PER_AU: Final = 149_597_870.7
_EARTH_RADIUS_KM: Final = 6378.137
_J2000_JD: Final = 2451545.0
_UNIX_J2000: Final = 946728000.0
_SECONDS_PER_DAY: Final = 86400.0

_BODIES: Final = {
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
}
_PROMOTION: Final = re.compile(
    r"https?://|\b(?:my|our) (?:channel|book|podcast|course)\b|\bsubscribe\b|"
    r"\bbuy now\b|\bdonate\b",
    re.IGNORECASE,
)
_CAVEAT: Final = (
    "A match shows a known object was in that direction, not what the witness saw. "
    "No match does not show an unknown craft. This is an unverified account."
)

_timescale: Final = load.timescale(builtin=True)


@dataclass(frozen=True, slots=True)
class WitnessReport:
    report_id: str
    occurred_at: datetime | None
    lat: float | None
    lon: float | None
    summary: str
    azimuth_deg: float | None = None
    elevation_deg: float | None = None


@dataclass(frozen=True, slots=True)
class Notam:
    notam_id: str
    lat: float
    lon: float
    radius_nm: float
    starts_at: datetime
    ends_at: datetime
    text: str


@dataclass(frozen=True, slots=True)
class UapResult:
    report: WitnessReport
    tag: str
    matched: tuple[str, ...]
    notams: tuple[str, ...]
    source_conduct: str  # 'witness_account' or 'promotional'
    receipt: DecompositionReceipt


def _number(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def parse_nuforc_record(record: dict[str, Any]) -> WitnessReport:
    """Normalize a NUFORC record. A time with no offset is kept out of the checks."""
    occurred: datetime | None = None
    stamp = record.get("occurred")
    if isinstance(stamp, str):
        try:
            parsed = datetime.fromisoformat(stamp)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.utcoffset() is not None:
            occurred = parsed
    return WitnessReport(
        report_id=str(record["id"]),
        occurred_at=occurred,
        lat=_number(record.get("latitude")),
        lon=_number(record.get("longitude")),
        summary=str(record.get("summary") or ""),
        azimuth_deg=_number(record.get("azimuth")),
        elevation_deg=_number(record.get("elevation")),
    )


def load_tle_text(text: str) -> list[EarthSatellite]:
    """Parse three-line or two-line element sets."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    satellites: list[EarthSatellite] = []
    for index, line in enumerate(lines):
        if line.startswith("1 ") and lines[index + 1 :][:1]:
            name = lines[index - 1] if index and lines[index - 1][:2] != "2 " else None
            satellites.append(EarthSatellite(line, lines[index + 1], name, _timescale))
    return satellites


def load_omm(records: Iterable[dict[str, Any]]) -> list[EarthSatellite]:
    """Parse CelesTrak GP or SupGP records in JSON OMM form."""
    return [EarthSatellite.from_omm(_timescale, record) for record in records]


def _julian_day(moment: datetime) -> float:
    return _J2000_JD + (moment.timestamp() - _UNIX_J2000) / _SECONDS_PER_DAY


def _sky_position(
    moment: datetime, lat: float, lon: float, body: int
) -> tuple[float, float]:
    """Return a body's azimuth, from north, and its altitude."""
    jd = _julian_day(moment)
    position = swe.calc_ut(jd, body, swe.FLG_MOSEPH)[0]
    azimuth, altitude, _ = swe.azalt(
        jd, swe.ECL2HOR, (lon, lat, 0.0), 0, 0, position[:3]
    )
    # Swiss Ephemeris counts azimuth from south.
    return (azimuth + 180.0) % 360.0, altitude


def _sun_vector_km(moment: datetime) -> NDArray[np.float64]:
    flags = swe.FLG_MOSEPH | swe.FLG_EQUATORIAL | swe.FLG_XYZ | swe.FLG_J2000
    xyz = swe.calc_ut(_julian_day(moment), swe.SUN, flags)[0][:3]
    return np.array(xyz, dtype=np.float64) * _KM_PER_AU


def angular_separation(
    az_a: Any, alt_a: Any, az_b: float, alt_b: float
) -> NDArray[np.float64]:
    """Return the angle between two sky directions, in degrees."""
    alt1, alt2 = np.radians(alt_a), np.radians(alt_b)
    cosine = np.sin(alt1) * np.sin(alt2) + np.cos(alt1) * np.cos(alt2) * np.cos(
        np.radians(np.asarray(az_a) - az_b)
    )
    return np.asarray(
        np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), dtype=np.float64
    )


def _azimuth_gap(azimuths: Any, azimuth: float) -> NDArray[np.float64]:
    return np.abs((np.asarray(azimuths) - azimuth + 180.0) % 360.0 - 180.0)


def satellite_track(
    satellite: EarthSatellite, lat: float, lon: float, moments: list[datetime]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Return azimuths, elevations, and whether the satellite is sunlit."""
    times = _timescale.from_datetimes(moments)
    site = wgs84.latlon(lat, lon)
    altitude, azimuth, _ = (satellite - site).at(times).altaz()

    # Cylindrical shadow: dark only behind the Earth and inside its radius.
    positions = satellite.at(times).position.km.T
    sun = _sun_vector_km(moments[len(moments) // 2])
    sun_unit = sun / np.linalg.norm(sun)
    along = positions @ sun_unit
    offset = np.linalg.norm(positions - np.outer(along, sun_unit), axis=1)
    sunlit = (along > 0) | (offset > _EARTH_RADIUS_KM)
    return azimuth.degrees, altitude.degrees, sunlit


def _is_current(satellite: EarthSatellite, moment: datetime) -> bool:
    return bool(abs(satellite.epoch.utc_datetime() - moment) <= MAX_ELEMENT_AGE)


def _direction_mask(
    report: WitnessReport, azimuths: Any, elevations: Any
) -> NDArray[np.bool_]:
    assert report.azimuth_deg is not None
    if report.elevation_deg is None:
        return _azimuth_gap(azimuths, report.azimuth_deg) <= ANGLE_TOLERANCE_DEG
    separation = angular_separation(
        azimuths, elevations, report.azimuth_deg, report.elevation_deg
    )
    return separation <= ANGLE_TOLERANCE_DEG


def _sample_moments(moment: datetime) -> list[datetime]:
    span = int(TIME_WINDOW.total_seconds())
    return [
        moment + timedelta(seconds=offset)
        for offset in range(-span, span + 1, TIME_STEP_SECONDS)
    ]


def _orbital_matches(
    report: WitnessReport, satellites: list[EarthSatellite]
) -> tuple[list[str], list[str]]:
    """Return satellites matching the direction, and those merely visible."""
    assert report.occurred_at and report.lat is not None and report.lon is not None
    _, sun_altitude = _sky_position(report.occurred_at, report.lat, report.lon, swe.SUN)
    if sun_altitude > MAX_SATELLITE_SUN_ALTITUDE_DEG:
        return [], []

    moments = _sample_moments(report.occurred_at)
    identified: list[str] = []
    visible: list[str] = []
    for satellite in satellites:
        azimuths, elevations, sunlit = satellite_track(
            satellite, report.lat, report.lon, moments
        )
        seen = sunlit & (elevations >= MIN_SATELLITE_ELEVATION_DEG)
        if not seen.any():
            continue
        name = satellite.name or str(satellite.model.satnum)
        visible.append(name)
        if (
            report.azimuth_deg is not None
            and (seen & _direction_mask(report, azimuths, elevations)).any()
        ):
            identified.append(name)
    return identified, visible


def _body_matches(report: WitnessReport) -> list[str]:
    assert report.occurred_at and report.lat is not None and report.lon is not None
    if report.azimuth_deg is None:
        return []
    moment = report.occurred_at
    if _sky_position(moment, report.lat, report.lon, swe.SUN)[1] > 0:
        return []
    matches: list[str] = []
    for name, body in _BODIES.items():
        azimuth, altitude = _sky_position(moment, report.lat, report.lon, body)
        if altitude > 0 and _direction_mask(report, azimuth, altitude).all():
            matches.append(name)
    return matches


def _active_notams(report: WitnessReport, notams: Iterable[Notam]) -> tuple[str, ...]:
    assert report.occurred_at and report.lat is not None and report.lon is not None
    return tuple(
        notam.notam_id
        for notam in notams
        if notam.starts_at <= report.occurred_at <= notam.ends_at
        and distance_km(report.lat, report.lon, notam.lat, notam.lon)
        <= notam.radius_nm * _KM_PER_NM + NOTAM_MARGIN_KM
    )


def source_conduct(summary: str) -> str:
    """Label a report that promotes something. This does not judge sincerity."""
    return "promotional" if _PROMOTION.search(summary) else "witness_account"


def classify_report(
    report: WitnessReport,
    satellites: Iterable[EarthSatellite],
    notams: Iterable[Notam] = (),
) -> UapResult:
    """Check one report against known objects and active airspace notices."""
    conduct = source_conduct(report.summary)
    if report.occurred_at is None or report.lat is None or report.lon is None:
        return _result(
            report,
            UNRESOLVED_REPORT,
            (),
            (),
            conduct,
            ("The report has no usable coordinate or time with a time zone.",),
        )

    current = [item for item in satellites if _is_current(item, report.occurred_at)]
    active = _active_notams(report, notams)
    bodies = _body_matches(report)
    identified, visible = _orbital_matches(report, current)
    has_direction = report.azimuth_deg is not None

    if identified:
        return _result(
            report,
            IDENTIFIED_ORBITAL_PASS,
            tuple(identified),
            active,
            conduct,
            (
                (
                    f"{', '.join(identified)} passed within {ANGLE_TOLERANCE_DEG:g} "
                    "degrees of the reported direction."
                ),
                "It was sunlit, and the sky at the witness's location was dark.",
            ),
        )
    if bodies:
        return _result(
            report,
            IDENTIFIED_CELESTIAL_BODY,
            tuple(bodies),
            active,
            conduct,
            (
                (
                    f"{', '.join(bodies)} stood within {ANGLE_TOLERANCE_DEG:g} degrees "
                    "of the reported direction."
                ),
            ),
        )
    if visible and not has_direction:
        return _result(
            report,
            POSSIBLE_ORBITAL_PASS,
            tuple(visible),
            active,
            conduct,
            (
                (
                    f"{len(visible)} sunlit satellites were above "
                    f"{MIN_SATELLITE_ELEVATION_DEG:g} degrees."
                ),
                "The report gives no direction, so none can be identified.",
            ),
        )
    if not current:
        return _result(
            report,
            INSUFFICIENT_CATALOG,
            (),
            active,
            conduct,
            (
                (
                    f"No satellite elements fall within {MAX_ELEMENT_AGE.days} days of "
                    "the report, so satellites were not ruled out."
                ),
            ),
        )
    return _result(
        report,
        UNCATALOGED_RESIDUAL,
        (),
        active,
        conduct,
        (
            (
                f"{len(current)} satellites and {len(_BODIES)} bright bodies were "
                "checked, and none matched."
            ),
            f"Active airspace notices nearby: {', '.join(active) or 'none'}.",
        ),
    )


def _result(
    report: WitnessReport,
    tag: str,
    matched: tuple[str, ...],
    notams: tuple[str, ...],
    conduct: str,
    reasons: tuple[str, ...],
) -> UapResult:
    return UapResult(
        report,
        tag,
        matched,
        notams,
        conduct,
        DecompositionReceipt(
            detector=DETECTOR,
            finding=f"Report {report.report_id} was tagged {tag}.",
            reasons=reasons,
            caveat=_CAVEAT,
            metrics={
                "tag": tag,
                "matched": list(matched),
                "notams": list(notams),
                "source_conduct": conduct,
            },
        ),
    )


def write_residual(conn: Connection[Any], result: UapResult) -> str:
    """Quarantine a report no known object explains."""
    if result.tag != UNCATALOGED_RESIDUAL or result.report.occurred_at is None:
        raise ValueError("only an uncataloged residual is written to quarantine")
    return write_observation(
        conn,
        domain=DOMAIN_IRRATIONAL,
        narrative_label=f"uap_residual:{result.report.report_id}",
        observed_text=result.report.summary,
        receipt=result.receipt,
        observed_at=result.report.occurred_at,
    )
