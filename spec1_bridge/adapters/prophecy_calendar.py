"""Eschaton: a rolling calendar of dates that tend to attract end-times claims.

Astronomy comes from pyswisseph's built-in Moshier ephemeris, so no data files are
needed. Religious dates are computed from their calendars. Faith observances are
listed because prediction claims attach to them, not because the observances are
suspect: the subject is viral panic, and believers are not its authors.

The score forecasts public attention. It does not predict that anything will
happen, and the weights are editorial judgment, not a fitted model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Final

import swisseph as swe

from spec1_bridge.sociology_telemetry import DecompositionReceipt

DETECTOR: Final = "eschaton_prophecy"
CALENDAR_DAYS: Final = 90
STACK_RADIUS_DAYS: Final = 3
HIGH_SUSCEPTIBILITY: Final = 40
MAX_SCORE: Final = 100

_J2000_JD: Final = 2451545.0
_J2000: Final = datetime(2000, 1, 1, 12, tzinfo=UTC)
_HEBREW_EPOCH_ORDINAL: Final = -1373427
_FLAGS: Final = swe.FLG_MOSEPH | swe.FLG_SPEED

_STATION_PLANETS: Final = {
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
}
_STATION_POINTS: Final = {"Mercury": 5}
_DEFAULT_STATION_POINTS: Final = 10
_SOLAR_TYPES: Final = (
    (swe.ECL_ANNULAR_TOTAL, "hybrid", 40),
    (swe.ECL_TOTAL, "total", 40),
    (swe.ECL_ANNULAR, "annular", 30),
    (swe.ECL_PARTIAL, "partial", 15),
)
_LUNAR_TYPES: Final = (
    (swe.ECL_TOTAL, "total", 30),
    (swe.ECL_PARTIAL, "partial", 15),
    (swe.ECL_PENUMBRAL, "penumbral", 5),
)
_FEAST_POINTS: Final = 15
_ANNIVERSARY_POINTS: Final = 10

# Dates that past end-times movements or claims attached significance to.
ANNIVERSARIES: Final[tuple[tuple[int, int, str], ...]] = (
    (1, 1, "Y2K rollover (2000)"),
    (5, 14, "Proclamation of the State of Israel (1948)"),
    (5, 21, "Harold Camping's predicted Rapture (2011)"),
    (6, 7, "Capture of the Old City of Jerusalem (1967)"),
    (9, 23, "David Meade's predicted Nibiru date (2017)"),
    (10, 21, "Harold Camping's revised end date (2011)"),
    (10, 22, "Millerite Great Disappointment (1844)"),
    (12, 21, "Maya calendar cycle end (2012)"),
)


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    on: date
    kind: str  # 'eclipse', 'station', 'observance', or 'anniversary'
    name: str
    points: int


@dataclass(frozen=True, slots=True)
class SusceptibilityPeak:
    on: date
    score: int
    events: tuple[CalendarEvent, ...]
    receipt: DecompositionReceipt


@dataclass(frozen=True, slots=True)
class ProphecyCalendar:
    starts_on: date
    ends_on: date
    events: tuple[CalendarEvent, ...]
    peaks: tuple[SusceptibilityPeak, ...]


def julian_day(moment: datetime) -> float:
    return _J2000_JD + (moment - _J2000).total_seconds() / 86400.0


def from_julian_day(value: float) -> datetime:
    return _J2000 + timedelta(days=value - _J2000_JD)


def _start_of(day: date) -> float:
    return julian_day(datetime.combine(day, time(), UTC))


def _label(flags: int, table: tuple[tuple[int, str, int], ...]) -> tuple[str, int]:
    for bit, label, points in table:
        if flags & bit:
            return label, points
    return "unclassified", 0


def eclipses_between(starts_on: date, ends_on: date) -> list[CalendarEvent]:
    """Return solar and lunar eclipses visible anywhere on Earth."""
    events: list[CalendarEvent] = []
    end = _start_of(ends_on + timedelta(days=1))
    searches = (
        (swe.sol_eclipse_when_glob, "solar", _SOLAR_TYPES),
        (swe.lun_eclipse_when, "lunar", _LUNAR_TYPES),
    )
    for search, body, table in searches:
        cursor = _start_of(starts_on)
        while True:
            flags, times = search(cursor, swe.FLG_MOSEPH)
            maximum = times[0]
            if maximum >= end:
                break
            label, points = _label(flags, table)
            events.append(
                CalendarEvent(
                    from_julian_day(maximum).date(),
                    "eclipse",
                    f"{label.capitalize()} {body} eclipse",
                    points,
                )
            )
            cursor = maximum + 1.0
    return events


def _longitude_speed(jd: float, planet: int) -> float:
    return float(swe.calc_ut(jd, planet, _FLAGS)[0][3])


def _refine_station(low: float, high: float, planet: int) -> float:
    low_retrograde = _longitude_speed(low, planet) < 0
    for _ in range(30):
        middle = (low + high) / 2
        if (_longitude_speed(middle, planet) < 0) == low_retrograde:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def retrograde_stations(starts_on: date, ends_on: date) -> list[CalendarEvent]:
    """Return days a planet's apparent motion reverses, Mercury through Saturn."""
    events: list[CalendarEvent] = []
    days = (ends_on - starts_on).days + 1
    for name, planet in _STATION_PLANETS.items():
        first = _start_of(starts_on)
        for offset in range(days):
            low, high = first + offset, first + offset + 1
            before = _longitude_speed(low, planet)
            after = _longitude_speed(high, planet)
            if (before < 0) == (after < 0):
                continue
            moment = from_julian_day(_refine_station(low, high, planet))
            motion = "retrograde" if after < 0 else "direct"
            events.append(
                CalendarEvent(
                    moment.date(),
                    "station",
                    f"{name} stations {motion}",
                    _STATION_POINTS.get(name, _DEFAULT_STATION_POINTS),
                )
            )
    return events


def _hebrew_elapsed_days(year: int) -> int:
    months = (235 * year - 234) // 19
    parts = 12084 + 13753 * months
    day = 29 * months + parts // 25920
    return day + 1 if (3 * (day + 1)) % 7 < 3 else day


def rosh_hashanah(gregorian_year: int) -> date:
    """Return 1 Tishri that falls in a Gregorian year."""
    year = gregorian_year + 3761
    previous = _hebrew_elapsed_days(year - 1)
    current = _hebrew_elapsed_days(year)
    following = _hebrew_elapsed_days(year + 1)
    correction = 0
    if following - current == 356:
        correction = 2
    elif current - previous == 382:
        correction = 1
    return date.fromordinal(_HEBREW_EPOCH_ORDINAL + current + correction)


def western_easter(year: int) -> date:
    """Return Easter Sunday by the anonymous Gregorian computus."""
    golden = year % 19
    century, rest = divmod(year, 100)
    leap_century, century_rest = divmod(century, 4)
    correction = (century - (century + 8) // 25 + 1) // 3
    epact = (19 * golden + century - leap_century - correction + 15) % 30
    leap_year, year_rest = divmod(rest, 4)
    weekday = (32 + 2 * century_rest + 2 * leap_year - epact - year_rest) % 7
    shift = (golden + 11 * epact + 22 * weekday) // 451
    month, day = divmod(epact + weekday - 7 * shift + 114, 31)
    return date(year, month, day + 1)


def observances(year: int) -> list[CalendarEvent]:
    """Return the feast and fast days of one Gregorian year."""
    new_year = rosh_hashanah(year)
    easter = western_easter(year)
    fast = new_year - timedelta(days=51)
    if fast.weekday() == 5:  # A fast on the Sabbath moves to Sunday.
        fast += timedelta(days=1)
    dates = {
        "Passover": new_year - timedelta(days=163),
        "Tisha B'Av": fast,
        "Rosh Hashanah": new_year,
        "Yom Kippur": new_year + timedelta(days=9),
        "Sukkot": new_year + timedelta(days=14),
        "Good Friday": easter - timedelta(days=2),
        "Easter": easter,
    }
    return [
        CalendarEvent(day, "observance", name, _FEAST_POINTS)
        for name, day in dates.items()
    ]


def anniversaries(year: int) -> list[CalendarEvent]:
    return [
        CalendarEvent(date(year, month, day), "anniversary", name, _ANNIVERSARY_POINTS)
        for month, day, name in ANNIVERSARIES
    ]


def _peak(day: date, events: list[CalendarEvent]) -> SusceptibilityPeak | None:
    near = tuple(
        item for item in events if abs((item.on - day).days) <= STACK_RADIUS_DAYS
    )
    score = min(MAX_SCORE, sum(item.points for item in near))
    if score < HIGH_SUSCEPTIBILITY:
        return None
    return SusceptibilityPeak(
        day,
        score,
        near,
        DecompositionReceipt(
            detector=DETECTOR,
            finding=f"{day.isoformat()} scores {score} for prophecy attention.",
            reasons=tuple(
                f"{item.name} on {item.on.isoformat()} adds {item.points} points."
                for item in near
            ),
            caveat=(
                "This forecasts when end-times claims may spread. It does not "
                "predict any event, and the point weights are editorial judgment."
            ),
            metrics={"score": score, "stack_radius_days": STACK_RADIUS_DAYS},
        ),
    )


def build_calendar(as_of: date, days: int = CALENDAR_DAYS) -> ProphecyCalendar:
    """Build the rolling calendar that begins on as_of."""
    ends_on = as_of + timedelta(days=days)
    years = range(as_of.year, ends_on.year + 1)
    candidates = [
        *eclipses_between(as_of, ends_on),
        *retrograde_stations(as_of, ends_on),
    ]
    for year in years:
        candidates.extend(observances(year))
        candidates.extend(anniversaries(year))
    events = sorted(
        (item for item in candidates if as_of <= item.on <= ends_on),
        key=lambda item: (item.on, item.name),
    )

    peaks: list[SusceptibilityPeak] = []
    for offset in range(days + 1):
        peak = _peak(as_of + timedelta(days=offset), events)
        if peak is not None:
            peaks.append(peak)
    return ProphecyCalendar(as_of, ends_on, tuple(events), tuple(peaks))
