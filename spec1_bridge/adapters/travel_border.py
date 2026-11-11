"""Travel and border: freight and passenger queues at major land ports of entry.

Signal: at a major crossing, either

- commercial (freight) and passenger standard-lane waits diverge by more than 45
  minutes, or
- either queue is longer than 90 minutes.

A major crossing is one CBP lists with at least three commercial lanes. That is
a capacity screen, not an official designation: it keeps the busy freight ports
(Detroit, Laredo, Otay Mesa, Blaine Pacific Highway) and leaves out crossings
whose one commercial lane cannot show a freight-versus-passenger pattern.

Source: U.S. CBP Border Wait Times (keyless). The poll reads every port from
https://bwt.cbp.gov/api/bwtnew, a JSON array the base adapter's object-only
pager cannot read, so this adapter reads it itself. The evidence is the
crossing's own record (https://bwt.cbp.gov/api/bwtnew/{port}), sealed as
tier_1_primary, and the evidence bar re-reads that sealed record: it must be the
same crossing, posting the same waits at the same update time, and still meet
the signal. CBP refreshes waits through the day, so a crossing whose waits
changed between poll and seal is rejected and caught again on a later poll.

Posted waits are CBP's estimates, entered by port staff or by automated sensors
(the record says which). They are a lead to check, not a measured delay.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from spec1_bridge.adapters.base_lazlong import (
    EvidenceTarget,
    LazLongAdapter,
    PollError,
    PollRequest,
    clamp_score,
    host_of,
    read_artifact_text,
    render_prompt,
)
from spec1_bridge.contracts import SourceArtifact

BWT_URL: Final = "https://bwt.cbp.gov/api/bwtnew"
BWT_HOST: Final = "bwt.cbp.gov"
KIND_CROSSING: Final = "cbp_bwt_crossing"

MAJOR_MIN_COMMERCIAL_LANES: Final = 3
DIVERGENCE_MINUTES: Final = 45
QUEUE_MINUTES: Final = 90

BASE_POINTS: Final = 50.0
POINTS_PER_MINUTE_OVER: Final = 0.5
BOTH_SIGNALS_POINTS: Final = 10.0

# CBP writes update times as "At 11:00 am PST", in the port's own zone.
_UPDATE_TIME: Final = re.compile(
    r"^At\s+(\d{1,2}):(\d{2})\s*(am|pm)\s+([A-Z]{3,4})$", re.IGNORECASE
)
_ZONE_OFFSETS: Final = {
    "AST": -4,
    "EST": -5,
    "EDT": -4,
    "CST": -6,
    "CDT": -5,
    "MST": -7,
    "MDT": -6,
    "PST": -8,
    "PDT": -7,
    "AKST": -9,
    "AKDT": -8,
}


@dataclass(frozen=True, slots=True)
class CrossingReading:
    """One crossing's posted standard-lane waits, as CBP published them."""

    port_number: str
    border: str
    port_name: str
    crossing_name: str
    date: str
    commercial_lanes: int | None
    commercial_delay: int | None
    commercial_update: str
    commercial_automation: str
    passenger_delay: int | None
    passenger_update: str
    passenger_automation: str

    @property
    def label(self) -> str:
        if self.crossing_name:
            return f"{self.port_name} ({self.crossing_name})"
        return self.port_name

    @property
    def is_major(self) -> bool:
        return (self.commercial_lanes or 0) >= MAJOR_MIN_COMMERCIAL_LANES

    @property
    def divergence(self) -> int | None:
        if self.commercial_delay is None or self.passenger_delay is None:
            return None
        return abs(self.commercial_delay - self.passenger_delay)

    @property
    def longest_queue(self) -> int | None:
        delays = [
            item
            for item in (self.commercial_delay, self.passenger_delay)
            if item is not None
        ]
        return max(delays) if delays else None

    @property
    def diverges(self) -> bool:
        return (self.divergence or 0) > DIVERGENCE_MINUTES

    @property
    def queue_exceeded(self) -> bool:
        return (self.longest_queue or 0) > QUEUE_MINUTES

    @property
    def flagged(self) -> bool:
        return self.is_major and (self.diverges or self.queue_exceeded)

    def observed_at(self) -> datetime:
        """The latest posted update time, as an aware datetime."""
        times = [
            moment
            for moment in (
                parse_update_time(self.date, self.commercial_update),
                parse_update_time(self.date, self.passenger_update),
            )
            if moment is not None
        ]
        if not times:
            raise ValueError(f"port {self.port_number} has no usable update time")
        return max(times)


def _int_or_none(value: Any) -> int | None:
    text = str(value if value is not None else "").strip()
    return int(text) if text.isdigit() else None


def _lanes(row: dict[str, Any], group: str) -> dict[str, Any]:
    block = row.get(group)
    return block if isinstance(block, dict) else {}


def _standard(row: dict[str, Any], group: str) -> dict[str, Any]:
    block = _lanes(row, group).get("standard_lanes")
    return block if isinstance(block, dict) else {}


def parse_update_time(day: str, update: str) -> datetime | None:
    """Combine CBP's MM/DD/YYYY date with an "At 11:00 am PST" update time."""
    match = _UPDATE_TIME.match(update.strip())
    if match is None:
        return None
    hour, minute, meridiem, zone = match.groups()
    offset = _ZONE_OFFSETS.get(zone.upper())
    if offset is None:
        return None
    try:
        local = datetime.strptime(day.strip(), "%m/%d/%Y")
    except ValueError:
        return None
    hours = int(hour) % 12 + (12 if meridiem.lower() == "pm" else 0)
    if int(minute) > 59:
        return None
    moment = local.replace(
        hour=hours, minute=int(minute), tzinfo=timezone(timedelta(hours=offset))
    )
    return moment.astimezone(UTC)


def parse_crossing(row: Any) -> CrossingReading | None:
    """Read one port record, or None if it is not a usable crossing record."""
    if not isinstance(row, dict):
        return None
    port_number = str(row.get("port_number") or "").strip()
    if not port_number:
        return None
    commercial = _standard(row, "commercial_vehicle_lanes")
    passenger = _standard(row, "passenger_vehicle_lanes")
    return CrossingReading(
        port_number=port_number,
        border=str(row.get("border") or "").strip(),
        port_name=str(row.get("port_name") or "").strip(),
        crossing_name=str(row.get("crossing_name") or "").strip(),
        date=str(row.get("date") or "").strip(),
        commercial_lanes=_int_or_none(
            _lanes(row, "commercial_vehicle_lanes").get("maximum_lanes")
        ),
        commercial_delay=_int_or_none(commercial.get("delay_minutes")),
        commercial_update=str(commercial.get("update_time") or "").strip(),
        commercial_automation=str(
            _lanes(row, "commercial_vehicle_lanes").get("cv_automation_type") or ""
        ).strip(),
        passenger_delay=_int_or_none(passenger.get("delay_minutes")),
        passenger_update=str(passenger.get("update_time") or "").strip(),
        passenger_automation=str(
            _lanes(row, "passenger_vehicle_lanes").get("pv_automation_type") or ""
        ).strip(),
    )


def _waits(reading: CrossingReading) -> dict[str, Any]:
    """The posted facts the evidence bar must find again in the sealed record."""
    return {
        "port_number": reading.port_number,
        "commercial_delay": reading.commercial_delay,
        "commercial_update": reading.commercial_update,
        "passenger_delay": reading.passenger_delay,
        "passenger_update": reading.passenger_update,
    }


def sealed_crossing(artifact: SourceArtifact) -> CrossingReading | None:
    """Read the crossing back from its sealed CBP record."""
    try:
        raw = Path(artifact.raw_payload_path).read_bytes()
    except OSError:
        return None
    text = read_artifact_text(raw, artifact.metadata.get("content_type"))
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    return parse_crossing(payload)


def _minutes(value: int | None) -> str:
    return "not posted" if value is None else f"{value} min"


class TravelBorderAdapter(LazLongAdapter):
    adapter_name = "travel_border"
    subject_domain = "border_crossings"

    # --- Polling ------------------------------------------------------------

    def poll_requests(self) -> list[PollRequest]:
        return [PollRequest("ports", BWT_URL)]

    def _fetch_all_pages(self, request: PollRequest) -> list[dict[str, Any]]:
        """bwtnew answers with a JSON array, which the base pager refuses."""
        response = self._get(request.url, request.params)
        try:
            payload = response.json()
        except ValueError as error:
            raise PollError(f"invalid JSON: {error}", retryable=True) from error
        if not isinstance(payload, list):
            raise PollError("response is not a JSON array", retryable=False)
        return self.extract_items(request, {"ports": payload})

    def extract_items(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rows = payload.get("ports", [])
        return (
            [row for row in rows if isinstance(row, dict)]
            if isinstance(rows, list)
            else []
        )

    def assemble_records(
        self, items: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for row in items.get("ports", []):
            reading = parse_crossing(row)
            if reading is not None and reading.flagged:
                records[reading.port_number] = {"reading": reading}
        return [records[key] for key in sorted(records)]

    # --- Lead definition ----------------------------------------------------

    @staticmethod
    def _reading(record: dict[str, Any]) -> CrossingReading:
        reading = record.get("reading")
        if isinstance(reading, CrossingReading):
            return reading
        # ValueError, not TypeError: harvest() records it as an error for retry.
        raise ValueError("record holds no CBP crossing reading")

    def record_key(self, raw_record: dict[str, Any]) -> str:
        """One lead per crossing per posted day, however long the queue lasts."""
        reading = self._reading(raw_record)
        return f"cbp:{reading.port_number}:{reading.date}"

    def record_timestamp(self, raw_record: dict[str, Any]) -> datetime:
        return self._reading(raw_record).observed_at()

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        reading = self._reading(raw_record)
        if not reading.flagged:
            return 0.0
        parts: list[float] = []
        if reading.diverges and reading.divergence is not None:
            over = reading.divergence - DIVERGENCE_MINUTES
            parts.append(BASE_POINTS + over * POINTS_PER_MINUTE_OVER)
        if reading.queue_exceeded and reading.longest_queue is not None:
            over = reading.longest_queue - QUEUE_MINUTES
            parts.append(BASE_POINTS + over * POINTS_PER_MINUTE_OVER)
        score = max(parts)
        if len(parts) == 2:
            score += BOTH_SIGNALS_POINTS
        return clamp_score(score)

    def evidence_targets(self, raw_record: dict[str, Any]) -> list[EvidenceTarget]:
        reading = self._reading(raw_record)
        return [
            EvidenceTarget(
                url=f"{BWT_URL}/{reading.port_number}",
                tier="tier_1_primary",
                kind=KIND_CROSSING,
                metadata=_waits(reading),
            )
        ]

    def verify_evidence_bar(self, artifacts: list[SourceArtifact]) -> bool:
        """One sealed CBP crossing record that still shows what was polled."""
        if len(artifacts) != 1:
            return False
        [artifact] = artifacts
        if (
            artifact.tier != "tier_1_primary"
            or host_of(artifact.origin_url) != BWT_HOST
            or artifact.metadata.get("kind") != KIND_CROSSING
        ):
            return False
        sealed = sealed_crossing(artifact)
        if sealed is None or not sealed.flagged:
            return False
        expected = {key: artifact.metadata.get(key) for key in _waits(sealed)}
        return _waits(sealed) == expected

    # --- Public text --------------------------------------------------------

    def public_interest_for(self, record: dict[str, Any]) -> tuple[str, str]:
        """Only this adapter's wording, never text carried in on a record."""
        return (
            self.default_public_question(record),
            self.default_public_interest_reason(record),
        )

    def default_public_question(self, record: dict[str, Any]) -> str:
        reading = self._reading(record)
        if reading.diverges:
            return (
                "What explains the gap between freight and passenger waits at the "
                f"{reading.label} port of entry?"
            )
        return f"What is holding up traffic at the {reading.label} port of entry?"

    def default_public_interest_reason(self, record: dict[str, Any]) -> str:
        reading = self._reading(record)
        return (
            f"CBP posted a {_minutes(reading.commercial_delay)} commercial wait and "
            f"a {_minutes(reading.passenger_delay)} passenger wait at "
            f"{reading.label} on {reading.date}; long queues at a major crossing "
            "delay goods, workers, and travelers who depend on it."
        )

    def receipt_details(self, record: dict[str, Any]) -> dict[str, Any]:
        reading = self._reading(record)
        return {
            "detector": self.adapter_name,
            **asdict(reading),
            "divergence_minutes": reading.divergence,
            "longest_queue_minutes": reading.longest_queue,
            "diverges": reading.diverges,
            "queue_exceeded": reading.queue_exceeded,
            "thresholds": {
                "divergence_minutes": DIVERGENCE_MINUTES,
                "queue_minutes": QUEUE_MINUTES,
                "major_min_commercial_lanes": MAJOR_MIN_COMMERCIAL_LANES,
            },
        }

    def build_paste_ready_prompt(
        self, record: dict[str, Any], artifacts: list[SourceArtifact]
    ) -> str:
        reading = self._reading(record)
        return render_prompt(
            f"Border queue signal: {reading.label}, {reading.border}",
            {
                "Port number": reading.port_number,
                "Commercial standard lanes": (
                    f"{_minutes(reading.commercial_delay)} "
                    f"({reading.commercial_update or 'no update time'}; "
                    f"{reading.commercial_automation or 'entry method not stated'})"
                ),
                "Passenger standard lanes": (
                    f"{_minutes(reading.passenger_delay)} "
                    f"({reading.passenger_update or 'no update time'}; "
                    f"{reading.passenger_automation or 'entry method not stated'})"
                ),
                "Freight/passenger divergence": _minutes(reading.divergence),
                "Commercial lanes at the crossing": reading.commercial_lanes,
                "Thresholds": (
                    f"divergence over {DIVERGENCE_MINUTES} min or a queue over "
                    f"{QUEUE_MINUTES} min"
                ),
            },
            artifacts,
            "CBP's posted waits are estimates. Check whether the queue persisted "
            "across later updates, whether lanes were closed or staffing changed, "
            "and what freight and commuters it held up, before treating it as news.",
        )
