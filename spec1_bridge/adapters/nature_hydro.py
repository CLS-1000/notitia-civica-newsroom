"""Nature and hydrology: streamflow surges on Pacific Northwest rivers.

Signal: on a gauged river in the Willamette, Clackamas, or Columbia basin, the
rate of change of discharge (dQ/dt) over a 15-minute step rises more than 3.0
standard deviations above that gauge's own 30-day baseline. A rise that fast is
the signature of a flash flood or a sudden release, so it is a lead to check,
not a flood finding.

Method, per gauge:

1. Read 30 days of discharge (parameter 00060, ft^3/s). Some gauges report every
   5 minutes, others every 15; only readings on the quarter hour are kept, so
   every gauge is compared on the same 15-minute step.
2. dQ/dt for each pair of consecutive quarter-hour readings, in ft^3/s per hour.
   A gap in the record breaks the series rather than averaging across it.
3. The baseline is every step older than the latest 2 hours; the recent window
   is the latest 2 hours. The signal is the largest recent rise, scored as
   (rise - baseline mean) / baseline standard deviation.

Source: USGS Water Data OGC API v1, continuous collection (keyless). The poll
asks for a fixed time interval, so the sealed copy of the same URL holds the
same readings, and the evidence bar recomputes the statistic from that sealed
copy: it must show the same peak, at the same time, still above 3.0 sigma.
USGS can revise provisional readings, so a recomputation that disagrees is
rejected rather than trusted.

The Columbia at The Dalles is regulated by dams upstream; a surge there may be
an operational release, which the prompt says.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Final
from urllib.parse import urlencode

from spec1_bridge.adapters.base_lazlong import (
    EvidenceTarget,
    LazLongAdapter,
    PollRequest,
    clamp_score,
    host_of,
    read_artifact_text,
    render_prompt,
)
from spec1_bridge.contracts import SourceArtifact

CONTINUOUS_URL: Final = (
    "https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items"
)
USGS_HOST: Final = "api.waterdata.usgs.gov"
KIND_SERIES: Final = "usgs_continuous_discharge"
DISCHARGE: Final = "00060"
PAGE_LIMIT: Final = 10000
PROPERTIES: Final = (
    "time,value,monitoring_location_id,parameter_code,unit_of_measure,approval_status"
)

BASELINE_DAYS: Final = 30
STEP: Final = timedelta(minutes=15)
RECENT_WINDOW: Final = timedelta(hours=2)
# Half of a 30-day baseline, so a gauge that was down for weeks is not scored.
MIN_BASELINE_STEPS: Final = 1440
SIGMA_THRESHOLD: Final = 3.0

BASE_POINTS: Final = 50.0
POINTS_PER_SIGMA_OVER: Final = 10.0


@dataclass(frozen=True, slots=True)
class Gauge:
    site: str
    name: str
    basin: str
    regulated: bool = False

    @property
    def location_id(self) -> str:
        return f"USGS-{self.site}"


# Names as USGS lists them in its monitoring-locations collection.
GAUGES: Final = (
    Gauge("14166000", "Willamette River at Harrisburg, OR", "Willamette"),
    Gauge("14174000", "Willamette River at Albany, OR", "Willamette"),
    Gauge("14191000", "Willamette River at Salem, OR", "Willamette"),
    Gauge("14210000", "Clackamas River at Estacada, OR", "Clackamas"),
    Gauge("14211010", "Clackamas River near Oregon City, OR", "Clackamas"),
    Gauge("14105700", "Columbia River at The Dalles, OR", "Columbia", regulated=True),
)
_BY_LOCATION: Final = {gauge.location_id: gauge for gauge in GAUGES}


@dataclass(frozen=True, slots=True)
class SurgeStats:
    peak_rate: float
    peak_at: datetime
    baseline_mean: float
    baseline_std: float
    sigma: float
    baseline_steps: int
    latest_discharge: float
    latest_at: datetime
    approval_status: str

    @property
    def surge(self) -> bool:
        return self.peak_rate > 0 and self.sigma > SIGMA_THRESHOLD

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_rate_cfs_per_hour": self.peak_rate,
            "peak_at": self.peak_at.isoformat(),
            "baseline_mean_cfs_per_hour": self.baseline_mean,
            "baseline_std_cfs_per_hour": self.baseline_std,
            "sigma": self.sigma,
            "baseline_steps": self.baseline_steps,
            "latest_discharge_cfs": self.latest_discharge,
            "latest_at": self.latest_at.isoformat(),
            "approval_status": self.approval_status,
        }


def _reading(row: Any) -> tuple[datetime, float, str] | None:
    if not isinstance(row, dict):
        return None
    try:
        moment = datetime.fromisoformat(str(row.get("time")))
        value = float(str(row.get("value")))
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None or not math.isfinite(value) or value < 0:
        return None
    return moment.astimezone(UTC), value, str(row.get("approval_status") or "")


def surge_stats(rows: list[dict[str, Any]]) -> SurgeStats | None:
    """Score a gauge's latest rise against its baseline; None if data is thin."""
    readings: dict[datetime, tuple[float, str]] = {}
    for row in rows:
        parsed = _reading(row)
        if parsed is None:
            continue
        moment, value, status = parsed
        if moment.minute % 15 == 0 and moment.second == 0 and moment.microsecond == 0:
            readings[moment] = (value, status)
    times = sorted(readings)
    if len(times) < 2:
        return None

    steps: list[tuple[datetime, float]] = []
    for before, after in pairwise(times):
        if after - before == STEP:
            rate = (readings[after][0] - readings[before][0]) / (
                STEP / timedelta(hours=1)
            )
            steps.append((after, rate))
    latest = times[-1]
    cutoff = latest - RECENT_WINDOW
    baseline = [rate for moment, rate in steps if moment <= cutoff]
    recent = [(moment, rate) for moment, rate in steps if moment > cutoff]
    if len(baseline) < MIN_BASELINE_STEPS or not recent:
        return None
    std = pstdev(baseline)
    if std <= 0:
        return None
    mean = fmean(baseline)
    peak_at, peak = max(recent, key=lambda item: (item[1], item[0]))
    return SurgeStats(
        peak_rate=peak,
        peak_at=peak_at,
        baseline_mean=mean,
        baseline_std=std,
        sigma=(peak - mean) / std,
        baseline_steps=len(baseline),
        latest_discharge=readings[latest][0],
        latest_at=latest,
        approval_status=readings[peak_at][1],
    )


def series_rows(payload: Any, location_id: str) -> list[dict[str, Any]] | None:
    """The discharge rows for one gauge in an OGC FeatureCollection."""
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), list):
        return None
    rows: list[dict[str, Any]] = []
    for feature in payload["features"]:
        properties = feature.get("properties") if isinstance(feature, dict) else None
        if not isinstance(properties, dict):
            continue
        if properties.get("monitoring_location_id", location_id) != location_id:
            return None
        if properties.get("parameter_code", DISCHARGE) != DISCHARGE:
            return None
        rows.append(properties)
    return rows


def sealed_stats(artifact: SourceArtifact) -> SurgeStats | None:
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
    rows = series_rows(payload, str(artifact.metadata.get("location_id")))
    return None if rows is None else surge_stats(rows)


def _interval(start: datetime, end: datetime) -> str:
    return f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}"


class NatureHydroAdapter(LazLongAdapter):
    adapter_name = "nature_hydro"
    subject_domain = "hydrology"

    # --- Polling ------------------------------------------------------------

    def _params(self, gauge: Gauge) -> dict[str, str]:
        # Whole minutes, so the interval and its sealed copy are reproducible.
        end = self._clock().astimezone(UTC).replace(second=0, microsecond=0)
        start = end - timedelta(days=BASELINE_DAYS)
        return {
            "f": "json",
            "monitoring_location_id": gauge.location_id,
            "parameter_code": DISCHARGE,
            "time": _interval(start, end),
            "limit": str(PAGE_LIMIT),
            "skipGeometry": "true",
            "properties": PROPERTIES,
        }

    def poll_requests(self) -> list[PollRequest]:
        return [
            PollRequest(gauge.location_id, CONTINUOUS_URL, self._params(gauge))
            for gauge in GAUGES
        ]

    def extract_items(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rows = series_rows(payload, request.name)
        if rows is None:
            return []
        query = request.params
        return [{**row, "_query": query} for row in rows]

    def next_request(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> PollRequest | None:
        links = payload.get("links")
        for link in links if isinstance(links, list) else []:
            if isinstance(link, dict) and link.get("rel") == "next":
                href = str(link.get("href") or "")
                if host_of(href) == USGS_HOST:
                    # The cursor link carries the whole query; keep the first
                    # page's params for sealing.
                    return PollRequest(request.name, href)
        return None

    def assemble_records(
        self, items: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for gauge in GAUGES:
            rows = items.get(gauge.location_id, [])
            queries = [row["_query"] for row in rows if row.get("_query")]
            stats = surge_stats(rows)
            if stats is None or not stats.surge or not queries:
                continue
            records.append({"gauge": gauge, "stats": stats, "query": queries[0]})
        return records

    # --- Lead definition ----------------------------------------------------

    @staticmethod
    def _parts(record: dict[str, Any]) -> tuple[Gauge, SurgeStats]:
        gauge, stats = record.get("gauge"), record.get("stats")
        if isinstance(gauge, Gauge) and isinstance(stats, SurgeStats):
            return gauge, stats
        # ValueError, not TypeError: harvest() records it as an error for retry.
        raise ValueError("record holds no scored USGS gauge")

    def record_key(self, raw_record: dict[str, Any]) -> str:
        """One lead per gauge per UTC day of the peak."""
        gauge, stats = self._parts(raw_record)
        return f"usgs:{gauge.site}:{stats.peak_at:%Y-%m-%d}"

    def record_timestamp(self, raw_record: dict[str, Any]) -> datetime:
        return self._parts(raw_record)[1].peak_at

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        _, stats = self._parts(raw_record)
        if not stats.surge:
            return 0.0
        over = stats.sigma - SIGMA_THRESHOLD
        return clamp_score(BASE_POINTS + over * POINTS_PER_SIGMA_OVER)

    def evidence_targets(self, raw_record: dict[str, Any]) -> list[EvidenceTarget]:
        gauge, stats = self._parts(raw_record)
        query = raw_record.get("query")
        if not query or not isinstance(query, dict):
            raise ValueError("record has no USGS query to seal")
        return [
            EvidenceTarget(
                url=f"{CONTINUOUS_URL}?{urlencode(query)}",
                tier="tier_1_primary",
                kind=KIND_SERIES,
                metadata={
                    "location_id": gauge.location_id,
                    "peak_at": stats.peak_at.isoformat(),
                    "peak_rate_cfs_per_hour": stats.peak_rate,
                },
            )
        ]

    def verify_evidence_bar(self, artifacts: list[SourceArtifact]) -> bool:
        """One sealed USGS series whose own readings reproduce the surge."""
        if len(artifacts) != 1:
            return False
        [artifact] = artifacts
        if (
            artifact.tier != "tier_1_primary"
            or host_of(artifact.origin_url) != USGS_HOST
            or artifact.metadata.get("kind") != KIND_SERIES
            or artifact.metadata.get("location_id") not in _BY_LOCATION
        ):
            return False
        stats = sealed_stats(artifact)
        if stats is None or not stats.surge:
            return False
        return stats.peak_at.isoformat() == artifact.metadata.get(
            "peak_at"
        ) and math.isclose(
            stats.peak_rate,
            float(artifact.metadata.get("peak_rate_cfs_per_hour", math.nan)),
        )

    # --- Public text --------------------------------------------------------

    def public_interest_for(self, record: dict[str, Any]) -> tuple[str, str]:
        """Only this adapter's wording, never text carried in on a record."""
        return (
            self.default_public_question(record),
            self.default_public_interest_reason(record),
        )

    def default_public_question(self, record: dict[str, Any]) -> str:
        gauge, _ = self._parts(record)
        return (
            f"What drove the sudden rise in flow on the {gauge.name}, and who "
            "downstream could be affected?"
        )

    def default_public_interest_reason(self, record: dict[str, Any]) -> str:
        gauge, stats = self._parts(record)
        status = stats.approval_status.lower() or "unlabelled"
        return (
            f"{status.capitalize()} USGS readings show discharge on the "
            f"{gauge.name} rising {stats.peak_rate:,.0f} cubic feet per second "
            f"per hour, {stats.sigma:.1f} standard deviations above its 30-day "
            "baseline; rises that fast can precede flooding for people and "
            "infrastructure downstream."
        )

    def receipt_details(self, record: dict[str, Any]) -> dict[str, Any]:
        gauge, stats = self._parts(record)
        return {
            "detector": self.adapter_name,
            "location_id": gauge.location_id,
            "gauge": gauge.name,
            "basin": gauge.basin,
            "regulated": gauge.regulated,
            **stats.as_dict(),
            "sigma_threshold": SIGMA_THRESHOLD,
            "step_minutes": int(STEP / timedelta(minutes=1)),
            "baseline_days": BASELINE_DAYS,
        }

    def build_paste_ready_prompt(
        self, record: dict[str, Any], artifacts: list[SourceArtifact]
    ) -> str:
        gauge, stats = self._parts(record)
        facts: dict[str, Any] = {
            "Gauge": f"{gauge.name} ({gauge.location_id}), {gauge.basin} basin",
            "Peak 15-minute rise": (
                f"{stats.peak_rate:,.0f} ft^3/s per hour at {stats.peak_at:%Y-%m-%d %H:%M} UTC"
            ),
            "30-day baseline dQ/dt": (
                f"mean {stats.baseline_mean:,.1f}, sd {stats.baseline_std:,.1f} "
                f"ft^3/s per hour over {stats.baseline_steps} steps"
            ),
            "Sigma above baseline": f"{stats.sigma:.1f} (threshold {SIGMA_THRESHOLD})",
            "Latest discharge": (
                f"{stats.latest_discharge:,.0f} ft^3/s at {stats.latest_at:%Y-%m-%d %H:%M} UTC"
            ),
            "Approval status": stats.approval_status or "not stated",
        }
        if gauge.regulated:
            facts["Regulation"] = (
                "dam-regulated reach; a surge may be an operational release"
            )
        return render_prompt(
            f"Streamflow surge: {gauge.name}",
            facts,
            artifacts,
            "Provisional USGS data can be revised. Check National Weather Service "
            "flood products and dam operators before reporting any flooding, and "
            "say what the readings do not show.",
        )
