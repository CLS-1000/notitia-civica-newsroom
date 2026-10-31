"""Delos: drug supply bottlenecks and shipping chokepoint drops.

A bottleneck is a Class I or II recall of a generic drug that is in shortage with
one company on the shortage list. That list names firms reporting a shortage, not
every approved manufacturer, so 'single source' is a lead to check, not a finding.

A chokepoint drop compares mean daily transits over the latest 7 days with the 30
days before them, which keeps weekly traffic patterns out of the comparison.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final
from urllib.parse import quote

from spec1_bridge.adapters.base_lazlong import (
    EvidenceTarget,
    LazLongAdapter,
    PollRequest,
    clamp_score,
    host_of,
    render_prompt,
)
from spec1_bridge.contracts import SourceArtifact

ENFORCEMENT_URL: Final = "https://api.fda.gov/drug/enforcement.json"
SHORTAGES_URL: Final = "https://api.fda.gov/drug/shortages.json"
PORTWATCH_URL: Final = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)
PORTWATCH_HOST: Final = "services9.arcgis.com"
FDA_HOST: Final = "api.fda.gov"

FDA_PAGE_SIZE: Final = 1000
PORTWATCH_PAGE_SIZE: Final = 1000
RECALL_LOOKBACK_DAYS: Final = 90
TRANSIT_LOOKBACK_DAYS: Final = 60
RECENT_WINDOW_DAYS: Final = 7
BASELINE_WINDOW_DAYS: Final = 30
DIVERGENCE_THRESHOLD: Final = 0.15

CLASS_POINTS: Final = {"Class I": 60.0, "Class II": 40.0}
SINGLE_SOURCE_POINTS: Final = 30.0
UNAVAILABLE_POINTS: Final = 10.0
CHOKEPOINT_BASE_POINTS: Final = 40.0
CHOKEPOINT_POINTS_PER_PERCENT: Final = 2.0

_DOSAGE_WORDS: Final = frozenset(
    {
        "capsule",
        "capsules",
        "for",
        "injectable",
        "injection",
        "oral",
        "solution",
        "suspension",
        "tablet",
        "tablets",
        "usp",
    }
)


def normalize_generic(name: str) -> str:
    """Reduce a drug name to its active ingredient words."""
    words = re.findall(r"[a-z]+", name.lower())
    return " ".join(word for word in words if word not in _DOSAGE_WORDS)


def _openfda_query_url(base: str, field_name: str, value: str) -> str:
    query = quote(f'{field_name}:"{value}"', safe=":")
    return f"{base}?search={query}"


def _fda_date(value: Any) -> datetime | None:
    try:
        return datetime.strptime(str(value), "%Y%m%d").replace(tzinfo=UTC)
    except ValueError:
        return None


def transit_divergence(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compare recent transits with their baseline; None if data is too thin."""
    daily: dict[str, int] = {}
    for row in rows:
        if row.get("date") and row.get("n_total") is not None:
            daily[str(row["date"])[:10]] = int(row["n_total"])

    days = sorted(daily)
    if len(days) < RECENT_WINDOW_DAYS + BASELINE_WINDOW_DAYS:
        return None
    recent_days = days[-RECENT_WINDOW_DAYS:]
    baseline_days = days[
        -(RECENT_WINDOW_DAYS + BASELINE_WINDOW_DAYS) : -RECENT_WINDOW_DAYS
    ]
    recent = sum(daily[day] for day in recent_days) / RECENT_WINDOW_DAYS
    baseline = sum(daily[day] for day in baseline_days) / BASELINE_WINDOW_DAYS
    if baseline <= 0:
        return None
    return {
        "recent_mean": recent,
        "baseline_mean": baseline,
        "divergence": (recent - baseline) / baseline,
        "latest_date": recent_days[-1],
    }


class DelosEconAdapter(LazLongAdapter):
    adapter_name = "delos_econ"
    subject_domain = "supply_chain"

    def poll_requests(self) -> list[PollRequest]:
        today = self._clock().date()
        recall_start = (today - timedelta(days=RECALL_LOOKBACK_DAYS)).strftime("%Y%m%d")
        transit_start = (today - timedelta(days=TRANSIT_LOOKBACK_DAYS)).isoformat()
        return [
            PollRequest(
                "enforcement",
                ENFORCEMENT_URL,
                {
                    "search": (
                        'classification:("Class I" OR "Class II") AND status:"Ongoing" '
                        f"AND report_date:[{recall_start} TO {today:%Y%m%d}]"
                    ),
                    "limit": str(FDA_PAGE_SIZE),
                },
            ),
            PollRequest(
                "shortages",
                SHORTAGES_URL,
                {"search": 'status:"Current"', "limit": str(FDA_PAGE_SIZE)},
            ),
            PollRequest(
                "transits",
                PORTWATCH_URL,
                {
                    "where": f"date >= DATE '{transit_start}'",
                    "outFields": "date,portid,portname,n_total",
                    "orderByFields": "ObjectId",
                    "resultRecordCount": str(PORTWATCH_PAGE_SIZE),
                    "resultOffset": "0",
                    "f": "json",
                },
            ),
        ]

    def extract_items(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if request.name == "transits":
            return [
                row["attributes"]
                for row in payload.get("features", [])
                if isinstance(row.get("attributes"), dict)
            ]
        rows = payload.get("results", [])
        return rows if isinstance(rows, list) else []

    def next_request(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> PollRequest | None:
        if request.name == "transits":
            if not payload.get("exceededTransferLimit"):
                return None
            offset = int(request.params["resultOffset"]) + PORTWATCH_PAGE_SIZE
            return PollRequest(
                request.name,
                request.url,
                {**request.params, "resultOffset": str(offset)},
            )

        results = payload.get("meta", {}).get("results", {})
        next_skip = int(results.get("skip", 0)) + FDA_PAGE_SIZE
        if next_skip >= int(results.get("total", 0)):
            return None
        return PollRequest(
            request.name, request.url, {**request.params, "skip": str(next_skip)}
        )

    @staticmethod
    def _bottlenecks(items: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        shortages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in items.get("shortages", []):
            generic = normalize_generic(str(row.get("generic_name", "")))
            if generic:
                shortages[generic].append(row)

        records: list[dict[str, Any]] = []
        recalls = {
            str(row.get("recall_number")): row for row in items.get("enforcement", [])
        }
        for recall_number, recall in recalls.items():
            description = " ".join(
                re.findall(
                    r"[a-z]+", str(recall.get("product_description", "")).lower()
                )
            )
            for generic, rows in shortages.items():
                companies = {str(row.get("company_name")) for row in rows}
                if len(companies) == 1 and f" {generic} " in f" {description} ":
                    records.append(
                        {
                            "kind": "fda_bottleneck",
                            "recall_number": recall_number,
                            "recall": recall,
                            "generic": generic,
                            "shortages": rows,
                            "company": next(iter(companies)),
                        }
                    )
        return records

    @staticmethod
    def _chokepoint_drops(
        items: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        ports: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in items.get("transits", []):
            ports[str(row.get("portid"))].append(row)

        records: list[dict[str, Any]] = []
        for port_id, rows in ports.items():
            stats = transit_divergence(rows)
            if stats is not None and stats["divergence"] < -DIVERGENCE_THRESHOLD:
                records.append(
                    {
                        "kind": "chokepoint_drop",
                        "port_id": port_id,
                        "port_name": rows[0].get("portname"),
                        **stats,
                    }
                )
        return records

    def assemble_records(
        self, items: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        return [*self._bottlenecks(items), *self._chokepoint_drops(items)]

    def record_key(self, raw_record: dict[str, Any]) -> str:
        if raw_record["kind"] == "fda_bottleneck":
            return f"fda:{raw_record['recall_number']}:{raw_record['generic']}"
        year, week, _ = date.fromisoformat(raw_record["latest_date"]).isocalendar()
        return f"portwatch:{raw_record['port_id']}:{year}-W{week:02d}"

    def record_timestamp(self, raw_record: dict[str, Any]) -> datetime:
        if raw_record["kind"] == "chokepoint_drop":
            return datetime.fromisoformat(raw_record["latest_date"]).replace(tzinfo=UTC)
        timestamp = _fda_date(raw_record["recall"].get("report_date"))
        if timestamp is None:
            raise ValueError(f"{raw_record['recall_number']} has no usable date")
        return timestamp

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        if raw_record["kind"] == "chokepoint_drop":
            percent = -raw_record["divergence"] * 100
            over_threshold = percent - DIVERGENCE_THRESHOLD * 100
            return clamp_score(
                CHOKEPOINT_BASE_POINTS + over_threshold * CHOKEPOINT_POINTS_PER_PERCENT
            )
        score = CLASS_POINTS.get(str(raw_record["recall"].get("classification")), 0.0)
        if score == 0.0:
            return 0.0
        score += SINGLE_SOURCE_POINTS
        availability = {
            str(row.get("availability", "")).lower() for row in raw_record["shortages"]
        }
        if "unavailable" in availability:
            score += UNAVAILABLE_POINTS
        return clamp_score(score)

    def evidence_targets(self, raw_record: dict[str, Any]) -> list[EvidenceTarget]:
        if raw_record["kind"] == "chokepoint_drop":
            return [
                EvidenceTarget(
                    url=(
                        f"{PORTWATCH_URL}?where="
                        + quote(f"portid = '{raw_record['port_id']}'", safe="")
                        + "&outFields=date,portid,portname,n_total"
                        "&orderByFields=date+DESC&resultRecordCount=37&f=json"
                    ),
                    tier="tier_1_primary",
                    kind="portwatch_transits",
                    metadata={"port_id": raw_record["port_id"]},
                )
            ]
        return [
            EvidenceTarget(
                url=_openfda_query_url(
                    ENFORCEMENT_URL, "recall_number", raw_record["recall_number"]
                ),
                tier="tier_1_primary",
                kind="fda_enforcement",
                excerpt=raw_record["recall"].get("reason_for_recall"),
            ),
            EvidenceTarget(
                url=_openfda_query_url(
                    SHORTAGES_URL,
                    "generic_name",
                    str(raw_record["shortages"][0]["generic_name"]),
                ),
                tier="tier_1_primary",
                kind="fda_shortage",
                excerpt=raw_record["shortages"][0].get("presentation"),
            ),
        ]

    def verify_evidence_bar(self, artifacts: list[SourceArtifact]) -> bool:
        kinds = {
            item.metadata.get("kind")
            for item in artifacts
            if item.tier == "tier_1_primary"
            and host_of(item.origin_url) in {FDA_HOST, PORTWATCH_HOST}
        }
        return {"fda_enforcement", "fda_shortage"} <= kinds or (
            "portwatch_transits" in kinds
        )

    def build_paste_ready_prompt(
        self, record: dict[str, Any], artifacts: list[SourceArtifact]
    ) -> str:
        if record["kind"] == "chokepoint_drop":
            return render_prompt(
                f"Chokepoint transit drop: {record['port_name']}",
                {
                    "Divergence": f"{record['divergence']:.1%}",
                    "Mean daily transits, latest 7 days": f"{record['recent_mean']:.1f}",
                    "Mean daily transits, prior 30 days": (
                        f"{record['baseline_mean']:.1f}"
                    ),
                    "Latest data date": record["latest_date"],
                },
                artifacts,
                "Assess what could explain this drop and what it would disrupt.",
            )
        recall = record["recall"]
        return render_prompt(
            f"Possible single-source bottleneck: {record['generic']}",
            {
                "Recall": f"{record['recall_number']} ({recall.get('classification')})",
                "Recalling firm": recall.get("recalling_firm"),
                "Reason": recall.get("reason_for_recall"),
                "Only company on shortage list": record["company"],
                "Availability": record["shortages"][0].get("availability"),
            },
            artifacts,
            "Check whether other approved manufacturers exist, since the shortage "
            "list names only firms reporting a shortage, then assess patient impact.",
        )
