"""Delos: drug supply bottlenecks, shortage cascades, and shipping chokepoint drops.

A bottleneck is a Class I or II recall of a generic drug that is in shortage with
one company on the shortage list. That list names firms reporting a shortage, not
every approved manufacturer, so 'single source' is a lead to check, not a finding.

A shortage cascade is a Class I or II recall of an active pharmaceutical
ingredient (API) or a sterile injectable whose drug has one or two approved
sources in the FDA NDC Directory, whether or not it is in shortage yet. Sources
are counted as distinct approved applications (ANDA, NDA, BLA) among finished
products with an unexpired listing, so a repackager selling another firm's
product does not count twice. A listing shows a product may be marketed, not that
it is, and the Directory does not say where a product is made: the count is a
screen for an editor to check, not a finding. Cascades score 80 or more and carry
priority HIGH in the lead's prompt.

A chokepoint drop compares mean daily transits over the latest 7 days with the 30
days before them, which keeps weekly traffic patterns out of the comparison.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final
from urllib.parse import quote

from spec1_bridge.adapters.base_adapter import (
    BaseAdapter,
    EvidenceTarget,
    PollError,
    PollRequest,
    clamp_score,
    host_of,
    render_prompt,
)
from spec1_bridge.contracts import SourceArtifact

ENFORCEMENT_URL: Final = "https://api.fda.gov/drug/enforcement.json"
SHORTAGES_URL: Final = "https://api.fda.gov/drug/shortages.json"
NDC_URL: Final = "https://api.fda.gov/drug/ndc.json"
PORTWATCH_URL: Final = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)
PORTWATCH_HOST: Final = "services9.arcgis.com"
FDA_HOST: Final = "api.fda.gov"

FDA_PAGE_SIZE: Final = 1000
PORTWATCH_PAGE_SIZE: Final = 1000
RECALL_LOOKBACK_DAYS: Final = 90
# Newest rows first, so 2000 rows is about 70 days across all 28 chokepoints.
PORTWATCH_MAX_ROWS: Final = 2000
RECENT_WINDOW_DAYS: Final = 7
BASELINE_WINDOW_DAYS: Final = 30
DIVERGENCE_THRESHOLD: Final = 0.15

CLASS_POINTS: Final = {"Class I": 60.0, "Class II": 40.0}
SINGLE_SOURCE_POINTS: Final = 30.0
UNAVAILABLE_POINTS: Final = 10.0
CHOKEPOINT_BASE_POINTS: Final = 40.0
CHOKEPOINT_POINTS_PER_PERCENT: Final = 2.0

# Cascade: a bottleneck generic with at most this many approved sources.
CASCADE_MAX_SOURCES: Final = 2
CASCADE_BASE_POINTS: Final = 80.0
CASCADE_CLASS_I_POINTS: Final = 10.0
CASCADE_SINGLE_SOURCE_POINTS: Final = 5.0
CASCADE_IN_SHORTAGE_POINTS: Final = 5.0
CASCADE_PRIORITY: Final = "HIGH"
APPROVED_CATEGORIES: Final = frozenset({"ANDA", "NDA", "NDA AUTHORIZED GENERIC", "BLA"})

_API_PATTERN: Final = re.compile(
    r"\b(?:API|active pharmaceutical ingredient|bulk drug substance|drug substance)\b",
    re.IGNORECASE,
)
_STERILE_INJECTABLE_PATTERN: Final = re.compile(
    r"\b(?:injections?|injectable|infusion|intravenous|intramuscular|subcutaneous|"
    r"intrathecal|sterile|for injection)\b",
    re.IGNORECASE,
)
_PARENTERAL_ROUTES: Final = frozenset(
    {"INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS", "INTRATHECAL", "INTRAVITREAL"}
)

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


def _openfda_block(recall: dict[str, Any]) -> dict[str, Any]:
    block = recall.get("openfda")
    return block if isinstance(block, dict) else {}


def recall_scope(recall: dict[str, Any]) -> str | None:
    """Return 'api' or 'sterile_injectable' when a recall can start a cascade."""
    description = str(recall.get("product_description", ""))
    if _API_PATTERN.search(description):
        return "api"
    openfda = _openfda_block(recall)
    routes = {str(route).upper() for route in openfda.get("route", []) or []}
    if _STERILE_INJECTABLE_PATTERN.search(description) or routes & _PARENTERAL_ROUTES:
        return "sterile_injectable"
    return None


def recall_generics(recall: dict[str, Any]) -> list[str]:
    """Return the normalized generic names openFDA attaches to a recall."""
    openfda = _openfda_block(recall)
    names = openfda.get("generic_name", []) or []
    found = [normalize_generic(str(name)) for name in names if str(name).strip()]
    return list(dict.fromkeys(name for name in found if name))


def approved_sources(listings: list[dict[str, Any]], today: date) -> dict[str, Any]:
    """Count distinct approved applications among a drug's current NDC listings.

    Repackagers and relabelers list under the application of the firm that makes
    the product, so counting applications rather than labelers keeps one source
    from counting twice. Unapproved categories carry no application to count.
    """
    applications: set[str] = set()
    labelers: set[str] = set()
    for row in listings:
        if row.get("finished") is False:
            continue
        expires = _fda_date(row.get("listing_expiration_date"))
        if expires is not None and expires.date() < today:
            continue
        category = str(row.get("marketing_category", "")).upper()
        application = str(row.get("application_number") or "").strip().upper()
        if category in APPROVED_CATEGORIES and application:
            applications.add(application)
            labelers.add(str(row.get("labeler_name", "")).strip())
    return {
        "applications": sorted(applications),
        "labelers": sorted(name for name in labelers if name),
        "listing_count": len(listings),
    }


def _ndc_search(generic: str) -> str:
    return f'generic_name:"{generic}" AND finished:true'


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


class SupplyChainAdapter(BaseAdapter):
    adapter_name = "delos_econ"
    subject_domain = "supply_chain"

    def poll_requests(self) -> list[PollRequest]:
        today = self._clock().date()
        recall_start = (today - timedelta(days=RECALL_LOOKBACK_DAYS)).strftime("%Y%m%d")
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
                    "where": "1=1",
                    "outFields": "*",
                    "orderByFields": "date DESC",
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
            if offset >= PORTWATCH_MAX_ROWS:
                return None
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
            scope = (
                recall_scope(recall)
                if str(recall.get("classification")) in CLASS_POINTS
                else None
            )
            in_description = [
                generic for generic in shortages if f" {generic} " in f" {description} "
            ]
            # A cascade can start before any shortage is reported, so openFDA's own
            # generic names count as well as names matched to the shortage list.
            generics = dict.fromkeys(
                [*in_description, *(recall_generics(recall) if scope else [])]
            )
            for generic in generics:
                rows = shortages.get(generic, [])
                companies = {str(row.get("company_name")) for row in rows}
                single = len(companies) == 1
                if not single and scope is None:
                    continue
                records.append(
                    {
                        "kind": "fda_bottleneck",
                        "recall_number": recall_number,
                        "recall": recall,
                        "generic": generic,
                        "shortages": rows,
                        "company": next(iter(companies)) if single else None,
                        "scope": scope,
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

    def enrich_record(self, raw_record: dict[str, Any]) -> dict[str, Any]:
        """Count a cascade candidate's approved sources in the NDC Directory."""
        if raw_record["kind"] != "fda_bottleneck" or not raw_record.get("scope"):
            return raw_record
        params = {"search": _ndc_search(raw_record["generic"]), "limit": "1000"}
        try:
            payload = self._get(NDC_URL, params).json()
        except PollError as error:
            # openFDA answers a search with no matches with HTTP 404.
            if str(error) != "HTTP 404":
                raise
            payload = {"results": []}
        except ValueError as error:
            raise PollError(f"invalid NDC JSON: {error}", retryable=True) from error
        listings = payload.get("results", []) if isinstance(payload, dict) else []
        sources = approved_sources(
            listings if isinstance(listings, list) else [], self._clock().date()
        )
        count = len(sources["applications"])
        cascade = 1 <= count <= CASCADE_MAX_SOURCES
        return {
            **raw_record,
            "ndc": sources,
            "cascade": cascade,
            "priority": CASCADE_PRIORITY if cascade else None,
        }

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        if raw_record["kind"] == "chokepoint_drop":
            percent = -raw_record["divergence"] * 100
            over_threshold = percent - DIVERGENCE_THRESHOLD * 100
            return clamp_score(
                CHOKEPOINT_BASE_POINTS + over_threshold * CHOKEPOINT_POINTS_PER_PERCENT
            )
        classification = str(raw_record["recall"].get("classification"))
        if classification not in CLASS_POINTS:
            return 0.0
        if raw_record.get("cascade"):
            score = CASCADE_BASE_POINTS
            if classification == "Class I":
                score += CASCADE_CLASS_I_POINTS
            if len(raw_record["ndc"]["applications"]) == 1:
                score += CASCADE_SINGLE_SOURCE_POINTS
            if raw_record["shortages"]:
                score += CASCADE_IN_SHORTAGE_POINTS
            return clamp_score(score)
        if raw_record.get("company") is None:
            return 0.0
        score = CLASS_POINTS[classification] + SINGLE_SOURCE_POINTS
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
        targets = [
            EvidenceTarget(
                url=_openfda_query_url(
                    ENFORCEMENT_URL, "recall_number", raw_record["recall_number"]
                ),
                tier="tier_1_primary",
                kind="fda_enforcement",
                excerpt=raw_record["recall"].get("reason_for_recall"),
            )
        ]
        if raw_record["shortages"]:
            targets.append(
                EvidenceTarget(
                    url=_openfda_query_url(
                        SHORTAGES_URL,
                        "generic_name",
                        str(raw_record["shortages"][0]["generic_name"]),
                    ),
                    tier="tier_1_primary",
                    kind="fda_shortage",
                    excerpt=raw_record["shortages"][0].get("presentation"),
                )
            )
        if raw_record.get("cascade"):
            # The same query that produced the count, so the count can be rechecked.
            targets.append(
                EvidenceTarget(
                    url=f"{NDC_URL}?search="
                    + quote(_ndc_search(raw_record["generic"]), safe=":")
                    + "&limit=1000",
                    tier="tier_1_primary",
                    kind="fda_ndc",
                    metadata={"applications": raw_record["ndc"]["applications"]},
                )
            )
        return targets

    def verify_evidence_bar(self, artifacts: list[SourceArtifact]) -> bool:
        kinds = {
            item.metadata.get("kind")
            for item in artifacts
            if item.tier == "tier_1_primary"
            and host_of(item.origin_url) in {FDA_HOST, PORTWATCH_HOST}
        }
        return (
            {"fda_enforcement", "fda_shortage"} <= kinds
            or {"fda_enforcement", "fda_ndc"} <= kinds
            or "portwatch_transits" in kinds
        )

    def default_public_question(self, record: dict[str, Any]) -> str:
        if record["kind"] == "chokepoint_drop":
            return (
                f"Why did ship transits through {record['port_name']} fall "
                f"{abs(record['divergence']):.0%}, and what could the drop disrupt?"
            )
        if record.get("cascade"):
            return (
                f"Could patients face a regional shortage of {record['generic']} "
                f"following FDA recall {record['recall_number']}?"
            )
        return (
            f"Could patients lose access to {record['generic']} after recall "
            f"{record['recall_number']}?"
        )

    def default_public_interest_reason(self, record: dict[str, Any]) -> str:
        if record["kind"] == "chokepoint_drop":
            return (
                "A sustained drop at a shipping chokepoint can delay goods and "
                "raise prices far from the port."
            )
        if record.get("cascade"):
            count = len(record["ndc"]["applications"])
            sources = (
                "only one approved source"
                if count == 1
                else "only two approved sources"
            )
            return (
                f"{record['generic'].capitalize()} has {sources} listed in the FDA NDC "
                "Directory, so a recall can leave patients with few or no alternatives."
            )
        return (
            "Patients who depend on a generic drug with one reported supplier can "
            "lose access when that supplier recalls it."
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
        if record.get("cascade"):
            sources = record["ndc"]
            return render_prompt(
                f"Possible shortage cascade: {record['generic']}",
                {
                    "Priority": record["priority"],
                    "Recall": f"{record['recall_number']} ({recall.get('classification')})",
                    "Recalling firm": recall.get("recalling_firm"),
                    "Product": recall.get("product_description"),
                    "Reason": recall.get("reason_for_recall"),
                    "Recall scope": str(record["scope"]).replace("_", " "),
                    "Approved sources in NDC Directory": (
                        f"{len(sources['applications'])} "
                        f"({', '.join(sources['applications'])})"
                    ),
                    "Labelers on those applications": ", ".join(sources["labelers"])
                    or "none named",
                    "On FDA shortage list": (
                        record["shortages"][0].get("availability")
                        if record["shortages"]
                        else "not listed"
                    ),
                },
                artifacts,
                "Confirm which approved sources are actually marketing the product, "
                "since an NDC listing does not prove supply, then assess how long "
                "existing stock covers patients and who is most exposed.",
            )
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
