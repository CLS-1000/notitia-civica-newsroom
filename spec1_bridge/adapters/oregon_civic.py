"""Ubik: Oregon Legislature measures, hearings, and LFO fiscal statements.

OLIS reports fiscal impact only as a status such as 'Fiscal impact issued'. Dollar
figures exist only in the LFO statement PDF, so a statement is downloaded for any
measure that has one. The amount is the largest figure in the statement, which
makes the >$1M signal a rule-based screen and not a costing of the measure.

OLIS has no gut-and-stuff flag either. A measure is flagged when an amendment was
adopted and the relating clause of its current version shares almost no words
with the introduced version.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Final
from urllib.parse import urljoin

from spec1_bridge.adapters.base_adapter import (
    BaseAdapter,
    EvidenceTarget,
    PollRequest,
    clamp_score,
    extract_pdf_text,
    host_of,
    normalize_whitespace,
    render_prompt,
)
from spec1_bridge.contracts import SourceArtifact

ODATA_BASE: Final = "https://api.oregonlegislature.gov/odata/ODataService.svc"
OLIS_HOST: Final = "olis.oregonlegislature.gov"
FISCAL_DOCUMENT_TYPES: Final = frozenset({"Fiscal Impact Statement", "Budget Report"})

FISCAL_THRESHOLD_USD: Final = 1_000_000.0
FISCAL_POINTS: Final = 50.0
EMERGENCY_POINTS: Final = 25.0
GUT_AND_STUFF_POINTS: Final = 35.0
GUT_AND_STUFF_MAX_OVERLAP: Final = 0.2

_AMOUNT: Final = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s*(million|billion))?", re.IGNORECASE
)
_MULTIPLIERS: Final = {"million": 1e6, "billion": 1e9}
_RELATING: Final = re.compile(r"relating to\s+(.+?)(?:;|\.|$)", re.IGNORECASE)
_STOPWORDS: Final = frozenset({"a", "an", "and", "for", "in", "of", "on", "the", "to"})

_ENTITIES: Final = {
    "measures": "Measures",
    "documents": "MeasureDocuments",
    "analysis": "MeasureAnalysisDocuments",
    "hearings": "CommitteeAgendaItems",
    "amendments": "CommitteeProposedAmendments",
}


def _measure_id(row: dict[str, Any]) -> str:
    return f"{row.get('MeasurePrefix', '')} {row.get('MeasureNumber', '')}".strip()


def _odata_date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value)[:19])
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC)


def _newest(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(rows, key=lambda row: str(row.get("CreatedDate") or ""), default=None)


def largest_amount(text: str) -> tuple[float, str] | None:
    """Return the largest dollar figure in text and the line it appears on."""
    best: tuple[float, str] | None = None
    for line in text.splitlines():
        for match in _AMOUNT.finditer(line):
            amount = float(match.group(1).replace(",", ""))
            unit = (match.group(2) or "").lower()
            amount *= _MULTIPLIERS.get(unit, 1.0)
            if best is None or amount > best[0]:
                best = (amount, normalize_whitespace(line))
    return best


def relating_clause_words(text: str) -> set[str]:
    """Return the significant words of a measure's relating clause."""
    match = _RELATING.search(normalize_whitespace(text))
    if match is None:
        return set()
    return set(re.findall(r"[a-z]+", match.group(1).lower())) - _STOPWORDS


def clauses_diverge(introduced: set[str], current: set[str]) -> bool:
    if not introduced or not current:
        return False
    overlap = len(introduced & current) / len(introduced | current)
    return overlap <= GUT_AND_STUFF_MAX_OVERLAP


class OregonCivicAdapter(BaseAdapter):
    adapter_name = "ubik_oregon"
    subject_domain = "oregon_legislature"

    def __init__(self, session_key: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._session_key = session_key

    def poll_requests(self) -> list[PollRequest]:
        params = {
            "$format": "json",
            "$filter": f"SessionKey eq '{self._session_key}'",
        }
        return [
            PollRequest(name, f"{ODATA_BASE}/{entity}", dict(params))
            for name, entity in _ENTITIES.items()
        ]

    def extract_items(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rows = payload.get("value", [])
        return rows if isinstance(rows, list) else []

    def next_request(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> PollRequest | None:
        link = payload.get("odata.nextLink") or payload.get("@odata.nextLink")
        if not link:
            return None
        return PollRequest(request.name, urljoin(f"{ODATA_BASE}/", str(link)))

    def assemble_records(
        self, items: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        related: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for name in ("documents", "analysis", "hearings", "amendments"):
            for row in items.get(name, []):
                related[_measure_id(row)][name].append(row)

        measures = {_measure_id(row): row for row in items.get("measures", [])}
        return [
            {
                "session_key": self._session_key,
                "measure_id": measure_id,
                "measure": measure,
                "documents": related[measure_id]["documents"],
                "analysis": related[measure_id]["analysis"],
                "hearings": related[measure_id]["hearings"],
                "amendments": related[measure_id]["amendments"],
            }
            for measure_id, measure in measures.items()
        ]

    @staticmethod
    def _fiscal_statement(record: dict[str, Any]) -> dict[str, Any] | None:
        return _newest(
            [
                row
                for row in record["analysis"]
                if row.get("DocumentType") in FISCAL_DOCUMENT_TYPES
                and not row.get("IsMinority")
            ]
        )

    def _pdf_text(self, url: str) -> str | None:
        return extract_pdf_text(self._get(url).content)

    def _amendment_replaced_measure(self, record: dict[str, Any]) -> bool:
        adopted = any(row.get("Meaning") == "Adopted" for row in record["amendments"])
        documents = record["documents"]
        introduced = next(
            (row for row in documents if row.get("VersionDescription") == "Introduced"),
            None,
        )
        current = _newest(documents)
        if not adopted or introduced is None or current in (None, introduced):
            return False
        assert current is not None
        first = self._pdf_text(str(introduced["DocumentUrl"]))
        latest = self._pdf_text(str(current["DocumentUrl"]))
        if first is None or latest is None:
            return False
        return clauses_diverge(
            relating_clause_words(first), relating_clause_words(latest)
        )

    def enrich_record(self, raw_record: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(raw_record)
        statement = self._fiscal_statement(raw_record)
        found: tuple[float, str] | None = None
        if statement is not None:
            text = self._pdf_text(str(statement["DocumentUrl"]))
            found = largest_amount(text) if text else None
        enriched["fiscal_amount_usd"] = found[0] if found else None
        enriched["fiscal_excerpt"] = found[1] if found else None
        enriched["gut_and_stuff"] = self._amendment_replaced_measure(raw_record)
        return enriched

    def record_key(self, raw_record: dict[str, Any]) -> str:
        statement = self._fiscal_statement(raw_record)
        statement_id = statement.get("MeasureAnalysisId") if statement else "none"
        return (
            f"{raw_record['session_key']}:{raw_record['measure_id']}:"
            f"v{len(raw_record['documents'])}:{statement_id}"
        )

    def record_timestamp(self, raw_record: dict[str, Any]) -> datetime:
        measure = raw_record["measure"]
        timestamp = _odata_date(measure.get("ModifiedDate")) or _odata_date(
            measure.get("CreatedDate")
        )
        if timestamp is None:
            raise ValueError(f"{raw_record['measure_id']} has no usable date")
        return timestamp

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        score = 0.0
        amount = raw_record.get("fiscal_amount_usd")
        if amount is not None and amount > FISCAL_THRESHOLD_USD:
            score += FISCAL_POINTS
        if raw_record["measure"].get("EmergencyClause") is True:
            score += EMERGENCY_POINTS
        if raw_record.get("gut_and_stuff"):
            score += GUT_AND_STUFF_POINTS
        return clamp_score(score)

    def evidence_targets(self, raw_record: dict[str, Any]) -> list[EvidenceTarget]:
        targets: list[EvidenceTarget] = []
        measure_text = _newest(raw_record["documents"])
        if measure_text is not None:
            targets.append(
                EvidenceTarget(
                    url=str(measure_text["DocumentUrl"]),
                    tier="tier_1_primary",
                    kind="olis_measure_text",
                    metadata={"version": measure_text.get("VersionDescription")},
                )
            )
        statement = self._fiscal_statement(raw_record)
        if statement is not None:
            targets.append(
                EvidenceTarget(
                    url=str(statement["DocumentUrl"]),
                    tier="tier_1_primary",
                    kind="lfo_fiscal_statement",
                    excerpt=raw_record.get("fiscal_excerpt"),
                    metadata={"version": statement.get("Version")},
                )
            )
        return targets

    def verify_evidence_bar(self, artifacts: list[SourceArtifact]) -> bool:
        return any(
            item.metadata.get("kind") in {"olis_measure_text", "lfo_fiscal_statement"}
            and item.tier == "tier_1_primary"
            and host_of(item.origin_url) == OLIS_HOST
            for item in artifacts
        )

    def default_public_question(self, record: dict[str, Any]) -> str:
        measure_id = record["measure_id"]
        if record.get("gut_and_stuff"):
            return (
                f"Why was the relating clause of Oregon {measure_id} replaced, and "
                "what does the measure do now?"
            )
        relating = record["measure"].get("RelatingTo") or "this measure"
        return f"What would Oregon {measure_id} ({relating}) change, and at what cost?"

    def default_public_interest_reason(self, record: dict[str, Any]) -> str:
        return (
            "A measure before the Oregon Legislature can change state law, and its "
            "fiscal statement shows the public money at stake."
        )

    def build_paste_ready_prompt(
        self, record: dict[str, Any], artifacts: list[SourceArtifact]
    ) -> str:
        measure = record["measure"]
        amount = record.get("fiscal_amount_usd")
        hearings = sorted(
            str(row.get("MeetingDate"))[:10]
            for row in record["hearings"]
            if row.get("MeetingDate")
        )
        return render_prompt(
            f"Oregon {record['session_key']} {record['measure_id']}: "
            f"{measure.get('RelatingTo') or 'relating clause unavailable'}",
            {
                "Current location": measure.get("CurrentLocation"),
                "LFO fiscal status": measure.get("FiscalImpact"),
                "Largest figure in LFO statement": (
                    f"${amount:,.0f}" if amount is not None else "not found"
                ),
                "Emergency clause": measure.get("EmergencyClause") is True,
                "Relating clause replaced by amendment": bool(
                    record.get("gut_and_stuff")
                ),
                "Committee hearings": ", ".join(hearings) or "none on record",
            },
            artifacts,
            "Assess what this measure does and whether the findings above hold up "
            "against the sealed text.",
        )
