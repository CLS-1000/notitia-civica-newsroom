"""Seldon: Congress.gov bills and House roll-call votes.

A lead needs the enrolled text of a bill, as sealed PDF or XML. Congress.gov API
v3 publishes roll-call votes for the House only, so a Senate-only passage shows
no vote here. The API key travels in a header, keeping it out of URLs, logs, and
the spool.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Final
from urllib.parse import parse_qsl, urlsplit

from spec1_bridge.adapters.base_lazlong import (
    EvidenceTarget,
    LazLongAdapter,
    PollError,
    PollRequest,
    clamp_score,
    host_of,
    render_prompt,
)
from spec1_bridge.contracts import SourceArtifact

API_BASE: Final = "https://api.congress.gov/v3"
ENROLLED_FORMATS: Final = {"PDF": "pdf", "Formatted XML": "xml"}
PAGE_SIZE: Final = "250"

ENROLLED_POINTS: Final = 50.0
PUBLIC_LAW_POINTS: Final = 25.0
PRESENTED_POINTS: Final = 15.0
ROLL_CALL_POINTS: Final = 15.0

_ENROLLED_FILE: Final = re.compile(r"enr\.(pdf|xml)$", re.IGNORECASE)
_ADVANCED_ACTION: Final = re.compile(
    r"became public law|signed by president|presented to president|"
    r"enrolled|veto",
    re.IGNORECASE,
)


def congress_for(moment: datetime) -> tuple[int, int]:
    """Return the Congress and session in progress at moment."""
    year = moment.year
    # A new Congress convenes on January 3 of odd years.
    if moment.month == 1 and moment.day < 3:
        year -= 1
    return (year - 1789) // 2 + 1, 1 if year % 2 == 1 else 2


def _bill_id(bill_type: Any, number: Any) -> str:
    return f"{str(bill_type).lower()}{number}"


def _is_enrolled_version(version: dict[str, Any]) -> bool:
    if "enrolled" in str(version.get("type", "")).lower():
        return True
    return any(
        _ENROLLED_FILE.search(str(item.get("url", "")))
        for item in version.get("formats", [])
    )


class SeldonCivicAdapter(LazLongAdapter):
    adapter_name = "seldon_civic"
    subject_domain = "federal_legislation"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        lookback_days: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        key = api_key or os.environ.get("CONGRESS_API_KEY")
        if not key:
            raise ValueError("a Congress.gov API key is required (CONGRESS_API_KEY)")
        self._api_key = key
        self._lookback_days = lookback_days

    def auth_headers(self) -> dict[str, str]:
        return {"X-Api-Key": self._api_key}

    def poll_requests(self) -> list[PollRequest]:
        now = self._clock()
        congress, session = congress_for(now)
        since = now - timedelta(days=self._lookback_days)
        return [
            PollRequest(
                "bills",
                f"{API_BASE}/bill/{congress}",
                {
                    "format": "json",
                    "limit": PAGE_SIZE,
                    "sort": "updateDate desc",
                    "fromDateTime": since.strftime("%Y-%m-%dT00:00:00Z"),
                },
            ),
            PollRequest(
                "votes",
                f"{API_BASE}/house-vote/{congress}/{session}",
                {"format": "json", "limit": PAGE_SIZE},
            ),
        ]

    def extract_items(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        key = "bills" if request.name == "bills" else "houseRollCallVotes"
        rows = payload.get(key, [])
        return rows if isinstance(rows, list) else []

    def next_request(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> PollRequest | None:
        pagination = payload.get("pagination")
        link = pagination.get("next") if isinstance(pagination, dict) else None
        if not link:
            return None
        parts = urlsplit(str(link))
        query = {
            key: value
            for key, value in parse_qsl(parts.query)
            if key.lower() != "api_key"
        }
        return PollRequest(
            request.name, f"{parts.scheme}://{parts.netloc}{parts.path}", query
        )

    def assemble_records(
        self, items: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        votes: dict[str, list[dict[str, Any]]] = {}
        for vote in items.get("votes", []):
            if vote.get("legislationNumber"):
                bill_id = _bill_id(
                    vote.get("legislationType"), vote["legislationNumber"]
                )
                votes.setdefault(bill_id, []).append(vote)

        bills = {
            _bill_id(bill.get("type"), bill.get("number")): bill
            for bill in items.get("bills", [])
        }
        return [
            {"bill_id": bill_id, "bill": bill, "votes": votes.get(bill_id, [])}
            for bill_id, bill in bills.items()
        ]

    def enrich_record(self, raw_record: dict[str, Any]) -> dict[str, Any]:
        bill = raw_record["bill"]
        latest_action = str((bill.get("latestAction") or {}).get("text", ""))
        enriched = {**raw_record, "enrolled_formats": {}}
        if _ADVANCED_ACTION.search(latest_action) is None:
            return enriched

        url = (
            f"{API_BASE}/bill/{bill['congress']}/{str(bill['type']).lower()}"
            f"/{bill['number']}/text"
        )
        response = self._get(url, {"format": "json"})
        try:
            versions = response.json().get("textVersions", [])
        except (ValueError, AttributeError) as error:
            raise PollError(f"bad text listing: {error}", retryable=True) from error

        for version in versions:
            if _is_enrolled_version(version):
                enriched["enrolled_formats"] = {
                    ENROLLED_FORMATS[item["type"]]: item["url"]
                    for item in version.get("formats", [])
                    if item.get("type") in ENROLLED_FORMATS and item.get("url")
                }
                break
        return enriched

    def record_key(self, raw_record: dict[str, Any]) -> str:
        bill = raw_record["bill"]
        return f"{bill.get('congress')}:{raw_record['bill_id']}:enrolled"

    def record_timestamp(self, raw_record: dict[str, Any]) -> datetime:
        bill = raw_record["bill"]
        value = (bill.get("latestAction") or {}).get("actionDate") or bill.get(
            "updateDate"
        )
        if not value:
            raise ValueError(f"{raw_record['bill_id']} has no usable date")
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        if not raw_record.get("enrolled_formats"):
            return 0.0
        score = ENROLLED_POINTS
        latest_action = str(
            (raw_record["bill"].get("latestAction") or {}).get("text", "")
        ).lower()
        if "became public law" in latest_action or "signed by" in latest_action:
            score += PUBLIC_LAW_POINTS
        elif "presented to president" in latest_action:
            score += PRESENTED_POINTS
        if raw_record["votes"]:
            score += ROLL_CALL_POINTS
        return clamp_score(score)

    def evidence_targets(self, raw_record: dict[str, Any]) -> list[EvidenceTarget]:
        return [
            EvidenceTarget(
                url=url,
                tier="tier_1_primary",
                kind="enrolled_text",
                metadata={"format": text_format},
            )
            for text_format, url in raw_record.get("enrolled_formats", {}).items()
        ]

    def verify_evidence_bar(self, artifacts: list[SourceArtifact]) -> bool:
        return any(
            item.metadata.get("kind") == "enrolled_text"
            and item.metadata.get("format") in {"pdf", "xml"}
            and item.tier == "tier_1_primary"
            and (host_of(item.origin_url) or "").endswith("congress.gov")
            and _ENROLLED_FILE.search(item.origin_url) is not None
            for item in artifacts
        )

    def build_paste_ready_prompt(
        self, record: dict[str, Any], artifacts: list[SourceArtifact]
    ) -> str:
        bill = record["bill"]
        latest = bill.get("latestAction") or {}
        rolls = ", ".join(
            f"#{vote.get('rollCallNumber')} ({vote.get('result')})"
            for vote in record["votes"]
        )
        return render_prompt(
            f"{bill.get('congress')}th Congress {bill.get('type')} "
            f"{bill.get('number')}: {bill.get('title')}",
            {
                "Latest action": f"{latest.get('actionDate')} {latest.get('text')}",
                "House roll-call votes": rolls or "none on record",
            },
            artifacts,
            "Summarize what the enrolled text enacts, quoting operative sections.",
        )
