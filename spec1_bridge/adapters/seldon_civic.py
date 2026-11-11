"""Seldon: federal law and regulation.

Two sources. The Federal Register needs no key and is always polled. Congress.gov
is polled only when a Congress.gov API key is given or set in CONGRESS_API_KEY;
without one Seldon runs keyless, on Federal Register rules alone.

- Congress.gov API v3: bills that are enrolled, presented or signed, with House
  roll-call votes and the bill's CBO cost estimates. A bill lead needs the
  enrolled text, as sealed PDF or XML. Congress.gov publishes roll-call votes for
  the House only, so a Senate-only passage shows no vote. CBO estimates appear
  in the bill detail as a title, date and link to cbo.gov, with no dollar figure,
  and cbo.gov refuses automated requests, so a lead links the estimate and never
  states its amount. The API key travels in a header, only to api.congress.gov.

- Federal Register API v1: significant final rules (type RULE, significant=1).
  A rule is flagged when its text states it is a major rule under the
  Congressional Review Act (5 U.S.C. 804(2)) or economically significant under
  E.O. 12866 section 3(f)(1), or when compliance starts within 90 days. The text
  states negatives as often as positives ("is not a major rule", "does not meet
  the criteria"), so only an affirmative statement counts. Compliance starts on
  an explicit compliance date in the rule's DATES section when one is stated,
  else on its effective date. A rule lead needs the rule itself, sealed from
  govinfo.gov (PDF) or federalregister.gov (full text).
"""

from __future__ import annotations

import os
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final
from urllib.parse import parse_qsl, urlencode, urlsplit

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
CONGRESS_API_HOST: Final = "api.congress.gov"
ENROLLED_FORMATS: Final = {"PDF": "pdf", "Formatted XML": "xml"}
PAGE_SIZE: Final = "250"

FR_API: Final = "https://www.federalregister.gov/api/v1/documents.json"
FR_HOST: Final = "www.federalregister.gov"
GOVINFO_HOST: Final = "www.govinfo.gov"
FR_PAGE_SIZE: Final = "100"
FR_FIELDS: Final = (
    "document_number",
    "title",
    "type",
    "action",
    "abstract",
    "agencies",
    "significant",
    "publication_date",
    "effective_on",
    "dates",
    "docket_ids",
    "regulation_id_numbers",
    "html_url",
    "pdf_url",
    "raw_text_url",
)
COMPLIANCE_WINDOW_DAYS: Final = 90

ENROLLED_POINTS: Final = 50.0
PUBLIC_LAW_POINTS: Final = 25.0
PRESENTED_POINTS: Final = 15.0
ROLL_CALL_POINTS: Final = 15.0
CBO_ESTIMATE_POINTS: Final = 10.0

SIGNIFICANT_RULE_POINTS: Final = 30.0
MAJOR_RULE_POINTS: Final = 40.0
NEAR_DEADLINE_POINTS: Final = 20.0

BILL_LABELS: Final = {
    "HR": "H.R.",
    "S": "S.",
    "HJRES": "H.J.Res.",
    "SJRES": "S.J.Res.",
    "HCONRES": "H.Con.Res.",
    "SCONRES": "S.Con.Res.",
    "HRES": "H.Res.",
    "SRES": "S.Res.",
}
CRA_MAJOR: Final = "a major rule under the Congressional Review Act"
ECONOMICALLY_SIGNIFICANT: Final = "economically significant under Executive Order 12866"

_GAP: Final = r"[\s\S]{0,100}?"
# Each pattern is an affirmative finding. "U.S.C." and "E.O." contain periods, so
# matches are windows of text, not sentences.
_MAJOR_FINDINGS: Final = (
    (
        re.compile(
            r"\b(?:meets|satisfies) the criteria\b" + _GAP + r"804\(2\)", re.IGNORECASE
        ),
        CRA_MAJOR,
    ),
    (re.compile(r"\bis an? [`'\"\u201c]*major rule\b", re.IGNORECASE), CRA_MAJOR),
    (
        re.compile(
            r"\beconomically significant\b" + _GAP + r"3\(f\)\(1\)", re.IGNORECASE
        ),
        ECONOMICALLY_SIGNIFICANT,
    ),
    (
        re.compile(
            r"\b3\(f\)\(1\)[\s\S]{0,60}?\beconomically significant\b", re.IGNORECASE
        ),
        ECONOMICALLY_SIGNIFICANT,
    ),
)
_NEGATION: Final = re.compile(r"\b(?:not|no|neither|nor|never)\b", re.IGNORECASE)
_NEGATION_LOOKBACK: Final = 40
_LONG_DATE: Final = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December) (\d{1,2}), (\d{4})\b"
)
_MONTHS: Final = {
    name: number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}

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


def bill_label(bill: dict[str, Any]) -> str:
    """Return a bill's citation, such as 'H.R. 1234'."""
    kind = str(bill.get("type", "")).upper()
    return f"{BILL_LABELS.get(kind, kind)} {bill.get('number')}"


def major_finding(text: str) -> tuple[str, str] | None:
    """Return (finding, excerpt) for an affirmative major or 3(f)(1) statement.

    A negation just before or inside the match ("is not a major rule", "not
    economically significant") disqualifies it. A Congressional Review Act
    finding is preferred, since it is the narrower designation.
    """
    flat = " ".join(text.split())
    found: list[tuple[str, str]] = []
    for pattern, finding in _MAJOR_FINDINGS:
        for match in pattern.finditer(flat):
            before = flat[max(0, match.start() - _NEGATION_LOOKBACK) : match.start()]
            if not _NEGATION.search(before) and not _NEGATION.search(match.group()):
                found.append((finding, match.group()))
                break
    for finding, excerpt in found:
        if finding == CRA_MAJOR:
            return finding, excerpt
    return found[0] if found else None


def _long_dates(text: str) -> list[date]:
    dates = []
    for month, day, year in _LONG_DATE.findall(text):
        try:
            dates.append(date(int(year), _MONTHS[month], int(day)))
        except ValueError:
            continue
    return dates


def compliance_date(dates_text: str) -> date | None:
    """Return the earliest date stated in a sentence of DATES about compliance."""
    found = [
        moment
        for sentence in re.split(r"(?<=\.)\s+", " ".join(dates_text.split()))
        if "complian" in sentence.lower()
        for moment in _long_dates(sentence)
    ]
    return min(found) if found else None


def _iso_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _long(moment: date) -> str:
    return f"{moment:%B} {moment.day}, {moment.year}"


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
        rule_lookback_days: int = 7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key or os.environ.get("CONGRESS_API_KEY") or None
        self._lookback_days = lookback_days
        self._rule_lookback_days = rule_lookback_days

    def _rules_request(self) -> PollRequest:
        since = (self._clock() - timedelta(days=self._rule_lookback_days)).date()
        query = [
            ("conditions[type][]", "RULE"),
            ("conditions[significant]", "1"),
            ("conditions[publication_date][gte]", since.isoformat()),
            ("order", "newest"),
            ("per_page", FR_PAGE_SIZE),
            *(("fields[]", name) for name in FR_FIELDS),
        ]
        # fields[] repeats, so the query lives in the URL rather than in params.
        return PollRequest("rules", f"{FR_API}?{urlencode(query)}")

    @property
    def polls_congress(self) -> bool:
        """Whether Congress.gov bills and votes are polled: only with a key."""
        return self._api_key is not None

    def auth_headers(self) -> dict[str, str]:
        return {"X-Api-Key": self._api_key} if self._api_key else {}

    def replays(self, request: PollRequest) -> bool:
        # A keyless run leaves spooled Congress.gov requests for a run with a key.
        return self.polls_congress or host_of(request.url) != CONGRESS_API_HOST

    def headers_for(self, url: str) -> dict[str, str]:
        # The key is for Congress.gov alone; the Federal Register needs none.
        return self.auth_headers() if host_of(url) == CONGRESS_API_HOST else {}

    def poll_requests(self) -> list[PollRequest]:
        if not self.polls_congress:
            return [self._rules_request()]
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
            self._rules_request(),
        ]

    def extract_items(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        key = {"bills": "bills", "rules": "results"}.get(
            request.name, "houseRollCallVotes"
        )
        rows = payload.get(key, [])
        return rows if isinstance(rows, list) else []

    def next_request(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> PollRequest | None:
        if request.name == "rules":
            link = payload.get("next_page_url")
            return PollRequest(request.name, str(link)) if link else None
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
        rules = {
            str(row["document_number"]): row
            for row in items.get("rules", [])
            if row.get("document_number")
            and str(row.get("type", "")).lower() == "rule"
            and row.get("significant") is not False
        }
        return [
            *(
                {
                    "kind": "bill",
                    "bill_id": bill_id,
                    "bill": bill,
                    "votes": votes.get(bill_id, []),
                }
                for bill_id, bill in bills.items()
            ),
            *({"kind": "fr_rule", "rule": rule} for rule in rules.values()),
        ]

    def _enrich_rule(self, raw_record: dict[str, Any]) -> dict[str, Any]:
        rule = raw_record["rule"]
        finding = None
        if rule.get("raw_text_url"):
            finding = major_finding(self._get(str(rule["raw_text_url"])).text)
        today = self._clock().date()
        explicit = compliance_date(str(rule.get("dates") or ""))
        effective = _iso_date(rule.get("effective_on"))
        starts = explicit if explicit and explicit >= today else effective
        window_end = today + timedelta(days=COMPLIANCE_WINDOW_DAYS)
        return {
            **raw_record,
            "finding": finding[0] if finding else None,
            "finding_excerpt": finding[1] if finding else None,
            "compliance_date": explicit,
            "starts": starts,
            "starts_is_compliance": starts is not None and starts == explicit,
            "near_deadline": starts is not None and today <= starts <= window_end,
        }

    def _cbo_estimates(self, bill: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{API_BASE}/bill/{bill['congress']}/{str(bill['type']).lower()}/{bill['number']}"
        try:
            detail = self._get(url, {"format": "json"}).json().get("bill", {})
        except (ValueError, AttributeError) as error:
            raise PollError(f"bad bill detail: {error}", retryable=True) from error
        estimates = detail.get("cboCostEstimates") if isinstance(detail, dict) else None
        if isinstance(estimates, dict):
            estimates = estimates.get("item", [])
        rows = [
            {
                "title": str(item.get("title", "")),
                "pubDate": str(item.get("pubDate", "")),
                "url": str(item.get("url", "")),
                "description": str(item.get("description", "")),
            }
            for item in estimates or []
            if isinstance(item, dict) and item.get("url")
        ]
        return sorted(rows, key=lambda item: item["pubDate"])

    def enrich_record(self, raw_record: dict[str, Any]) -> dict[str, Any]:
        if raw_record.get("kind") == "fr_rule":
            return self._enrich_rule(raw_record)
        bill = raw_record["bill"]
        latest_action = str((bill.get("latestAction") or {}).get("text", ""))
        enriched = {**raw_record, "enrolled_formats": {}, "cbo_estimates": []}
        if _ADVANCED_ACTION.search(latest_action) is None:
            return enriched
        enriched["cbo_estimates"] = self._cbo_estimates(bill)

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
        if raw_record.get("kind") == "fr_rule":
            return f"fr:{raw_record['rule']['document_number']}"
        bill = raw_record["bill"]
        return f"{bill.get('congress')}:{raw_record['bill_id']}:enrolled"

    def record_timestamp(self, raw_record: dict[str, Any]) -> datetime:
        if raw_record.get("kind") == "fr_rule":
            published = _iso_date(raw_record["rule"].get("publication_date"))
            if published is None:
                raise ValueError(
                    f"fr:{raw_record['rule']['document_number']} has no date"
                )
            return datetime(published.year, published.month, published.day, tzinfo=UTC)
        bill = raw_record["bill"]
        value = (bill.get("latestAction") or {}).get("actionDate") or bill.get(
            "updateDate"
        )
        if not value:
            raise ValueError(f"{raw_record['bill_id']} has no usable date")
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        if raw_record.get("kind") == "fr_rule":
            if not raw_record.get("finding") and not raw_record.get("near_deadline"):
                return 0.0
            score = SIGNIFICANT_RULE_POINTS
            if raw_record.get("finding"):
                score += MAJOR_RULE_POINTS
            if raw_record.get("near_deadline"):
                score += NEAR_DEADLINE_POINTS
            return clamp_score(score)
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
        if raw_record.get("cbo_estimates"):
            score += CBO_ESTIMATE_POINTS
        return clamp_score(score)

    def evidence_targets(self, raw_record: dict[str, Any]) -> list[EvidenceTarget]:
        if raw_record.get("kind") == "fr_rule":
            rule = raw_record["rule"]
            number = str(rule["document_number"])
            targets = []
            if rule.get("pdf_url"):
                targets.append(
                    EvidenceTarget(
                        url=str(rule["pdf_url"]),
                        tier="tier_1_primary",
                        kind="fr_rule_pdf",
                        metadata={"document_number": number},
                    )
                )
            if rule.get("raw_text_url"):
                targets.append(
                    EvidenceTarget(
                        url=str(rule["raw_text_url"]),
                        tier="tier_1_primary",
                        kind="fr_rule_text",
                        excerpt=raw_record.get("finding_excerpt"),
                        metadata={"document_number": number},
                    )
                )
            return targets
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
        return self._statute_sealed(artifacts) or self._rule_sealed(artifacts)

    @staticmethod
    def _rule_sealed(artifacts: list[SourceArtifact]) -> bool:
        """The rule itself: govinfo's official PDF or the Federal Register text."""
        for item in artifacts:
            number = str(item.metadata.get("document_number") or "")
            if (
                item.tier != "tier_1_primary"
                or not number
                or number not in item.origin_url
            ):
                continue
            host = host_of(item.origin_url)
            kind = item.metadata.get("kind")
            if kind == "fr_rule_pdf" and host == GOVINFO_HOST:
                return True
            if (
                kind == "fr_rule_text"
                and host == FR_HOST
                and "/documents/full_text/" in item.origin_url
            ):
                return True
        return False

    @staticmethod
    def _statute_sealed(artifacts: list[SourceArtifact]) -> bool:
        return any(
            item.metadata.get("kind") == "enrolled_text"
            and item.metadata.get("format") in {"pdf", "xml"}
            and item.tier == "tier_1_primary"
            and (host_of(item.origin_url) or "").endswith("congress.gov")
            and _ENROLLED_FILE.search(item.origin_url) is not None
            for item in artifacts
        )

    def default_public_question(self, record: dict[str, Any]) -> str:
        if record.get("kind") == "fr_rule":
            title = str(record["rule"].get("title", "")).strip().rstrip(".")
            return f"What does the rule \u201c{title}\u201d change in federal law or regulation?"
        return f"What does {bill_label(record['bill'])} change in federal law or regulation?"

    def default_public_interest_reason(self, record: dict[str, Any]) -> str:
        if record.get("kind") == "fr_rule":
            return self._rule_reason(record)
        estimates = record.get("cbo_estimates") or []
        if estimates:
            latest = estimates[-1]
            published = _iso_date(latest["pubDate"])
            when = f" on {_long(published)}" if published else ""
            return (
                f"The Congressional Budget Office published a cost estimate for "
                f"{bill_label(record['bill'])}{when} ({latest['url']})."
            )
        return (
            "Enrolled federal legislation changes law that applies across the "
            "country, and the House roll calls show how representatives voted."
        )

    @staticmethod
    def _rule_reason(record: dict[str, Any]) -> str:
        finding = record.get("finding")
        kind = "major federal rule" if finding else "significant federal rule"
        parts = []
        starts = record.get("starts")
        if record.get("near_deadline") and starts:
            if record.get("starts_is_compliance"):
                parts.append(
                    f"Compliance with this {kind} is required starting {_long(starts)}."
                )
            else:
                parts.append(f"This {kind} takes effect on {_long(starts)}.")
        if finding:
            parts.append(f"The rule states it is {finding}.")
        return " ".join(parts)

    def build_paste_ready_prompt(
        self, record: dict[str, Any], artifacts: list[SourceArtifact]
    ) -> str:
        if record.get("kind") == "fr_rule":
            rule = record["rule"]
            agencies = ", ".join(
                str(item.get("name") or item.get("raw_name"))
                for item in rule.get("agencies") or []
                if isinstance(item, dict)
            )
            starts = record.get("starts")
            return render_prompt(
                f"Federal Register final rule {rule.get('document_number')}: "
                f"{rule.get('title')}",
                {
                    "Agencies": agencies or "not listed",
                    "Action": rule.get("action"),
                    "Published": rule.get("publication_date"),
                    "Effective": rule.get("effective_on") or "not stated",
                    "Compliance date (from DATES)": (
                        record["compliance_date"].isoformat()
                        if record.get("compliance_date")
                        else "none stated"
                    ),
                    f"Compliance starts within {COMPLIANCE_WINDOW_DAYS} days": (
                        f"yes, {starts.isoformat()}"
                        if record.get("near_deadline") and starts
                        else "no"
                    ),
                    "Major or 3(f)(1) finding": record.get("finding") or "none stated",
                    "Docket": ", ".join(rule.get("docket_ids") or []) or "none listed",
                    "RIN": ", ".join(rule.get("regulation_id_numbers") or [])
                    or "none listed",
                },
                artifacts,
                "Summarize what the rule requires, of whom, and from when, quoting "
                "its regulatory text and DATES section, and say whether it adds or "
                "removes obligations.",
            )
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
                "CBO cost estimates": "; ".join(
                    f"{item['title']} ({item['pubDate'][:10]}): {item['url']}"
                    for item in record.get("cbo_estimates") or []
                )
                or "none listed",
            },
            artifacts,
            "Summarize what the enrolled text enacts, quoting operative sections.",
        )
