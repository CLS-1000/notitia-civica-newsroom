"""Mazer Defense (Engine 6): defense procurement integrity.

Signals:

- A major sole-source award: a Department of Defense contract whose USAspending
  record shows it was not competed (extent competed B, C, G, or NDO) and whose
  total obligation is at least $50,000,000.
- A sustained GAO bid protest against a DoD award, from a published GAO
  decision that sustains the protest.

DoD Inspector General audit reports linked to an award are sealed with it as
corroboration; they never make a lead on their own.

Sources:

- USAspending API v2 (keyless). Awards are found with spending_by_award, which
  is POST-only; the base adapter polls with GET, so this adapter POSTs its own
  search, with the JSON body kept in the request's params so a failed search
  spools and replays like any other. Every decision is made from the award's
  own record (GET /api/v2/awards/{id}/), which is what gets sealed, never from
  the search row.
- GAO bid protest decisions and DoD IG reports, supplied by editors as dockets
  (a JSONL file, one docket per line): gao.gov and dodig.mil refuse automated
  requests and GAO publishes no API or feed. A docket names its official URL,
  the award it concerns, and a finding quoted verbatim from the decision. It is
  only a claim until sealed: the evidence bar re-reads the sealed decision and
  requires its docket number, the sustain wording (for GAO), the awardee as
  named, and the finding, verbatim. Dockets from any other host (defense
  blogs, forums, leak sites) are rejected when loaded. If an official source
  refuses the fetch, the lead is an error for retry, never a lead without it.

Fail closed: a record whose award is not DoD, or which is neither a major
sole-source award nor linked to a sustained protest, is not a lead. The public
question and reason come only from this adapter, and the reason states only
what the sealed records show: "sole-source" only for an uncompeted award, and a
GAO finding only for a sealed sustained decision.

Not implemented: cost-plus ceiling growth over 25%. USAspending's transactions
API carries no per-modification ceiling, and the FPDS ATOM feed that did was
retired in 2026 for the SAM.gov Contract Awards API, which needs an API key.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

import httpx

from spec1_bridge.adapters.base_lazlong import (
    _MAX_BACKOFF_SECONDS,
    _RETRYABLE_STATUS,
    EvidenceTarget,
    LazLongAdapter,
    PollError,
    PollRequest,
    clamp_score,
    error_in_body,
    excerpt_is_verbatim,
    host_of,
    normalize_whitespace,
    read_artifact_text,
    render_prompt,
)
from spec1_bridge.contracts import SourceArtifact

logger = logging.getLogger(__name__)

USASPENDING_SEARCH: Final = (
    "https://api.usaspending.gov/api/v2/search/spending_by_award/"
)
USASPENDING_AWARD: Final = "https://api.usaspending.gov/api/v2/awards/"
USASPENDING_HOST: Final = "api.usaspending.gov"
SEARCH_REQUEST: Final = "usaspending:sole_source"

GAO_HOSTS: Final = frozenset({"gao.gov", "www.gao.gov"})
DODIG_HOSTS: Final = frozenset({"dodig.mil", "www.dodig.mil", "media.defense.gov"})

DOD_NAME: Final = "Department of Defense"
DOD_TOPTIER_CODE: Final = "097"
CONTRACT_TYPE_CODES: Final = ("A", "B", "C", "D")
# FPDS extent competed: not available for competition, not competed, not
# competed under simplified acquisition, non-competitive delivery order.
SOLE_SOURCE_CODES: Final = frozenset({"B", "C", "G", "NDO"})
SOLE_SOURCE_THRESHOLD: Final = 50_000_000.0

PAGE_SIZE: Final = 100
MAX_SEARCH_PAGES: Final = 5
LOOKBACK_DAYS: Final = 30

AWARD_TIER: Final = "tier_1_primary"
DOCKET_TIER: Final = "tier_1_primary"
KIND_AWARD: Final = "usaspending_award"
KIND_GAO: Final = "gao_decision"
KIND_DODIG: Final = "dodig_report"

SOLE_SOURCE_SCORE: Final = 60.0
LARGE_AWARD: Final = 500_000_000.0
LARGE_AWARD_POINTS: Final = 10.0
PROTEST_SCORE: Final = 65.0
SUSTAINED_PROTEST_POINTS: Final = 25.0
AUDIT_POINTS: Final = 5.0

_GAO_DOCKET: Final = re.compile(r"^B-\d{6}(?:\.\d+)*$")
_DODIG_DOCKET: Final = re.compile(r"^DODIG-\d{4}-\d{3}$")
_SUSTAINED: Final = re.compile(
    r"\b(?:we sustain the protests?|the protests? (?:is|are) sustained|"
    r"protests? sustained|we therefore sustain)\b",
    re.IGNORECASE,
)
_OUTCOMES: Final = frozenset({"sustained", "denied", "dismissed"})


# --- Dockets ------------------------------------------------------------------


class DocketRejected(ValueError):
    """A docket line that is not an official, well-formed record."""


@dataclass(frozen=True, slots=True)
class Docket:
    """A GAO decision or DoD IG report an editor linked to an award.

    Every text field is a claim until the evidence bar finds it, verbatim, in
    the sealed document.
    """

    kind: str  # KIND_GAO or KIND_DODIG
    number: str  # B-423456.2 or DODIG-2026-045
    url: str
    award_id: str
    issued: date
    finding: str
    outcome: str = ""  # GAO only: sustained, denied, dismissed
    awardee_as_named: str = ""  # GAO only

    @property
    def sustained(self) -> bool:
        return self.kind == KIND_GAO and self.outcome == "sustained"

    @property
    def counts(self) -> bool:
        """Whether this docket is sealed: sustained protests and IG audits."""
        return self.sustained or self.kind == KIND_DODIG

    @classmethod
    def from_dict(cls, row: Any) -> Docket:
        if not isinstance(row, dict):
            raise DocketRejected("docket is not an object")
        kind = {"gao_protest": KIND_GAO, "dodig_audit": KIND_DODIG}.get(
            str(row.get("kind"))
        )
        if kind is None:
            raise DocketRejected(f"unknown docket kind {row.get('kind')!r}")
        number = str(row.get("docket") or "").strip()
        url = str(row.get("url") or "").strip()
        host = host_of(url).lower()
        if kind == KIND_GAO:
            if not _GAO_DOCKET.match(number):
                raise DocketRejected(f"not a GAO B-number: {number!r}")
            if host not in GAO_HOSTS or not url.startswith("https://"):
                raise DocketRejected(f"{number}: not an official gao.gov URL: {url!r}")
        else:
            if not _DODIG_DOCKET.match(number):
                raise DocketRejected(f"not a DoD IG report number: {number!r}")
            if host not in DODIG_HOSTS or not url.startswith("https://"):
                raise DocketRejected(f"{number}: not an official DoD IG URL: {url!r}")
        award_id = str(row.get("award_id") or "").strip()
        if not award_id:
            raise DocketRejected(f"{number}: names no USAspending award")
        try:
            issued = date.fromisoformat(str(row.get("issued") or ""))
        except ValueError as error:
            raise DocketRejected(f"{number}: no issue date") from error
        finding = normalize_whitespace(str(row.get("finding") or ""))
        if not finding:
            raise DocketRejected(f"{number}: no verbatim finding")
        outcome = str(row.get("outcome") or "").strip().casefold()
        awardee = normalize_whitespace(str(row.get("awardee_as_named") or ""))
        if kind == KIND_GAO:
            if outcome not in _OUTCOMES:
                raise DocketRejected(f"{number}: unknown outcome {outcome!r}")
            if not awardee:
                raise DocketRejected(f"{number}: does not say how it names the awardee")
        return cls(kind, number, url, award_id, issued, finding, outcome, awardee)


def load_dockets_jsonl(path: Path) -> tuple[list[Docket], list[str]]:
    """Read dockets, one JSON object per line: (accepted, rejection reasons)."""
    dockets: list[Docket] = []
    rejected: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                dockets.append(Docket.from_dict(json.loads(line)))
            except (ValueError, DocketRejected) as error:
                rejected.append(f"line {number}: {error}")
    return dockets, rejected


# --- Awards -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AwardFacts:
    """What a USAspending award record says, read without inference."""

    award_id: str
    piid: str
    recipient: str
    total_obligation: float
    extent_competed: str
    extent_competed_description: str
    pricing: str
    agency: str
    agency_code: str
    subagency: str
    date_signed: date | None

    @property
    def is_dod(self) -> bool:
        return self.agency == DOD_NAME and self.agency_code == DOD_TOPTIER_CODE

    @property
    def sole_source(self) -> bool:
        return self.extent_competed in SOLE_SOURCE_CODES

    @property
    def major_sole_source(self) -> bool:
        return (
            self.is_dod
            and self.sole_source
            and self.total_obligation >= SOLE_SOURCE_THRESHOLD
        )


def award_url(award_id: str) -> str:
    return f"{USASPENDING_AWARD}{quote(award_id, safe='')}/"


def _dig(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _string(value: Any) -> str:
    return normalize_whitespace(str(value)) if value is not None else ""


def award_facts(detail: Any, award_id: str | None = None) -> AwardFacts | None:
    """Read an award record, or None if it is not a usable contract record.

    The agency is DoD only when the record says so by name and toptier code.
    """
    if not isinstance(detail, dict):
        return None
    generated = _string(detail.get("generated_unique_award_id"))
    if not generated or (award_id is not None and generated != award_id):
        return None
    if _string(detail.get("category")).casefold() != "contract":
        return None
    amount = detail.get("total_obligation")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return None
    recipient = _string(_dig(detail, "recipient", "recipient_name"))
    if not recipient:
        return None
    toptier = _dig(detail, "awarding_agency", "toptier_agency") or {}
    name = _string(_dig(toptier, "name"))
    code = _string(_dig(toptier, "code"))
    contract = detail.get("latest_transaction_contract_data") or {}
    try:
        signed = date.fromisoformat(_string(detail.get("date_signed"))[:10])
    except ValueError:
        signed = None
    return AwardFacts(
        award_id=generated,
        piid=_string(detail.get("piid")),
        recipient=recipient,
        total_obligation=float(amount),
        extent_competed=_string(_dig(contract, "extent_competed")).upper(),
        extent_competed_description=_string(
            _dig(contract, "extent_competed_description")
        ),
        pricing=_string(_dig(contract, "type_of_contract_pricing_description")),
        agency=name,
        agency_code=code,
        subagency=_string(_dig(detail, "awarding_agency", "subtier_agency", "name")),
        date_signed=signed,
    )


def format_amount(value: float) -> str:
    """75_400_000 -> '75.4 million'; 1_250_000_000 -> '1.25 billion'."""
    if value >= 1e9:
        text, unit = f"{value / 1e9:.2f}", "billion"
    elif value >= 1e6:
        text, unit = f"{value / 1e6:.1f}", "million"
    else:
        return f"{value:,.0f}"
    return f"{text.rstrip('0').rstrip('.')} {unit}"


# --- Evidence bar on sealed payloads ------------------------------------------


def _sealed_text(artifact: SourceArtifact) -> str | None:
    try:
        raw = Path(artifact.raw_payload_path).read_bytes()
    except OSError:
        return None
    return read_artifact_text(raw, artifact.metadata.get("content_type"))


def sealed_award_holds(artifact: SourceArtifact) -> AwardFacts | None:
    """The award as its sealed record shows it, if it matches what was claimed."""
    if artifact.tier != AWARD_TIER or host_of(artifact.origin_url) != USASPENDING_HOST:
        return None
    text = _sealed_text(artifact)
    if text is None:
        return None
    try:
        facts = award_facts(json.loads(text), artifact.metadata.get("award_id"))
    except ValueError:
        return None
    if facts is None or not facts.is_dod:
        return None
    if facts.total_obligation != artifact.metadata.get("total_obligation"):
        return None
    if facts.extent_competed != artifact.metadata.get("extent_competed"):
        return None
    return facts


def sealed_docket_holds(artifact: SourceArtifact) -> bool:
    """A sealed GAO decision or DoD IG report that says what its docket claims."""
    kind = artifact.metadata.get("kind")
    hosts = (
        GAO_HOSTS if kind == KIND_GAO else DODIG_HOSTS if kind == KIND_DODIG else None
    )
    if hosts is None or artifact.tier != DOCKET_TIER:
        return False
    if host_of(artifact.origin_url).lower() not in hosts:
        return False
    text = _sealed_text(artifact)
    if not text:
        return False
    number = str(artifact.metadata.get("docket") or "")
    finding = str(artifact.metadata.get("finding") or "")
    if not number or number.split(".")[0] not in text:
        return False
    if not excerpt_is_verbatim(finding, text):
        return False
    if kind == KIND_GAO:
        awardee = str(artifact.metadata.get("awardee_as_named") or "")
        return bool(
            _SUSTAINED.search(normalize_whitespace(text))
            and awardee
            and excerpt_is_verbatim(awardee, text)
        )
    return True


# --- Adapter ------------------------------------------------------------------


class MazerDefenseAdapter(LazLongAdapter):
    adapter_name = "mazer_defense"
    subject_domain = "security_defense"

    def __init__(
        self,
        dockets: Sequence[Docket] = (),
        *,
        lookback_days: int = LOOKBACK_DAYS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        self._lookback_days = lookback_days
        unique = {(item.kind, item.number, item.award_id): item for item in dockets}
        self._dockets = sorted(unique.values(), key=lambda item: item.number)

    # --- Polling ------------------------------------------------------------

    def _search_body(self, page: int = 1) -> dict[str, Any]:
        today = self._clock().astimezone(UTC).date()
        start = today - timedelta(days=self._lookback_days)
        return {
            "filters": {
                "award_type_codes": list(CONTRACT_TYPE_CODES),
                "agencies": [{"type": "awarding", "tier": "toptier", "name": DOD_NAME}],
                "time_period": [
                    {"start_date": start.isoformat(), "end_date": today.isoformat()}
                ],
                "award_amounts": [{"lower_bound": SOLE_SOURCE_THRESHOLD}],
                "extent_competed_type_codes": sorted(SOLE_SOURCE_CODES),
            },
            "fields": [
                "Award ID",
                "Recipient Name",
                "Award Amount",
                "Awarding Agency",
                "generated_internal_id",
            ],
            "sort": "Award Amount",
            "order": "desc",
            "limit": PAGE_SIZE,
            "page": page,
        }

    def poll_requests(self) -> list[PollRequest]:
        # The POST body travels in params, so the spool keeps it verbatim.
        return [
            PollRequest(
                SEARCH_REQUEST,
                USASPENDING_SEARCH,
                {"body": json.dumps(self._search_body(), sort_keys=True)},
            )
        ]

    def _post(self, url: str, body: dict[str, Any]) -> httpx.Response:
        """POST with the same backoff and failure rules as the base _get."""
        failure = ""
        for attempt in range(self._max_attempts):
            if attempt > 0:
                delay = self._base_delay * 2 ** (attempt - 1)
                self._sleep(min(delay, _MAX_BACKOFF_SECONDS))
            try:
                response = self._http().post(
                    url, json=body, headers=self.headers_for(url)
                )
            except httpx.TransportError as error:
                failure = f"{type(error).__name__}: {error}"
                continue
            if response.status_code in _RETRYABLE_STATUS:
                failure = f"HTTP {response.status_code}"
                continue
            if response.is_error:
                raise PollError(f"HTTP {response.status_code}", retryable=False)
            return response
        raise PollError(failure, retryable=True)

    def _fetch_all_pages(self, request: PollRequest) -> list[dict[str, Any]]:
        if request.name != SEARCH_REQUEST:
            return super()._fetch_all_pages(request)
        try:
            body = json.loads(request.params["body"])
        except (KeyError, ValueError) as error:
            raise PollError(
                f"unreadable search body: {error}", retryable=False
            ) from error
        if not isinstance(body, dict):
            raise PollError("search body is not an object", retryable=False)
        items: list[dict[str, Any]] = []
        for page in range(1, MAX_SEARCH_PAGES + 1):
            response = self._post(request.url, {**body, "page": page})
            try:
                payload = response.json()
            except ValueError as error:
                raise PollError(f"invalid JSON: {error}", retryable=True) from error
            if not isinstance(payload, dict):
                raise PollError("response is not a JSON object", retryable=False)
            problem = error_in_body(payload)
            if problem:
                raise PollError(problem, retryable=True)
            items.extend(self.extract_items(request, payload))
            if _dig(payload, "page_metadata", "hasNext") is not True:
                break
        return items

    def extract_items(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rows = payload.get("results")
        if not isinstance(rows, list):
            return []
        return [
            {"award_id": str(row["generated_internal_id"]).strip()}
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("generated_internal_id"), str)
            and row["generated_internal_id"].strip()
        ]

    def assemble_records(
        self, items: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """One record per award: search hits, plus awards an editor docketed."""
        award_ids = dict.fromkeys(
            [
                *(row["award_id"] for row in items.get(SEARCH_REQUEST, [])),
                *(item.award_id for item in self._dockets),
            ]
        )
        return [
            {
                "kind": "award",
                "award_id": award_id,
                "dockets": [
                    item for item in self._dockets if item.award_id == award_id
                ],
            }
            for award_id in award_ids
        ]

    def enrich_record(self, raw_record: dict[str, Any]) -> dict[str, Any]:
        """Read the award's own USAspending record."""
        award_id = raw_record["award_id"]
        try:
            detail = self._get(award_url(award_id)).json()
        except ValueError as error:
            raise PollError(f"bad award JSON: {error}", retryable=True) from error
        facts = award_facts(detail, award_id)
        if facts is None:
            logger.info(
                "%s: %s is not a usable contract record", self.adapter_name, award_id
            )
        return {**raw_record, "facts": facts}

    # --- Lead definition ----------------------------------------------------

    def record_key(self, raw_record: dict[str, Any]) -> str:
        if raw_record.get("kind") != "award":
            raise ValueError(f"unknown record kind: {raw_record.get('kind')!r}")
        counted = sorted(item.number for item in raw_record["dockets"] if item.counts)
        key = f"usaspending:{raw_record['award_id']}"
        return f"{key}:{'+'.join(counted)}" if counted else key

    def record_timestamp(self, raw_record: dict[str, Any]) -> datetime:
        facts: AwardFacts | None = raw_record.get("facts")
        if facts is None:
            raise ValueError("award has not been read from USAspending")
        days = [item.issued for item in raw_record["dockets"] if item.counts]
        if facts.date_signed is not None:
            days.append(facts.date_signed)
        if not days:
            raise ValueError(f"{facts.award_id} has no signed date")
        latest = max(days)
        return datetime(latest.year, latest.month, latest.day, tzinfo=UTC)

    @staticmethod
    def _sustained(record: dict[str, Any]) -> list[Docket]:
        return [item for item in record["dockets"] if item.sustained]

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        facts: AwardFacts | None = raw_record.get("facts")
        if facts is None or not facts.is_dod:
            return 0.0
        sustained = self._sustained(raw_record)
        if facts.major_sole_source:
            score = SOLE_SOURCE_SCORE
            if facts.total_obligation >= LARGE_AWARD:
                score += LARGE_AWARD_POINTS
            if sustained:
                score += SUSTAINED_PROTEST_POINTS
        elif sustained:
            score = PROTEST_SCORE
        else:
            return 0.0
        if any(item.kind == KIND_DODIG for item in raw_record["dockets"]):
            score += AUDIT_POINTS
        return clamp_score(score)

    def evidence_targets(self, raw_record: dict[str, Any]) -> list[EvidenceTarget]:
        facts: AwardFacts | None = raw_record.get("facts")
        if facts is None:
            return []
        targets = [
            EvidenceTarget(
                url=award_url(facts.award_id),
                tier=AWARD_TIER,
                kind=KIND_AWARD,
                excerpt=facts.recipient,
                metadata={
                    "award_id": facts.award_id,
                    "total_obligation": facts.total_obligation,
                    "extent_competed": facts.extent_competed,
                    "major_sole_source": facts.major_sole_source,
                },
            )
        ]
        targets.extend(
            EvidenceTarget(
                url=item.url,
                tier=DOCKET_TIER,
                kind=item.kind,
                excerpt=item.finding,
                metadata={
                    "docket": item.number,
                    "award_id": item.award_id,
                    "finding": item.finding,
                    "outcome": item.outcome,
                    "awardee_as_named": item.awardee_as_named,
                    "issued": item.issued.isoformat(),
                },
            )
            for item in raw_record["dockets"]
            if item.counts
        )
        return targets

    def verify_evidence_bar(self, artifacts: list[SourceArtifact]) -> bool:
        """Official records only, each re-read and holding what it claims.

        Any artifact from another host, or a docket whose sealed text does not
        hold, fails the whole lead rather than being dropped from it.
        """
        awards = [item for item in artifacts if item.metadata.get("kind") == KIND_AWARD]
        dockets = [
            item for item in artifacts if item.metadata.get("kind") != KIND_AWARD
        ]
        if len(awards) != 1:
            return False
        facts = sealed_award_holds(awards[0])
        if facts is None:
            return False
        if not all(
            sealed_docket_holds(item)
            and item.metadata.get("award_id") == facts.award_id
            for item in dockets
        ):
            return False
        sustained = any(item.metadata.get("kind") == KIND_GAO for item in dockets)
        return facts.major_sole_source or sustained

    # --- Public text --------------------------------------------------------

    def public_interest_for(self, record: dict[str, Any]) -> tuple[str, str]:
        """Only this adapter's wording, never text carried in on a record."""
        return (
            self.default_public_question(record),
            self.default_public_interest_reason(record),
        )

    def default_public_question(self, record: dict[str, Any]) -> str:
        facts = self._facts(record)
        return (
            "What do federal procurement and audit records show regarding the "
            f"${format_amount(facts.total_obligation)} DoD award to {facts.recipient}?"
        )

    def default_public_interest_reason(self, record: dict[str, Any]) -> str:
        facts = self._facts(record)
        amount = format_amount(facts.total_obligation)
        sustained = self._sustained(record)
        if facts.major_sole_source:
            first = f"USAspending dockets confirm a ${amount} sole-source obligation"
        else:
            first = f"USAspending dockets confirm a ${amount} DoD obligation"
        if sustained:
            decisions = "; ".join(
                f'{item.number}: "{item.finding}"' for item in sustained
            )
            return (
                f"{first}; GAO bid protest dockets document a sustained protest "
                f"({decisions})."
            )
        return f"{first}; no sustained GAO bid protest has been linked to it."

    def _facts(self, record: dict[str, Any]) -> AwardFacts:
        facts: AwardFacts | None = record.get("facts")
        if facts is None or not facts.is_dod:
            raise ValueError("record is not a DoD award read from USAspending")
        if not (facts.major_sole_source or self._sustained(record)):
            raise ValueError("award is neither a major sole-source award nor protested")
        return facts

    def receipt_details(self, record: dict[str, Any]) -> dict[str, Any]:
        facts = self._facts(record)
        return {
            "detector": self.adapter_name,
            "award_id": facts.award_id,
            "piid": facts.piid,
            "recipient": facts.recipient,
            "awarding_subagency": facts.subagency,
            "total_obligation": facts.total_obligation,
            "extent_competed": (
                f"{facts.extent_competed} ({facts.extent_competed_description})"
            ),
            "pricing": facts.pricing,
            "major_sole_source": facts.major_sole_source,
            "sole_source_threshold": SOLE_SOURCE_THRESHOLD,
            "dockets": [
                {
                    "docket": item.number,
                    "kind": item.kind,
                    "outcome": item.outcome or None,
                    "issued": item.issued.isoformat(),
                    "finding": item.finding,
                    "sealed": item.counts,
                    "linked_by": "editor",
                }
                for item in record["dockets"]
            ],
            "not_checked": (
                "cost-plus ceiling growth: no keyless source reports "
                "per-modification ceilings"
            ),
        }

    def build_paste_ready_prompt(
        self, record: dict[str, Any], artifacts: list[SourceArtifact]
    ) -> str:
        facts = self._facts(record)
        dockets = record["dockets"]
        return render_prompt(
            f"Defense procurement: ${format_amount(facts.total_obligation)} award to "
            f"{facts.recipient}",
            {
                "Award": f"{facts.piid or facts.award_id} ({facts.subagency or DOD_NAME})",
                "Total obligation": f"${facts.total_obligation:,.2f}",
                "Extent competed": (
                    f"{facts.extent_competed} ({facts.extent_competed_description})"
                ),
                "Pricing": facts.pricing or "not reported",
                "Sustained GAO protests": "; ".join(
                    f"{item.number} ({item.issued})"
                    for item in dockets
                    if item.sustained
                )
                or "none linked",
                "DoD IG reports": "; ".join(
                    f"{item.number} ({item.issued})"
                    for item in dockets
                    if item.kind == KIND_DODIG
                )
                or "none linked",
            },
            artifacts,
            "From the sealed records alone, say what was awarded, to whom, for how "
            "much, and on what competitive basis, quoting any GAO or DoD IG finding. "
            "A sole-source award is lawful when justified; do not imply wrongdoing "
            "the records do not state. Name the justification if the records give it.",
        )
