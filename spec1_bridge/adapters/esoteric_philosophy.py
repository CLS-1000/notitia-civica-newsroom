"""Gnosis Mystic (Engine 11): esoteric movements, their lineages, and the record.

A movement is given by its name, its state, and the URL of its own public
description of itself. That page is sealed, and its text is the only basis for
mapping the movement to a historical tradition: a lineage is named when the
movement's own words use at least two of that tradition's distinctive terms.
Each tradition's period comes from LINEAGES, a short reference table that cites
the dated texts it rests on, for an editor to check.

The movement's public record is then read: the Oregon Secretary of State
business registry (data.oregon.gov), IRS exempt-organization data and Form 990
financials (ProPublica Nonprofit Explorer), and, with COURTLISTENER_API_TOKEN
set, federal court dockets that name the organization as a party.

The ethical line, as this engine draws it: it never characterizes a faith
community. It reports what the records show, and the receipt lists the checks
that raise questions for an editor:

- no state registration, or no IRS exemption record;
- officer compensation above a quarter of revenue in the latest Form 990;
- liabilities above assets;
- filings that stopped more than three years ago, except for churches, which
  the law does not require to file;
- court dockets naming the organization.

These are prompts for reporting, not findings. Leaders are never searched by
name, since a name match is not a person match. Public filings cannot show the
absence of coercion or of undisclosed assets, so no lead says they do.

Indigenous entheogenic traditions are noted but never mapped as a lineage:
whether a movement's practice descends from one is for that community to say.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final
from urllib.parse import urlencode

from spec1_bridge.adapters.base_adapter import (
    BaseAdapter,
    EvidenceTarget,
    PollError,
    PollRequest,
    host_of,
    normalize_whitespace,
    read_artifact_text,
    render_prompt,
)
from spec1_bridge.contracts import SourceArtifact

OREGON_REGISTRY: Final = "https://data.oregon.gov/resource/tckn-sxa6.json"
PROPUBLICA_SEARCH: Final = (
    "https://projects.propublica.org/nonprofits/api/v2/search.json"
)
PROPUBLICA_ORG: Final = (
    "https://projects.propublica.org/nonprofits/api/v2/organizations"
)
COURTLISTENER_SEARCH: Final = "https://www.courtlistener.com/api/rest/v4/search/"
COURTLISTENER_HOST: Final = "www.courtlistener.com"
TOKEN_ENV: Final = "COURTLISTENER_API_TOKEN"

SELF_DESCRIPTION_TIER: Final = "tier_3_secondary"
REGISTRY_TIER: Final = "tier_1_primary"
FILINGS_TIER: Final = "tier_2_institutional"

MIN_LINEAGE_TERMS: Final = 2
MAX_OFFICER_SHARE: Final = 0.25
STALE_FILING_YEARS: Final = 3
CHURCH_FOUNDATION_CODE: Final = 10  # IRS: a church, not required to file a 990
CHARITY_SUBSECTION: Final = 3  # 501(c)(3)

LINEAGE_SCORE: Final = 50.0
REVIEW_SCORE: Final = 65.0


@dataclass(frozen=True, slots=True)
class Lineage:
    name: str
    period: str
    reference: str
    terms: tuple[str, ...]


LINEAGES: Final = (
    Lineage(
        "Gnostic",
        "the 4th century CE",
        "the Nag Hammadi codices, 4th-century Coptic manuscripts found in 1945",
        (
            "gnosis",
            "gnostic",
            "pleroma",
            "demiurge",
            "aeon",
            "archon",
            "nag hammadi",
            "pistis sophia",
            "valentinian",
        ),
    ),
    Lineage(
        "Hermetic",
        "the 2nd and 3rd centuries CE",
        "the Greek Corpus Hermeticum, generally dated to the 2nd and 3rd centuries CE",
        (
            "hermetic",
            "hermeticism",
            "hermes trismegistus",
            "corpus hermeticum",
            "emerald tablet",
            "poimandres",
            "as above, so below",
            "kybalion",
        ),
    ),
    Lineage(
        "Rosicrucian",
        "1614",
        "the Fama Fraternitatis, first printed at Kassel in 1614",
        (
            "rosicrucian",
            "rosy cross",
            "rose cross",
            "christian rosenkreuz",
            "fama fraternitatis",
            "confessio fraternitatis",
            "chymical wedding",
        ),
    ),
)
_ENTHEOGENIC: Final = re.compile(
    r"\b(?:ayahuasca|peyote|psilocybin|san pedro|huachuma|iboga|entheogen\w*|"
    r"plant medicine|sacred medicine ceremon\w*)\b",
    re.IGNORECASE,
)


# --- Inputs and matching ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Movement:
    """A movement as the desk names it, with its own public self-description."""

    name: str
    region: str  # a city or area, for the public question
    state: str  # two-letter
    description_url: str

    @property
    def key(self) -> str:
        flat = re.sub(r"[^a-z0-9]+", "-", self.name.casefold()).strip("-")
        return f"{self.state.upper()}:{flat}"


@dataclass(frozen=True, slots=True)
class LineageMatch:
    lineage: Lineage
    terms: tuple[str, ...]
    excerpt: str


def _excerpt(text: str, term: str) -> str:
    flat = normalize_whitespace(text)
    found = flat.casefold().find(term)
    if found == -1:
        return ""
    low, high = max(0, found - 120), min(len(flat), found + len(term) + 120)
    while low > 0 and not flat[low - 1].isspace():
        low -= 1
    while high < len(flat) and not flat[high].isspace():
        high += 1
    return flat[low:high].strip()


def match_lineages(text: str) -> list[LineageMatch]:
    """Traditions whose distinctive terms the text uses at least twice over.

    Terms are matched as whole words, so "aeon" is not found in "paeony".
    """
    folded = normalize_whitespace(text).casefold()
    matches = []
    for lineage in LINEAGES:
        terms = tuple(
            term
            for term in lineage.terms
            if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", folded)
        )
        if len(terms) >= MIN_LINEAGE_TERMS:
            matches.append(LineageMatch(lineage, terms, _excerpt(text, terms[0])))
    return sorted(matches, key=lambda item: (-len(item.terms), item.lineage.name))


def mentions_entheogens(text: str) -> bool:
    return _ENTHEOGENIC.search(text) is not None


# --- Public record ----------------------------------------------------------------


def _upper_name(name: str) -> str:
    return normalize_whitespace(name).upper().replace("'", "''")


def registry_request(movement: Movement) -> PollRequest:
    return PollRequest(
        f"registry:{movement.key}",
        OREGON_REGISTRY,
        {
            "$where": f"business_name = '{_upper_name(movement.name)}'",
            "$limit": "20",
        },
    )


def registry_url(movement: Movement) -> str:
    request = registry_request(movement)
    return f"{request.url}?{urlencode(request.params)}"


def irs_request(movement: Movement) -> PollRequest:
    return PollRequest(
        f"irs:{movement.key}",
        PROPUBLICA_SEARCH,
        {"q": movement.name, "state[id]": movement.state.upper()},
    )


def court_request(movement: Movement) -> PollRequest:
    return PollRequest(
        f"courts:{movement.key}",
        COURTLISTENER_SEARCH,
        {"q": f'"{normalize_whitespace(movement.name)}"', "type": "r"},
    )


def _same_name(left: str, right: str) -> bool:
    def flat(text: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())

    return flat(left) == flat(right)


@dataclass(frozen=True, slots=True)
class Registration:
    registry_number: str
    entity_type: str
    registered: str

    @property
    def nonprofit(self) -> bool:
        return "NONPROFIT" in self.entity_type.upper()


def registration(movement: Movement, rows: list[dict[str, Any]]) -> Registration | None:
    for row in rows:
        if _same_name(str(row.get("business_name") or ""), movement.name):
            return Registration(
                registry_number=str(row.get("registry_number") or ""),
                entity_type=str(row.get("entity_type") or ""),
                registered=str(row.get("registry_date") or "")[:10],
            )
    return None


def irs_match(movement: Movement, rows: list[dict[str, Any]]) -> int | None:
    """The EIN of the exempt organization with exactly the movement's name."""
    for row in rows:
        if (
            _same_name(str(row.get("name") or ""), movement.name)
            and str(row.get("ein") or "").isdigit()
        ):
            return int(row["ein"])
    return None


@dataclass(frozen=True, slots=True)
class Finances:
    ein: int
    subsection: int | None
    church: bool
    latest_year: int | None
    revenue: float | None
    assets: float | None
    liabilities: float | None
    officer_share: float | None


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def finances(payload: Any) -> Finances:
    """What ProPublica's IRS data says about an organization and its latest 990."""
    if not isinstance(payload, dict) or not isinstance(
        payload.get("organization"), dict
    ):
        # Malformed upstream data, retried like any bad payload.
        raise ValueError("no organization in the response")  # noqa: TRY004
    org = payload["organization"]
    filings = [
        item
        for item in payload.get("filings_with_data") or []
        if isinstance(item, dict) and isinstance(item.get("tax_prd_yr"), int)
    ]
    latest = max(filings, key=lambda item: item["tax_prd_yr"], default=None)
    share = None
    if latest is not None:
        revenue = _number(latest.get("totrevenue"))
        paid = _number(latest.get("compnsatncurrofcr"))
        if revenue and revenue > 0 and paid is not None:
            share = paid / revenue
    return Finances(
        ein=int(org.get("ein") or 0),
        subsection=org.get("subsection_code")
        if isinstance(org.get("subsection_code"), int)
        else None,
        church=org.get("foundation_code") == CHURCH_FOUNDATION_CODE,
        latest_year=latest["tax_prd_yr"] if latest else None,
        revenue=_number(latest.get("totrevenue")) if latest else None,
        assets=_number(latest.get("totassetsend")) if latest else None,
        liabilities=_number(latest.get("totliabend")) if latest else None,
        officer_share=share,
    )


def review_questions(
    registered: Registration | None,
    money: Finances | None,
    dockets: int | None,
    year: int,
    registry_checked: bool = True,
) -> list[str]:
    """Questions the public record raises, in plain words. None is a finding."""
    found = []
    if registry_checked and registered is None:
        found.append("no Oregon business registration under this exact name")
    elif registry_checked and registered is not None and not registered.nonprofit:
        found.append(f"registered in Oregon as a {registered.entity_type.lower()}")
    if money is None:
        found.append("no IRS exempt-organization record under this exact name")
    else:
        if money.subsection != CHARITY_SUBSECTION:
            found.append("IRS record is not a 501(c)(3) organization")
        if money.officer_share is not None and money.officer_share > MAX_OFFICER_SHARE:
            found.append(
                f"officer compensation was {money.officer_share:.0%} of revenue in "
                f"the {money.latest_year} Form 990"
            )
        if (
            money.assets is not None
            and money.liabilities is not None
            and money.liabilities > money.assets
        ):
            found.append(
                f"liabilities exceeded assets in the {money.latest_year} Form 990"
            )
        if not money.church and (
            money.latest_year is None or year - money.latest_year > STALE_FILING_YEARS
        ):
            found.append("no Form 990 data for more than three years")
    if dockets:
        found.append(f"{dockets} federal court docket(s) name the organization")
    return found


# --- Adapter --------------------------------------------------------------------------


class EsotericPhilosophyAdapter(BaseAdapter):
    adapter_name = "gnosis_mystic"
    subject_domain = "esoteric_philosophy"

    def __init__(
        self,
        movements: list[Movement],
        courtlistener_token: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._movements = {item.key: item for item in movements}
        self._token = courtlistener_token or os.environ.get(TOKEN_ENV) or None

    @property
    def checks_courts(self) -> bool:
        return self._token is not None

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self._token}"} if self._token else {}

    def headers_for(self, url: str) -> dict[str, str]:
        # The token is for CourtListener alone.
        return self.auth_headers() if host_of(url) == COURTLISTENER_HOST else {}

    def replays(self, request: PollRequest) -> bool:
        return self.checks_courts or host_of(request.url) != COURTLISTENER_HOST

    def poll_requests(self) -> list[PollRequest]:
        requests = []
        for movement in self._movements.values():
            if _in_oregon(movement):
                requests.append(registry_request(movement))
            requests.append(irs_request(movement))
            if self.checks_courts:
                requests.append(court_request(movement))
        return requests

    def _fetch_all_pages(self, request: PollRequest) -> list[dict[str, Any]]:
        # Socrata answers with a JSON array, which the base reader refuses.
        if request.url != OREGON_REGISTRY:
            return super()._fetch_all_pages(request)
        try:
            rows = self._get(request.url, request.params).json()
        except ValueError as error:
            raise PollError(f"invalid JSON: {error}", retryable=True) from error
        if not isinstance(rows, list):
            raise PollError("registry response is not a list", retryable=False)
        return [row for row in rows if isinstance(row, dict)]

    def extract_items(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if request.url == PROPUBLICA_SEARCH:
            rows = payload.get("organizations", [])
            return (
                [row for row in rows if isinstance(row, dict)]
                if isinstance(rows, list)
                else []
            )
        if request.url == COURTLISTENER_SEARCH:
            return [{"count": payload.get("count")}]
        return []

    def assemble_records(
        self, items: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        records = []
        for movement in self._movements.values():
            courts = items.get(f"courts:{movement.key}") if self.checks_courts else None
            count = courts[0].get("count") if courts else None
            records.append(
                {
                    "kind": "movement",
                    "movement": movement,
                    "registration": registration(
                        movement, items.get(f"registry:{movement.key}", [])
                    ),
                    "ein": irs_match(movement, items.get(f"irs:{movement.key}", [])),
                    "dockets": int(count) if isinstance(count, int) else None,
                }
            )
        return records

    def enrich_record(self, raw_record: dict[str, Any]) -> dict[str, Any]:
        movement: Movement = raw_record["movement"]
        response = self._get(movement.description_url)
        text = read_artifact_text(
            response.content, response.headers.get("content-type")
        )
        matches = match_lineages(text or "")
        enriched = {
            **raw_record,
            "lineages": matches,
            "entheogenic": mentions_entheogens(text or ""),
            "finances": None,
        }
        if not matches:
            return enriched
        if raw_record["ein"] is not None:
            try:
                payload = self._get(f"{PROPUBLICA_ORG}/{raw_record['ein']}.json").json()
                enriched["finances"] = finances(payload)
            except ValueError as error:
                raise PollError(f"bad IRS data: {error}", retryable=True) from error
        enriched["questions"] = review_questions(
            raw_record["registration"],
            enriched["finances"],
            raw_record["dockets"],
            self._clock().year,
            registry_checked=_in_oregon(movement),
        )
        return enriched

    def record_key(self, raw_record: dict[str, Any]) -> str:
        movement: Movement = raw_record["movement"]
        return movement.key

    def record_timestamp(self, raw_record: dict[str, Any]) -> datetime:
        # A lineage has no event date; the lead is as fresh as this reading.
        return self._clock()

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        if not raw_record.get("lineages"):
            return 0.0
        return REVIEW_SCORE if raw_record.get("questions") else LINEAGE_SCORE

    def evidence_targets(self, raw_record: dict[str, Any]) -> list[EvidenceTarget]:
        movement: Movement = raw_record["movement"]
        first: LineageMatch = raw_record["lineages"][0]
        targets = [
            EvidenceTarget(
                url=movement.description_url,
                tier=SELF_DESCRIPTION_TIER,
                kind="self_description",
                excerpt=first.excerpt or None,
                metadata={"lineage": first.lineage.name},
            )
        ]
        if raw_record["registration"] is not None:
            targets.append(
                EvidenceTarget(
                    url=registry_url(movement),
                    tier=REGISTRY_TIER,
                    kind="state_registry",
                    metadata={
                        "registry_number": raw_record["registration"].registry_number
                    },
                )
            )
        if raw_record["finances"] is not None:
            targets.append(
                EvidenceTarget(
                    url=f"{PROPUBLICA_ORG}/{raw_record['ein']}.json",
                    tier=FILINGS_TIER,
                    kind="irs_filings",
                    metadata={"ein": raw_record["ein"]},
                )
            )
        return targets

    def verify_evidence_bar(self, artifacts: list[SourceArtifact]) -> bool:
        """The movement's own words, and at least one public record of it."""
        kinds = {item.metadata.get("kind") for item in artifacts}
        return "self_description" in kinds and bool(
            kinds & {"state_registry", "irs_filings"}
        )

    def receipt_details(self, record: dict[str, Any]) -> dict[str, Any]:
        registered: Registration | None = record["registration"]
        money: Finances | None = record["finances"]
        return {
            "detector": self.adapter_name,
            "lineages": [
                {
                    "lineage": item.lineage.name,
                    "period": item.lineage.period,
                    "reference": item.lineage.reference,
                    "terms_in_self_description": list(item.terms),
                    "excerpt": item.excerpt,
                }
                for item in record["lineages"]
            ],
            "indigenous_entheogenic_practice_mentioned": record["entheogenic"],
            "state_registration": (
                {
                    "registry_number": registered.registry_number,
                    "entity_type": registered.entity_type,
                    "registered": registered.registered,
                }
                if registered
                else None
            ),
            "irs": (
                {
                    "ein": money.ein,
                    "subsection": money.subsection,
                    "church": money.church,
                    "latest_990_year": money.latest_year,
                    "revenue": money.revenue,
                    "assets": money.assets,
                    "liabilities": money.liabilities,
                    "officer_compensation_share": (
                        round(money.officer_share, 4)
                        if money.officer_share is not None
                        else None
                    ),
                }
                if money
                else None
            ),
            "state_registry_checked": _in_oregon(record["movement"]),
            "court_dockets_naming_organization": record["dockets"],
            "courts_checked": self.checks_courts,
            "questions_for_review": record["questions"],
            "not_searched": "Leaders' names, which can match other people.",
            "cannot_show": (
                "Shared vocabulary shows how a movement describes itself, not that "
                "it descends from a tradition. Public filings cannot show the "
                "absence of coercion or of undisclosed assets. Nothing here "
                "characterizes the community or its members."
            ),
        }

    def default_public_question(self, record: dict[str, Any]) -> str:
        movement: Movement = record["movement"]
        return (
            "What historical esoteric tradition or philosophical lineage informs the "
            f"{movement.name} in {movement.region}?"
        )

    def default_public_interest_reason(self, record: dict[str, Any]) -> str:
        first: LineageMatch = record["lineages"][0]
        roots = (
            f"Its own published teachings use the vocabulary of the {first.lineage.name} "
            f"tradition, whose texts date to {first.lineage.period}"
        )
        if record["questions"]:
            return (
                f"{roots}; its public filings raise questions listed for editorial "
                "review."
            )
        courts = (
            ""
            if self.checks_courts
            else ", though federal court dockets were not checked"
        )
        records = (
            "Oregon and IRS records list"
            if _in_oregon(record["movement"])
            else ("IRS records list")
        )
        return (
            f"{roots}; {records} it as a registered nonprofit, and its filings show "
            f"none of the financial warning signs this desk checks{courts}."
        )

    def build_paste_ready_prompt(
        self, record: dict[str, Any], artifacts: list[SourceArtifact]
    ) -> str:
        details = self.receipt_details(record)
        return render_prompt(
            f"Esoteric movement: {record['movement'].name} ({record['movement'].region})",
            {
                "Lineage vocabulary": "; ".join(
                    f"{item['lineage']} ({', '.join(item['terms_in_self_description'])})"
                    for item in details["lineages"]
                ),
                "Reference for each period": "; ".join(
                    item["reference"] for item in details["lineages"]
                ),
                "State registration": details["state_registration"] or "none found",
                "IRS record": details["irs"] or "none found",
                "Court dockets naming it": (
                    details["court_dockets_naming_organization"]
                    if details["courts_checked"]
                    else "not checked"
                ),
                "Questions for review": "; ".join(details["questions_for_review"])
                or "none",
            },
            artifacts,
            "Describe the tradition the movement's own words draw on, and what its "
            "public records show, quoting the sources. Treat members and their faith "
            "with respect. Do not call the movement a cult or say it is free of "
            "wrongdoing; say only what the records show.",
        )


def _in_oregon(movement: Movement) -> bool:
    """Only Oregon's registry is read; other states' registries are not."""
    return movement.state.upper() == "OR"
