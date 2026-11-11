"""Ubik Spectral (Engine 9): living folklore and the record beneath it.

Haunting and spectral accounts are what people say they experienced at a place.
This engine asks what the documented history of that place holds. It never
treats an account as proof of anything physical, and it never seals one as
evidence: accounts stay unverified narratives, and only archival records are
sealed.

A site has a sighting pattern when at least two accounts, after commercial
promotion is set aside, name it within a kilometre of each other. For each such
site, Chronicling America (the Library of Congress's digitized historical
newspapers, through the loc.gov API) is searched for the site's name in the
site's state. A page counts only when its full OCR text names the site beside a
fatal fire, explosion, mine collapse, or shipwreck, with no "no one was killed"
before it. The page's full text is sealed, and the passage is its verbatim
excerpt.

Each newspaper is a separate publisher, though the Library of Congress serves
them all, so two newspapers reporting the same disaster are the two independent
sources the core Corroboration Gate asks for. One page is a lead triage rejects.

The reason cites acoustic science as a proposed mechanism, never as a finding
about the site: no site measurement is claimed, because none is available.

Witness dignity: accounts carry no author, are never quoted in the public
question or reason, and are described as accounts, not as hoaxes or proof.
Ghost tours, haunted attractions, and ticketed events are filtered out, since
promotion is not community memory.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Final
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

from spec1_bridge.adapters.base_adapter import (
    BaseAdapter,
    EvidenceTarget,
    PollError,
    PollRequest,
    host_of,
    normalize_whitespace,
    render_prompt,
)
from spec1_bridge.adapters.void_crossjoin import distance_km
from spec1_bridge.contracts import SourceArtifact
from spec1_bridge.sociology_telemetry import DOMAIN_IRRATIONAL

CHRONAM_SEARCH: Final = "https://www.loc.gov/collections/chronicling-america/"
FULLTEXT_SERVICE: Final = "https://tile.loc.gov/text-services/word-coordinates-service"
FULLTEXT_HOST: Final = "tile.loc.gov"
SEARCH_PAGE_SIZE: Final = "50"
MAX_PAGES_CHECKED: Final = 10
MAX_PAGES_SEALED: Final = 3

ARCHIVE_TIER: Final = "tier_2_institutional"
ARCHIVE_KIND: Final = "chronam_page_text"
DOMAIN: Final = DOMAIN_IRRATIONAL

MIN_REPORTS: Final = 2
PATTERN_RADIUS_KM: Final = 1.0
CONTEXT_CHARS: Final = 240
LORE_SCORE: Final = 75.0
SIGNAL: Final = "ARCHIVAL_TRAGEDY_CORRELATION"

INFRASOUND_NOTE: Final = (
    "acoustic science literature has proposed low-frequency infrasound (around "
    "18.9 Hz) from industrial machinery, dams, or ventilation as a physical "
    "explanation for spatial dread and visual anomalies."
)
CAVEAT: Final = (
    "The accounts are what people reported experiencing, not evidence of a "
    "physical event, and a disaster in the archive does not show it caused them. "
    "No acoustic measurement was made at the site; infrasound is a proposed "
    "mechanism, cited from the literature."
)

# Ordered: a mine fire is a mine collapse story before it is a fire story.
_EVENTS: Final = (
    (
        "mine collapse",
        re.compile(r"\b(?:cave-? ?in|caved in|mine (?:collapse|disaster))\b"),
    ),
    ("explosion", re.compile(r"\b(?:explosion|exploded|blew up)\b")),
    (
        "shipwreck",
        re.compile(
            r"\b(?:wreck(?:ed)?|shipwreck|foundered|capsized|went down|ran aground)\b"
        ),
    ),
    ("fire", re.compile(r"\b(?:fire|blaze|conflagration|burned|burning)\b")),
)
_FATAL: Final = re.compile(
    r"\b(?:killed|perished|lost (?:their|his|her) li(?:fe|ves)|burned to death|"
    r"drowned|crushed to death|fatally|bodies|death toll|dead)\b"
)
_NO_DEATHS: Final = re.compile(
    r"\b(?:no one|nobody|none|no lives were|no person|not a (?:man|soul))\b[^.]{0,30}$"
)
_PROMOTION: Final = re.compile(
    r"\b(?:tickets?|admission|book (?:now|online|your|a)|ghost tours?|"
    r"haunted (?:house|attraction|tour|trail|hayride|experience|maze)|"
    r"guided tours?|reservations?|promo code|use code|discount|"
    r"family fun|open nightly|limited spots|gift shop|follow us|subscribe|"
    r"our (?:tour|tours|website|channel|podcast|book))\b|https?://|www\.|\$\s?\d",
    re.IGNORECASE,
)
_COMMERCIAL_HOSTS: Final = frozenset(
    {
        "eventbrite.com",
        "ticketmaster.com",
        "viator.com",
        "tripadvisor.com",
        "groupon.com",
        "fareharbor.com",
        "peek.com",
        "getyourguide.com",
        "airbnb.com",
    }
)
_LCCN: Final = re.compile(r"/data/(sn\d+|[a-z]{1,3}\d{8,10})/", re.IGNORECASE)


# --- Records ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoreReport:
    """One spectral account of a place. It never names who gave it."""

    report_id: str
    text: str
    site: str
    locality: str
    state: str
    latitude: float
    longitude: float
    reported_at: datetime
    source_url: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> LoreReport:
        reported = datetime.fromisoformat(str(row["reported_at"]))
        return cls(
            report_id=str(row["report_id"]),
            text=str(row["text"]),
            site=normalize_whitespace(str(row["site"])),
            locality=normalize_whitespace(str(row.get("locality") or "")),
            state=normalize_whitespace(str(row["state"])),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            reported_at=reported if reported.tzinfo else reported.replace(tzinfo=UTC),
            source_url=str(row.get("source_url") or ""),
        )


@dataclass(frozen=True, slots=True)
class SightingPattern:
    """Accounts of one site, close together on the map."""

    site: str
    locality: str
    state: str
    reports: tuple[LoreReport, ...]
    set_aside: int  # promotional accounts filtered out

    @property
    def key(self) -> str:
        return f"{_site_key(self.state)}:{_site_key(self.site)}"

    @property
    def centroid(self) -> tuple[float, float]:
        return (
            fmean(item.latitude for item in self.reports),
            fmean(item.longitude for item in self.reports),
        )

    @property
    def location(self) -> str:
        state = self.state.title()
        if self.locality:
            return f"{self.site} in {self.locality}, {state}"
        return f"{self.site}, {state}"


@dataclass(frozen=True, slots=True)
class ArchivalPage:
    """A Chronicling America page that names the site beside a fatal disaster."""

    page_url: str
    fulltext_url: str
    lccn: str
    newspaper: str
    published: date
    event: str
    excerpt: str


def load_reports_jsonl(path: Path) -> list[LoreReport]:
    """Read accounts, one JSON object per line. Blank lines are skipped."""
    with path.open(encoding="utf-8") as handle:
        return [
            LoreReport.from_dict(json.loads(line)) for line in handle if line.strip()
        ]


# --- Witness dignity ----------------------------------------------------------


def is_promotional(report: LoreReport) -> bool:
    """Whether an account is selling something: a tour, a ticket, an attraction."""
    host = host_of(report.source_url).lower().removeprefix("www.")
    if any(host == item or host.endswith(f".{item}") for item in _COMMERCIAL_HOSTS):
        return True
    return _PROMOTION.search(report.text) is not None


def _site_key(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def sighting_patterns(reports: Sequence[LoreReport]) -> list[SightingPattern]:
    """Group accounts by site, keep those near the group's centre, and keep
    sites with at least MIN_REPORTS distinct accounts."""
    groups: defaultdict[tuple[str, str], list[LoreReport]] = defaultdict(list)
    set_aside: defaultdict[tuple[str, str], int] = defaultdict(int)
    for report in reports:
        key = (_site_key(report.state), _site_key(report.site))
        if not key[1]:
            continue
        if is_promotional(report):
            set_aside[key] += 1
        else:
            groups[key].append(report)
    patterns = []
    for key, members in groups.items():
        unique = list({item.report_id: item for item in members}.values())
        lat = fmean(item.latitude for item in unique)
        lon = fmean(item.longitude for item in unique)
        near = [
            item
            for item in unique
            if distance_km(lat, lon, item.latitude, item.longitude) <= PATTERN_RADIUS_KM
        ]
        if len(near) < MIN_REPORTS:
            continue
        first = min(near, key=lambda item: item.reported_at)
        patterns.append(
            SightingPattern(
                site=first.site,
                locality=first.locality,
                state=first.state,
                reports=tuple(sorted(near, key=lambda item: item.reported_at)),
                set_aside=set_aside[key],
            )
        )
    return sorted(patterns, key=lambda item: item.key)


# --- Buried historical record -------------------------------------------------


def _first(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def fulltext_url(result: dict[str, Any]) -> str | None:
    """The full-text service URL for a search result's page, from its segment."""
    link = str(result.get("word_coordinates_url") or "")
    segment = parse_qs(urlsplit(link).query).get("segment")
    if not segment:
        return None
    query = {"segment": segment[0], "format": "alto_xml", "full_text": "1"}
    return f"{FULLTEXT_SERVICE}?{urlencode(query)}"


def lccn_of(url: str) -> str | None:
    """The newspaper's Library of Congress control number in a page URL."""
    match = _LCCN.search(unquote(url))
    return match.group(1).lower() if match else None


def page_text(payload: Any) -> str | None:
    """The full OCR text in a full-text service response."""
    if not isinstance(payload, dict):
        return None
    for value in payload.values():
        if isinstance(value, dict) and isinstance(value.get("full_text"), str):
            return str(value["full_text"])
    return None


def fatal_disaster(site: str, text: str) -> tuple[str, str] | None:
    """(event, excerpt) where text names site beside a fatal disaster.

    The excerpt is the passage as printed, whitespace collapsed, so it stays
    verbatim in the sealed page.
    """
    flat = normalize_whitespace(text)
    folded = flat.casefold()
    wanted = normalize_whitespace(site).casefold()
    if not wanted:
        return None
    start = 0
    while (found := folded.find(wanted, start)) != -1:
        start = found + len(wanted)
        low = max(0, found - CONTEXT_CHARS)
        high = min(len(flat), start + CONTEXT_CHARS)
        while low > 0 and not flat[low - 1].isspace():
            low -= 1
        while high < len(flat) and not flat[high].isspace():
            high += 1
        window = folded[low:high]
        fatal = next(
            (
                match
                for match in _FATAL.finditer(window)
                if not _NO_DEATHS.search(window[: match.start()])
            ),
            None,
        )
        if fatal is None:
            continue
        for event, pattern in _EVENTS:
            if pattern.search(window):
                return event, flat[low:high].strip()
    return None


# --- Adapter ------------------------------------------------------------------


class LivingFolkloreAdapter(BaseAdapter):
    adapter_name = "ubik_spectral"
    subject_domain = "anomalous_lore"

    def __init__(self, reports: Sequence[LoreReport], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._patterns = sighting_patterns(reports)

    @property
    def patterns(self) -> list[SightingPattern]:
        return list(self._patterns)

    def poll_requests(self) -> list[PollRequest]:
        return [
            PollRequest(
                f"archive:{pattern.key}",
                CHRONAM_SEARCH,
                {
                    "dl": "page",
                    "ops": "PHRASE",
                    "qs": pattern.site,
                    "searchType": "advanced",
                    "location_state": pattern.state.casefold(),
                    "c": SEARCH_PAGE_SIZE,
                    "fo": "json",
                },
            )
            for pattern in self._patterns
        ]

    def extract_items(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rows = payload.get("results", [])
        return (
            [row for row in rows if isinstance(row, dict)]
            if isinstance(rows, list)
            else []
        )

    def assemble_records(
        self, items: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        records = []
        for pattern in self._patterns:
            rows = items.get(f"archive:{pattern.key}", [])
            if rows:
                records.append(
                    {"kind": "lore_site", "pattern": pattern, "results": rows}
                )
        return records

    def enrich_record(self, raw_record: dict[str, Any]) -> dict[str, Any]:
        """Read each candidate page's full text and keep the fatal disasters."""
        pattern: SightingPattern = raw_record["pattern"]
        wanted = pattern.site.casefold()
        candidates = sorted(
            (
                row
                for row in raw_record["results"]
                if wanted
                in normalize_whitespace(_first(row.get("description"))).casefold()
                or wanted in str(row.get("title", "")).casefold()
            ),
            key=lambda row: str(row.get("date", "")),
        )[:MAX_PAGES_CHECKED]
        pages: list[ArchivalPage] = []
        for row in candidates:
            link = fulltext_url(row)
            published = _iso_date(row.get("date"))
            lccn = lccn_of(link or "")
            if link is None or published is None or lccn is None:
                continue
            try:
                text = page_text(self._get(link).json())
            except ValueError as error:
                raise PollError(f"bad page text: {error}", retryable=True) from error
            found = fatal_disaster(pattern.site, text or "")
            if found is None:
                continue
            pages.append(
                ArchivalPage(
                    page_url=str(row.get("id", "")).replace("http://", "https://", 1),
                    fulltext_url=link,
                    lccn=lccn,
                    newspaper=_first(row.get("partof_title")),
                    published=published,
                    event=found[0],
                    excerpt=found[1],
                )
            )
        return {**raw_record, "pages": _one_event(pages)}

    def record_key(self, raw_record: dict[str, Any]) -> str:
        # One lead per site. The key is read before enrich_record runs, so it
        # cannot depend on the pages, and a known site costs no page fetches.
        pattern: SightingPattern = raw_record["pattern"]
        return pattern.key

    def record_timestamp(self, raw_record: dict[str, Any]) -> datetime:
        pattern: SightingPattern = raw_record["pattern"]
        return pattern.reports[-1].reported_at

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        return LORE_SCORE if raw_record.get("pages") else 0.0

    def evidence_targets(self, raw_record: dict[str, Any]) -> list[EvidenceTarget]:
        return [
            EvidenceTarget(
                url=page.fulltext_url,
                tier=ARCHIVE_TIER,
                kind=ARCHIVE_KIND,
                excerpt=page.excerpt,
                metadata={
                    "lccn": page.lccn,
                    "newspaper": page.newspaper,
                    "published": page.published.isoformat(),
                    "page_url": page.page_url,
                    "event": page.event,
                },
            )
            for page in raw_record["pages"]
        ]

    def verify_evidence_bar(self, artifacts: list[SourceArtifact]) -> bool:
        """A newspaper page's full text, sealed from the Library of Congress."""
        return any(
            item.metadata.get("kind") == ARCHIVE_KIND
            and item.tier == ARCHIVE_TIER
            and host_of(item.origin_url) == FULLTEXT_HOST
            and lccn_of(item.origin_url) == item.metadata.get("lccn")
            for item in artifacts
        )

    def receipt_details(self, record: dict[str, Any]) -> dict[str, Any]:
        pattern: SightingPattern = record["pattern"]
        lat, lon = pattern.centroid
        return {
            "detector": self.adapter_name,
            "signal": SIGNAL,
            "domain": DOMAIN,
            "evidentiary_status": "unverified_narrative",
            "site": pattern.location,
            "accounts": len(pattern.reports),
            "accounts_set_aside_as_promotion": pattern.set_aside,
            "pattern_radius_km": PATTERN_RADIUS_KM,
            "centroid": [round(lat, 5), round(lon, 5)],
            "first_account": pattern.reports[0].reported_at.isoformat(),
            "last_account": pattern.reports[-1].reported_at.isoformat(),
            "archival_pages": [
                {
                    "newspaper": page.newspaper,
                    "lccn": page.lccn,
                    "published": page.published.isoformat(),
                    "event": page.event,
                    "page_url": page.page_url,
                    "excerpt": page.excerpt,
                }
                for page in record["pages"]
            ],
            "infrasound": "cited from the literature as a proposed mechanism; not measured",
            "cannot_show": CAVEAT,
        }

    def default_public_question(self, record: dict[str, Any]) -> str:
        return (
            "What historical event or environmental factor is tied to the spectral "
            f"accounts reported in {record['pattern'].location}?"
        )

    def default_public_interest_reason(self, record: dict[str, Any]) -> str:
        first: ArchivalPage = record["pages"][0]
        return (
            f"Archival records document a fatal {first.event} at this site in "
            f"{first.published.year}; {INFRASOUND_NOTE}"
        )

    def build_paste_ready_prompt(
        self, record: dict[str, Any], artifacts: list[SourceArtifact]
    ) -> str:
        pattern: SightingPattern = record["pattern"]
        pages: list[ArchivalPage] = record["pages"]
        return render_prompt(
            f"Living folklore: spectral accounts of {pattern.location}",
            {
                "Accounts (unverified, no authors)": len(pattern.reports),
                "Promotional accounts set aside": pattern.set_aside,
                "Accounts span": (
                    f"{pattern.reports[0].reported_at.date()} to "
                    f"{pattern.reports[-1].reported_at.date()}"
                ),
                "Archival pages": "; ".join(
                    f"{page.newspaper} ({page.published}): fatal {page.event}"
                    for page in pages
                ),
                "Infrasound": "proposed mechanism from the literature; not measured here",
            },
            artifacts,
            "From the sealed pages alone, say what happened at the site, when, and "
            "to whom, quoting each newspaper. Describe the accounts as accounts, "
            "with respect for the community that keeps them, and do not say the "
            "disaster caused them.",
        )


def _iso_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _one_event(pages: list[ArchivalPage]) -> list[ArchivalPage]:
    """The earliest disaster, as each newspaper first reported it.

    Pages of a different kind of event, or from another year, are a different
    story and are left out, so the sources corroborate one event.
    """
    if not pages:
        return []
    first = pages[0]
    by_paper: dict[str, ArchivalPage] = {}
    for page in pages:
        if page.event != first.event or page.published.year != first.published.year:
            continue
        by_paper.setdefault(page.lccn, page)
    return list(by_paper.values())[:MAX_PAGES_SEALED]
