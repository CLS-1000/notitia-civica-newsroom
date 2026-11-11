"""Importer for CLS-1000/pdx-1i signals.

Reads pdx-1i's ground-truth store (pdx1_signals.jsonl), one
pdx1.models.IntelligenceRecord per line exactly as pdx-1i writes it with
model_dump_json(), seals each record's source URL into custody, and emits
Spec1LeadPacket objects for newsroom triage.

The record fields this reads, from pdx-1i src/pdx1/models.py:

    record_id, source_type, pattern, outcome, priority, tier, anomaly,
    entity_ids, url, published_at, origin

origin is how pdx-1i read the payload: live, cache (last-good after a failed
fetch) or fixture (an invented test payload). Only live and cache are public
records. A record with any other origin, or none (written before pdx-1i stamped
it), is skipped as not_live before anything is fetched.

evidence_url, when pdx-1i sets one, is a machine-readable URL that returns
exactly this record (WA PDC: the receipt's data.wa.gov row). It is what gets
sealed; url, the human link pdx-1i's brief cites, is kept in the metadata.

priority is ELEVATED | STANDARD | MONITOR. tier is pdx-1i's own evidence
grade, HARD_RECORD | REPORTED | INFERRED. anomaly is null unless the record
is a sigma anomaly, and otherwise carries sigma, observed, baseline and tier.

The score is a rule-based screen: a floor from the priority band, raised by
the anomaly's sigma when there is one. INFERRED records never pass the
evidence bar, because pdx-1i derived them by correlation and no filed record
states them. A record whose sealed source has no readable text is skipped,
never given an invented excerpt. Every packet is analysis_tier='rule_based'.

A sealed page that cannot be about its record is skipped as a hollow source.
Such a page holds text, so it would pass the verbatim check, but it does not
state the record. A source is hollow when:

- its URL is a known search form or landing page (checked against the URL asked
  for and the URL the fetch ended at), or it carries a host's search-form marker;
- it is an empty answer (Socrata's `[]` for an unknown id);
- the visible main content of an HTML page, read by spec1_bridge.page_content
  and never the raw markup, shows nothing, shows a not-found or not-released
  notice, is a search form or is a list of links;
- the record's pattern names an identifier (a transaction, receipt or filing
  number, or a bill number) and the page's visible content never states it;
- one page is sealed for two different records in the same run and does not
  state either record's own pattern;
- it is an OLIS page or API answer (oregonlegislature.gov) whose bill title
  ("Relating to ...") or legislative session does not match the record's
  (see olis_mismatch_reason).

Leads from WA PDC sources (pdc.wa.gov) are subject_domain washington_civic;
every other lead is oregon_civic (see subject_domain_for).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx

from newsroom.harlan_eternity.custody import (
    DEFAULT_ARCHIVE_DIR,
    CustodyError,
    fetch_and_seal_source,
)
from spec1_bridge.adapters.base_adapter import (
    clamp_score,
    excerpt_is_verbatim,
    normalize_whitespace,
    read_artifact_text,
    render_prompt,
)
from spec1_bridge.contracts import SourceArtifact, Spec1LeadPacket, public_interest
from spec1_bridge.page_content import (
    PageContent,
    hollow_page_reason,
    looks_like_html,
    read_main_content,
)

logger = logging.getLogger(__name__)

ADAPTER_NAME: Final = "pdx1_importer"
SUBJECT_DOMAIN: Final = "oregon_civic"
WASHINGTON_DOMAIN: Final = "washington_civic"
WASHINGTON_HOSTS: Final = ("pdc.wa.gov",)
ARTIFACT_KIND: Final = "pdx1_record_source"
PDX1_NAMESPACE: Final = uuid.UUID("a3b89f21-7c42-4f1b-8e2a-9e123456789a")

PRIORITY_RANK: Final = {"MONITOR": 0, "STANDARD": 1, "ELEVATED": 2}
# Floors sit either side of mycroft's MIN_ANOMALY_SCORE (40): MONITOR alone does
# not clear triage, STANDARD and ELEVATED do.
PRIORITY_SCORE: Final = {"MONITOR": 30.0, "STANDARD": 45.0, "ELEVATED": 60.0}
# 3 sigma, pdx-1i's publish-eligible TIER_1 band, maps to 60; 5 sigma caps at 100.
POINTS_PER_SIGMA: Final = 20.0
RECORD_TIERS: Final = frozenset({"HARD_RECORD", "REPORTED", "INFERRED"})
PUBLIC_ORIGINS: Final = frozenset({"live", "cache"})
EXCERPT_CHARS: Final = 300
# A text run this long reads as a sentence of the record, not a label or a menu.
MIN_EXCERPT_WORDS: Final = 8

# Pages that list or search filings rather than state one. Seen on Kali
# 2026-11-01: ORESTAR transaction links without a session land on the search
# form; SEI and PDC links in pdx-1i point at landing pages.
LANDING_PAGES: Final = frozenset(
    {
        ("www.oregon.gov", "/ogec/pages/sei-filings.aspx"),
        ("www.pdc.wa.gov", "/browse/campaign-explorer"),
        ("www.pdc.wa.gov", "/political-disclosure-reporting-data"),
    }
)
# Text that only a search form carries, by host.
SEARCH_FORM_MARKERS: Final = {
    "secure.sos.state.or.us": "cneSearchForm",
}

# Identifiers a pdx-1i pattern names its record by: ORE-2026-4417392 (ORESTAR),
# WA-2026-778341 (WA PDC), SEI-2026-00871 (OGEC), and Oregon measure numbers.
RECORD_NUMBER: Final = re.compile(r"\b([A-Z]{2,5})-(\d{4})-(\d{3,})\b")
MEASURE_NUMBER: Final = re.compile(
    r"\b(HB|SB|HJR|SJR|HCR|SCR|HJM|SJM|HM|SM)\s?(\d{1,4})\b"
)
MEASURE_NAMES: Final = {"HB": "House Bill", "SB": "Senate Bill"}

# OLIS: the legislative information system and its OData API.
OLIS_HOST: Final = "oregonlegislature.gov"
# A session key as OLIS writes it in its URLs: 2026R1, 2025S1.
SESSION_KEY: Final = re.compile(r"(?<![0-9A-Za-z])(\d{4}[RS]\d)(?![0-9A-Za-z])")
# A session as pdx-1i's pattern names it: "in the 2026 Regular session",
# "in the 2025 First Special Session".
SESSION_NAME: Final = re.compile(
    r"\b(\d{4})\s+(?:(first|second|third|fourth)\s+)?(regular|special)\s+session\b",
    re.IGNORECASE,
)
SESSION_ORDINALS: Final = {"first": 1, "second": 2, "third": 3, "fourth": 4}
# A "Relating to" clause, ended by the pattern's ", in the <session>", a ";", a
# sentence stop or a closing JSON quote (the API's "RelatingTo" value).
RELATING_TO: Final = re.compile(
    r"relating\s+to\s+(.+?)(?=,\s*in\s+the\s|[;.\"]|$)", re.IGNORECASE
)
# The record's own title: what follows "OLIS record for HB 4092," up to the
# pattern's ", in the <session>" (2026 Regular, 2026R1), or else up to the
# first sentence stop. Live records title a measure by its CatchLine first.
RECORD_TITLE_TO_SESSION: Final = re.compile(
    r"\brecord\s+for\s+[A-Z]{1,4}\s?\d{1,5}\s*,\s*(.+?)"
    r"(?=,\s*in\s+the\s+(?:\d{4}[RS]\d|\d{4}\s+(?:(?:first|second|third|fourth)\s+)?"
    r"(?:regular|special))\b)",
    re.IGNORECASE,
)
RECORD_TITLE_TO_STOP: Final = re.compile(
    r"\brecord\s+for\s+[A-Z]{1,4}\s?\d{1,5}\s*,\s*(.+?)(?=\.(?:\s|$)|$)",
    re.IGNORECASE,
)
SUBJECT_STOPWORDS: Final = frozenset(
    {"a", "an", "and", "as", "by", "for", "in", "of", "on", "or", "the", "to", "with"}
)

# The tier of a source whose content has not passed the hollow-source checks.
UNCHECKED_TIER: Final = "tier_4_unverified"

# Hosts that publish filed public records. Anything else is secondary.
PRIMARY_HOSTS: Final = (
    "secure.sos.state.or.us",  # ORESTAR
    "sos.oregon.gov",
    "oregonlegislature.gov",  # OLIS (olis., api.)
    "apps.oregon.gov",  # SEI
    "oregon.gov",
    "data.wa.gov",  # WA PDC open data
    "pdc.wa.gov",
    "portland.gov",
    "multco.us",
    "clackamas.us",
    "washingtoncountyor.gov",
    "oregonmetro.gov",
    "clark.wa.gov",
)


class Pdx1RecordError(ValueError):
    """A store line is not a usable pdx-1i IntelligenceRecord."""


@dataclass(frozen=True, slots=True)
class Pdx1Record:
    record_id: str
    source_type: str
    pattern: str
    outcome: str
    priority: str
    tier: str
    sigma: float | None
    anomaly: dict[str, Any] | None
    entity_ids: list[str]
    url: str | None
    published_at: datetime
    origin: str | None = None
    evidence_url: str | None = None
    public_question: str = ""
    public_interest_reason: str = ""

    @property
    def source_url(self) -> str | None:
        """The URL to seal: the record-level evidence URL when there is one."""
        return self.evidence_url or self.url


@dataclass(slots=True)
class Pdx1Import:
    """Outcome of one import. skipped counts records by why they were left out."""

    packets: list[Spec1LeadPacket] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)


def _host(url: str) -> str:
    return (httpx.URL(url).host or "").lower()


def classify_source_tier(url: str) -> str:
    """The highest tier a document from this host can have.

    A ceiling, not a grade: pdx-1i sources take their tier from
    assign_source_tier, which also needs the content checks to pass.
    """
    host = _host(url)
    for primary in PRIMARY_HOSTS:
        if host == primary or host.endswith("." + primary):
            return "tier_1_primary"
    return "tier_3_secondary"


def assign_source_tier(source_url: str, final_url: str, hollow: str | None) -> str:
    """The tier a sealed pdx-1i source earns: its host's, only if its content passed.

    A hollow source (hollow is hollow_source_reason's answer) is
    tier_4_unverified whatever its host, so an error page on a primary host can
    never reach triage as tier_1_primary even if it were not skipped. A fetch
    that ended on another host takes the lower of the two hosts' tiers.
    """
    if hollow is not None:
        return UNCHECKED_TIER
    asked, served = classify_source_tier(source_url), classify_source_tier(final_url)
    return asked if asked == served else "tier_3_secondary"


def is_landing_page(url: str) -> bool:
    parsed = httpx.URL(url)
    return (
        (parsed.host or "").lower(),
        parsed.path.lower().rstrip("/"),
    ) in LANDING_PAGES


def is_empty_answer(text: str) -> bool:
    """A query that matched nothing: Socrata answers `[]` for an unknown id."""
    return text.strip() in {"[]", "{}", "null"}


def is_search_form(url: str, text: str) -> bool:
    marker = SEARCH_FORM_MARKERS.get(_host(url))
    return marker is not None and marker in text


def record_identifiers(pattern: str) -> list[re.Pattern[str]]:
    """What a page must say to be about the record the pattern describes.

    A record number matches in full or by its final number on its own, since a
    filing page may show 4417392 for ORE-2026-4417392. A measure matches as
    HB 4092, HB4092 or House Bill 4092.
    """
    found: list[re.Pattern[str]] = []
    for prefix, year, number in RECORD_NUMBER.findall(pattern):
        found.append(
            re.compile(
                rf"\b{prefix}-{year}-{number}\b|(?<![\d-]){number}(?!\d)", re.IGNORECASE
            )
        )
    for prefix, number in MEASURE_NUMBER.findall(pattern):
        names = "|".join(
            re.escape(name) for name in (prefix, MEASURE_NAMES.get(prefix)) if name
        )
        found.append(re.compile(rf"\b(?:{names})\s?{number}\b", re.IGNORECASE))
    return found


def states_record(record: Pdx1Record, page: PageContent) -> bool:
    """The page's visible content names the record, when the pattern names one."""
    wanted = record_identifiers(record.pattern)
    return not wanted or any(item.search(page.text) for item in wanted)


def _host_is(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def subject_domain_for(record: Pdx1Record) -> str:
    """washington_civic for a lead whose source is on pdc.wa.gov, else oregon_civic.

    Both the human url and the evidence_url count, so a WA PDC receipt sealed
    from its data.wa.gov row keeps the Washington domain of its pdc.wa.gov link.
    """
    for url in (record.url, record.evidence_url):
        if url and any(_host_is(_host(url), host) for host in WASHINGTON_HOSTS):
            return WASHINGTON_DOMAIN
    return SUBJECT_DOMAIN


def is_olis_url(url: str) -> bool:
    return _host_is(_host(url), OLIS_HOST)


def record_subject(text: str) -> str | None:
    """The subject of the first "Relating to" clause in text, or None."""
    match = RELATING_TO.search(normalize_whitespace(text))
    if match is None:
        return None
    return match.group(1).strip(" ,").lower() or None


def record_session(text: str) -> str | None:
    """The OLIS session key the text names: "2026 Regular session" is 2026R1."""
    match = SESSION_NAME.search(text)
    if match is not None:
        year, ordinal, kind = match.groups()
        number = SESSION_ORDINALS[ordinal.lower()] if ordinal else 1
        return f"{year}{'R' if kind.lower() == 'regular' else 'S'}{number}"
    key = SESSION_KEY.search(text)
    return key.group(1) if key else None


def record_title(text: str) -> str | None:
    """The record's title (its CatchLine for live OLIS records), or None."""
    flat = normalize_whitespace(text)
    match = RECORD_TITLE_TO_SESSION.search(flat) or RECORD_TITLE_TO_STOP.search(flat)
    if match is None:
        return None
    title = match.group(1).strip(" ,.")
    if re.match(r"in\s+the\s", title, re.IGNORECASE) and record_session(title):
        return None  # "OLIS record for HB 4092, in the 2026 Regular session."
    return title or None


def _folded(text: str) -> str:
    return normalize_whitespace(text).lower()


def url_session(url: str) -> str | None:
    """The session key in an OLIS URL: /liz/2025R1/... or ('2026R1')."""
    match = SESSION_KEY.search(httpx.URL(url).path)
    return match.group(1) if match else None


def _subject_words(subject: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", subject.lower())) - SUBJECT_STOPWORDS
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words}


def subjects_match(ours: str, theirs: str) -> bool:
    """Whether two "Relating to" subjects are the same subject.

    The rule is deliberately strict. Each subject is lowercased, split into
    words, stripped of stopwords (a, the, of, to, ...) and singularised by
    dropping one trailing "s" from words longer than three letters. The
    subjects match only when the smaller word set is non-empty and every one of
    its words is in the larger set. So "child safety system" matches "child
    safety systems" and "taxes" matches "taxes; prescribing an effective date",
    but subjects that share only some words, or none, do not match.
    """
    mine, other = _subject_words(ours), _subject_words(theirs)
    smaller, larger = sorted((mine, other), key=len)
    return bool(smaller) and smaller <= larger


def olis_mismatch_reason(
    record: Pdx1Record, final_url: str, text: str, page: PageContent | None
) -> str | None:
    """Why an OLIS source is not the record's measure, or None if it may be.

    Applies when the record's source URL or the URL the fetch ended at is on
    oregonlegislature.gov (olis., api.). The record's subject and session come
    from its pattern ("Relating to ..., in the 2026 Regular session"). The
    source's subject is the first "Relating to" clause of the page's visible
    content (its bill title), or of the raw text for a non-HTML answer (the
    API's RelatingTo). The source's session is the key in each OLIS URL
    (/liz/2026R1/...), which must equal the record's.

    The session check always applies: a record or URL that names no session
    cannot be checked, so the source is rejected.

    The subject check depends on what the record states. A record with a
    "Relating to" subject is matched against the source's (subjects_match).
    Live pdx-1i OLIS records instead take their title from the measure's
    CatchLine ("Prohibits a retailer ..."), a plain sentence that is not the
    bill's RelatingTo. For those, the record's title must appear verbatim, up
    to whitespace and case, in the page's visible main content (the overview
    shows it as "Catchline/Summary: ...") or in the raw text of a non-HTML
    answer. A record that states neither a subject nor a title is rejected.
    """
    urls = [url for url in (record.source_url, final_url) if url and is_olis_url(url)]
    if not urls:
        return None
    session = record_session(record.pattern)
    if session is None:
        return "is an OLIS source but the record states no session to check"
    for url in urls:
        served = url_session(url)
        if served is None:
            return "is an OLIS source whose URL names no session"
        if served != session:
            return f"is session {served}, not the record's session {session}"
    ours = record_subject(record.pattern)
    if ours is None:
        title = record_title(record.pattern)
        if title is None:
            return "is an OLIS source but the record states no subject to check"
        if _folded(title) not in _folded(page.text if page is not None else text):
            return f"does not state the record's title {title!r} (subject mismatch)"
        return None
    theirs = record_subject(page.text if page is not None else text)
    if theirs is None:
        return "is an OLIS source that states no subject"
    if not subjects_match(ours, theirs):
        return f"states subject {theirs!r}, not the record's {ours!r}"
    return None


def hollow_source_reason(
    record: Pdx1Record, final_url: str, text: str, page: PageContent | None
) -> str | None:
    """Why a sealed source cannot state its record, or None if it may.

    page is the HTML page's visible main content, or None for JSON, PDF or plain
    text. evidence_url payloads are JSON rows that pdx-1i fetched by the
    record's own id, so only the empty answer applies to them, plus the OLIS
    subject and session check for an oregonlegislature.gov API answer.
    """
    if is_landing_page(final_url):
        return "is a landing page"
    if is_search_form(final_url, text):
        return "is a search form"
    if is_empty_answer(text):
        return "is an empty answer"
    if page is None:
        return olis_mismatch_reason(record, final_url, text, None)
    reason = hollow_page_reason(page)
    if reason is not None:
        return reason
    if not states_record(record, page):
        return "does not state the record's identifier"
    return olis_mismatch_reason(record, final_url, text, page)


def _optional_text(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    return value if isinstance(value, str) else ""


def parse_record(line: str) -> Pdx1Record:
    """Validate one JSONL line against the fields pdx-1i writes."""
    try:
        raw = json.loads(line)
    except ValueError as error:
        raise Pdx1RecordError(f"not JSON: {error}") from error
    if not isinstance(raw, dict):
        raise Pdx1RecordError("not a JSON object")

    def text(name: str) -> str:
        value = raw.get(name)
        if not isinstance(value, str) or not value:
            raise Pdx1RecordError(f"{name} missing or not a string")
        return value

    priority = text("priority")
    if priority not in PRIORITY_RANK:
        raise Pdx1RecordError(f"unknown priority {priority!r}")
    tier = text("tier")
    if tier not in RECORD_TIERS:
        raise Pdx1RecordError(f"unknown tier {tier!r}")

    anomaly = raw.get("anomaly")
    sigma: float | None = None
    if anomaly is not None:
        if not isinstance(anomaly, dict) or not isinstance(
            anomaly.get("sigma"), int | float
        ):
            raise Pdx1RecordError("anomaly present without a numeric sigma")
        sigma = float(anomaly["sigma"])

    try:
        published_at = datetime.fromisoformat(text("published_at"))
    except ValueError as error:
        raise Pdx1RecordError(f"bad published_at: {error}") from error
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)

    url = raw.get("url")
    entity_ids = raw.get("entity_ids") or []
    return Pdx1Record(
        record_id=text("record_id"),
        source_type=text("source_type"),
        pattern=_optional_text(raw, "pattern"),
        outcome=_optional_text(raw, "outcome"),
        priority=priority,
        tier=tier,
        sigma=sigma,
        anomaly=anomaly,
        entity_ids=[str(item) for item in entity_ids]
        if isinstance(entity_ids, list)
        else [],
        url=url if isinstance(url, str) and url else None,
        published_at=published_at,
        origin=raw.get("origin") if isinstance(raw.get("origin"), str) else None,
        evidence_url=_optional_text(raw, "evidence_url") or None,
        public_question=_optional_text(raw, "public_question"),
        public_interest_reason=_optional_text(raw, "public_interest_reason"),
    )


def score_record(record: Pdx1Record) -> float:
    """Priority floor, raised by the anomaly's sigma. A rule-based screen."""
    score = PRIORITY_SCORE[record.priority]
    if record.sigma is not None:
        score = max(score, abs(record.sigma) * POINTS_PER_SIGMA)
    return clamp_score(score)


def lead_id_for(record: Pdx1Record) -> str:
    return str(uuid.uuid5(PDX1_NAMESPACE, record.record_id))


def _clip(text: str) -> str:
    """At most EXCERPT_CHARS, cut at a word boundary. Still a span of text."""
    if len(text) <= EXCERPT_CHARS:
        return text
    cut = text[:EXCERPT_CHARS]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).strip()


def pick_excerpt(record: Pdx1Record, text: str, page: PageContent | None = None) -> str:
    """The record's own pattern text if the source states it, else a span of its body.

    For an HTML page (page is its visible main content) the excerpt is one raw
    text run of that content: never <head>, <script>, <style>, nav, link or
    form-control text. The first run long enough to read as a sentence is used,
    preferring one that names the record's identifier, else the longest run. A run holds no markup
    or character references, so it is a verbatim span of the sealed payload,
    which is what the corroboration gate (excerpt_is_verbatim on the payload)
    and the grounding gate (normalize_text on source_registry.verbatim_text)
    compare against. Returns "" if no span qualifies; the lead then fails the
    evidence bar rather than carry boilerplate.

    Other payloads (JSON rows, PDF text, plain text) have no boilerplate, so
    their excerpt is the opening of the text, as before.
    """
    pattern_stated = bool(record.pattern) and excerpt_is_verbatim(record.pattern, text)
    if page is None:
        if pattern_stated:
            return record.pattern
        return normalize_whitespace(text)[:EXCERPT_CHARS].strip()

    if pattern_stated and normalize_whitespace(record.pattern) in page.text:
        return record.pattern
    sentences = [run for run in page.text_runs if len(run.split()) >= MIN_EXCERPT_WORDS]
    wanted = record_identifiers(record.pattern)
    naming = [run for run in sentences if any(item.search(run) for item in wanted)]
    candidates = naming or sentences
    if not candidates:
        # A short record page: its longest body run, still never boilerplate.
        candidates = sorted(page.text_runs, key=len, reverse=True)[:1]
    if not candidates:
        return ""
    excerpt = _clip(candidates[0])
    return excerpt if excerpt_is_verbatim(excerpt, text) else ""


_STATE_FOR_DOMAIN: Final = {SUBJECT_DOMAIN: "Oregon", WASHINGTON_DOMAIN: "Washington"}


def public_interest_for(record: Pdx1Record) -> tuple[str, str]:
    """Return the record's own question and reason, else neutral defaults."""
    kind = record.source_type.replace("_", " ") or "civic"
    seats = f" about {', '.join(record.entity_ids)}" if record.entity_ids else ""
    state = _STATE_FOR_DOMAIN.get(subject_domain_for(record), "local")
    return public_interest(
        record.public_question,
        record.public_interest_reason,
        f"What does {kind} record {record.record_id} show{seats}, and does the "
        "sealed source support it?",
        f"It is a public record of how {state} officials and public bodies act on "
        "residents' behalf.",
    )


def build_prompt(record: Pdx1Record, artifact: SourceArtifact) -> str:
    facts: dict[str, Any] = {
        "pdx-1i record": record.record_id,
        "Source type": record.source_type,
        "Pattern (pdx-1i summary, not verbatim)": record.pattern or "(none)",
        "pdx-1i outcome / priority": f"{record.outcome or '?'} / {record.priority}",
        "pdx-1i evidence tier": record.tier,
        "Seats / entities": ", ".join(record.entity_ids) or "none",
        "Anomaly": json.dumps(record.anomaly, sort_keys=True)
        if record.anomaly
        else "none",
        "Published": record.published_at.isoformat(),
    }
    return render_prompt(
        f"CIVIC RECORD: pdx-1i {record.record_id}",
        facts,
        [artifact],
        (
            "Draft a nonpartisan, factual brief on this civic record from the sealed "
            "source. Refer to officials by seat, as pdx-1i does."
        ),
    )


def _drop_shared_pages(result: Pdx1Import) -> None:
    """Drop leads whose sealed page was sealed for another record too and does not
    state the lead's own pattern. One filing that backs two records keeps both."""
    counts = Counter(p.source_artifacts[0].sha256_hash for p in result.packets)
    shared = {digest for digest, count in counts.items() if count > 1}
    if not shared:
        return
    kept: list[Spec1LeadPacket] = []
    for packet in result.packets:
        artifact = packet.source_artifacts[0]
        if artifact.sha256_hash in shared and not artifact.metadata["pattern_stated"]:
            result.skipped["hollow_source"] += 1
            result.errors.append(
                f"{artifact.metadata['pdx1_record_id']}: {artifact.origin_url} "
                "is the same page sealed for another record"
            )
        else:
            kept.append(packet)
    result.packets = kept


def import_pdx1_signals(
    store_path: Path | str,
    allocate_source_id: Callable[[], str],
    *,
    is_known: Callable[[str], bool] = lambda _lead_id: False,
    min_priority: str = "STANDARD",
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    transport: httpx.BaseTransport | None = None,
) -> Pdx1Import:
    """Read pdx1_signals.jsonl and emit leads for records at or above min_priority.

    Sources are fetched and sealed only for new leads at or above min_priority,
    so a re-run over the same store fetches nothing.
    """
    if min_priority not in PRIORITY_RANK:
        raise ValueError(f"min_priority must be one of {sorted(PRIORITY_RANK)}")
    floor = PRIORITY_RANK[min_priority]
    result = Pdx1Import()
    file_path = Path(store_path)
    if not file_path.exists():
        logger.warning("PDX-1i signals file not found: %s", file_path)
        result.errors.append(f"store not found: {file_path}")
        return result

    seen: set[str] = set()
    with file_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = parse_record(line)
            except Pdx1RecordError as error:
                result.skipped["malformed"] += 1
                result.errors.append(f"line {number}: {error}")
                continue
            if record.origin not in PUBLIC_ORIGINS:
                result.skipped["not_live"] += 1
                continue
            if PRIORITY_RANK[record.priority] < floor:
                result.skipped["below_priority"] += 1
                continue
            source_url = record.source_url
            if source_url is None:
                result.skipped["no_url"] += 1
                continue
            if is_landing_page(source_url):
                result.skipped["hollow_source"] += 1
                result.errors.append(
                    f"{record.record_id}: {source_url} is a landing page"
                )
                continue
            lead_id = lead_id_for(record)
            if lead_id in seen or is_known(lead_id):
                result.skipped["known"] += 1
                continue
            seen.add(lead_id)

            # Nothing about the content is known yet, so the blob is sealed as
            # unverified; the artifact's tier is decided after the checks.
            try:
                receipt = fetch_and_seal_source(
                    source_url,
                    allocate_source_id(),
                    UNCHECKED_TIER,
                    archive_dir=archive_dir,
                    transport=transport,
                )
            except (httpx.HTTPError, CustodyError, ValueError, OSError) as error:
                result.skipped["seal_failed"] += 1
                result.errors.append(
                    f"{record.record_id}: could not seal {source_url}: {error}"
                )
                continue

            archive_path = Path(receipt["archive_path"])
            text = read_artifact_text(
                archive_path.read_bytes(), receipt["content_type"]
            )
            if not text or not text.strip():
                result.skipped["no_source_text"] += 1
                continue
            final_url = receipt["final_url"]
            page = (
                read_main_content(text)
                if looks_like_html(text, receipt["content_type"])
                else None
            )
            hollow = hollow_source_reason(record, final_url, text, page)
            tier = assign_source_tier(source_url, final_url, hollow)
            if hollow is not None:
                result.skipped["hollow_source"] += 1
                result.errors.append(f"{record.record_id}: {final_url} {hollow}")
                continue

            artifact = SourceArtifact(
                source_id=receipt["source_id"],
                tier=tier,
                origin_url=source_url,
                sha256_hash=receipt["raw_sha256"],
                raw_payload_path=str(archive_path),
                captured_at=datetime.fromisoformat(receipt["retrieved_at"]),
                metadata={
                    "kind": ARTIFACT_KIND,
                    "content_type": receipt["content_type"],
                    "engine": "pdx-1i",
                    "subject_domain": subject_domain_for(record),
                    "pdx1_record_id": record.record_id,
                    "source_type": record.source_type,
                    "pdx1_tier": record.tier,
                    "pdx1_origin": record.origin,
                    "pdx1_url": record.url,
                    "entity_ids": record.entity_ids,
                    "pattern_stated": bool(
                        record.pattern and excerpt_is_verbatim(record.pattern, text)
                    ),
                },
            )
            excerpt = pick_excerpt(record, text, page)
            question, reason = public_interest_for(record)
            result.packets.append(
                Spec1LeadPacket(
                    lead_id=lead_id,
                    timestamp=record.published_at,
                    adapter_name=ADAPTER_NAME,
                    anomaly_score=score_record(record),
                    evidence_bar_passed=record.tier != "INFERRED" and bool(excerpt),
                    source_artifacts=[artifact],
                    verbatim_excerpts={artifact.source_id: excerpt},
                    paste_ready_prompt=build_prompt(record, artifact),
                    analysis_tier="rule_based",
                    public_question=question,
                    public_interest_reason=reason,
                )
            )

    _drop_shared_pages(result)
    logger.info(
        "pdx-1i: %d leads, skipped %s, %d errors",
        len(result.packets),
        dict(result.skipped),
        len(result.errors),
    )
    return result
