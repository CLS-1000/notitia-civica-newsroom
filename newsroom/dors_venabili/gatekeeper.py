"""Deterministic pre-publication gates.

Nothing here calls a model, and every issue a gate returns blocks publication.
Each check works from text alone, so the same draft always gets the same verdict.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Final

_TEASERS: Final = ("here is why", "here's why", "what to know")
_BANNED_HEADLINE_TOKENS: Final = (
    "slams",
    "blasts",
    "destroys",
    "bombshell",
    "shocking",
    "quietly",
    "exposed",
    "radical",
    "extremist",
)
_WEASEL_PHRASES: Final = ("critics say", "experts say")
_LOADED_TERMS: Final = ("admits", "concedes")
# Agenda, scheme, and plot are ordinary civic words, so only adversarial framing is
# flagged. "Committee agenda" and "plot of land" pass.
_LOADED_FRAMING: Final = (
    re.compile(r"\b(hidden|secret|political)\s+agenda\b", re.IGNORECASE),
    re.compile(r"\b(nefarious|corrupt)\s+scheme\b", re.IGNORECASE),
    re.compile(r"\bplot(ted)?\s+to\b", re.IGNORECASE),
)
_ALLEGATION_TERMS: Final = (
    "bribed",
    "defrauded",
    "embezzled",
    "embezzlement",
    "extorted",
    "forged",
    "fraud",
    "fraudulent",
    "laundered",
    "laundering",
    "stole",
    "theft",
)

_TAG: Final = re.compile(r"\[S-\d+\]")
_TAG_LIKE: Final = re.compile(r"\[\s*[Ss]\s*-[^\]]*\]")
_QUOTE: Final = re.compile(r'"([^"]*)"')
_CURLY_QUOTES: Final = str.maketrans({"“": '"', "”": '"'})
_SSN: Final = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")
_PHONE: Final = re.compile(
    r"(?<![\d-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"
)
_ADDRESS: Final = re.compile(
    r"\b\d{1,6}\s+(?:[NSEW]\.?\s+)?(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Court|Ct|Boulevard|Blvd|"
    r"Way|Place|Pl|Terrace|Ter)\b"
)
_BUSINESS_ADDRESS: Final = re.compile(
    r"\b(?:suite|ste\.?|floor|headquarters|offices?|capitol|courthouse|city hall)\b",
    re.IGNORECASE,
)
_ADDRESS_CONTEXT_CHARS: Final = 40
_MAX_ROLE_SHARE: Final = 0.60
PRIMARY_TIER: Final = "tier_1_primary"
SINGLE_SOURCE_BADGE: Final = "FACT BRIEF"


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE) is not None


def normalize_text(text: str) -> str:
    """Collapse whitespace and straighten double quotes for comparison."""
    return " ".join(text.translate(_CURLY_QUOTES).split())


def split_paragraphs(body: str) -> list[str]:
    """Return prose blocks. Markdown headings and rules are not paragraphs."""
    blocks = (block.strip() for block in re.split(r"\n\s*\n", body))
    return [
        block
        for block in blocks
        if block and not block.startswith("#") and set(block) - set("-*_ ")
    ]


def extract_citations(body: str) -> list[str]:
    """Return every source ID cited in body, in order, with repeats."""
    return [tag[1:-1] for tag in _TAG.findall(body)]


def _without_quotes(text: str) -> str:
    return _QUOTE.sub(" ", text.translate(_CURLY_QUOTES))


def audit_headline(headline: str) -> list[str]:
    """Return every reason this headline may not run."""
    issues: list[str] = []
    if not headline.strip():
        issues.append("headline is empty")
    if "?" in headline:
        issues.append("headline contains a question mark")
    lowered = headline.translate({0x2019: "'"}).lower()
    issues.extend(
        f"forward-referencing teaser: {teaser!r}"
        for teaser in _TEASERS
        if teaser in lowered
    )
    issues.extend(
        f"banned headline token: {token!r}"
        for token in _BANNED_HEADLINE_TOKENS
        if _contains_term(headline, token)
    )
    return issues


def audit_loaded_language(body: str) -> list[str]:
    """Flag the paper's own loaded wording. Verified direct quotes are exempt."""
    prose = _without_quotes(body)
    issues = [
        f"weasel words: {phrase!r}"
        for phrase in _WEASEL_PHRASES
        if _contains_term(prose, phrase)
    ]
    issues.extend(
        f"loaded framing: {term!r}"
        for term in _LOADED_TERMS
        if _contains_term(prose, term)
    )
    issues.extend(
        f"loaded framing: {match.group()!r}"
        for pattern in _LOADED_FRAMING
        for match in pattern.finditer(prose)
    )
    return issues


def _check_quotes(
    paragraph: str, cited: list[str], source_texts: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Return unverified quotes and structural problems for one paragraph."""
    straightened = paragraph.translate(_CURLY_QUOTES)
    if straightened.count('"') % 2:
        return [], ["unbalanced double quotes"]

    searchable = [normalize_text(source_texts[source_id]) for source_id in cited]
    failures = []
    for quote in _QUOTE.findall(straightened):
        wanted = normalize_text(_TAG.sub("", quote))
        if wanted and not any(wanted in text for text in searchable):
            failures.append(wanted)
    return failures, []


def verify_grounding(body: str, source_texts: dict[str, str]) -> dict[str, Any]:
    """Check every paragraph's citations, and every quote against what it cites."""
    paragraphs: list[dict[str, Any]] = []
    all_issues: list[str] = []
    unknown_ids: list[str] = []
    quote_failures: list[dict[str, Any]] = []

    for index, paragraph in enumerate(split_paragraphs(body), start=1):
        issues: list[str] = []
        tags = extract_citations(paragraph)
        valid = [tag for tag in tags if tag in source_texts]
        unknown = sorted({tag for tag in tags if tag not in source_texts})
        malformed = [
            item for item in _TAG_LIKE.findall(paragraph) if not _TAG.fullmatch(item)
        ]

        issues.extend(f"unknown source ID {tag}" for tag in unknown)
        issues.extend(f"malformed citation {item}" for item in malformed)
        unknown_ids.extend(unknown)
        if not valid:
            issues.append("no valid [S-XXXX] citation")

        failures, structural = _check_quotes(paragraph, valid, source_texts)
        issues.extend(structural)
        for quote in failures:
            issues.append(f"quote not verbatim in cited source: {quote!r}")
            quote_failures.append({"paragraph": index, "quote": quote, "cited": valid})

        paragraphs.append(
            {
                "index": index,
                "citations": tags,
                "valid_citations": valid,
                "issues": issues,
                "passed": not issues,
            }
        )
        all_issues.extend(f"paragraph {index}: {issue}" for issue in issues)

    if not paragraphs:
        all_issues.append("draft has no paragraphs")

    return {
        "passed": not all_issues,
        "paragraphs": paragraphs,
        "unknown_ids": sorted(set(unknown_ids)),
        "quote_failures": quote_failures,
        "issues": all_issues,
    }


def _home_address_issues(body: str) -> list[str]:
    issues = []
    for match in _ADDRESS.finditer(body):
        start = max(0, match.start() - _ADDRESS_CONTEXT_CHARS)
        context = body[start : match.end() + _ADDRESS_CONTEXT_CHARS]
        if _BUSINESS_ADDRESS.search(context) is None:
            issues.append(f"possible non-business address: {match.group()!r}")
    return issues


def scrub_pii_and_defamation(
    body: str, is_private_person: bool, right_of_reply_logged: bool
) -> list[str]:
    """Flag PII, and allegations against private people who had no reply."""
    issues = [f"possible SSN: {match}" for match in _SSN.findall(body)]
    issues.extend(f"possible phone number: {match}" for match in _PHONE.findall(body))
    issues.extend(_home_address_issues(body))

    if is_private_person and not right_of_reply_logged:
        issues.extend(
            f"allegation against a private person without right of reply: {term!r}"
            for term in _ALLEGATION_TERMS
            if _contains_term(body, term)
        )
    return issues


def normalize_badge(badge_type: str) -> str:
    """Reduce '[FACT BRIEF]' and 'fact_brief' to one spelling."""
    return " ".join(badge_type.strip("[] ").replace("_", " ").upper().split())


def check_source_diversity(
    citations: list[str],
    roles: dict[str, str],
    *,
    badge_type: str = "",
    source_tiers: dict[str, str] | None = None,
) -> list[str]:
    """Require that no role category exceeds 60% of citations.

    The one exception is a FACT BRIEF whose only citation is a Tier 1 primary
    document. Every other badge, including DEEP EXPLAINER and INVESTIGATION, is
    held to the ceiling however few citations it has.
    """
    if not citations:
        return ["no citations to assess"]
    if (
        len(citations) == 1
        and normalize_badge(badge_type) == SINGLE_SOURCE_BADGE
        and (source_tiers or {}).get(citations[0]) == PRIMARY_TIER
    ):
        return []

    issues = sorted(
        f"no role assigned to {source_id}"
        for source_id in set(citations)
        if not roles.get(source_id)
    )
    counts = Counter(roles.get(source_id) or "unassigned" for source_id in citations)
    total = len(citations)
    issues.extend(
        f"role {role!r} is {count / total:.0%} of citations, limit is "
        f"{_MAX_ROLE_SHARE:.0%}"
        for role, count in counts.most_common()
        if count / total > _MAX_ROLE_SHARE
    )
    return issues


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    label: str
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues


def run_all_gates(
    *,
    headline: str,
    body: str,
    source_texts: dict[str, str],
    roles: dict[str, str],
    is_private_person: bool,
    right_of_reply_logged: bool,
    badge_type: str = "",
    source_tiers: dict[str, str] | None = None,
) -> list[GateResult]:
    """Run the five gates. Both the desk and the sign-off call this."""
    return [
        GateResult("headline", "Headline standards", audit_headline(headline)),
        GateResult("language", "Loaded language", audit_loaded_language(body)),
        GateResult(
            "grounding",
            "Citations and verbatim quotes",
            verify_grounding(body, source_texts)["issues"],
        ),
        GateResult(
            "pii_defamation",
            "PII and defamation",
            scrub_pii_and_defamation(body, is_private_person, right_of_reply_logged),
        ),
        GateResult(
            "diversity",
            "Source diversity",
            check_source_diversity(
                extract_citations(body),
                roles,
                badge_type=badge_type,
                source_tiers=source_tiers,
            ),
        ),
    ]
