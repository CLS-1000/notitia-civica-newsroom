"""Claim matcher: narratives that fact-checkers already evaluated, back again.

Sources: the Google Fact Check Tools API (claims:search), which returns
ClaimReview markup that fact-checkers publish with their reviews, and the
review pages themselves.

Every claims:search result goes into a local ClaimReview cache, and each
incoming Domain 6 narrative cluster is matched against the whole cache, so a
claim reviewed years ago still matches after the API stops ranking it. Without
an API key (GOOGLE_FACTCHECK_API_KEY) the matcher runs on the cache alone. The
key travels in a header, only to factchecktools.googleapis.com.

A cluster is a zombie when its text shares at least 65% of its 5-character
shingles (Jaccard) with a documented claim that a fact-checker rated false or
misleading more than 180 days before the cluster was first seen. MinHash over
those shingles picks the candidates; the decision is the exact Jaccard, so a
threshold never depends on hash luck.

Each review page is sealed as tier_2_institutional, one per publisher, so the
core Corroboration Gate, which counts distinct publisher domains, sees two
independent fact-checkers. A single fact-check is a lead that triage rejects.

What a match shows, and no more: the new text reuses the wording of a claim
that fact-checkers evaluated. It does not show who posted it, why, or that the
new post is false, since a recycled claim can carry new facts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol

from newsroom.harlan_eternity.custody import normalize_domain
from spec1_bridge.adapters.base_lazlong import (
    EvidenceTarget,
    LazLongAdapter,
    PollRequest,
    host_of,
    render_prompt,
)
from spec1_bridge.adapters.hawkes_narrative_engine import (
    SHINGLE_SIZE,
    SIGNATURE_SIZE,
    estimated_similarity,
    minhash_signature,
    shingles,
)
from spec1_bridge.contracts import SourceArtifact
from spec1_bridge.sociology_telemetry import DOMAIN_FAKE_NEWS, StreamPost

FACTCHECK_API: Final = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
FACTCHECK_HOST: Final = "factchecktools.googleapis.com"
API_KEY_ENV: Final = "GOOGLE_FACTCHECK_API_KEY"
DEFAULT_CACHE_PATH: Final = Path("data/claimreview_cache.db")
PAGE_SIZE: Final = "50"
QUERY_MAX_CHARS: Final = 200

FACT_CHECK_TIER: Final = "tier_2_institutional"
FACT_CHECK_KIND: Final = "fact_check_review"
DOMAIN: Final = DOMAIN_FAKE_NEWS

MATCH_THRESHOLD: Final = 0.65
ZOMBIE_AGE_DAYS: Final = 180
ZOMBIE_SCORE: Final = 85.0
ZOMBIE_PRIORITY: Final = "HIGH"
ZOMBIE_SIGNAL: Final = "ZOMBIE_CLAIM_RECYCLED"
CLUSTER_LOOKBACK_DAYS: Final = 30

# Two signature rows per band: a pair at Jaccard 0.65 shares a band with
# probability 1 - (1 - 0.65**2)**32, above 0.99999, so candidates are not missed.
LSH_ROWS: Final = 2
LSH_BANDS: Final = SIGNATURE_SIZE // LSH_ROWS
RECEIPT_SHINGLE_SAMPLE: Final = 24
QUESTION_CLAIM_MAX_CHARS: Final = 160

# Ratings that state the claim is false or misleading. A claim rated true or
# half true is not a debunk, and its return is not a zombie.
_DEBUNK: Final = re.compile(
    r"\b(?:false|fake|pants on fire|incorrect|inaccurate|misleading|"
    r"no evidence|unsupported|unproven|unfounded|fabricated|hoax|baseless|"
    r"debunked|wrong|not true|untrue|misattributed|doctored|altered|"
    r"out of context|missing context|scam)\b",
    re.IGNORECASE,
)
_SPACE: Final = re.compile(r"\s+")


# --- Records ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactCheckReview:
    """One fact-checker's ClaimReview of a claim."""

    publisher_name: str
    publisher_site: str
    url: str
    title: str
    review_date: date | None
    textual_rating: str
    language_code: str

    @property
    def is_debunk(self) -> bool:
        return _DEBUNK.search(self.textual_rating) is not None

    @property
    def publisher(self) -> str:
        """Who reviewed it, for independence: the site, else the page's host."""
        site = self.publisher_site or host_of(self.url)
        return normalize_domain(site)

    def as_dict(self) -> dict[str, Any]:
        return {
            "publisher_name": self.publisher_name,
            "publisher_site": self.publisher_site,
            "url": self.url,
            "title": self.title,
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "textual_rating": self.textual_rating,
            "language_code": self.language_code,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> FactCheckReview:
        return cls(
            publisher_name=str(row.get("publisher_name") or ""),
            publisher_site=str(row.get("publisher_site") or ""),
            url=str(row.get("url") or ""),
            title=str(row.get("title") or ""),
            review_date=_iso_date(row.get("review_date")),
            textual_rating=str(row.get("textual_rating") or ""),
            language_code=str(row.get("language_code") or ""),
        )


@dataclass(frozen=True, slots=True)
class DocumentedClaim:
    """A claim as fact-checkers documented it, with every review of it."""

    text: str
    claimant: str
    claim_date: date | None
    reviews: tuple[FactCheckReview, ...]

    @property
    def key(self) -> str:
        return claim_key(self.text)


@dataclass(frozen=True, slots=True)
class NarrativeCluster:
    """Near-duplicate posts from Domain 6, as one narrative. No authors."""

    text: str
    first_seen: datetime
    last_seen: datetime
    size: int = 1
    platforms: tuple[str, ...] = ()

    @classmethod
    def from_posts(cls, posts: Sequence[StreamPost]) -> NarrativeCluster:
        """Represent a cluster by its earliest post, the likeliest original."""
        if not posts:
            raise ValueError("a cluster needs at least one post")
        ordered = sorted(posts, key=lambda post: post.at)
        return cls(
            text=ordered[0].text,
            first_seen=ordered[0].at,
            last_seen=ordered[-1].at,
            size=len(ordered),
            platforms=tuple(sorted({post.platform for post in ordered})),
        )

    @property
    def fingerprint(self) -> str:
        return claim_key(self.text)


@dataclass(frozen=True, slots=True)
class ClaimMatch:
    claim: DocumentedClaim
    jaccard: float
    minhash_estimate: float


# --- Text ---------------------------------------------------------------------


def _iso_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _normalized(text: str) -> str:
    """The text shingles() sees: NFKC, casefolded, single-spaced."""
    # Matches hawkes_narrative_engine's normalization, so spans line up with shingles.
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(folded.split())


def claim_key(text: str) -> str:
    return hashlib.sha256(_normalized(text).encode()).hexdigest()


def jaccard(left: str, right: str) -> float:
    """Exact Jaccard similarity of two texts' 5-character shingle sets."""
    a, b = shingles(left), shingles(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def query_text(text: str) -> str:
    """The cluster text as a claims:search query, cut at a word boundary."""
    flat = _SPACE.sub(" ", text).strip()
    if len(flat) <= QUERY_MAX_CHARS:
        return flat
    return flat[:QUERY_MAX_CHARS].rsplit(" ", 1)[0]


def _runs(covered: list[bool], wanted: bool) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate([*covered, not wanted]):
        if flag == wanted and start is None:
            start = index
        elif flag != wanted and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _preserved_spans(text: str, covered: list[bool]) -> list[str]:
    return [
        piece
        for start, end in _runs(covered, True)
        if (piece := text[start:end].strip())
    ]


def _altered_spans(text: str, covered: list[bool]) -> list[str]:
    """Uncovered runs widened to whole words, so 'ultnomah' reads 'multnomah'."""
    spans: list[str] = []
    for start, end in _runs(covered, False):
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        while end < len(text) and not text[end].isspace():
            end += 1
        piece = text[start:end].strip()
        if piece and piece not in spans:
            spans.append(piece)
    return spans


def shingle_receipt(incoming: str, documented: str) -> dict[str, Any]:
    """Show the editor which wording the two texts share, shingle by shingle.

    preserved_spans are runs of the incoming text covered by shared shingles,
    which is the copy-paste proof; altered_spans are what changed.
    """
    text = _normalized(incoming)
    ours, theirs = shingles(incoming), shingles(documented)
    shared = ours & theirs
    covered = [False] * len(text)
    ordered: list[str] = []
    for index in range(len(text) - SHINGLE_SIZE + 1):
        piece = text[index : index + SHINGLE_SIZE]
        if piece in shared:
            for offset in range(SHINGLE_SIZE):
                covered[index + offset] = True
            if piece not in ordered:
                ordered.append(piece)
    union = ours | theirs
    return {
        "method": (
            f"{SHINGLE_SIZE}-character shingles of the casefolded text; exact "
            f"Jaccard decides, MinHash ({SIGNATURE_SIZE} seeds) finds candidates"
        ),
        "jaccard": round(len(shared) / len(union), 4) if union else 0.0,
        "shared_shingles": len(shared),
        "incoming_shingles": len(ours),
        "documented_shingles": len(theirs),
        "union_shingles": len(union),
        "sample_shared_shingles": ordered[:RECEIPT_SHINGLE_SAMPLE],
        "preserved_spans": _preserved_spans(text, covered),
        "altered_spans": _altered_spans(text, covered),
    }


# --- Cache --------------------------------------------------------------------


def parse_claims(payload: dict[str, Any]) -> list[DocumentedClaim]:
    """Read the claims in a claims:search response. Malformed rows are skipped."""
    claims = []
    for row in payload.get("claims") or []:
        if not isinstance(row, dict) or not str(row.get("text") or "").strip():
            continue
        reviews = []
        for item in row.get("claimReview") or []:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            publisher = item.get("publisher")
            publisher = publisher if isinstance(publisher, dict) else {}
            reviews.append(
                FactCheckReview(
                    publisher_name=str(publisher.get("name") or ""),
                    publisher_site=str(publisher.get("site") or ""),
                    url=str(item["url"]),
                    title=str(item.get("title") or ""),
                    review_date=_iso_date(item.get("reviewDate")),
                    textual_rating=str(item.get("textualRating") or ""),
                    language_code=str(item.get("languageCode") or ""),
                )
            )
        if reviews:
            claims.append(
                DocumentedClaim(
                    text=_SPACE.sub(" ", str(row["text"])).strip(),
                    claimant=str(row.get("claimant") or ""),
                    claim_date=_iso_date(row.get("claimDate")),
                    reviews=tuple(reviews),
                )
            )
    return claims


class ClaimReviewCache:
    """Local SQLite store of every documented claim the matcher has seen."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    claim_key  TEXT PRIMARY KEY,
                    claim_text TEXT NOT NULL,
                    claimant   TEXT NOT NULL,
                    claim_date TEXT,
                    reviews    TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def add(self, claims: Sequence[DocumentedClaim]) -> None:
        """Store claims, merging reviews of the same claim by review URL."""
        with self._connect() as conn:
            for claim in claims:
                row = conn.execute(
                    "SELECT reviews FROM claims WHERE claim_key = ?", (claim.key,)
                ).fetchone()
                merged = {
                    item["url"]: item for item in (json.loads(row[0]) if row else [])
                }
                merged.update({item.url: item.as_dict() for item in claim.reviews})
                conn.execute(
                    """
                    INSERT INTO claims
                        (claim_key, claim_text, claimant, claim_date, reviews,
                         updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (claim_key) DO UPDATE
                    SET reviews = excluded.reviews, updated_at = excluded.updated_at
                    """,
                    (
                        claim.key,
                        claim.text,
                        claim.claimant,
                        claim.claim_date.isoformat() if claim.claim_date else None,
                        json.dumps(sorted(merged.values(), key=lambda r: r["url"])),
                        datetime.now(UTC).isoformat(),
                    ),
                )

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT count(*) FROM claims").fetchone()
        return int(row[0]) if row else 0

    def all(self) -> list[DocumentedClaim]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT claim_text, claimant, claim_date, reviews FROM claims "
                "ORDER BY claim_key"
            ).fetchall()
        return [
            DocumentedClaim(
                text=text,
                claimant=claimant,
                claim_date=_iso_date(claimed),
                reviews=tuple(FactCheckReview.from_dict(r) for r in json.loads(raw)),
            )
            for text, claimant, claimed, raw in rows
        ]


# --- Matching -----------------------------------------------------------------


@dataclass(slots=True)
class ClaimIndex:
    """MinHash LSH over documented claims, for candidate lookup."""

    claims: list[DocumentedClaim]
    _signatures: list[tuple[int, ...] | None] = field(default_factory=list)
    _buckets: dict[tuple[int, tuple[int, ...]], list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        buckets: defaultdict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        for index, claim in enumerate(self.claims):
            signature = minhash_signature(claim.text)
            self._signatures.append(signature)
            if signature is not None:
                for band, piece in _bands(signature):
                    buckets[(band, piece)].append(index)
        self._buckets = dict(buckets)

    def matches(
        self, text: str, threshold: float = MATCH_THRESHOLD
    ) -> list[ClaimMatch]:
        """Documented claims at or above threshold, most similar first."""
        signature = minhash_signature(text)
        if signature is None:
            return []
        candidates = {
            index
            for band, piece in _bands(signature)
            for index in self._buckets.get((band, piece), [])
        }
        found = []
        for index in candidates:
            claim = self.claims[index]
            score = jaccard(text, claim.text)
            if score >= threshold:
                other = self._signatures[index]
                assert other is not None
                found.append(
                    ClaimMatch(claim, score, estimated_similarity(signature, other))
                )
        return sorted(found, key=lambda item: (-item.jaccard, item.claim.key))


def _bands(signature: Sequence[int]) -> Iterator[tuple[int, tuple[int, ...]]]:
    for band in range(LSH_BANDS):
        start = band * LSH_ROWS
        yield band, tuple(signature[start : start + LSH_ROWS])


def zombie_reviews(
    claim: DocumentedClaim, first_seen: datetime
) -> list[FactCheckReview]:
    """Debunks made more than ZOMBIE_AGE_DAYS before first_seen, one per publisher.

    Each publisher's earliest such review is kept, so every review is an
    independent fact-checker.
    """
    cutoff = first_seen.date() - timedelta(days=ZOMBIE_AGE_DAYS)
    earliest: dict[str, FactCheckReview] = {}
    for review in claim.reviews:
        if not review.is_debunk or review.review_date is None:
            continue
        if review.review_date >= cutoff:
            continue
        held = earliest.get(review.publisher)
        if held is None or review.review_date < (held.review_date or cutoff):
            earliest[review.publisher] = review
    return sorted(
        earliest.values(), key=lambda item: (item.review_date or cutoff, item.publisher)
    )


def find_zombie(
    cluster: NarrativeCluster, index: ClaimIndex
) -> tuple[ClaimMatch, list[FactCheckReview]] | None:
    """The closest documented claim the cluster recycles, with its old debunks."""
    for match in index.matches(cluster.text):
        reviews = zombie_reviews(match.claim, cluster.first_seen)
        if reviews:
            return match, reviews
    return None


def _publishers(reviews: Sequence[FactCheckReview]) -> str:
    names = list(
        dict.fromkeys(item.publisher_name or item.publisher for item in reviews)
    )
    if len(names) <= 2:
        return " and ".join(names)
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _long(moment: date) -> str:
    return f"{moment:%B} {moment.day}, {moment.year}"


def _quoted_claim(text: str) -> str:
    flat = _SPACE.sub(" ", text).strip().replace("'", "’")
    if len(flat) <= QUESTION_CLAIM_MAX_CHARS:
        return flat
    return flat[:QUESTION_CLAIM_MAX_CHARS].rsplit(" ", 1)[0] + "…"


# --- Domain 6 clusters from the ledger -----------------------------------------


class _Rows(Protocol):
    def fetchall(self) -> list[tuple[Any, ...]]: ...


class _Ledger(Protocol):
    def execute(self, query: str, params: tuple[Any, ...] = ...) -> _Rows: ...


def _float(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def load_narrative_clusters(
    conn: _Ledger, now: datetime, lookback_days: int = CLUSTER_LOOKBACK_DAYS
) -> list[NarrativeCluster]:
    """Domain 6 narratives quarantined in the last lookback_days, one per text.

    The Hawkes engine writes each cluster's representative text, first sighting,
    and receipt metrics (events, span_hours, platforms). Repeat sightings of one
    narrative merge: earliest first sighting, latest last sighting, largest size.
    """
    rows = conn.execute(
        """
        SELECT observed_text, observed_at, metrics
        FROM sociology_telemetry.narrative_observations
        WHERE domain = %s AND observed_at >= %s
        ORDER BY observed_at, observation_id
        """,
        (DOMAIN, now - timedelta(days=lookback_days)),
    ).fetchall()
    merged: dict[str, NarrativeCluster] = {}
    for text, observed_at, metrics in rows:
        if not isinstance(text, str) or not text.strip():
            continue
        receipt = metrics.get("receipt") if isinstance(metrics, dict) else None
        figures = receipt.get("metrics") if isinstance(receipt, dict) else None
        figures = figures if isinstance(figures, dict) else {}
        platforms = figures.get("platforms")
        cluster = NarrativeCluster(
            text=text,
            first_seen=observed_at,
            last_seen=observed_at + timedelta(hours=_float(figures.get("span_hours"))),
            size=max(1, int(_float(figures.get("events")))),
            platforms=tuple(
                sorted(str(item) for item in platforms)
                if isinstance(platforms, list | dict)
                else ()
            ),
        )
        held = merged.get(cluster.fingerprint)
        if held is not None:
            cluster = NarrativeCluster(
                text=held.text,
                first_seen=min(held.first_seen, cluster.first_seen),
                last_seen=max(held.last_seen, cluster.last_seen),
                size=max(held.size, cluster.size),
                platforms=tuple(sorted({*held.platforms, *cluster.platforms})),
            )
        merged[cluster.fingerprint] = cluster
    return list(merged.values())


# --- Adapter ------------------------------------------------------------------


class ClaimMatcherAdapter(LazLongAdapter):
    adapter_name = "claim_matcher"
    subject_domain = "narrative_integrity"

    def __init__(
        self,
        clusters: Sequence[NarrativeCluster],
        api_key: str | None = None,
        *,
        cache_path: Path = DEFAULT_CACHE_PATH,
        language_code: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._clusters = list(clusters)
        self._api_key = api_key or os.environ.get(API_KEY_ENV) or None
        self._cache = ClaimReviewCache(cache_path)
        self._language_code = language_code

    @property
    def cache(self) -> ClaimReviewCache:
        return self._cache

    @property
    def syncs(self) -> bool:
        """Whether a harvest pulls fresh ClaimReview results: only with a key."""
        return self._api_key is not None

    def auth_headers(self) -> dict[str, str]:
        return {"X-Goog-Api-Key": self._api_key} if self._api_key else {}

    def headers_for(self, url: str) -> dict[str, str]:
        # The key is for the Fact Check Tools API alone, never a review page.
        return self.auth_headers() if host_of(url) == FACTCHECK_HOST else {}

    def replays(self, request: PollRequest) -> bool:
        return self._api_key is not None

    def poll_requests(self) -> list[PollRequest]:
        if self._api_key is None:
            return []
        requests = []
        for query in dict.fromkeys(query_text(item.text) for item in self._clusters):
            if not query:
                continue
            params = {"query": query, "pageSize": PAGE_SIZE}
            if self._language_code:
                params["languageCode"] = self._language_code
            requests.append(PollRequest("claims", FACTCHECK_API, params))
        return requests

    def extract_items(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [payload]

    def assemble_records(
        self, items: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        fetched = [
            claim
            for payload in items.get("claims", [])
            for claim in parse_claims(payload)
        ]
        if fetched:
            self._cache.add(fetched)
        index = ClaimIndex(self._cache.all())
        records = []
        for cluster in self._clusters:
            found = find_zombie(cluster, index)
            if found is None:
                continue
            match, reviews = found
            records.append(
                {
                    "kind": "zombie_claim",
                    "cluster": cluster,
                    "match": match,
                    "reviews": reviews,
                    "receipt": shingle_receipt(cluster.text, match.claim.text),
                }
            )
        return records

    def record_key(self, raw_record: dict[str, Any]) -> str:
        claim = raw_record["match"].claim
        return f"{claim.key[:16]}:{raw_record['cluster'].fingerprint[:16]}"

    def record_timestamp(self, raw_record: dict[str, Any]) -> datetime:
        seen: datetime = raw_record["cluster"].last_seen
        return seen if seen.tzinfo else seen.replace(tzinfo=UTC)

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        return ZOMBIE_SCORE if raw_record.get("reviews") else 0.0

    def evidence_targets(self, raw_record: dict[str, Any]) -> list[EvidenceTarget]:
        claim = raw_record["match"].claim
        return [
            EvidenceTarget(
                url=review.url,
                tier=FACT_CHECK_TIER,
                kind=FACT_CHECK_KIND,
                metadata={
                    "publisher": review.publisher_name,
                    "publisher_site": review.publisher,
                    "review_date": (
                        review.review_date.isoformat() if review.review_date else None
                    ),
                    "textual_rating": review.textual_rating,
                    "claim_key": claim.key,
                },
            )
            for review in raw_record["reviews"]
        ]

    def verify_evidence_bar(self, artifacts: list[SourceArtifact]) -> bool:
        """A fact-check page sealed from its publisher's own site.

        One is enough to be a lead; the core Corroboration Gate then asks for two
        independent fact-checkers before the lead enters triage.
        """
        return any(
            item.metadata.get("kind") == FACT_CHECK_KIND
            and item.tier == FACT_CHECK_TIER
            and _on_site(item.origin_url, str(item.metadata.get("publisher_site", "")))
            for item in artifacts
        )

    def receipt_details(self, record: dict[str, Any]) -> dict[str, Any]:
        match: ClaimMatch = record["match"]
        cluster: NarrativeCluster = record["cluster"]
        return {
            "detector": self.adapter_name,
            "signal": ZOMBIE_SIGNAL,
            "domain": DOMAIN,
            "priority": ZOMBIE_PRIORITY,
            "incoming_text": cluster.text,
            "cluster_size": cluster.size,
            "cluster_platforms": list(cluster.platforms),
            "first_seen": cluster.first_seen.isoformat(),
            "documented_claim": match.claim.text,
            "documented_claimant": match.claim.claimant,
            "minhash_estimate": round(match.minhash_estimate, 4),
            "shingles": record["receipt"],
            "fact_checks": [item.as_dict() for item in record["reviews"]],
            "cannot_show": (
                "Shared wording shows the text was copied from, or shares an origin "
                "with, a claim fact-checkers evaluated. It does not show who posted "
                "it or why, or that the new post is false."
            ),
        }

    def default_public_question(self, record: dict[str, Any]) -> str:
        claim = _quoted_claim(record["cluster"].text)
        return (
            f"Is the claim '{claim}' a recycled narrative previously evaluated by "
            f"{_publishers(record['reviews'])}?"
        )

    def default_public_interest_reason(self, record: dict[str, Any]) -> str:
        reviews: list[FactCheckReview] = record["reviews"]
        first = reviews[0].review_date
        assert first is not None
        if record["receipt"]["altered_spans"]:
            tail = "core syntax is preserved with localized details altered."
        else:
            tail = "the wording is preserved unchanged."
        return (
            f"5-gram MinHash matched a documented claim from {_long(first)} by "
            f"{_publishers(reviews)}; {tail}"
        )

    def build_paste_ready_prompt(
        self, record: dict[str, Any], artifacts: list[SourceArtifact]
    ) -> str:
        match: ClaimMatch = record["match"]
        receipt = record["receipt"]
        return render_prompt(
            "Recycled claim: a Domain 6 narrative matches a documented fact-check",
            {
                "Circulating text": record["cluster"].text,
                "Documented claim": match.claim.text,
                "Jaccard (5-character shingles)": f"{match.jaccard:.3f}",
                "Shared shingles": (
                    f"{receipt['shared_shingles']} of {receipt['union_shingles']}"
                ),
                "Preserved wording": " | ".join(receipt["preserved_spans"]) or "none",
                "Altered wording": " | ".join(receipt["altered_spans"]) or "none",
                "Fact-checks": "; ".join(
                    f"{item.publisher_name or item.publisher} rated it "
                    f"“{item.textual_rating}” on {item.review_date} "
                    f"({item.url})"
                    for item in record["reviews"]
                ),
            },
            artifacts,
            "Compare the circulating text with each fact-check: say which details "
            "are new, whether the fact-checks' findings still apply to them, and "
            "quote each review's conclusion.",
        )


def _on_site(url: str, site: str) -> bool:
    host = normalize_domain(host_of(url))
    wanted = normalize_domain(site)
    return bool(host and wanted) and (host == wanted or host.endswith(f".{wanted}"))
