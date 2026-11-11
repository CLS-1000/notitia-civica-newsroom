"""The ClaimReview zombie matcher, tested offline against mocked Fact Check Tools
payloads and review pages, plus Seldon's keyless run.

Every request goes through test_phase1_adapters.Router, which fails the test on any
URL it was not given.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Self

import httpx
import psycopg
import pytest
from conftest import Conn
from test_phase1_adapters import (
    NOW,
    Router,
    counter,
    ollama,
    options,
    pdf_response,
    spool_rows,
)
from test_seldon_civic import PDF_URL, TEXT_URL, rule, text_response

from spec1_bridge import mycroft_triage
from spec1_bridge.adapters.claim_matcher import (
    FACT_CHECK_TIER,
    FACTCHECK_API,
    ZOMBIE_AGE_DAYS,
    ClaimIndex,
    ClaimMatcherAdapter,
    DocumentedClaim,
    FactCheckReview,
    NarrativeCluster,
    jaccard,
    parse_claims,
    query_text,
    shingle_receipt,
)
from spec1_bridge.adapters.seldon_civic import (
    CONGRESS_API_HOST,
    FR_API,
    SeldonCivicAdapter,
)
from spec1_bridge.contracts import SourceArtifact
from spec1_bridge.mycroft_triage import (
    independent_publishers,
    run_adapter,
    triage_packets,
)
from spec1_bridge.sociology_telemetry import StreamPost

FACTCHECK_KEY = "test-factcheck-key-do-not-spool"

DOCUMENTED = (
    "Poll workers in Maricopa County were caught on camera shredding thousands of "
    "mail-in ballots the night before the election, and officials are covering it up."
)
# The same claim with its place swapped, as recycled rumors usually arrive.
LOCALIZED = DOCUMENTED.replace("Maricopa", "Multnomah")
UNRELATED = (
    "The county library will extend its weekend hours this winter and add a free "
    "homework help program for students in grades three through eight."
)

FIRST_SEEN = NOW - timedelta(hours=6)
FAKE_ARTIFACT = SourceArtifact(
    source_id="S-1",
    tier=FACT_CHECK_TIER,
    origin_url="https://example.org/",
    sha256_hash="0" * 64,
    raw_payload_path="/nonexistent",
    captured_at=NOW,
)
OLD_REVIEW = date(2024, 11, 20)  # about two years before FIRST_SEEN

POLITIFACT_URL = "https://www.politifact.com/factchecks/2024/nov/20/ballots-shredded/"
AP_URL = "https://apnews.com/article/fact-check-ballots-shredded-0001"
REUTERS_URL = "https://www.reuters.com/fact-check/ballots-shredded-2024-11-22/"


def review(
    url: str = POLITIFACT_URL,
    *,
    name: str = "PolitiFact",
    site: str = "politifact.com",
    reviewed: date | None = OLD_REVIEW,
    rating: str = "False",
) -> dict[str, Any]:
    return {
        "publisher": {"name": name, "site": site},
        "url": url,
        "title": "No, poll workers were not filmed shredding ballots",
        "reviewDate": f"{reviewed.isoformat()}T00:00:00Z" if reviewed else None,
        "textualRating": rating,
        "languageCode": "en",
    }


def politifact(**overrides: Any) -> dict[str, Any]:
    return review(POLITIFACT_URL, **overrides)


def ap(**overrides: Any) -> dict[str, Any]:
    return review(AP_URL, name="AP News", site="apnews.com", **overrides)


def reuters(**overrides: Any) -> dict[str, Any]:
    return review(REUTERS_URL, name="Reuters", site="reuters.com", **overrides)


def claim_payload(*reviews: dict[str, Any], text: str = DOCUMENTED) -> dict[str, Any]:
    return {
        "claims": [
            {
                "text": text,
                "claimant": "Viral social media posts",
                "claimDate": "2024-11-18T00:00:00Z",
                "claimReview": list(reviews),
            }
        ]
    }


def page(title: str) -> httpx.Response:
    return httpx.Response(
        200,
        text=f"<html><head><title>{title}</title></head><body>Rated False.</body></html>",
        headers={"content-type": "text/html; charset=utf-8"},
    )


def router(payload: dict[str, Any] | None = None) -> Router:
    return Router(
        {
            FACTCHECK_API: payload if payload is not None else {"claims": []},
            POLITIFACT_URL: page("PolitiFact"),
            AP_URL: page("AP Fact Check"),
            REUTERS_URL: page("Reuters Fact Check"),
        }
    )


def cluster(
    text: str = LOCALIZED, first_seen: datetime = FIRST_SEEN
) -> NarrativeCluster:
    return NarrativeCluster(
        text=text,
        first_seen=first_seen,
        last_seen=first_seen + timedelta(hours=3),
        size=42,
        platforms=("bluesky",),
    )


def matcher(
    tmp_path: Path,
    mock: Router,
    clusters: list[NarrativeCluster] | None = None,
    api_key: str | None = FACTCHECK_KEY,
) -> ClaimMatcherAdapter:
    return ClaimMatcherAdapter(
        clusters if clusters is not None else [cluster()],
        api_key,
        cache_path=tmp_path / "claimreview.db",
        **options(tmp_path, mock),
    )


def harvest(
    tmp_path: Path,
    payload: dict[str, Any],
    clusters: list[NarrativeCluster] | None = None,
) -> Any:
    return matcher(tmp_path, router(payload), clusters).harvest(counter())


# --- Similarity -----------------------------------------------------------------


def test_exact_copy_has_jaccard_one() -> None:
    assert jaccard(DOCUMENTED, DOCUMENTED) == 1.0
    assert jaccard(DOCUMENTED, f"  {DOCUMENTED.upper()} ") == 1.0


def test_word_swapped_local_mutation_stays_above_threshold() -> None:
    assert 0.65 <= jaccard(LOCALIZED, DOCUMENTED) < 1.0
    twice = LOCALIZED.replace("election", "vote")
    assert 0.65 <= jaccard(twice, DOCUMENTED) < jaccard(LOCALIZED, DOCUMENTED)


def test_unrelated_text_is_far_below_threshold() -> None:
    assert jaccard(UNRELATED, DOCUMENTED) < 0.40


def test_too_short_text_matches_nothing() -> None:
    assert jaccard("hi", DOCUMENTED) == 0.0
    index = ClaimIndex(parse_claims(claim_payload(politifact())))
    assert index.matches("hi") == []


def test_index_finds_the_closest_claim_first() -> None:
    other = DOCUMENTED.replace("thousands of", "hundreds of")
    index = ClaimIndex(
        parse_claims(claim_payload(politifact()))
        + parse_claims(claim_payload(ap(), text=other))
    )
    found = index.matches(DOCUMENTED)
    assert [item.claim.text for item in found] == [DOCUMENTED, other]
    assert found[0].jaccard == 1.0 and found[0].minhash_estimate == 1.0


def test_receipt_shows_the_copied_wording_and_the_altered_detail() -> None:
    receipt = shingle_receipt(LOCALIZED, DOCUMENTED)
    assert receipt["altered_spans"] == ["multnomah"]
    assert "county were caught on camera shredding thousands" in " ".join(
        receipt["preserved_spans"]
    )
    assert receipt["sample_shared_shingles"][:3] == ["poll ", "oll w", "ll wo"]
    assert receipt["jaccard"] == round(jaccard(LOCALIZED, DOCUMENTED), 4)
    assert receipt["shared_shingles"] < receipt["union_shingles"]


def test_receipt_of_an_exact_copy_alters_nothing() -> None:
    receipt = shingle_receipt(DOCUMENTED, DOCUMENTED)
    assert receipt["altered_spans"] == []
    assert receipt["jaccard"] == 1.0


# --- ClaimReview payloads -------------------------------------------------------


def test_parse_claims_reads_claimreview() -> None:
    [claim] = parse_claims(claim_payload(politifact(), ap()))
    assert claim.text == DOCUMENTED
    assert claim.claim_date == date(2024, 11, 18)
    assert [item.publisher for item in claim.reviews] == [
        "politifact.com",
        "apnews.com",
    ]
    assert claim.reviews[0].review_date == OLD_REVIEW
    assert claim.reviews[0].is_debunk


@pytest.mark.parametrize(
    "payload",
    [
        {"claims": [{"text": "", "claimReview": [politifact()]}]},
        {"claims": [{"text": DOCUMENTED, "claimReview": []}]},
        {"claims": [{"text": DOCUMENTED, "claimReview": [{"title": "no url"}]}]},
        {"claims": ["not an object"]},
        {},
    ],
)
def test_malformed_claims_are_skipped(payload: dict[str, Any]) -> None:
    assert parse_claims(payload) == []


@pytest.mark.parametrize(
    ("rating", "debunk"),
    [
        ("False", True),
        ("Pants on Fire", True),
        ("Mostly False", True),
        ("Missing context", True),
        ("Not true", True),
        ("True", False),
        ("Mostly True", False),
        ("Half True", False),
    ],
)
def test_only_false_or_misleading_ratings_are_debunks(
    rating: str, debunk: bool
) -> None:
    item = FactCheckReview(
        "P", "p.org", "https://p.org/x", "", OLD_REVIEW, rating, "en"
    )
    assert item.is_debunk is debunk


def test_cache_merges_reviews_of_the_same_claim(tmp_path: Path) -> None:
    adapter = matcher(tmp_path, router())
    adapter.cache.add(parse_claims(claim_payload(politifact())))
    adapter.cache.add(parse_claims(claim_payload(ap(), politifact())))
    [claim] = adapter.cache.all()
    assert sorted(item.url for item in claim.reviews) == [AP_URL, POLITIFACT_URL]


def test_query_text_is_cut_at_a_word_boundary() -> None:
    long = "word " * 100
    assert len(query_text(long)) <= 200 and not query_text(long).endswith(" ")
    assert query_text("  short\n claim ") == "short claim"


# --- Polling --------------------------------------------------------------------


def test_key_travels_as_a_header_only_to_the_api(tmp_path: Path) -> None:
    mock = router(claim_payload(politifact(), ap()))
    matcher(tmp_path, mock).harvest(counter())

    api = mock.calls_to(FACTCHECK_API)
    pages = [item for item in mock.requests if item not in api]
    assert api and pages
    assert all(item.headers["x-goog-api-key"] == FACTCHECK_KEY for item in api)
    assert not any("x-goog-api-key" in item.headers for item in pages)
    assert not any(FACTCHECK_KEY in str(item.url) for item in mock.requests)
    assert api[0].url.params["query"] == LOCALIZED


def test_keyless_matcher_matches_the_cache_without_calling_the_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    mock = router()
    adapter = matcher(tmp_path, mock, api_key=None)
    adapter.cache.add(parse_claims(claim_payload(politifact(), ap())))

    result = adapter.harvest(counter())
    assert len(result.packets) == 1
    assert mock.calls_to(FACTCHECK_API) == []


def test_a_cached_claim_still_matches_after_the_api_stops_returning_it(
    tmp_path: Path,
) -> None:
    harvest(tmp_path, claim_payload(politifact(), ap()), [cluster(UNRELATED)])
    later = matcher(tmp_path, router({"claims": []}))
    assert len(later.harvest(counter()).packets) == 1


# --- Zombie leads ---------------------------------------------------------------


def test_exact_match_debunk_is_a_high_priority_zombie(tmp_path: Path) -> None:
    [packet] = harvest(
        tmp_path, claim_payload(politifact(), ap()), [cluster(DOCUMENTED)]
    ).packets

    assert packet.adapter_name == "claim_matcher"
    assert packet.anomaly_score == 85.0
    details = packet.receipt_details
    assert details["priority"] == "HIGH"
    assert details["domain"] == 6
    assert details["shingles"]["jaccard"] == 1.0
    assert packet.public_interest_reason == (
        "5-gram MinHash matched a documented claim from November 20, 2024 by "
        "AP News and PolitiFact; the wording is preserved unchanged."
    )


def test_localized_mutation_is_a_zombie_with_its_proof(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, claim_payload(politifact(), ap())).packets

    assert packet.anomaly_score == 85.0
    assert packet.public_question == (
        f"Is the claim '{LOCALIZED}' a recycled narrative previously evaluated by "
        "AP News and PolitiFact?"
    )
    assert packet.public_interest_reason == (
        "5-gram MinHash matched a documented claim from November 20, 2024 by "
        "AP News and PolitiFact; core syntax is preserved with localized details "
        "altered."
    )
    shingles = packet.receipt_details["shingles"]
    assert shingles["altered_spans"] == ["multnomah"]
    assert shingles["jaccard"] >= 0.65
    assert "Altered wording: multnomah" in packet.paste_ready_prompt


def test_unrelated_text_is_not_a_lead(tmp_path: Path) -> None:
    mock = router(claim_payload(politifact(), ap()))
    result = matcher(tmp_path, mock, [cluster(UNRELATED)]).harvest(counter())
    assert result.packets == [] and result.evidence_rejections == []
    assert mock.calls_to(POLITIFACT_URL) == []


def test_question_quotes_a_long_claim_shortly(tmp_path: Path) -> None:
    record = {
        "cluster": cluster(f"{LOCALIZED} {LOCALIZED}"),
        "reviews": [FactCheckReview.from_dict({"publisher_name": "A", "url": AP_URL})],
    }
    question = matcher(tmp_path, router()).default_public_question(record)
    assert "\u2026'" in question and len(question) < 260


@pytest.mark.parametrize(
    ("age_days", "zombie"),
    [
        (ZOMBIE_AGE_DAYS - 1, False),
        (ZOMBIE_AGE_DAYS, False),
        (ZOMBIE_AGE_DAYS + 1, True),
        (900, True),
    ],
)
def test_only_debunks_older_than_180_days_make_a_zombie(
    tmp_path: Path, age_days: int, zombie: bool
) -> None:
    reviewed = FIRST_SEEN.date() - timedelta(days=age_days)
    payload = claim_payload(politifact(reviewed=reviewed), ap(reviewed=reviewed))
    assert bool(harvest(tmp_path, payload).packets) is zombie


def test_recent_review_is_not_sealed_beside_an_old_one(tmp_path: Path) -> None:
    recent = FIRST_SEEN.date() - timedelta(days=30)
    [packet] = harvest(
        tmp_path, claim_payload(politifact(), ap(reviewed=recent))
    ).packets
    assert [item.origin_url for item in packet.source_artifacts] == [POLITIFACT_URL]


@pytest.mark.parametrize(
    "overrides",
    [{"rating": "True"}, {"rating": "Mostly True"}, {"reviewed": None}],
    ids=["true", "mostly-true", "undated"],
)
def test_reviews_that_are_not_dated_debunks_do_not_count(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    payload = claim_payload(politifact(**overrides), ap(**overrides))
    assert harvest(tmp_path, payload).packets == []


def test_fact_check_pages_are_sealed_as_tier_two(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, claim_payload(politifact(), ap())).packets
    assert [(item.origin_url, item.tier) for item in packet.source_artifacts] == [
        (AP_URL, FACT_CHECK_TIER),
        (POLITIFACT_URL, FACT_CHECK_TIER),
    ]
    assert [item.metadata["review_date"] for item in packet.source_artifacts] == [
        "2024-11-20",
        "2024-11-20",
    ]


def test_one_publisher_is_sealed_once(tmp_path: Path) -> None:
    second = politifact() | {"url": f"{POLITIFACT_URL}follow-up/"}
    mock = router(claim_payload(politifact(), second))
    mock.routes[f"{POLITIFACT_URL}follow-up/"] = page("PolitiFact again")
    [packet] = matcher(tmp_path, mock).harvest(counter()).packets
    assert [item.origin_url for item in packet.source_artifacts] == [POLITIFACT_URL]


def test_review_page_off_the_publishers_site_fails_the_evidence_bar(
    tmp_path: Path,
) -> None:
    mirror = "https://mirror.example.net/politifact-copy"
    payload = claim_payload(politifact() | {"url": mirror})
    mock = router(payload)
    mock.routes[mirror] = page("copy")
    result = matcher(tmp_path, mock).harvest(counter())
    assert result.packets == [] and len(result.evidence_rejections) == 1


def test_cluster_from_posts_uses_the_earliest_post() -> None:
    posts = [
        StreamPost("bluesky", LOCALIZED, FIRST_SEEN + timedelta(hours=2)),
        StreamPost("gdelt", DOCUMENTED, FIRST_SEEN),
    ]
    made = NarrativeCluster.from_posts(posts)
    assert made.text == DOCUMENTED and made.first_seen == FIRST_SEEN
    assert made.size == 2 and made.platforms == ("bluesky", "gdelt")
    with pytest.raises(ValueError):
        NarrativeCluster.from_posts([])


def test_documented_claim_key_ignores_case_and_spacing() -> None:
    a = DocumentedClaim(DOCUMENTED, "", None, ())
    b = DocumentedClaim(f" {DOCUMENTED.upper()}  ", "", None, ())
    assert a.key == b.key


# --- Corroboration in triage ----------------------------------------------------


def test_one_fact_check_is_rejected_by_triage(tmp_path: Path, db: Conn) -> None:
    [packet] = harvest(tmp_path, claim_payload(politifact())).packets
    report = triage_packets([packet], "narrative_integrity", db, ollama()[0], NOW)
    assert report.written == []
    assert [(item.gate, item.reason) for item in report.rejected] == [
        ("corroboration", "needs one primary or 2 independent publisher domains")
    ]


def test_independent_publishers_counts_domains_not_urls() -> None:
    def at(url: str) -> Any:
        return replace(FAKE_ARTIFACT, origin_url=url)

    same_outlet = [at(POLITIFACT_URL), at(f"{POLITIFACT_URL}follow-up/")]
    assert independent_publishers(same_outlet) == {"www.politifact.com"}
    assert independent_publishers([at(POLITIFACT_URL), at(AP_URL)]) == {
        "www.politifact.com",
        "apnews.com",
    }
    assert independent_publishers([at("not a url"), at("ftp://x.org/a")]) == set()


def test_two_urls_from_one_outlet_are_rejected_by_triage(
    tmp_path: Path, db: Conn
) -> None:
    [packet] = harvest(tmp_path, claim_payload(politifact())).packets
    [sealed] = packet.source_artifacts
    second = replace(
        sealed, source_id="S-900", origin_url=f"{POLITIFACT_URL}follow-up/"
    )
    packet.source_artifacts = [sealed, second]
    report = triage_packets([packet], "narrative_integrity", db, ollama()[0], NOW)
    assert report.written == []
    assert [item.gate for item in report.rejected] == ["corroboration"]


def test_two_outlets_pass_triage_even_with_the_same_page(
    tmp_path: Path, db: Conn
) -> None:
    [packet] = harvest(tmp_path, claim_payload(politifact())).packets
    [sealed] = packet.source_artifacts
    packet.source_artifacts = [
        sealed,
        replace(sealed, source_id="S-901", origin_url=AP_URL),
    ]
    report = triage_packets([packet], "narrative_integrity", db, ollama()[0], NOW)
    assert report.rejected == [] and report.written == [packet.lead_id]


@pytest.mark.parametrize("extra", [[ap], [ap, reuters]], ids=["two", "three"])
def test_independent_fact_checks_pass_triage(
    tmp_path: Path, db: Conn, extra: list[Any]
) -> None:
    payload = claim_payload(politifact(), *(make() for make in extra))
    [packet] = harvest(tmp_path, payload).packets
    report = triage_packets([packet], "narrative_integrity", db, ollama()[0], NOW)
    assert report.rejected == [] and report.written == [packet.lead_id]


def test_zombie_reaches_the_ledger_with_sealed_reviews(
    tmp_path: Path, db: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    enricher, _ = ollama()
    adapter = matcher(tmp_path, router(claim_payload(politifact(), ap())))
    report = run_adapter(adapter, db, enricher)
    assert report.rejected == [] and len(report.written) == 1

    row = db.execute(
        """
        SELECT anomaly_score, subject_domain, public_question,
               decomposition_receipt -> 'detector', source_ids
        FROM triage_records WHERE adapter_name = 'claim_matcher'
        """
    ).fetchone()
    assert row is not None
    score, domain, question, detector, source_ids = row
    assert (score, domain) == (85.0, "narrative_integrity")
    assert question.startswith("Is the claim 'Poll workers in Multnomah County")
    assert detector["priority"] == "HIGH"
    assert detector["shingles"]["altered_spans"] == ["multnomah"]
    assert [item["review_date"] for item in detector["fact_checks"]] == [
        "2024-11-20",
        "2024-11-20",
    ]

    sealed = db.execute(
        "SELECT url, tier FROM source_registry WHERE source_id = ANY(%s) ORDER BY url",
        (source_ids,),
    ).fetchall()
    assert sealed == [(AP_URL, FACT_CHECK_TIER), (POLITIFACT_URL, FACT_CHECK_TIER)]

    # The same narrative on the next run is already known.
    again = run_adapter(
        matcher(tmp_path, router(claim_payload(politifact(), ap()))), db, enricher
    )
    assert again.written == [] and again.rejected == []


# --- Seldon without a Congress.gov key -------------------------------------------


def fr_router() -> Router:
    return Router(
        {
            FR_API: {"count": 1, "results": [rule()]},
            TEXT_URL: text_response(
                "OIRA has determined that this rule meets the criteria set forth in "
                "5 U.S.C. 804(2)."
            ),
            PDF_URL: pdf_response("Lead Service Line Replacement final rule"),
        }
    )


def test_seldon_runs_keyless_on_federal_register_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    mock = fr_router()
    adapter = SeldonCivicAdapter(**options(tmp_path, mock))
    assert not adapter.polls_congress

    [packet] = adapter.harvest(counter()).packets
    assert packet.adapter_name == "seldon_civic"
    assert {item.url.host for item in mock.requests} == {
        "www.federalregister.gov",
        "www.govinfo.gov",
    }
    assert not any("x-api-key" in item.headers for item in mock.requests)


def test_keyless_seldon_reaches_the_ledger(
    tmp_path: Path, db: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    adapter = SeldonCivicAdapter(**options(tmp_path, fr_router()))
    report = run_adapter(adapter, db, ollama()[0])
    assert report.rejected == [] and len(report.written) == 1


def test_keyless_seldon_leaves_spooled_congress_requests_for_a_keyed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    outage = Router(
        {
            "https://api.congress.gov/": httpx.Response(503),
            FR_API: {"count": 0, "results": []},
        }
    )
    SeldonCivicAdapter("k", **options(tmp_path, outage)).poll_sources()
    spooled = [row[1] for row in spool_rows(tmp_path)]
    assert sorted(spooled) == ["bills", "votes"]

    keyless_mock = Router({FR_API: {"count": 0, "results": []}})
    SeldonCivicAdapter(**options(tmp_path, keyless_mock)).poll_sources()
    assert all(item.url.host != CONGRESS_API_HOST for item in keyless_mock.requests)
    assert sorted(row[1] for row in spool_rows(tmp_path)) == spooled


def test_triage_always_registers_seldon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    seen: list[str] = []

    class Ledger:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def fake_run(adapter: Any, _conn: Any, _enricher: Any) -> Any:
        seen.append(adapter.adapter_name)
        return mycroft_triage.TriageReport()

    monkeypatch.setattr(mycroft_triage, "run_adapter", fake_run)
    monkeypatch.setattr(psycopg, "connect", lambda *_a, **_k: Ledger())
    assert mycroft_triage.main(["--dsn", "dbname=unused"]) == 0
    assert "seldon_civic" in seen
