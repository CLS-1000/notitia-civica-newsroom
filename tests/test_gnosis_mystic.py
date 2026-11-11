"""Gnosis Mystic (Engine 11), tested offline against mocked self-description
pages, Oregon registry rows, ProPublica IRS data, and CourtListener counts.

The payload shapes copy live responses from data.oregon.gov (Socrata, a JSON
array), the ProPublica Nonprofit Explorer v2 API, and CourtListener v4 search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import Conn
from test_phase1_adapters import NOW, Router, counter, ollama, options

from spec1_bridge.adapters.gnosis_mystic import (
    COURTLISTENER_SEARCH,
    LINEAGE_SCORE,
    OREGON_REGISTRY,
    PROPUBLICA_ORG,
    PROPUBLICA_SEARCH,
    REVIEW_SCORE,
    Finances,
    GnosisMysticAdapter,
    Movement,
    Registration,
    finances,
    match_lineages,
    mentions_entheogens,
    registration,
    review_questions,
)
from spec1_bridge.contracts import public_interest
from spec1_bridge.mycroft_triage import run_adapter, triage_packets

TOKEN = "test-courtlistener-token-do-not-spool"
SITE = "https://lightofhermes.example.org/about"
MOVEMENT = Movement("Lodge of the Emerald Tablet", "Portland", "OR", SITE)
EIN = 931234567

HERMETIC_PAGE = (
    "<html><body><h1>About the Lodge</h1><p>We study the Corpus Hermeticum and the "
    "teachings attributed to Hermes Trismegistus. Our monthly circle reads the "
    "Emerald Tablet together: as above, so below.</p></body></html>"
)
ROSICRUCIAN_PAGE = (
    "<p>Our fellowship honours the Rosicrucian manifestos, the Fama Fraternitatis "
    "and the Confessio Fraternitatis, and the legend of Christian Rosenkreuz.</p>"
)
VAGUE_PAGE = "<p>A welcoming circle for meditation, Sophia and inner peace.</p>"
ENTHEOGEN_PAGE = (
    "<p>Retreats with ayahuasca ceremonies in the gnostic spirit; we seek gnosis "
    "of the pleroma.</p>"
)


def registry_row(
    name: str = MOVEMENT.name, entity: str = "DOMESTIC NONPROFIT CORPORATION"
) -> dict[str, Any]:
    return {
        "registry_number": "1234567",
        "business_name": name.upper(),
        "entity_type": entity,
        "registry_date": "1998-03-14T00:00:00.000",
        "city": "PORTLAND",
        "state": "OR",
    }


def irs_org(**filing: Any) -> dict[str, Any]:
    latest = {
        "tax_prd_yr": 2025,
        "totrevenue": 182000,
        "totassetsend": 410000,
        "totliabend": 25000,
        "compnsatncurrofcr": 12000,
        "pdf_url": None,
    } | filing
    return {
        "organization": {
            "ein": EIN,
            "name": MOVEMENT.name,
            "state": "OR",
            "subsection_code": 3,
            "foundation_code": 15,
            "ntee_code": "X20",
        },
        "filings_with_data": [latest, latest | {"tax_prd_yr": 2024}],
    }


def router(
    page: str = HERMETIC_PAGE,
    *,
    registry: list[dict[str, Any]] | None = None,
    irs: list[dict[str, Any]] | None = None,
    org: dict[str, Any] | None = None,
    dockets: int = 0,
) -> Router:
    return Router(
        {
            SITE: httpx.Response(
                200, text=page, headers={"content-type": "text/html; charset=utf-8"}
            ),
            OREGON_REGISTRY: httpx.Response(
                200,
                json=registry if registry is not None else [registry_row()],
            ),
            PROPUBLICA_SEARCH: {
                "total_results": 1,
                "organizations": irs
                if irs is not None
                else [
                    {"ein": EIN, "name": MOVEMENT.name, "state": "OR", "subseccd": 3}
                ],
            },
            f"{PROPUBLICA_ORG}/{EIN}.json": org if org is not None else irs_org(),
            COURTLISTENER_SEARCH: {"count": dockets, "results": []},
        }
    )


def adapter(
    tmp_path: Path,
    mock: Router,
    movements: list[Movement] | None = None,
    token: str | None = TOKEN,
) -> GnosisMysticAdapter:
    return GnosisMysticAdapter(
        movements or [MOVEMENT], token, **options(tmp_path, mock)
    )


def harvest(tmp_path: Path, mock: Router, **kwargs: Any) -> Any:
    return adapter(tmp_path, mock, **kwargs).harvest(counter())


# --- Lineage mapping --------------------------------------------------------------


def test_hermetic_vocabulary_maps_to_the_hermetic_tradition() -> None:
    [found] = match_lineages(HERMETIC_PAGE)
    assert found.lineage.name == "Hermetic"
    assert set(found.terms) >= {
        "corpus hermeticum",
        "hermes trismegistus",
        "emerald tablet",
    }
    assert found.lineage.period == "the 2nd and 3rd centuries CE"
    assert "Corpus Hermeticum" in found.excerpt


def test_rosicrucian_vocabulary_dates_to_1614() -> None:
    [found] = match_lineages(ROSICRUCIAN_PAGE)
    assert (found.lineage.name, found.lineage.period) == ("Rosicrucian", "1614")
    assert "Fama Fraternitatis" in found.lineage.reference


def test_one_shared_word_is_not_a_lineage() -> None:
    assert match_lineages(VAGUE_PAGE) == []
    assert match_lineages("Sophia's gnostic mood ring.") == []


def test_terms_match_whole_words_only() -> None:
    assert match_lineages("Paeony gardens and hermeticallysealed archonite jars.") == []


def test_strongest_lineage_comes_first() -> None:
    text = f"{HERMETIC_PAGE} We also read the Fama Fraternitatis and the Rosicrucian texts."
    names = [item.lineage.name for item in match_lineages(text)]
    assert names == ["Hermetic", "Rosicrucian"]


def test_entheogenic_practice_is_noted_never_mapped() -> None:
    assert mentions_entheogens(ENTHEOGEN_PAGE)
    assert not mentions_entheogens(HERMETIC_PAGE)
    names = [item.lineage.name for item in match_lineages(ENTHEOGEN_PAGE)]
    assert names == ["Gnostic"]  # the gnostic vocabulary, not an indigenous lineage


# --- The ethical line: public-record checks -----------------------------------------


def money(**overrides: Any) -> Finances:
    values: dict[str, Any] = {
        "ein": EIN,
        "subsection": 3,
        "church": False,
        "latest_year": 2025,
        "revenue": 182000.0,
        "assets": 410000.0,
        "liabilities": 25000.0,
        "officer_share": 0.066,
    }
    return Finances(**(values | overrides))


REGISTERED = Registration("1234567", "DOMESTIC NONPROFIT CORPORATION", "1998-03-14")


def test_clean_record_raises_no_questions() -> None:
    assert review_questions(REGISTERED, money(), 0, 2026) == []


@pytest.mark.parametrize(
    ("changes", "question"),
    [
        (
            {"officer_share": 0.41},
            "officer compensation was 41% of revenue in the 2025 Form 990",
        ),
        ({"liabilities": 500000.0}, "liabilities exceeded assets in the 2025 Form 990"),
        ({"subsection": 4}, "IRS record is not a 501(c)(3) organization"),
        ({"latest_year": 2019}, "no Form 990 data for more than three years"),
        ({"latest_year": None}, "no Form 990 data for more than three years"),
    ],
)
def test_financial_checks(changes: dict[str, Any], question: str) -> None:
    assert review_questions(REGISTERED, money(**changes), 0, 2026) == [question]


def test_a_church_that_files_no_990_is_within_the_law() -> None:
    assert (
        review_questions(REGISTERED, money(church=True, latest_year=None), 0, 2026)
        == []
    )


def test_missing_registrations_and_dockets_are_questions() -> None:
    assert review_questions(None, None, 2, 2026) == [
        "no Oregon business registration under this exact name",
        "no IRS exempt-organization record under this exact name",
        "2 federal court docket(s) name the organization",
    ]
    for_profit = Registration("9", "DOMESTIC BUSINESS CORPORATION", "2020-01-01")
    assert review_questions(for_profit, money(), None, 2026) == [
        "registered in Oregon as a domestic business corporation"
    ]


def test_other_states_registries_are_not_held_against_a_movement() -> None:
    assert review_questions(None, money(), 0, 2026, registry_checked=False) == []


def test_registration_and_irs_need_the_exact_name() -> None:
    near = registry_row("Lodge of the Emerald Tablet Bookstore LLC")
    assert registration(MOVEMENT, [near]) is None
    assert registration(MOVEMENT, [registry_row()]) == REGISTERED


def test_finances_read_the_latest_990() -> None:
    found = finances(irs_org(compnsatncurrofcr=91000))
    assert (found.latest_year, found.subsection, found.church) == (2025, 3, False)
    assert found.officer_share == pytest.approx(0.5)
    with pytest.raises(ValueError):
        finances({"error": "not found"})


# --- Polling and credentials ----------------------------------------------------------


def test_token_goes_only_to_courtlistener(tmp_path: Path) -> None:
    mock = router()
    harvest(tmp_path, mock)
    court = mock.calls_to(COURTLISTENER_SEARCH)
    others = [item for item in mock.requests if item not in court]
    assert court and others
    assert all(item.headers["authorization"] == f"Token {TOKEN}" for item in court)
    assert not any("authorization" in item.headers for item in others)
    assert not any(TOKEN in str(item.url) for item in mock.requests)
    assert court[0].url.params["q"] == f'"{MOVEMENT.name}"'


def test_without_a_token_courts_are_not_checked_and_the_reason_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COURTLISTENER_API_TOKEN", raising=False)
    mock = router()
    [packet] = harvest(tmp_path, mock, token=None).packets
    assert mock.calls_to(COURTLISTENER_SEARCH) == []
    assert packet.receipt_details["courts_checked"] is False
    assert packet.public_interest_reason.endswith(
        "though federal court dockets were not checked."
    )


def test_registry_is_queried_by_exact_upper_case_name(tmp_path: Path) -> None:
    mock = router()
    harvest(tmp_path, mock)
    # Polled once, then fetched again when the lead seals the same query.
    queries = {item.url.params["$where"] for item in mock.calls_to(OREGON_REGISTRY)}
    assert queries == {"business_name = 'LODGE OF THE EMERALD TABLET'"}


def test_quotes_in_names_are_escaped_for_the_registry(tmp_path: Path) -> None:
    movement = Movement("St. Germain's Circle", "Salem", "OR", SITE)
    mock = router(registry=[])
    harvest(tmp_path, mock, movements=[movement])
    [query] = mock.calls_to(OREGON_REGISTRY)
    assert query.url.params["$where"] == "business_name = 'ST. GERMAIN''S CIRCLE'"


def test_out_of_state_movement_skips_the_oregon_registry(tmp_path: Path) -> None:
    movement = Movement(MOVEMENT.name, "Seattle", "WA", SITE)
    mock = router()
    [packet] = harvest(tmp_path, mock, movements=[movement]).packets
    assert mock.calls_to(OREGON_REGISTRY) == []
    assert (
        "; IRS records list it as a registered nonprofit"
        in packet.public_interest_reason
    )


# --- Lead generation and Five-Mask standards --------------------------------------------


def test_clean_lineage_lead(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router()).packets
    assert packet.adapter_name == "gnosis_mystic"
    assert packet.anomaly_score == LINEAGE_SCORE
    assert packet.public_question == (
        "What historical esoteric tradition or philosophical lineage informs the "
        "Lodge of the Emerald Tablet in Portland?"
    )
    assert packet.public_interest_reason == (
        "Its own published teachings use the vocabulary of the Hermetic tradition, "
        "whose texts date to the 2nd and 3rd centuries CE; Oregon and IRS records list "
        "it as a registered nonprofit, and its filings show none of the financial "
        "warning signs this desk checks."
    )


def test_five_mask_format(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router()).packets
    question, reason = packet.public_question, packet.public_interest_reason
    assert public_interest(None, None, question, reason) == (question, reason)
    assert question.endswith("?") and reason.endswith(".")
    for text in (question, reason):
        assert text == " ".join(text.split())
    for word in ("cult", "predatory", "coercive", "confirm", "proves", "guarantee"):
        assert word not in f"{question} {reason}".lower()


def test_record_questions_change_the_reason_not_the_question(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(org=irs_org(compnsatncurrofcr=91000))).packets
    assert packet.anomaly_score == REVIEW_SCORE
    assert packet.public_interest_reason.endswith(
        "; its public filings raise questions listed for editorial review."
    )
    assert packet.receipt_details["questions_for_review"] == [
        "officer compensation was 50% of revenue in the 2025 Form 990"
    ]
    assert "cult" not in packet.paste_ready_prompt.split("Do not call")[0].lower()


def test_dockets_naming_the_organization_are_a_question(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(dockets=3)).packets
    assert packet.receipt_details["court_dockets_naming_organization"] == 3
    assert (
        "3 federal court docket(s) name the organization"
        in packet.receipt_details["questions_for_review"]
    )


def test_no_lineage_no_lead(tmp_path: Path) -> None:
    mock = router(VAGUE_PAGE)
    assert harvest(tmp_path, mock).packets == []
    assert mock.calls_to(f"{PROPUBLICA_ORG}/") == []


def test_lead_seals_self_description_registry_and_irs(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router()).packets
    sealed = {item.metadata["kind"]: item for item in packet.source_artifacts}
    assert set(sealed) == {"self_description", "state_registry", "irs_filings"}
    assert sealed["self_description"].tier == "tier_3_secondary"
    assert sealed["state_registry"].tier == "tier_1_primary"
    assert sealed["irs_filings"].tier == "tier_2_institutional"
    [excerpt] = packet.verbatim_excerpts.values()
    assert "Corpus Hermeticum" in excerpt


def test_self_description_alone_fails_the_evidence_bar(tmp_path: Path) -> None:
    mock = router(registry=[], irs=[])
    result = harvest(tmp_path, mock)
    assert result.packets == [] and len(result.evidence_rejections) == 1


def test_receipt_lists_what_was_and_was_not_searched(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(ENTHEOGEN_PAGE)).packets
    details = packet.receipt_details
    assert details["indigenous_entheogenic_practice_mentioned"] is True
    assert [item["lineage"] for item in details["lineages"]] == ["Gnostic"]
    assert "Leaders' names" in details["not_searched"]
    assert "absence of coercion" in details["cannot_show"]
    assert (
        details["irs"]["ein"] == EIN
        and details["state_registration"]["registry_number"] == "1234567"
    )


# --- Triage -------------------------------------------------------------------------------


def test_lineage_lead_reaches_the_ledger(
    tmp_path: Path, db: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    report = run_adapter(adapter(tmp_path, router()), db, ollama()[0])
    assert report.rejected == [] and len(report.written) == 1
    row = db.execute(
        "SELECT anomaly_score, subject_domain FROM triage_records "
        "WHERE adapter_name = 'gnosis_mystic'"
    ).fetchone()
    assert row == (LINEAGE_SCORE, "esoteric_philosophy")

    again = run_adapter(adapter(tmp_path, router()), db, ollama()[0])
    assert again.written == [] and again.rejected == []


def test_lead_without_a_primary_record_needs_two_publishers(
    tmp_path: Path, db: Conn
) -> None:
    movement = Movement(MOVEMENT.name, "Seattle", "WA", SITE)
    [packet] = harvest(tmp_path, router(), movements=[movement]).packets
    # Self-description (example.org) and IRS data (propublica.org) are two publishers.
    report = triage_packets([packet], "esoteric_philosophy", db, ollama()[0], NOW)
    assert report.rejected == [] and report.written == [packet.lead_id]
