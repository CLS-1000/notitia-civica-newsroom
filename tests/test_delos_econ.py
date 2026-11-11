"""Delos shortage-cascade predictor, tested offline against mocked openFDA payloads.

Every request goes through test_phase1_adapters.Router, which fails the test on any
URL it was not given, so nothing here touches the network.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from conftest import Conn
from test_phase1_adapters import (
    Router,
    counter,
    fda_payload,
    ollama,
    options,
)

from spec1_bridge.adapters.delos_econ import (
    ENFORCEMENT_URL,
    NDC_URL,
    PORTWATCH_URL,
    SHORTAGES_URL,
    DelosEconAdapter,
    approved_sources,
    recall_generics,
    recall_scope,
)
from spec1_bridge.mycroft_triage import run_adapter

RECALL = {
    "recall_number": "D-0200-2026",
    "classification": "Class II",
    "status": "Ongoing",
    "recalling_firm": "Acme Sterile LLC",
    "product_description": "Vecuronium Bromide for Injection, 10 mg single-dose vial",
    "reason_for_recall": "Lack of sterility assurance: microbial growth in media fill",
    "report_date": "20261001",
    "openfda": {"generic_name": ["VECURONIUM BROMIDE"], "route": ["INTRAVENOUS"]},
}
SHORTAGE = {
    "generic_name": "Vecuronium Bromide for Injection",
    "company_name": "Acme Sterile LLC",
    "availability": "Limited Availability",
    "presentation": "Vecuronium Bromide, Injection, 10 mg vial (NDC 12345-678-01)",
    "status": "Current",
}


def listing(
    application: str,
    labeler: str = "Acme Sterile LLC",
    *,
    category: str = "ANDA",
    expires: str = "20271231",
    finished: bool = True,
) -> dict[str, Any]:
    return {
        "generic_name": "Vecuronium Bromide",
        "labeler_name": labeler,
        "application_number": application,
        "marketing_category": category,
        "finished": finished,
        "listing_expiration_date": expires,
        "dosage_form": "INJECTION, POWDER, LYOPHILIZED, FOR SOLUTION",
    }


ONE_SOURCE = [listing("ANDA090001")]
TWO_SOURCES = [listing("ANDA090001"), listing("ANDA090002", "Beta Pharma")]
THREE_SOURCES = [*TWO_SOURCES, listing("ANDA090003", "Gamma Labs")]


def router(
    *,
    recalls: list[dict[str, Any]] | None = None,
    shortages: list[dict[str, Any]] | None = None,
    ndc: list[dict[str, Any]] | httpx.Response | None = None,
) -> Router:
    recall_rows = recalls if recalls is not None else [RECALL]
    shortage_rows = shortages if shortages is not None else []
    ndc_route: Any = (
        ndc if isinstance(ndc, httpx.Response) else fda_payload(ndc or ONE_SOURCE)
    )
    return Router(
        {
            f"{ENFORCEMENT_URL}?search=classification": fda_payload(recall_rows),
            f"{SHORTAGES_URL}?search=status": fda_payload(shortage_rows),
            f"{ENFORCEMENT_URL}?search=recall_number": fda_payload(recall_rows[:1]),
            f"{SHORTAGES_URL}?search=generic_name": fda_payload(shortage_rows[:1]),
            NDC_URL: ndc_route,
            f"{PORTWATCH_URL}?where=1%3D1": {"features": []},
        }
    )


def adapter(tmp_path: Path, mock: Router) -> DelosEconAdapter:
    return DelosEconAdapter(**options(tmp_path, mock))


def harvest(tmp_path: Path, mock: Router) -> Any:
    return adapter(tmp_path, mock).harvest(counter())


# --- Which recalls can start a cascade ---------------------------------------


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Heparin Sodium API, 25 kg drum", "api"),
        ("Active Pharmaceutical Ingredient: Cisplatin, bulk", "api"),
        ("Bulk drug substance lot 44", "api"),
        ("Vecuronium Bromide for Injection, 10 mg", "sterile_injectable"),
        ("Cefazolin Sodium Injection, 1 g", "sterile_injectable"),
        ("Sterile Water, 10 mL vial", "sterile_injectable"),
        ("Potassium Chloride in Dextrose infusion bag", "sterile_injectable"),
        ("Metformin Hydrochloride Tablets, 500 mg", None),
        ("Rapid Release Capsules", None),
    ],
)
def test_recall_scope_reads_the_product(description: str, expected: str | None) -> None:
    assert recall_scope({"product_description": description}) == expected


def test_recall_scope_reads_parenteral_routes() -> None:
    recall = {
        "product_description": "Vecuronium 10 mg",
        "openfda": {"route": ["intravenous"]},
    }
    assert recall_scope(recall) == "sterile_injectable"
    oral = {"product_description": "Vecuronium 10 mg", "openfda": {"route": ["ORAL"]}}
    assert recall_scope(oral) is None


def test_recall_scope_survives_a_malformed_openfda_block() -> None:
    assert recall_scope({"product_description": "Pills", "openfda": "oops"}) is None


def test_recall_generics_normalizes_and_dedupes() -> None:
    recall = {
        "openfda": {
            "generic_name": ["VECURONIUM BROMIDE", "Vecuronium Bromide Injection", " "]
        }
    }
    assert recall_generics(recall) == ["vecuronium bromide"]
    assert recall_generics({}) == []


# --- Counting approved sources -----------------------------------------------


TODAY = date(2026, 10, 15)


def test_repackagers_on_one_application_count_once() -> None:
    rows = [
        listing("ANDA090001", "Acme Sterile LLC"),
        listing("ANDA090001", "Repack Health"),
        listing("anda090001", "Hospital Relabel Co"),
    ]
    sources = approved_sources(rows, TODAY)
    assert sources["applications"] == ["ANDA090001"]
    assert sources["labelers"] == [
        "Acme Sterile LLC",
        "Hospital Relabel Co",
        "Repack Health",
    ]


@pytest.mark.parametrize(
    "row",
    [
        listing("ANDA090009", expires="20260101"),
        listing("", category="ANDA"),
        listing("PART000001", category="UNAPPROVED DRUG OTHER"),
        listing("ANDA090009", finished=False),
    ],
    ids=["expired", "no-application", "unapproved", "unfinished"],
)
def test_listings_that_are_not_approved_current_sources_do_not_count(
    row: dict[str, Any],
) -> None:
    assert approved_sources([listing("ANDA090001"), row], TODAY)["applications"] == [
        "ANDA090001"
    ]


@pytest.mark.parametrize("category", ["NDA", "NDA AUTHORIZED GENERIC", "BLA", "anda"])
def test_other_approved_categories_count(category: str) -> None:
    rows = [listing("APP1", category=category)]
    assert approved_sources(rows, TODAY)["applications"] == ["APP1"]


def test_a_listing_with_no_expiry_still_counts() -> None:
    row = listing("ANDA090001") | {"listing_expiration_date": None}
    assert approved_sources([row], TODAY)["applications"] == ["ANDA090001"]


# --- Assembling candidates -----------------------------------------------------


def test_injectable_recall_is_a_candidate_before_any_shortage(tmp_path: Path) -> None:
    records = adapter(tmp_path, router()).poll_sources()
    assert len(records) == 1
    record = records[0]
    assert (record["generic"], record["scope"], record["shortages"]) == (
        "vecuronium bromide",
        "sterile_injectable",
        [],
    )
    assert record["company"] is None


def test_oral_recall_without_a_shortage_is_not_a_candidate(tmp_path: Path) -> None:
    recall = RECALL | {
        "product_description": "Vecuronium Bromide Tablets",
        "openfda": {"generic_name": ["VECURONIUM BROMIDE"], "route": ["ORAL"]},
    }
    assert adapter(tmp_path, router(recalls=[recall])).poll_sources() == []


def test_class_iii_recall_is_not_a_candidate(tmp_path: Path) -> None:
    recall = RECALL | {"classification": "Class III"}
    assert adapter(tmp_path, router(recalls=[recall])).poll_sources() == []


def test_shortage_and_openfda_names_make_one_candidate(tmp_path: Path) -> None:
    records = adapter(tmp_path, router(shortages=[SHORTAGE])).poll_sources()
    assert [item["generic"] for item in records] == ["vecuronium bromide"]
    assert records[0]["company"] == "Acme Sterile LLC"


# --- Scoring and the lead -----------------------------------------------------


@pytest.mark.parametrize(
    ("classification", "ndc", "shortages", "score"),
    [
        ("Class II", TWO_SOURCES, [], 80.0),
        ("Class II", ONE_SOURCE, [], 85.0),
        ("Class II", TWO_SOURCES, [SHORTAGE], 85.0),
        ("Class I", TWO_SOURCES, [], 90.0),
        ("Class I", ONE_SOURCE, [SHORTAGE], 100.0),
    ],
)
def test_cascade_scores_at_least_eighty(
    tmp_path: Path,
    classification: str,
    ndc: list[dict[str, Any]],
    shortages: list[dict[str, Any]],
    score: float,
) -> None:
    recall = RECALL | {"classification": classification}
    result = harvest(tmp_path, router(recalls=[recall], shortages=shortages, ndc=ndc))
    assert result.errors == []
    assert [packet.anomaly_score for packet in result.packets] == [score]


def test_cascade_lead_states_priority_question_and_reason(tmp_path: Path) -> None:
    packet = harvest(tmp_path, router(ndc=TWO_SOURCES)).packets[0]
    assert packet.public_question == (
        "Could patients face a regional shortage of vecuronium bromide following "
        "FDA recall D-0200-2026?"
    )
    assert packet.public_interest_reason == (
        "Vecuronium bromide has only two approved sources listed in the FDA NDC "
        "Directory, so a recall can leave patients with few or no alternatives."
    )
    prompt = packet.paste_ready_prompt
    assert "Possible shortage cascade: vecuronium bromide" in prompt
    assert "Priority: HIGH" in prompt
    assert "Approved sources in NDC Directory: 2 (ANDA090001, ANDA090002)" in prompt
    assert "On FDA shortage list: not listed" in prompt
    assert "an NDC listing does not prove supply" in prompt


def test_single_source_reason_says_one(tmp_path: Path) -> None:
    packet = harvest(tmp_path, router(ndc=ONE_SOURCE)).packets[0]
    assert "has only one approved source listed" in packet.public_interest_reason


def test_cascade_seals_the_recall_and_the_ndc_query(tmp_path: Path) -> None:
    mock = router(ndc=TWO_SOURCES)
    packet = harvest(tmp_path, mock).packets[0]
    kinds = {item.metadata["kind"] for item in packet.source_artifacts}
    assert kinds == {"fda_enforcement", "fda_ndc"}

    sealed = next(i for i in packet.source_artifacts if i.metadata["kind"] == "fda_ndc")
    assert sealed.tier == "tier_1_primary"
    assert sealed.metadata["applications"] == ["ANDA090001", "ANDA090002"]
    query = parse_qs(urlsplit(sealed.origin_url).query)
    assert query["search"] == ['generic_name:"vecuronium bromide" AND finished:true']

    enrichment = mock.calls_to(NDC_URL)[0]
    assert enrichment.url.params["search"] == query["search"][0]


def test_cascade_in_shortage_also_seals_the_shortage(tmp_path: Path) -> None:
    packet = harvest(tmp_path, router(shortages=[SHORTAGE], ndc=ONE_SOURCE)).packets[0]
    kinds = {item.metadata["kind"] for item in packet.source_artifacts}
    assert kinds == {"fda_enforcement", "fda_shortage", "fda_ndc"}
    assert "On FDA shortage list: Limited Availability" in packet.paste_ready_prompt


def test_three_sources_is_not_a_cascade(tmp_path: Path) -> None:
    result = harvest(tmp_path, router(ndc=THREE_SOURCES))
    assert result.packets == [] and result.errors == []


def test_three_sources_still_leaves_a_shortage_bottleneck(tmp_path: Path) -> None:
    result = harvest(tmp_path, router(shortages=[SHORTAGE], ndc=THREE_SOURCES))
    [packet] = result.packets
    assert "Possible single-source bottleneck" in packet.paste_ready_prompt
    assert "Priority: HIGH" not in packet.paste_ready_prompt
    assert packet.public_question == (
        "Could patients lose access to vecuronium bromide after recall D-0200-2026?"
    )


def test_repackagers_do_not_hide_a_single_source(tmp_path: Path) -> None:
    rows = [listing("ANDA090001", name) for name in ("Acme", "Repack A", "Repack B")]
    [packet] = harvest(tmp_path, router(ndc=rows)).packets
    assert packet.anomaly_score == 85.0


def test_expired_listings_can_reveal_a_duopoly(tmp_path: Path) -> None:
    rows = [*TWO_SOURCES, listing("ANDA090003", "Gone Labs", expires="20250101")]
    [packet] = harvest(tmp_path, router(ndc=rows)).packets
    assert packet.anomaly_score == 80.0


def test_no_ndc_listing_is_not_flagged(tmp_path: Path) -> None:
    not_found = httpx.Response(
        404, json={"error": {"code": "NOT_FOUND", "message": "No matches found!"}}
    )
    result = harvest(tmp_path, router(ndc=not_found))
    assert result.packets == [] and result.errors == []


def test_ndc_outage_is_retried_not_flagged(tmp_path: Path) -> None:
    result = harvest(tmp_path, router(ndc=httpx.Response(503)))
    assert result.packets == []
    assert len(result.errors) == 1 and "D-0200-2026" in result.errors[0]


def test_ndc_query_is_not_made_for_oral_bottlenecks(tmp_path: Path) -> None:
    recall = RECALL | {
        "product_description": "Vecuronium Bromide Tablets",
        "openfda": {"generic_name": ["VECURONIUM BROMIDE"], "route": ["ORAL"]},
    }
    mock = router(recalls=[recall], shortages=[SHORTAGE])
    [packet] = harvest(tmp_path, mock).packets
    assert mock.calls_to(NDC_URL) == []
    assert "Possible single-source bottleneck" in packet.paste_ready_prompt


# --- End to end ----------------------------------------------------------------


def test_cascade_reaches_the_ledger(
    db: Conn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    enricher, _ = ollama()
    report = run_adapter(adapter(tmp_path, router(ndc=TWO_SOURCES)), db, enricher)

    assert report.rejected == [] and len(report.written) == 1
    row = db.execute(
        """
        SELECT anomaly_score, public_question, public_interest_reason,
               paste_ready_prompt, cardinality(source_ids)
        FROM triage_records WHERE adapter_name = 'delos_econ'
        """
    ).fetchone()
    assert row is not None
    score, question, reason, prompt, sources = row
    assert score == 80.0
    assert question.startswith("Could patients face a regional shortage of ")
    assert reason.startswith("Vecuronium bromide has only two approved sources")
    assert "Priority: HIGH" in prompt
    assert sources == 2
