"""Seldon's federal sources, tested offline against mocked Congress.gov and Federal
Register payloads.

Every request goes through test_phase1_adapters.Router, which fails the test on any
URL it was not given. The rule texts below reuse the wording of real 2026 final
rules, including the negative findings ("is not a major rule", "does not meet the
criteria") that a naive keyword match would flag.
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
    API_KEY,
    CONGRESS_API,
    ENROLLED_XML,
    Router,
    artifact,
    congress_bill,
    counter,
    ollama,
    options,
    pdf_response,
    text_versions,
)

from spec1_bridge.adapters.seldon_civic import (
    CRA_MAJOR,
    ECONOMICALLY_SIGNIFICANT,
    FR_API,
    SeldonCivicAdapter,
    bill_label,
    compliance_date,
    major_finding,
)
from spec1_bridge.mycroft_triage import run_adapter

DOC = "2026-30001"
PDF_URL = f"https://www.govinfo.gov/content/pkg/FR-2026-10-14/pdf/{DOC}.pdf"
TEXT_URL = (
    f"https://www.federalregister.gov/documents/full_text/text/2026/10/14/{DOC}.txt"
)

MAJOR_TEXT = (
    "Pursuant to Subtitle E of the Small Business Regulatory Enforcement Fairness Act\n"
    "of 1996, also known as the Congressional Review Act, OIRA has determined that\n"
    "this rule meets the criteria set forth in 5 U.S.C.\n804(2). Therefore, this rule "
    "shall be submitted to each House of the Congress."
)
NOT_MAJOR_TEXT = (
    "OIRA has determined that this final rule is not a ``major rule'' as defined by "
    "the Congressional Review Act (CRA), 5 U.S.C. 801-808. This rulemaking is "
    "significant under section 3(f) of Executive Order 12866 but not under section "
    "3(f)(1). The agency considered the Unfunded Mandates Reform Act and the "
    "Congressional Review Act (5 U.S.C. 804(2))."
)


def rule(**overrides: Any) -> dict[str, Any]:
    return {
        "document_number": DOC,
        "title": "Safe Drinking Water: Lead Service Line Replacement.",
        "type": "Rule",
        "action": "Final rule.",
        "abstract": "This rule requires water systems to replace lead service lines.",
        "agencies": [{"name": "Environmental Protection Agency"}],
        "significant": True,
        "publication_date": "2026-10-14",
        "effective_on": "2026-12-01",
        "dates": "This final rule is effective December 1, 2026.",
        "docket_ids": ["EPA-HQ-OW-2026-0001"],
        "regulation_id_numbers": ["2040-AG15"],
        "html_url": f"https://www.federalregister.gov/documents/2026/10/14/{DOC}/x",
        "pdf_url": PDF_URL,
        "raw_text_url": TEXT_URL,
    } | overrides


CBO_ESTIMATE = {
    "title": "H.R. 1234, Water Infrastructure Act",
    "pubDate": "2026-09-20T15:00:00Z",
    "url": "https://www.cbo.gov/publication/61234",
    "description": "As ordered reported by the House Committee on Transportation.",
}


def text_response(body: str) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/plain"})


def router(
    *,
    bills: list[dict[str, Any]] | None = None,
    cbo: Any = None,
    rules: list[dict[str, Any]] | None = None,
    rule_text: str = MAJOR_TEXT,
    rule_pages: list[dict[str, Any]] | None = None,
) -> Router:
    routes: dict[str, Any] = {
        f"{CONGRESS_API}/bill/119?": {"bills": bills if bills is not None else []},
        f"{CONGRESS_API}/house-vote/119/2": {"houseRollCallVotes": []},
        f"{CONGRESS_API}/bill/119/hr/1234?": {
            "bill": {
                "number": "1234",
                "cboCostEstimates": cbo if cbo is not None else [],
            }
        },
        f"{CONGRESS_API}/bill/119/hr/1234/text": text_versions(
            ("Enrolled Bill", "Formatted XML", ENROLLED_XML)
        ),
        ENROLLED_XML: httpx.Response(
            200, content=b"<bill/>", headers={"content-type": "application/xml"}
        ),
        FR_API: {"count": 1, "results": rules if rules is not None else [rule()]},
        TEXT_URL: text_response(rule_text),
        PDF_URL: pdf_response("Lead Service Line Replacement final rule"),
    }
    for number, page in enumerate(rule_pages or [], start=2):
        routes[
            f"https://www.federalregister.gov/api/v1/documents.json?page={number}"
        ] = page
    return Router(routes)


def adapter(tmp_path: Path, mock: Router) -> SeldonCivicAdapter:
    return SeldonCivicAdapter(API_KEY, **options(tmp_path, mock))


def harvest(tmp_path: Path, mock: Router) -> Any:
    return adapter(tmp_path, mock).harvest(counter())


# --- Reading a rule's own findings ----------------------------------------------


@pytest.mark.parametrize(
    ("text", "finding"),
    [
        pytest.param(MAJOR_TEXT, CRA_MAJOR, id="meets-804(2)-across-lines"),
        pytest.param(
            "OIRA has found that this rule meets the criteria at 5 U.S.C. 804(2).",
            CRA_MAJOR,
            id="meets-criteria-at",
        ),
        pytest.param(
            "This final rule is a ``major rule'' under the CRA.",
            CRA_MAJOR,
            id="is-a-major-rule",
        ),
        pytest.param(
            "This final rule has been determined to be economically significant under "
            "Executive Order 12866 section 3(f)(1) and was reviewed by OMB.",
            ECONOMICALLY_SIGNIFICANT,
            id="economically-significant",
        ),
        pytest.param(
            "This rule is significant under section 3(f)(1) because it is economically "
            "significant.",
            ECONOMICALLY_SIGNIFICANT,
            id="3(f)(1)-first",
        ),
    ],
)
def test_affirmative_findings_are_found(text: str, finding: str) -> None:
    result = major_finding(text)
    assert result is not None and result[0] == finding


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(NOT_MAJOR_TEXT, id="is-not-major-and-3f-but-not-3f1"),
        pytest.param(
            "OIRA has determined this rule does not meet the criteria listed in "
            "5 U.S.C. 804(2).",
            id="does-not-meet",
        ),
        pytest.param(
            "The agency finds the rule is not economically significant under section "
            "3(f)(1) of E.O. 12866.",
            id="not-economically-significant",
        ),
        pytest.param("No finding is stated here.", id="nothing"),
    ],
)
def test_negative_and_passing_mentions_are_not_findings(text: str) -> None:
    assert major_finding(text) is None


def test_congressional_review_act_finding_is_preferred() -> None:
    text = (
        "This rule is economically significant under E.O. 12866 section 3(f)(1). "
        "OIRA has determined that this rule meets the criteria set forth in "
        "5 U.S.C. 804(2)."
    )
    assert major_finding(text) == (
        CRA_MAJOR,
        "meets the criteria set forth in 5 U.S.C. 804(2)",
    )


@pytest.mark.parametrize(
    ("dates", "expected"),
    [
        ("This rule is effective December 7, 2026.", None),
        (
            (
                "Effective Date: This rule is effective December 7, 2026. Compliance "
                "Dates: Compliance with the reporting requirement is required by March 1, "
                "2027."
            ),
            date(2027, 3, 1),
        ),
        (
            (
                "Compliance date: Regulated entities must comply by June 1, 2027, and "
                "small entities by January 4, 2027."
            ),
            date(2027, 1, 4),
        ),
        ("Compliance with these rules is required by February 30, 2027.", None),
        ("", None),
    ],
)
def test_compliance_date_comes_only_from_compliance_sentences(
    dates: str, expected: date | None
) -> None:
    assert compliance_date(dates) == expected


@pytest.mark.parametrize(
    ("kind", "label"),
    [
        ("HR", "H.R. 12"),
        ("S", "S. 12"),
        ("hjres", "H.J.Res. 12"),
        ("SRES", "S.Res. 12"),
    ],
)
def test_bill_labels(kind: str, label: str) -> None:
    assert bill_label({"type": kind, "number": "12"}) == label


# --- Polling the Federal Register ------------------------------------------------


def test_rules_query_asks_for_significant_final_rules(tmp_path: Path) -> None:
    mock = router()
    adapter(tmp_path, mock).poll_sources()
    [request] = mock.calls_to(FR_API)
    query = parse_qs(urlsplit(str(request.url)).query)
    assert query["conditions[type][]"] == ["RULE"]
    assert query["conditions[significant]"] == ["1"]
    assert query["conditions[publication_date][gte]"] == ["2026-10-08"]
    assert {
        "document_number",
        "effective_on",
        "dates",
        "raw_text_url",
        "pdf_url",
    } <= set(query["fields[]"])


def test_congress_key_never_goes_to_the_federal_register(tmp_path: Path) -> None:
    mock = router(bills=[congress_bill()], cbo=[CBO_ESTIMATE])
    harvest(tmp_path, mock)
    for request in mock.requests:
        sent = "x-api-key" in request.headers
        assert sent == (request.url.host == "api.congress.gov"), request.url


def test_rules_follow_next_page_url(tmp_path: Path) -> None:
    second = rule(document_number="2026-30002")
    page_two = {"count": 2, "results": [second]}
    first_page = {
        "count": 2,
        "results": [rule()],
        "next_page_url": "https://www.federalregister.gov/api/v1/documents.json?page=2",
    }
    mock = router(rule_pages=[page_two])
    mock.routes[FR_API] = first_page
    records = adapter(tmp_path, mock).poll_sources()
    assert [item["rule"]["document_number"] for item in records] == [DOC, "2026-30002"]


@pytest.mark.parametrize(
    "row",
    [
        rule(type="Proposed Rule"),
        rule(significant=False),
        rule(document_number=None),
    ],
    ids=["proposed", "not-significant", "no-number"],
)
def test_only_significant_final_rules_are_records(
    tmp_path: Path, row: dict[str, Any]
) -> None:
    assert adapter(tmp_path, router(rules=[row])).poll_sources() == []


# --- Rule leads ------------------------------------------------------------------


def test_major_rule_taking_effect_soon(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router()).packets
    assert packet.anomaly_score == 90.0
    assert packet.public_question == (
        "What does the rule “Safe Drinking Water: Lead Service Line Replacement” "
        "change in federal law or regulation?"
    )
    assert packet.public_interest_reason == (
        "This major federal rule takes effect on December 1, 2026. The rule states it "
        f"is {CRA_MAJOR}."
    )
    prompt = packet.paste_ready_prompt
    assert f"Federal Register final rule {DOC}" in prompt
    assert "Compliance starts within 90 days: yes, 2026-12-01" in prompt
    assert f"Major or 3(f)(1) finding: {CRA_MAJOR}" in prompt
    assert "Docket: EPA-HQ-OW-2026-0001" in prompt
    assert "whether it adds or removes obligations" in prompt


def test_rule_lead_seals_the_rule_and_cites_its_finding(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router()).packets
    kinds = {item.metadata["kind"]: item for item in packet.source_artifacts}
    assert set(kinds) == {"fr_rule_pdf", "fr_rule_text"}
    assert kinds["fr_rule_pdf"].origin_url == PDF_URL
    text = kinds["fr_rule_text"]
    assert packet.verbatim_excerpts == {
        text.source_id: "meets the criteria set forth in 5 U.S.C. 804(2)"
    }


def test_explicit_compliance_date_is_named_as_compliance(tmp_path: Path) -> None:
    dates = (
        "This rule is effective December 1, 2026. Compliance date: Systems must comply "
        "by January 4, 2027."
    )
    [packet] = harvest(tmp_path, router(rules=[rule(dates=dates)])).packets
    assert packet.public_interest_reason.startswith(
        "Compliance with this major federal rule is required starting January 4, 2027."
    )


def test_past_compliance_date_falls_back_to_effective_date(tmp_path: Path) -> None:
    dates = "Compliance date: Systems were required to comply by January 4, 2026."
    [packet] = harvest(tmp_path, router(rules=[rule(dates=dates)])).packets
    assert packet.public_interest_reason.startswith(
        "This major federal rule takes effect on December 1, 2026."
    )


def test_major_rule_with_a_distant_date(tmp_path: Path) -> None:
    [packet] = harvest(
        tmp_path, router(rules=[rule(effective_on="2027-03-01")])
    ).packets
    assert packet.anomaly_score == 70.0
    assert packet.public_interest_reason == f"The rule states it is {CRA_MAJOR}."


def test_significant_rule_taking_effect_soon(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(rule_text=NOT_MAJOR_TEXT)).packets
    assert packet.anomaly_score == 50.0
    assert packet.public_interest_reason == (
        "This significant federal rule takes effect on December 1, 2026."
    )
    assert "Major or 3(f)(1) finding: none stated" in packet.paste_ready_prompt
    assert packet.verbatim_excerpts == {}


@pytest.mark.parametrize(
    "effective_on",
    ["2027-03-01", "2026-09-01", None],
    ids=["too-far", "already-in-effect", "no-date"],
)
def test_ordinary_significant_rule_is_not_a_lead(
    tmp_path: Path, effective_on: str | None
) -> None:
    mock = router(rules=[rule(effective_on=effective_on)], rule_text=NOT_MAJOR_TEXT)
    result = harvest(tmp_path, mock)
    assert result.packets == [] and result.errors == []


def test_rule_without_full_text_is_judged_on_dates_alone(tmp_path: Path) -> None:
    mock = router(rules=[rule(raw_text_url=None)])
    [packet] = harvest(tmp_path, mock).packets
    assert packet.anomaly_score == 50.0
    assert mock.calls_to(TEXT_URL) == []
    assert {item.metadata["kind"] for item in packet.source_artifacts} == {
        "fr_rule_pdf"
    }


def test_rule_with_nothing_to_seal_is_not_a_lead(tmp_path: Path) -> None:
    mock = router(rules=[rule(raw_text_url=None, pdf_url=None)])
    assert harvest(tmp_path, mock).packets == []


def test_rule_text_outage_is_retried(tmp_path: Path) -> None:
    mock = router()
    mock.routes[TEXT_URL] = httpx.Response(503)
    result = harvest(tmp_path, mock)
    assert result.packets == []
    assert len(result.errors) == 1 and f"fr:{DOC}" in result.errors[0]


# --- The statutory evidence bar for rules ------------------------------------------


@pytest.mark.parametrize(
    ("url", "kind", "number", "tier", "ok"),
    [
        (PDF_URL, "fr_rule_pdf", DOC, "tier_1_primary", True),
        (TEXT_URL, "fr_rule_text", DOC, "tier_1_primary", True),
        (
            f"https://evil.example.com/{DOC}.pdf",
            "fr_rule_pdf",
            DOC,
            "tier_1_primary",
            False,
        ),
        (PDF_URL, "fr_rule_pdf", "2026-99999", "tier_1_primary", False),
        (
            f"https://www.federalregister.gov/documents/2026/10/14/{DOC}/x",
            "fr_rule_text",
            DOC,
            "tier_1_primary",
            False,
        ),
        (PDF_URL, "fr_rule_pdf", DOC, "tier_2_institutional", False),
        (PDF_URL, "fr_rule_text", DOC, "tier_1_primary", False),
    ],
    ids=[
        "govinfo-pdf",
        "fr-full-text",
        "host",
        "wrong-document",
        "not-full-text",
        "tier",
        "kind",
    ],
)
def test_rule_evidence_bar(
    tmp_path: Path, url: str, kind: str, number: str, tier: str, ok: bool
) -> None:
    item = artifact(
        tmp_path,
        origin_url=url,
        tier=tier,
        metadata={"kind": kind, "document_number": number},
    )
    assert adapter(tmp_path, router()).verify_evidence_bar([item]) is ok


# --- Bills and CBO estimates ----------------------------------------------------------


def test_bill_with_cbo_estimate_links_it(tmp_path: Path) -> None:
    mock = router(bills=[congress_bill()], cbo=[CBO_ESTIMATE], rules=[])
    [packet] = harvest(tmp_path, mock).packets
    assert packet.anomaly_score == 85.0
    assert packet.public_question == (
        "What does H.R. 1234 change in federal law or regulation?"
    )
    assert packet.public_interest_reason == (
        "The Congressional Budget Office published a cost estimate for H.R. 1234 on "
        "September 20, 2026 (https://www.cbo.gov/publication/61234)."
    )
    assert (
        "CBO cost estimates: H.R. 1234, Water Infrastructure Act (2026-09-20): "
        "https://www.cbo.gov/publication/61234"
    ) in packet.paste_ready_prompt


def test_cbo_amounts_are_never_stated(tmp_path: Path) -> None:
    mock = router(bills=[congress_bill()], cbo=[CBO_ESTIMATE], rules=[])
    [packet] = harvest(tmp_path, mock).packets
    assert "$" not in packet.public_interest_reason
    assert not any(request.url.host == "www.cbo.gov" for request in mock.requests)


def test_latest_cbo_estimate_is_the_one_cited(tmp_path: Path) -> None:
    older = CBO_ESTIMATE | {
        "pubDate": "2026-03-02T00:00:00Z",
        "url": "https://www.cbo.gov/publication/60001",
    }
    mock = router(bills=[congress_bill()], cbo=[CBO_ESTIMATE, older], rules=[])
    [packet] = harvest(tmp_path, mock).packets
    assert packet.public_interest_reason.endswith(
        "(https://www.cbo.gov/publication/61234)."
    )


def test_cbo_estimates_in_item_form_are_read(tmp_path: Path) -> None:
    mock = router(bills=[congress_bill()], cbo={"item": [CBO_ESTIMATE]}, rules=[])
    [packet] = harvest(tmp_path, mock).packets
    assert packet.anomaly_score == 85.0


def test_bill_without_cbo_estimate_keeps_its_reason(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(bills=[congress_bill()], rules=[])).packets
    assert packet.anomaly_score == 75.0
    assert packet.public_interest_reason.startswith("Enrolled federal legislation")
    assert "CBO cost estimates: none listed" in packet.paste_ready_prompt


def test_bill_detail_is_not_fetched_for_bills_in_committee(tmp_path: Path) -> None:
    mock = router(bills=[congress_bill("Referred to the Committee.")], rules=[])
    assert harvest(tmp_path, mock).packets == []
    assert mock.calls_to(f"{CONGRESS_API}/bill/119/hr/1234?") == []


def test_cbo_estimate_cannot_replace_the_enrolled_text(tmp_path: Path) -> None:
    mock = router(bills=[congress_bill()], cbo=[CBO_ESTIMATE], rules=[])
    mock.routes[f"{CONGRESS_API}/bill/119/hr/1234/text"] = {"textVersions": []}
    assert harvest(tmp_path, mock).packets == []


# --- End to end ------------------------------------------------------------------------


def test_bill_and_rule_reach_the_ledger(
    db: Conn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    enricher, _ = ollama()
    mock = router(bills=[congress_bill()], cbo=[CBO_ESTIMATE])
    report = run_adapter(adapter(tmp_path, mock), db, enricher)

    assert report.rejected == [] and len(report.written) == 2
    rows = db.execute(
        """
        SELECT anomaly_score, public_question FROM triage_records
        WHERE adapter_name = 'seldon_civic' ORDER BY anomaly_score
        """
    ).fetchall()
    assert rows == [
        (85.0, "What does H.R. 1234 change in federal law or regulation?"),
        (
            90.0,
            (
                "What does the rule “Safe Drinking Water: Lead Service Line "
                "Replacement” change in federal law or regulation?"
            ),
        ),
    ]
