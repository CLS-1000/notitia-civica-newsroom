"""Klein Biomed (Engine 4), tested offline against mocked PubMed, ClinicalTrials.gov,
and openFDA responses.

Every adapter request goes through test_phase1_adapters.Router, which fails the
test on any URL it was not given, and the autouse guard below fails any request
or socket connection that bypasses the mocks.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import httpx
import pytest
from test_phase1_adapters import Router, counter, options

from spec1_bridge.adapters.base_lazlong import PollRequest
from spec1_bridge.adapters.klein_biomed import (
    CTGOV_REQUEST,
    CTGOV_SEARCH,
    CTGOV_STUDY,
    EFETCH,
    ESEARCH,
    FDA_REQUEST,
    KIND_CTGOV,
    KIND_FDA,
    KIND_PUBMED,
    OPENFDA_DRUGSFDA,
    PAGE_SIZE,
    PRIMARY_TIER,
    PUBMED_REQUEST,
    UNAVAILABLE,
    KleinBiomedAdapter,
    assess_trial,
    ctgov_url,
    efetch_url,
    endpoint_is_harm,
    fda_application_url,
    fda_approvals,
    format_percent,
    p_value_phrase,
    p_value_significant,
    parse_pubmed_xml,
    pubmed_ineligibility,
    registry_pmids,
    trial_metrics,
    trial_public_text,
)

PMID = "39000001"
NCT = "NCT04882191"
ENDPOINT = "Number of Participants With Cardiovascular Death or Hospitalization"
# 558/3305 - 432/3304 = 0.168835... - 0.130751... = 3.8084 percentage points
ARR = (558 / 3305 - 432 / 3304) * 100
RCT = ("D016449", "Randomized Controlled Trial")
JOURNAL_ARTICLE = ("D016428", "Journal Article")
HUMANS = ("D006801", "Humans")
ANIMALS = ("D000818", "Animals")
APPLICATION = "NDA219876"


# --- Network guard --------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Fail any real HTTP request or socket connection."""
    attempts: list[str] = []

    def refuse_http(_self: Any, request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        raise AssertionError(f"real network request: {request.url}")

    def refuse_socket(_self: Any, address: Any) -> None:
        attempts.append(str(address))
        raise AssertionError(f"real socket connection: {address}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse_http)
    monkeypatch.setattr(socket.socket, "connect", refuse_socket)
    yield attempts
    assert attempts == []


# --- Fixtures shaped like live responses ------------------------------------------


def pubmed_xml(
    pmid: str = PMID,
    *,
    status: str = "MEDLINE",
    types: tuple[tuple[str, str], ...] = (JOURNAL_ARTICLE, RCT),
    mesh: tuple[tuple[str, str], ...] = (HUMANS,),
    ncts: tuple[str, ...] = (NCT,),
    grants: tuple[str, ...] = ("NHLBI NIH HHS",),
    title: str = "Empagliflozin in Patients with Chronic Kidney Disease.",
    electronic: tuple[str, str, str] | None = ("2026", "10", "01"),
    history: tuple[str, str, str] | None = ("2026", "10", "02"),
    retracted: bool = False,
) -> bytes:
    def ymd(tag: str, attrs: str, parts: tuple[str, str, str]) -> str:
        year, month, day = parts
        return (
            f"<{tag} {attrs}><Year>{year}</Year><Month>{month}</Month>"
            f"<Day>{day}</Day></{tag}>"
        )

    banks = (
        "<DataBankList><DataBank><DataBankName>ClinicalTrials.gov</DataBankName>"
        "<AccessionNumberList>"
        + "".join(f"<AccessionNumber>{item}</AccessionNumber>" for item in ncts)
        + "</AccessionNumberList></DataBank></DataBankList>"
        if ncts
        else ""
    )
    grant_list = (
        "<GrantList>"
        + "".join(f"<Grant><Agency>{escape(item)}</Agency></Grant>" for item in grants)
        + "</GrantList>"
        if grants
        else ""
    )
    corrections = (
        '<CommentsCorrectionsList><CommentsCorrections RefType="RetractionIn">'
        "<RefSource>Lancet. 2026</RefSource></CommentsCorrections>"
        "</CommentsCorrectionsList>"
        if retracted
        else ""
    )
    return (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<PubmedArticleSet><PubmedArticle>"
        f'<MedlineCitation Status="{status}" Owner="NLM">'
        f'<PMID Version="1">{pmid}</PMID>'
        "<Article><Journal><Title>The New England journal of medicine</Title></Journal>"
        f"<ArticleTitle>{escape(title)}</ArticleTitle>"
        + grant_list
        + banks
        + "<PublicationTypeList>"
        + "".join(
            f'<PublicationType UI="{ui}">{name}</PublicationType>' for ui, name in types
        )
        + "</PublicationTypeList>"
        + (
            ymd("ArticleDate", 'DateType="Electronic"', electronic)
            if electronic
            else ""
        )
        + "</Article>"
        + "<MeshHeadingList>"
        + "".join(
            f'<MeshHeading><DescriptorName UI="{ui}">{name}</DescriptorName></MeshHeading>'
            for ui, name in mesh
        )
        + "</MeshHeadingList>"
        + corrections
        + "</MedlineCitation><PubmedData><History>"
        + (ymd("PubMedPubDate", 'PubStatus="pubmed"', history) if history else "")
        + "</History></PubmedData></PubmedArticle></PubmedArticleSet>"
    ).encode()


def study(
    nct: str = NCT,
    *,
    enrollment: dict[str, Any] | None = None,
    allocation: str = "RANDOMIZED",
    study_type: str = "INTERVENTIONAL",
    conditions: list[str] | None = None,
    sponsor: dict[str, Any] | None = None,
    endpoint: str = ENDPOINT,
    treatment: tuple[str, str] = ("432", "3304"),
    comparator: tuple[str, str] = ("558", "3305"),
    p_values: tuple[str, ...] = ("<0.001",),
    references: list[dict[str, Any]] | None = None,
    outcomes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    outcome = {
        "type": "PRIMARY",
        "title": endpoint,
        "paramType": "COUNT_OF_PARTICIPANTS",
        "unitOfMeasure": "Participants",
        "groups": [
            {"id": "OG000", "title": "Empagliflozin"},
            {"id": "OG001", "title": "Placebo"},
        ],
        "denoms": [
            {
                "units": "Participants",
                "counts": [
                    {"groupId": "OG000", "value": treatment[1]},
                    {"groupId": "OG001", "value": comparator[1]},
                ],
            }
        ],
        "classes": [
            {
                "categories": [
                    {
                        "measurements": [
                            {"groupId": "OG000", "value": treatment[0]},
                            {"groupId": "OG001", "value": comparator[0]},
                        ]
                    }
                ]
            }
        ],
        "analyses": [
            {"groupIds": ["OG000", "OG001"], "pValue": value} for value in p_values
        ],
    }
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct},
            "conditionsModule": {
                "conditions": conditions
                if conditions is not None
                else ["Chronic Kidney Disease"]
            },
            "designModule": {
                "studyType": study_type,
                "designInfo": {"allocation": allocation},
                "enrollmentInfo": enrollment
                if enrollment is not None
                else {"count": 6609, "type": "ACTUAL"},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": sponsor
                if sponsor is not None
                else {"name": "Boehringer Ingelheim", "class": "INDUSTRY"}
            },
            "armsInterventionsModule": {
                "armGroups": [
                    {
                        "label": "Empagliflozin",
                        "type": "EXPERIMENTAL",
                        "interventionNames": ["Drug: Empagliflozin"],
                    },
                    {
                        "label": "Placebo",
                        "type": "PLACEBO_COMPARATOR",
                        "interventionNames": ["Drug: Placebo"],
                    },
                ]
            },
            "referencesModule": {"references": references or []},
        },
        "resultsSection": {
            "outcomeMeasuresModule": {
                "outcomeMeasures": outcomes if outcomes is not None else [outcome]
            }
        },
    }


def drugsfda(
    number: str = APPLICATION,
    *,
    submissions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "application_number": number,
        "sponsor_name": "EXAMPLE PHARMA INC",
        "products": [{"brand_name": "EXAMPLIX"}],
        "openfda": {"generic_name": ["EXAMPLIMAB"]},
        "submissions": submissions
        if submissions is not None
        else [
            {
                "submission_type": "ORIG",
                "submission_number": "1",
                "submission_status": "AP",
                "submission_status_date": "20261001",
            }
        ],
    }


def router(
    *,
    pmids: tuple[str, ...] = (PMID,),
    articles: dict[str, bytes] | None = None,
    studies: dict[str, Any] | None = None,
    registry_hits: list[dict[str, Any]] | None = None,
    applications: list[dict[str, Any]] | None = None,
) -> Router:
    article_map = articles if articles is not None else {PMID: pubmed_xml()}
    study_map = studies if studies is not None else {NCT: study()}
    application_list = applications if applications is not None else []

    def efetch(request: httpx.Request) -> httpx.Response:
        body = article_map.get(request.url.params.get("id", ""))
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, content=body, headers={"content-type": "text/xml"})

    def openfda(request: httpx.Request) -> httpx.Response:
        search = request.url.params.get("search", "")
        if search.startswith("application_number:"):
            wanted = search.split('"')[1]
            rows = [
                row for row in application_list if row["application_number"] == wanted
            ]
            return httpx.Response(200, json={"results": rows})
        if not application_list:
            return httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})
        return httpx.Response(
            200,
            json={
                "meta": {"results": {"total": len(application_list)}},
                "results": application_list,
            },
        )

    routes: dict[str, Any] = {
        ESEARCH: {"esearchresult": {"count": str(len(pmids)), "idlist": list(pmids)}},
        EFETCH: efetch,
        CTGOV_SEARCH: {"studies": registry_hits or []},
        OPENFDA_DRUGSFDA: openfda,
    }
    for nct, body in study_map.items():
        routes[f"{CTGOV_STUDY}{nct}"] = body
    # Any other registry record is not found.
    routes[CTGOV_STUDY] = httpx.Response(404)
    return Router(routes)


def adapter(tmp_path: Path, mock: Router) -> KleinBiomedAdapter:
    return KleinBiomedAdapter(**options(tmp_path, mock))


def harvest(tmp_path: Path, mock: Router) -> Any:
    return adapter(tmp_path, mock).harvest(counter())


def enriched(tmp_path: Path, mock: Router, pmid: str = PMID) -> dict[str, Any]:
    klein = adapter(tmp_path, mock)
    with klein._session():
        return klein.enrich_record({"kind": "trial", "pmid": pmid})


def sealed_files(tmp_path: Path) -> list[Path]:
    archive = tmp_path / "archive"
    return sorted(archive.glob("*.blob")) if archive.exists() else []


def public_text(packet: Any) -> str:
    return f"{packet.public_question} {packet.public_interest_reason}"


# --- PubMed eligibility ------------------------------------------------------------


def test_eligible_human_rct_has_no_ineligibility() -> None:
    record = parse_pubmed_xml(pubmed_xml())
    assert record is not None
    assert record.pmid == PMID
    assert record.nct_ids == (NCT,)
    assert record.grant_agencies == ("NHLBI NIH HHS",)
    assert record.published == datetime(2026, 10, 1, tzinfo=UTC)
    assert pubmed_ineligibility(record) == []


def test_animal_study_is_ineligible() -> None:
    record = parse_pubmed_xml(pubmed_xml(mesh=(ANIMALS,)))
    assert record is not None
    reasons = pubmed_ineligibility(record)
    assert "MeSH does not include Humans" in reasons
    assert "MeSH includes Animals" in reasons


def test_study_tagged_both_humans_and_animals_fails_closed() -> None:
    record = parse_pubmed_xml(pubmed_xml(mesh=(HUMANS, ANIMALS)))
    assert record is not None
    assert pubmed_ineligibility(record) == ["MeSH includes Animals"]


def test_preprint_is_ineligible_even_when_typed_as_an_rct() -> None:
    record = parse_pubmed_xml(
        pubmed_xml(status="PubMed-not-MEDLINE", types=(RCT, ("D000076942", "Preprint")))
    )
    assert record is not None
    reasons = pubmed_ineligibility(record)
    assert "excluded publication type: Preprint" in reasons
    assert any("not indexed for MEDLINE" in item for item in reasons)


@pytest.mark.parametrize(
    "excluded",
    [
        ("D016454", "Review"),
        ("D000078182", "Systematic Review"),
        ("D017418", "Meta-Analysis"),
        ("D064888", "Observational Study"),
        ("D002363", "Case Reports"),
        ("D016421", "Editorial"),
        ("D016422", "Letter"),
        ("D016420", "Comment"),
    ],
)
def test_excluded_publication_types_are_ineligible(excluded: tuple[str, str]) -> None:
    record = parse_pubmed_xml(pubmed_xml(types=(JOURNAL_ARTICLE, RCT, excluded)))
    assert record is not None
    assert pubmed_ineligibility(record) == [f"excluded publication type: {excluded[1]}"]


def test_non_rct_human_study_is_ineligible() -> None:
    record = parse_pubmed_xml(
        pubmed_xml(types=(JOURNAL_ARTICLE, ("D064888", "Observational Study")))
    )
    assert record is not None
    reasons = pubmed_ineligibility(record)
    assert "publication type is not Randomized Controlled Trial" in reasons


def test_title_and_abstract_wording_is_never_evidence() -> None:
    record = parse_pubmed_xml(
        pubmed_xml(
            title="A peer-reviewed randomized controlled trial in human patients",
            types=(JOURNAL_ARTICLE,),
            mesh=(),
        )
    )
    assert record is not None
    reasons = pubmed_ineligibility(record)
    assert "publication type is not Randomized Controlled Trial" in reasons
    assert "MeSH does not include Humans" in reasons


def test_article_not_yet_indexed_for_medline_is_ineligible() -> None:
    record = parse_pubmed_xml(pubmed_xml(status="In-Process"))
    assert record is not None
    assert pubmed_ineligibility(record) == [
        "not indexed for MEDLINE (status 'In-Process')"
    ]


def test_retracted_trial_is_ineligible() -> None:
    record = parse_pubmed_xml(pubmed_xml(retracted=True))
    assert record is not None
    assert "retracted or under an expression of concern" in pubmed_ineligibility(record)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"<not-closed",
        b"<PubmedArticleSet></PubmedArticleSet>",
        pubmed_xml().replace(b"</PubmedArticleSet>", b"")
        + pubmed_xml().split(b"<PubmedArticleSet>")[1],
        pubmed_xml(pmid="abc"),
        (
            b"<PubmedArticleSet><PubmedArticle><MedlineCitation/></PubmedArticle>"
            b"</PubmedArticleSet>"
        ),
    ],
)
def test_malformed_pubmed_payload_is_not_a_record(raw: bytes) -> None:
    assert parse_pubmed_xml(raw) is None
    assert assess_trial(parse_pubmed_xml(raw)).ineligible == (
        "PubMed record missing or malformed",
    )


def test_assessment_rejects_a_record_for_another_pmid() -> None:
    assessment = assess_trial(parse_pubmed_xml(pubmed_xml()), study(), pmid="1")
    assert not assessment.qualified
    assert assessment.pubmed is None


def test_eligible_trial_without_registration_does_not_qualify() -> None:
    assessment = assess_trial(parse_pubmed_xml(pubmed_xml(ncts=())), study())
    assert not assessment.qualified
    assert "no ClinicalTrials.gov registration linked" in assessment.reasons()


def test_two_registrations_are_ambiguous() -> None:
    assessment = assess_trial(
        parse_pubmed_xml(pubmed_xml(ncts=(NCT, "NCT00000001"))), study()
    )
    assert (
        "more than one ClinicalTrials.gov registration linked" in assessment.reasons()
    )


# --- Registry metrics ---------------------------------------------------------------


def test_metrics_are_read_from_reported_counts() -> None:
    metrics = trial_metrics(NCT, study(), ("NHLBI NIH HHS",))
    assert metrics.complete, metrics.missing
    assert metrics.n == 6609
    assert metrics.treatment == "Empagliflozin"
    assert metrics.comparator == "Placebo"
    assert metrics.condition == "Chronic Kidney Disease"
    assert metrics.endpoint == ENDPOINT
    assert metrics.units == "Participants"
    assert (metrics.treatment_events, metrics.treatment_total) == (432, 3304)
    assert (metrics.comparator_events, metrics.comparator_total) == (558, 3305)
    assert metrics.arr_percent == pytest.approx(ARR)
    assert metrics.arr_percent == pytest.approx(3.8084, abs=1e-4)
    assert metrics.p_value == "<0.001"
    assert metrics.funding == (
        "Boehringer Ingelheim (industry lead sponsor); grants from NHLBI NIH HHS"
    )
    receipt = metrics.as_dict()
    assert receipt["arr_calculation"] == (
        "(558/3305) - (432/3304) = 3.8084 percentage points"
    )


def test_arr_is_absolute_not_relative() -> None:
    metrics = trial_metrics(
        NCT, study(treatment=("50", "1000"), comparator=("100", "1000"))
    )
    # Relative reduction would be 50%; the absolute difference is 5 points.
    assert metrics.arr_percent == pytest.approx(5.0)


def test_estimated_enrollment_is_not_n() -> None:
    metrics = trial_metrics(NCT, study(enrollment={"count": 7000, "type": "ESTIMATED"}))
    assert metrics.n is None
    assert "actual enrollment (N) not reported" in metrics.missing


def test_missing_p_value_is_missing() -> None:
    metrics = trial_metrics(NCT, study(p_values=()))
    assert metrics.p_value is None
    assert "p-value not reported" in metrics.missing


@pytest.mark.parametrize("reported", ["0.21", ">0.05", "0.05", "NS", ""])
def test_non_significant_or_unreadable_p_value_does_not_qualify(reported: str) -> None:
    metrics = trial_metrics(NCT, study(p_values=(reported,)))
    assert not metrics.complete


def test_conflicting_p_values_are_missing() -> None:
    metrics = trial_metrics(NCT, study(p_values=("0.01", "0.03")))
    assert metrics.p_value is None
    assert "more than one p-value reported for the comparison" in metrics.missing


def test_undisclosed_funding_is_missing() -> None:
    metrics = trial_metrics(NCT, study(sponsor={"name": "", "class": "OTHER"}))
    assert metrics.funding is None
    assert "funding source not reported" in metrics.missing
    assert metrics.as_dict()["funding"] == UNAVAILABLE


def test_grants_alone_are_a_funding_source() -> None:
    metrics = trial_metrics(NCT, study(sponsor={}), ("NCI NIH HHS",))
    assert metrics.funding == "grants from NCI NIH HHS"


@pytest.mark.parametrize(
    "endpoint",
    [
        "Number of Participants With Overall Survival at 5 Years",
        "Number of Participants With Complete Response",
        "Number of Participants Achieving Remission",
        "Number of Participants With Event-Free Survival",
        "Number of Participants Who Completed the Questionnaire",
    ],
)
def test_endpoint_without_established_harm_direction_has_no_arr(endpoint: str) -> None:
    assert not endpoint_is_harm(endpoint)
    metrics = trial_metrics(NCT, study(endpoint=endpoint))
    assert metrics.arr_percent is None
    assert "endpoint direction not established; ARR unavailable" in metrics.missing


def test_more_events_on_treatment_is_not_a_reduction() -> None:
    metrics = trial_metrics(NCT, study(treatment=("600", "3304")))
    assert metrics.arr_percent is None
    assert any("no absolute risk reduction" in item for item in metrics.missing)


def test_two_primary_count_outcomes_are_ambiguous() -> None:
    one = study()["resultsSection"]["outcomeMeasuresModule"]["outcomeMeasures"][0]
    metrics = trial_metrics(NCT, study(outcomes=[one, {**one, "title": "Stroke"}]))
    assert metrics.arr_percent is None
    assert any(
        "more than one primary count outcome" in item for item in metrics.missing
    )


def test_three_arm_outcome_is_ambiguous() -> None:
    record = study()
    outcome = record["resultsSection"]["outcomeMeasuresModule"]["outcomeMeasures"][0]
    outcome["groups"].append({"id": "OG002", "title": "Empagliflozin 25 mg"})
    metrics = trial_metrics(NCT, record)
    assert any(
        "one experimental arm with one comparator" in item for item in metrics.missing
    )


def test_no_posted_results_is_missing() -> None:
    metrics = trial_metrics(NCT, study(outcomes=[]))
    assert (
        "no primary count-of-participants outcome in posted results" in metrics.missing
    )


@pytest.mark.parametrize(
    ("treatment", "comparator"),
    [
        (("", "3304"), ("558", "3305")),
        (("432", "0"), ("558", "3305")),
        (("4000", "3304"), ("558", "3305")),
    ],
)
def test_bad_counts_give_no_arr(
    treatment: tuple[str, str], comparator: tuple[str, str]
) -> None:
    metrics = trial_metrics(NCT, study(treatment=treatment, comparator=comparator))
    assert metrics.arr_percent is None
    assert "event counts and denominators not reported for both arms" in metrics.missing


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        "text",
        {"protocolSection": {"identificationModule": {"nctId": "NCT00000001"}}},
    ],
)
def test_malformed_registry_record_is_missing(body: Any) -> None:
    metrics = trial_metrics(NCT, body)
    assert not metrics.complete
    assert metrics.n is None and metrics.arr_percent is None


def test_registry_must_report_randomized_interventional_design() -> None:
    metrics = trial_metrics(NCT, study(allocation="NA", study_type="OBSERVATIONAL"))
    assert "registry study type is not interventional" in metrics.missing
    assert "registry allocation is not randomized" in metrics.missing


@pytest.mark.parametrize(
    ("reported", "significant", "phrase"),
    [
        ("<0.001", True, "p<0.001"),
        ("0.002", True, "p=0.002"),
        ("= 0.04", True, "p=0.04"),
        ("≤0.05", True, "p≤0.05"),
        ("0.05", False, "p=0.05"),
        ("1e-5", True, "p=1e-5"),
    ],
)
def test_p_value_is_kept_as_reported(
    reported: str, significant: bool, phrase: str
) -> None:
    assert p_value_significant(reported) is significant
    assert p_value_phrase(reported) == phrase


@pytest.mark.parametrize(
    ("value", "text"), [(3.8084, "3.8"), (12.0, "12"), (0.04, "0.04"), (0.1, "0.1")]
)
def test_format_percent(value: float, text: str) -> None:
    assert format_percent(value) == text


# --- Public text -----------------------------------------------------------------


def test_qualified_trial_uses_both_templates() -> None:
    assessment = assess_trial(parse_pubmed_xml(pubmed_xml()), study())
    assert assessment.qualified
    question, reason = trial_public_text(assessment)
    assert question == (
        "What did the clinical trial for Empagliflozin demonstrate in human patients "
        "with Chronic Kidney Disease?"
    )
    assert reason == (
        "Peer-reviewed human trial (N=6609) demonstrated an absolute risk reduction "
        "of 3.8% (p<0.001); funded by Boehringer Ingelheim (industry lead sponsor); "
        "grants from NHLBI NIH HHS."
    )


@pytest.mark.parametrize(
    ("article", "registry"),
    [
        (pubmed_xml(mesh=(ANIMALS,)), study()),
        (pubmed_xml(types=(("D000076942", "Preprint"),)), study()),
        (pubmed_xml(types=(JOURNAL_ARTICLE,)), study()),
        (pubmed_xml(), study(p_values=())),
        (pubmed_xml(), study(enrollment={})),
        (pubmed_xml(grants=()), study(sponsor={})),
        (pubmed_xml(), study(endpoint="Number of Participants With Response")),
        (pubmed_xml(), None),
    ],
)
def test_ineligible_or_incomplete_evidence_gets_no_trial_statement(
    article: bytes, registry: Any
) -> None:
    assessment = assess_trial(parse_pubmed_xml(article), registry)
    assert not assessment.qualified
    with pytest.raises(ValueError, match="does not qualify"):
        trial_public_text(assessment)


# --- openFDA approvals -----------------------------------------------------------


def test_original_nda_approval_is_read() -> None:
    [approval] = fda_approvals(drugsfda())
    assert approval.application_number == APPLICATION
    assert approval.approved_on == date(2026, 10, 1)
    assert approval.drug == "EXAMPLIX (EXAMPLIMAB)"
    assert approval.key == f"{APPLICATION}:ORIG1"


@pytest.mark.parametrize(
    "application",
    [
        drugsfda("ANDA212345"),
        drugsfda(
            submissions=[
                {
                    "submission_type": "SUPPL",
                    "submission_number": "4",
                    "submission_status": "AP",
                    "submission_status_date": "20261001",
                }
            ]
        ),
        drugsfda(
            submissions=[
                {
                    "submission_type": "ORIG",
                    "submission_number": "1",
                    "submission_status": "TA",
                    "submission_status_date": "20261001",
                }
            ]
        ),
        drugsfda(
            submissions=[
                {
                    "submission_type": "ORIG",
                    "submission_number": "1",
                    "submission_status": "AP",
                    "submission_status_date": "2026-10-01",
                }
            ]
        ),
        drugsfda(
            submissions=[
                {
                    "submission_type": "ORIG",
                    "submission_status": "AP",
                    "submission_status_date": "20261001",
                }
            ]
        ),
        {"application_number": 7},
        "NDA219876",
        None,
    ],
)
def test_other_fda_records_are_not_approvals(application: Any) -> None:
    assert fda_approvals(application) == []


# --- Polling, extraction, and assembly ---------------------------------------------


def test_poll_requests_cover_pubmed_registry_and_fda(tmp_path: Path) -> None:
    requests = {item.name: item for item in adapter(tmp_path, router()).poll_requests()}
    assert set(requests) == {PUBMED_REQUEST, CTGOV_REQUEST, FDA_REQUEST}
    pubmed = requests[PUBMED_REQUEST].params
    assert pubmed["datetype"] == "mhda"
    assert (pubmed["mindate"], pubmed["maxdate"]) == ("2026/09/15", "2026/10/15")
    assert (
        "RANGE[2026-09-15,2026-10-15]"
        in requests[CTGOV_REQUEST].params["filter.advanced"]
    )
    assert "[20260915 TO 20261015]" in requests[FDA_REQUEST].params["search"]
    for item in requests.values():
        assert not any("key" in name.casefold() for name in item.params)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"esearchresult": None},
        {"esearchresult": {"idlist": "1"}},
        {"results": "x"},
        {"studies": [None, {"protocolSection": {}}]},
    ],
)
def test_malformed_poll_payloads_give_no_items(tmp_path: Path, payload: Any) -> None:
    klein = adapter(tmp_path, router())
    for request in klein.poll_requests():
        assert klein.extract_items(request, payload) == []


def test_extract_items(tmp_path: Path) -> None:
    klein = adapter(tmp_path, router())
    requests = {item.name: item for item in klein.poll_requests()}
    assert klein.extract_items(
        requests[PUBMED_REQUEST], {"esearchresult": {"idlist": [PMID, "x1"]}}
    ) == [{"pmid": PMID}]
    hits = klein.extract_items(
        requests[CTGOV_REQUEST],
        {"studies": [study(references=[{"pmid": PMID, "type": "RESULT"}])]},
    )
    assert hits == [{"nct_id": NCT, "pmids": [PMID]}]
    assert klein.extract_items(requests[FDA_REQUEST], {"results": [drugsfda(), 3]}) == [
        drugsfda()
    ]


def test_registry_prefers_result_references_over_derived() -> None:
    references = [
        {"pmid": "111", "type": "BACKGROUND"},
        {"pmid": "222", "type": "DERIVED"},
        {"pmid": "333", "type": "RESULT"},
    ]
    assert registry_pmids(study(references=references)) == ("333",)
    assert registry_pmids(study(references=references[:2])) == ("222",)
    assert registry_pmids(study(references=references[:1])) == ()


def test_pagination(tmp_path: Path) -> None:
    klein = adapter(tmp_path, router())
    requests = {item.name: item for item in klein.poll_requests()}
    nxt = klein.next_request(
        requests[PUBMED_REQUEST], {"esearchresult": {"count": "250"}}
    )
    assert nxt is not None and nxt.params["retstart"] == str(PAGE_SIZE)
    assert (
        klein.next_request(requests[PUBMED_REQUEST], {"esearchresult": {"count": "40"}})
        is None
    )
    nxt = klein.next_request(
        requests[FDA_REQUEST], {"meta": {"results": {"total": 101}}}
    )
    assert nxt is not None and nxt.params["skip"] == str(PAGE_SIZE)
    nxt = klein.next_request(requests[CTGOV_REQUEST], {"nextPageToken": "abc"})
    assert nxt is not None and nxt.params["pageToken"] == "abc"
    assert klein.next_request(requests[CTGOV_REQUEST], {}) is None
    capped = PollRequest(PUBMED_REQUEST, ESEARCH, {"retstart": "400"})
    assert klein.next_request(capped, {"esearchresult": {"count": "9999"}}) is None


def test_assembly_merges_one_trial_found_by_pmid_and_nct(tmp_path: Path) -> None:
    records = adapter(tmp_path, router()).assemble_records(
        {
            PUBMED_REQUEST: [{"pmid": PMID}, {"pmid": PMID}],
            CTGOV_REQUEST: [
                {"nct_id": NCT, "pmids": [PMID]},
                {"nct_id": "NCT00000002", "pmids": []},
                {"nct_id": "NCT00000003", "pmids": ["1", "2"]},
                {"nct_id": "bad", "pmids": ["5"]},
            ],
            FDA_REQUEST: [
                drugsfda(),
                drugsfda(
                    "NDA000001",
                    submissions=[
                        {
                            "submission_type": "ORIG",
                            "submission_number": "1",
                            "submission_status": "AP",
                            "submission_status_date": "20200101",
                        }
                    ],
                ),
                {"junk": True},
            ],
        }
    )
    klein = adapter(tmp_path, router())
    keys = [klein.record_key(item) for item in records]
    assert keys == [
        f"pubmed:{PMID}",
        "ctgov:NCT00000002",
        "ctgov:NCT00000003",
        f"fda:{APPLICATION}:ORIG1",
    ]
    assert records[0] == {"kind": "trial", "pmid": PMID, "nct_id": NCT}


def test_two_registrations_claiming_one_article_conflict(tmp_path: Path) -> None:
    [record] = adapter(tmp_path, router()).assemble_records(
        {
            CTGOV_REQUEST: [
                {"nct_id": NCT, "pmids": [PMID]},
                {"nct_id": "NCT00000009", "pmids": [PMID]},
            ]
        }
    )
    assert record["nct_conflict"] is True


def test_record_key_rejects_unknown_kinds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown record kind"):
        adapter(tmp_path, router()).record_key({"kind": "rumor"})


# --- Timestamps --------------------------------------------------------------------


def test_trial_timestamp_is_the_electronic_publication_date(tmp_path: Path) -> None:
    record = enriched(tmp_path, router())
    stamp = adapter(tmp_path, router()).record_timestamp(record)
    assert stamp == datetime(2026, 10, 1, tzinfo=UTC)
    assert stamp.tzinfo is not None


def test_trial_timestamp_falls_back_to_the_pubmed_date(tmp_path: Path) -> None:
    mock = router(articles={PMID: pubmed_xml(electronic=None)})
    record = enriched(tmp_path, mock)
    assert adapter(tmp_path, mock).record_timestamp(record) == datetime(
        2026, 10, 2, tzinfo=UTC
    )


def test_partial_dates_are_never_padded(tmp_path: Path) -> None:
    raw = pubmed_xml(electronic=None, history=None)
    record = parse_pubmed_xml(raw)
    assert record is not None and record.published is None
    assessment = assess_trial(record, study())
    assert "no full publication date" in assessment.reasons()
    klein = adapter(tmp_path, router())
    with pytest.raises(ValueError, match="no full publication date"):
        klein.record_timestamp(
            {"kind": "trial", "pmid": PMID, "assessment": assessment}
        )


def test_unread_trial_has_no_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not been read"):
        adapter(tmp_path, router()).record_timestamp({"kind": "trial", "pmid": PMID})


def test_fda_timestamp_is_the_approval_date(tmp_path: Path) -> None:
    [approval] = fda_approvals(drugsfda())
    stamp = adapter(tmp_path, router()).record_timestamp(
        {"kind": "fda_approval", "approval": approval}
    )
    assert stamp == datetime(2026, 10, 1, tzinfo=UTC)


# --- Evidence targets ----------------------------------------------------------------


def test_qualified_trial_targets_pubmed_and_registry(tmp_path: Path) -> None:
    record = enriched(tmp_path, router())
    targets = adapter(tmp_path, router()).evidence_targets(record)
    assert [(item.kind, item.url, item.tier) for item in targets] == [
        (KIND_PUBMED, efetch_url(PMID), PRIMARY_TIER),
        (KIND_CTGOV, ctgov_url(NCT), PRIMARY_TIER),
    ]
    assert targets[1].excerpt == ENDPOINT
    assert targets[1].metadata["absolute_risk_reduction_percent"] == pytest.approx(ARR)


def test_unqualified_trial_has_no_targets(tmp_path: Path) -> None:
    mock = router(articles={PMID: pubmed_xml(mesh=(ANIMALS,))})
    assert adapter(tmp_path, mock).evidence_targets(enriched(tmp_path, mock)) == []


def test_fda_approval_targets_the_openfda_record(tmp_path: Path) -> None:
    [approval] = fda_approvals(drugsfda())
    [target] = adapter(tmp_path, router()).evidence_targets(
        {"kind": "fda_approval", "approval": approval}
    )
    assert target.kind == KIND_FDA
    assert target.url == fda_application_url(APPLICATION)
    assert target.url.startswith("https://api.fda.gov/")
    assert target.metadata["approved_on"] == "2026-10-01"


# --- Harvest end to end --------------------------------------------------------------


def test_qualified_trial_becomes_a_lead(tmp_path: Path) -> None:
    mock = router()
    result = harvest(tmp_path, mock)
    assert result.errors == [] and result.evidence_rejections == []
    [packet] = result.packets
    assert packet.adapter_name == "klein_biomed"
    assert packet.anomaly_score == 85.0  # N >= 1000
    assert packet.timestamp == datetime(2026, 10, 1, tzinfo=UTC)
    assert packet.public_question.startswith(
        "What did the clinical trial for Empagliflozin demonstrate"
    )
    assert "Peer-reviewed human trial (N=6609)" in packet.public_interest_reason
    assert "absolute risk reduction of 3.8% (p<0.001)" in packet.public_interest_reason
    kinds = sorted(item.metadata["kind"] for item in packet.source_artifacts)
    assert kinds == [KIND_CTGOV, KIND_PUBMED]
    assert all(len(item.sha256_hash) == 64 for item in packet.source_artifacts)
    assert len(sealed_files(tmp_path)) == 2
    assert set(packet.verbatim_excerpts.values()) == {PMID, ENDPOINT}
    receipt = packet.receipt_details
    assert receipt["record_type"] == "human_rct"
    assert receipt["metrics"]["comparator"] == "Placebo"
    assert receipt["not_qualified_because"] == []
    assert "Absolute risk reduction: 3.8 percentage points" in packet.paste_ready_prompt


def test_trial_found_only_by_nct_number_qualifies(tmp_path: Path) -> None:
    """The article names no registration; the registry names the article."""
    registered = study(references=[{"pmid": PMID, "type": "RESULT"}])
    mock = router(
        pmids=(),
        articles={PMID: pubmed_xml(ncts=())},
        studies={NCT: registered},
        registry_hits=[registered],
    )
    [packet] = harvest(tmp_path, mock).packets
    assert "N=6609" in packet.public_interest_reason


def test_trial_found_by_both_identifiers_is_one_lead(tmp_path: Path) -> None:
    registered = study(references=[{"pmid": PMID, "type": "DERIVED"}])
    mock = router(studies={NCT: registered}, registry_hits=[registered])
    assert len(harvest(tmp_path, mock).packets) == 1


def test_registry_hit_without_one_article_is_not_called_peer_reviewed(
    tmp_path: Path,
) -> None:
    registered = study(references=[])
    mock = router(pmids=(), studies={NCT: registered}, registry_hits=[registered])
    result = harvest(tmp_path, mock)
    assert result.packets == [] and result.errors == []
    assert sealed_files(tmp_path) == []


def test_registry_claim_on_an_article_naming_another_trial_fails(
    tmp_path: Path,
) -> None:
    registered = study(references=[{"pmid": PMID, "type": "RESULT"}])
    mock = router(
        pmids=(),
        articles={PMID: pubmed_xml(ncts=("NCT00000001",))},
        studies={NCT: registered},
        registry_hits=[registered],
    )
    assert harvest(tmp_path, mock).packets == []


def test_registry_that_does_not_name_the_article_fails(tmp_path: Path) -> None:
    assessment = assess_trial(
        parse_pubmed_xml(pubmed_xml(ncts=())), study(references=[]), nct_hint=NCT
    )
    assert assessment.reasons() == [
        "registry record does not name this article as its results"
    ]


@pytest.mark.parametrize(
    "article",
    [
        pubmed_xml(mesh=(ANIMALS,)),
        pubmed_xml(types=(("D000076942", "Preprint"),), status="PubMed-not-MEDLINE"),
        pubmed_xml(types=(JOURNAL_ARTICLE, ("D064888", "Observational Study"))),
        pubmed_xml(types=(JOURNAL_ARTICLE, RCT, ("D016454", "Review"))),
        b"<garbage",
    ],
)
def test_ineligible_article_makes_no_lead_and_seals_nothing(
    tmp_path: Path, article: bytes
) -> None:
    mock = router(articles={PMID: article})
    result = harvest(tmp_path, mock)
    assert result.packets == [] and result.errors == []
    assert sealed_files(tmp_path) == []
    assert not any(str(item.url).startswith(CTGOV_STUDY) for item in mock.requests)


@pytest.mark.parametrize(
    "registry",
    [
        study(p_values=()),
        study(enrollment={"count": 6609, "type": "ESTIMATED"}),
        study(sponsor={}),
        study(outcomes=[]),
        study(p_values=("0.31",)),
    ],
)
def test_incomplete_metrics_make_no_lead(
    tmp_path: Path, registry: dict[str, Any]
) -> None:
    mock = router(articles={PMID: pubmed_xml(grants=())}, studies={NCT: registry})
    result = harvest(tmp_path, mock)
    assert result.packets == []
    assert sealed_files(tmp_path) == []


def test_unregistered_trial_is_unavailable_not_an_error(tmp_path: Path) -> None:
    result = harvest(tmp_path, router(studies={}))
    assert result.packets == [] and result.errors == []


def test_registry_outage_is_an_error_for_retry(tmp_path: Path) -> None:
    result = harvest(tmp_path, router(studies={NCT: httpx.Response(503)}))
    assert result.packets == []
    assert len(result.errors) == 1 and result.errors[0].startswith(f"pubmed:{PMID}:")


def test_fda_approval_lead_claims_no_trial_finding(tmp_path: Path) -> None:
    mock = router(pmids=(), applications=[drugsfda()])
    [packet] = harvest(tmp_path, mock).packets
    assert packet.anomaly_score == 60.0
    assert packet.public_question == (
        "What did the FDA approve on 2026-10-01 in application NDA219876 for "
        "EXAMPLIX (EXAMPLIMAB), and what trial evidence supports it?"
    )
    text = public_text(packet)
    assert "Peer-reviewed human trial" not in text
    assert "demonstrated" not in text
    assert "N=" not in text
    assert "so none is claimed" in packet.public_interest_reason
    [artifact] = packet.source_artifacts
    assert artifact.metadata["kind"] == KIND_FDA
    receipt = packet.receipt_details
    for name in (
        "trial_evidence",
        "sample_size_n",
        "absolute_risk_reduction_percent",
        "p_value",
        "funding",
    ):
        assert receipt[name] == UNAVAILABLE


def test_record_cannot_carry_its_own_public_text(tmp_path: Path) -> None:
    klein = adapter(tmp_path, router())
    record = enriched(tmp_path, router(articles={PMID: pubmed_xml(mesh=(ANIMALS,))}))
    record |= {
        "public_question": "What did the trial demonstrate in humans?",
        "public_interest_reason": "Peer-reviewed human trial (N=10) demonstrated ...",
    }
    with pytest.raises(ValueError, match="does not qualify"):
        klein.public_interest_for(record)


def test_known_lead_is_skipped_before_any_fetch(tmp_path: Path) -> None:
    mock = router()
    klein = adapter(tmp_path, mock)
    result = klein.harvest(counter(), is_known=lambda _lead: True)
    assert result.packets == []
    assert not any(str(item.url).startswith(EFETCH) for item in mock.requests)


# --- Evidence bar on sealed payloads -----------------------------------------------


def sealed_trial(tmp_path: Path) -> tuple[KleinBiomedAdapter, Any]:
    mock = router()
    klein = adapter(tmp_path, mock)
    [packet] = klein.harvest(counter()).packets
    return klein, packet


def test_evidence_bar_holds_on_the_sealed_payloads(tmp_path: Path) -> None:
    klein, packet = sealed_trial(tmp_path)
    assert klein.verify_evidence_bar(packet.source_artifacts)


def test_evidence_bar_rereads_the_sealed_registry_record(tmp_path: Path) -> None:
    klein, packet = sealed_trial(tmp_path)
    registry = next(
        item for item in packet.source_artifacts if item.metadata["kind"] == KIND_CTGOV
    )
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(study(p_values=("0.4",))))
    swapped = [
        replace(item, raw_payload_path=str(forged)) if item is registry else item
        for item in packet.source_artifacts
    ]
    assert not klein.verify_evidence_bar(swapped)


def test_evidence_bar_rejects_metrics_that_differ_from_the_sealed_record(
    tmp_path: Path,
) -> None:
    klein, packet = sealed_trial(tmp_path)
    changed = [
        replace(item, metadata={**item.metadata, "sample_size_n": 99999})
        if item.metadata["kind"] == KIND_CTGOV
        else item
        for item in packet.source_artifacts
    ]
    assert not klein.verify_evidence_bar(changed)


def test_evidence_bar_rejects_wrong_hosts_and_missing_sources(tmp_path: Path) -> None:
    klein, packet = sealed_trial(tmp_path)
    artifacts = packet.source_artifacts
    assert not klein.verify_evidence_bar(artifacts[:1])
    assert not klein.verify_evidence_bar([])
    moved = [replace(item, origin_url="https://example.org/x") for item in artifacts]
    assert not klein.verify_evidence_bar(moved)
    gone = [replace(item, raw_payload_path="/nonexistent") for item in artifacts]
    assert not klein.verify_evidence_bar(gone)


def test_fda_evidence_bar_rereads_the_sealed_approval(tmp_path: Path) -> None:
    klein = adapter(tmp_path, router(pmids=(), applications=[drugsfda()]))
    [packet] = klein.harvest(counter()).packets
    assert klein.verify_evidence_bar(packet.source_artifacts)
    [artifact] = packet.source_artifacts
    other_date = replace(
        artifact, metadata={**artifact.metadata, "approved_on": "2026-01-01"}
    )
    assert not klein.verify_evidence_bar([other_date])
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"results": []}))
    assert not klein.verify_evidence_bar(
        [replace(artifact, raw_payload_path=str(empty))]
    )


def test_harvest_records_evidence_rejection_when_the_seal_fails_the_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock = router()
    klein = adapter(tmp_path, mock)
    monkeypatch.setattr(klein, "verify_evidence_bar", lambda _artifacts: False)
    result = klein.harvest(counter())
    assert result.packets == []
    assert result.evidence_rejections == [f"pubmed:{PMID}"]
