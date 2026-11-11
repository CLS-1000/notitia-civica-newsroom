"""Klein Biomed (Engine 4): clinical evidence for science and health reporting.

Two kinds of record can become a lead, and nothing else can:

- A peer-reviewed human randomized controlled trial. PubMed is searched for
  records newly indexed for MEDLINE as randomized controlled trials in humans,
  and each record's own metadata decides eligibility: it must be indexed for
  MEDLINE (the NLM's selection of peer-reviewed journals), carry the
  publication type Randomized Controlled Trial, carry the MeSH heading Humans
  and not Animals, and carry none of the excluded types (preprint, review,
  meta-analysis, observational study, case report, editorial, letter, comment,
  retraction). Nothing is inferred from the title or abstract.
- An FDA original approval of a new drug or biologic (NDA or BLA, not a
  generic ANDA), read from openFDA's Drugs@FDA records.

A trial can be found by either identifier. PubMed is searched by PMID as
above, and ClinicalTrials.gov is searched for randomized interventional
studies whose results were newly posted; a registration that names one PubMed
results article (a RESULT reference, else an NLM-derived one) brings that
PMID in. Both paths meet on the PMID, so one trial is one lead, and either way
the article must pass the PubMed checks above: a registration with no single
eligible article is never called peer-reviewed.

An eligible trial becomes a lead only when its results are reported in its
ClinicalTrials.gov registration, linked by NCT number from either side. From the
registry results this reads, without estimating anything:

- N: the actual enrollment.
- ARR: computed only from the reported event counts and denominators of the
  single primary count-of-participants outcome with one experimental and one
  comparator arm, as (comparator events / comparator N) minus (experimental
  events / experimental N), in percentage points. It is an absolute
  difference, never a relative one. It is called a risk reduction only when it
  is positive and the endpoint names a harm (death, hospitalization, an event)
  and no benefit (survival, response, remission); otherwise the direction is
  not established and the ARR is unavailable.
- p: the p-value reported by the registry's analysis of those two arms, kept
  as reported. The trial is said to have "demonstrated" a reduction only when
  that p-value is below 0.05.
- Funding: the registry's lead sponsor and its class, plus PubMed grant
  agencies.

If any of these is missing or ambiguous, the trial is not a lead. The public
question and reason use the trial templates only for a trial whose every
statement is supported. An FDA approval gets its own question and reason,
which claim nothing about trials: its trial fields are marked unavailable.

The evidence bar re-reads the sealed PubMed and registry payloads and repeats
the whole assessment on them, so a lead stands on what was sealed, not on what
was polled.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode
from xml.etree import ElementTree

from spec1_bridge.adapters.base_lazlong import (
    EvidenceTarget,
    LazLongAdapter,
    PollError,
    PollRequest,
    host_of,
    render_prompt,
)
from spec1_bridge.contracts import SourceArtifact

logger = logging.getLogger(__name__)

ESEARCH: Final = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH: Final = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EUTILS_HOST: Final = "eutils.ncbi.nlm.nih.gov"
CTGOV_STUDY: Final = "https://clinicaltrials.gov/api/v2/studies/"
CTGOV_SEARCH: Final = "https://clinicaltrials.gov/api/v2/studies"
CTGOV_HOST: Final = "clinicaltrials.gov"
OPENFDA_DRUGSFDA: Final = "https://api.fda.gov/drug/drugsfda.json"
OPENFDA_HOST: Final = "api.fda.gov"

PUBMED_REQUEST: Final = "pubmed:rct"
FDA_REQUEST: Final = "fda:approvals"
CTGOV_REQUEST: Final = "ctgov:results"
# Narrows the search only; each record's own metadata still decides eligibility.
PUBMED_TERM: Final = (
    '"randomized controlled trial"[pt] AND humans[mh] AND medline[sb] '
    "NOT animals[mh:noexp] NOT preprint[pt]"
)
PAGE_SIZE: Final = 100
MAX_RESULTS: Final = 500
LOOKBACK_DAYS: Final = 30

PRIMARY_TIER: Final = "tier_1_primary"
KIND_PUBMED: Final = "pubmed_record"
KIND_CTGOV: Final = "ctgov_results"
KIND_FDA: Final = "fda_approval"

TRIAL_SCORE: Final = 75.0
LARGE_TRIAL_BONUS: Final = 10.0
LARGE_TRIAL_N: Final = 1000
FDA_SCORE: Final = 60.0
SIGNIFICANCE: Final = 0.05

UNAVAILABLE: Final = "unavailable"

_HUMANS: Final = ("D006801", "humans")
_ANIMALS: Final = ("D000818", "animals")
_RCT: Final = ("D016449", "randomized controlled trial")
_EXCLUDED_TYPES: Final = frozenset(
    {
        "preprint",
        "review",
        "systematic review",
        "scoping review",
        "meta-analysis",
        "observational study",
        "case reports",
        "editorial",
        "letter",
        "comment",
        "news",
        "published erratum",
        "retracted publication",
        "retraction of publication",
        "expression of concern",
        "clinical trial protocol",
        "guideline",
        "practice guideline",
    }
)
_RETRACTION_REFS: Final = frozenset({"retractionin", "expressionofconcernin"})
_COMPARATOR_ARMS: Final = frozenset(
    {"PLACEBO_COMPARATOR", "ACTIVE_COMPARATOR", "SHAM_COMPARATOR", "NO_INTERVENTION"}
)
_HARM: Final = re.compile(
    r"\b(?:death|deaths|died|mortality|fatal|hospitali[sz]ations?|admissions?|"
    r"readmissions?|stroke|infarction|heart failure|events?|progression|"
    r"recurrence|relapse|infections?|fractures?|exacerbations?|amputations?|"
    r"dialysis|kidney failure|incidence|thrombos[ie]s|embolism|bleeding|"
    r"transmission|seroconversion)\b",
    re.IGNORECASE,
)
_BENEFIT: Final = re.compile(
    r"\b(?:surviv\w*|alive|respon\w*|remission|cured?|free|success\w*|improv\w*|"
    r"recover\w*|resolution|heal\w*|clearance|abstinen\w*|achiev\w*|attain\w*|"
    r"eradicat\w*|normali[sz]\w*)\b",
    re.IGNORECASE,
)
_P_VALUE: Final = re.compile(r"^\s*(<=|<|≤|=)?\s*(\d*\.?\d+(?:[eE]-?\d+)?)\s*$")
_NCT: Final = re.compile(r"^NCT\d{8}$")
_MONTHS: Final = {
    name: number
    for number, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun")
        + ("jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
_SPONSOR_CLASS: Final = {
    "INDUSTRY": "industry",
    "NIH": "NIH",
    "FED": "U.S. federal",
    "OTHER_GOV": "government",
    "NETWORK": "research network",
    "INDIV": "individual",
}


# --- PubMed -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PubMedRecord:
    """The PubMed metadata that decides whether an article is eligible."""

    pmid: str
    title: str
    journal: str
    medline_status: str
    publication_types: tuple[tuple[str, str], ...]  # (UI, name)
    mesh: tuple[tuple[str, str], ...]  # (UI, descriptor)
    nct_ids: tuple[str, ...]
    grant_agencies: tuple[str, ...]
    retracted: bool
    published: datetime | None


def efetch_url(pmid: str) -> str:
    return f"{EFETCH}?{urlencode({'db': 'pubmed', 'id': pmid, 'retmode': 'xml'})}"


def _text(node: ElementTree.Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def _date_of(node: ElementTree.Element | None) -> datetime | None:
    """A full year, month, and day, or None. A partial date is never padded."""
    if node is None:
        return None
    year, month, day = (_text(node.find(part)) for part in ("Year", "Month", "Day"))
    month_number = (
        int(month) if month.isdigit() else _MONTHS.get(month[:3].casefold(), 0)
    )
    if not (year.isdigit() and day.isdigit() and month_number):
        return None
    try:
        return datetime(int(year), month_number, int(day), tzinfo=UTC)
    except ValueError:
        return None


def parse_pubmed_xml(raw: bytes) -> PubMedRecord | None:
    """Read the one article in an efetch response, or None if it is not one."""
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        return None
    articles = root.findall("PubmedArticle")
    if len(articles) != 1:
        return None
    citation = articles[0].find("MedlineCitation")
    article = citation.find("Article") if citation is not None else None
    if citation is None or article is None:
        return None
    pmid = _text(citation.find("PMID"))
    if not pmid.isdigit():
        return None
    nct_ids = tuple(
        _text(number)
        for bank in article.findall("DataBankList/DataBank")
        if _text(bank.find("DataBankName")).casefold() == "clinicaltrials.gov"
        for number in bank.findall("AccessionNumberList/AccessionNumber")
        if _NCT.match(_text(number))
    )
    published = None
    for stamp in article.findall("ArticleDate"):
        if stamp.get("DateType", "").casefold() == "electronic":
            published = _date_of(stamp)
    if published is None:
        for stamp in articles[0].findall("PubmedData/History/PubMedPubDate"):
            if stamp.get("PubStatus") == "pubmed":
                published = _date_of(stamp)
    return PubMedRecord(
        pmid=pmid,
        title=_text(article.find("ArticleTitle")),
        journal=_text(article.find("Journal/Title")),
        medline_status=citation.get("Status", ""),
        publication_types=tuple(
            (item.get("UI", ""), _text(item))
            for item in article.findall("PublicationTypeList/PublicationType")
        ),
        mesh=tuple(
            (item.get("UI", ""), _text(item))
            for item in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
        ),
        nct_ids=tuple(dict.fromkeys(nct_ids)),
        grant_agencies=tuple(
            dict.fromkeys(
                agency
                for agency in (
                    _text(item) for item in article.findall("GrantList/Grant/Agency")
                )
                if agency
            )
        ),
        retracted=any(
            item.get("RefType", "").casefold() in _RETRACTION_REFS
            for item in citation.findall("CommentsCorrectionsList/CommentsCorrections")
        ),
        published=published,
    )


def _has(pairs: tuple[tuple[str, str], ...], wanted: tuple[str, str]) -> bool:
    ui, name = wanted
    return any(item_ui == ui or item.casefold() == name for item_ui, item in pairs)


def pubmed_ineligibility(record: PubMedRecord) -> list[str]:
    """Why an article is not a peer-reviewed human RCT. Empty when it is."""
    reasons = []
    if record.medline_status != "MEDLINE":
        reasons.append(f"not indexed for MEDLINE (status {record.medline_status!r})")
    if not _has(record.publication_types, _RCT):
        reasons.append("publication type is not Randomized Controlled Trial")
    excluded = sorted(
        name
        for _ui, name in record.publication_types
        if name.casefold() in _EXCLUDED_TYPES
    )
    if excluded:
        reasons.append(f"excluded publication type: {', '.join(excluded)}")
    if record.retracted:
        reasons.append("retracted or under an expression of concern")
    if not _has(record.mesh, _HUMANS):
        reasons.append("MeSH does not include Humans")
    if _has(record.mesh, _ANIMALS):
        reasons.append("MeSH includes Animals")
    return reasons


# --- ClinicalTrials.gov results -----------------------------------------------


@dataclass(frozen=True, slots=True)
class TrialMetrics:
    """What the registry reports. Each missing value has a reason instead."""

    nct_id: str
    treatment: str | None = None
    comparator: str | None = None
    condition: str | None = None
    endpoint: str | None = None
    units: str | None = None
    n: int | None = None
    treatment_events: int | None = None
    treatment_total: int | None = None
    comparator_events: int | None = None
    comparator_total: int | None = None
    arr_percent: float | None = None
    p_value: str | None = None
    funding: str | None = None
    missing: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict[str, Any]:
        calculation = None
        if self.arr_percent is not None:
            calculation = (
                f"({self.comparator_events}/{self.comparator_total}) - "
                f"({self.treatment_events}/{self.treatment_total}) = "
                f"{self.arr_percent:.4f} percentage points"
            )
        return {
            "nct_id": self.nct_id,
            "treatment": self.treatment or UNAVAILABLE,
            "comparator": self.comparator or UNAVAILABLE,
            "condition": self.condition or UNAVAILABLE,
            "endpoint": self.endpoint or UNAVAILABLE,
            "units": self.units or UNAVAILABLE,
            "sample_size_n": self.n if self.n is not None else UNAVAILABLE,
            "absolute_risk_reduction_percent": (
                round(self.arr_percent, 4)
                if self.arr_percent is not None
                else UNAVAILABLE
            ),
            "arr_calculation": calculation or UNAVAILABLE,
            "p_value": self.p_value or UNAVAILABLE,
            "funding": self.funding or UNAVAILABLE,
            "missing": list(self.missing),
        }


def ctgov_url(nct_id: str) -> str:
    return f"{CTGOV_STUDY}{nct_id}?format=json"


def _dig(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _dicts(value: Any) -> list[dict[str, Any]]:
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _count(value: Any) -> int | None:
    text = str(value).strip() if value is not None else ""
    return int(text) if text.isdigit() else None


def _intervention(name: str) -> str:
    """'Drug: Empagliflozin' -> 'Empagliflozin'."""
    return name.split(": ", 1)[1] if ": " in name else name


def p_value_significant(reported: str) -> bool:
    """Whether a reported p-value is below SIGNIFICANCE. Unreadable is not."""
    match = _P_VALUE.match(reported)
    if match is None:
        return False
    operator, number = match.groups()
    value = float(number)
    if operator in ("<", "<=", "≤"):
        return value <= SIGNIFICANCE
    return value < SIGNIFICANCE


def p_value_phrase(reported: str) -> str:
    """'0.002' -> 'p=0.002'; '<0.001' -> 'p<0.001'. The value stays as reported."""
    text = reported.strip()
    if text[:1] in "<≤=":
        return f"p{text.replace(' ', '')}"
    return f"p={text}"


def endpoint_is_harm(title: str) -> bool:
    """Whether fewer participants with this endpoint is better, by its wording.

    A count can only be called a risk reduction when the endpoint is a harm.
    The registry does not say which way an outcome points, so an endpoint that
    names no harm, or names any benefit, leaves the direction unknown.
    """
    return bool(_HARM.search(title)) and not _BENEFIT.search(title)


def _funding(study: dict[str, Any], grant_agencies: tuple[str, ...]) -> str | None:
    sponsors = _dig(study, "protocolSection", "sponsorCollaboratorsModule") or {}
    parts = []
    lead = sponsors.get("leadSponsor") if isinstance(sponsors, dict) else None
    if isinstance(lead, dict) and str(lead.get("name") or "").strip():
        label = _SPONSOR_CLASS.get(str(lead.get("class") or ""))
        role = f"{label} lead sponsor" if label else "lead sponsor"
        parts.append(f"{str(lead['name']).strip()} ({role})")
    if grant_agencies:
        parts.append(f"grants from {', '.join(grant_agencies)}")
    return "; ".join(parts) or None


def trial_metrics(
    nct_id: str, study: Any, grant_agencies: tuple[str, ...] = ()
) -> TrialMetrics:
    """Read N, ARR, p, and funding from a ClinicalTrials.gov v2 study record.

    Nothing is estimated: a value the registry does not report, or reports
    ambiguously, is missing, with the reason.
    """
    if not isinstance(study, dict):
        return TrialMetrics(nct_id, missing=("registry record is malformed",))
    missing: list[str] = []
    protocol = study.get("protocolSection") or {}
    if _dig(protocol, "identificationModule", "nctId") != nct_id:
        return TrialMetrics(nct_id, missing=("registry record is for another trial",))
    design = _dig(protocol, "designModule") or {}
    if design.get("studyType") != "INTERVENTIONAL":
        missing.append("registry study type is not interventional")
    if _dig(design, "designInfo", "allocation") != "RANDOMIZED":
        missing.append("registry allocation is not randomized")

    enrollment = _dig(design, "enrollmentInfo") or {}
    n = _count(enrollment.get("count")) if enrollment.get("type") == "ACTUAL" else None
    if n is None or n == 0:
        n = None
        missing.append("actual enrollment (N) not reported")

    conditions = [
        str(item).strip()
        for item in (_dig(protocol, "conditionsModule", "conditions") or [])
        if str(item).strip()
    ]
    condition = ", ".join(conditions) or None
    if condition is None:
        missing.append("condition not reported")

    funding = _funding(study, grant_agencies)
    if funding is None:
        missing.append("funding source not reported")

    arms = {
        str(arm.get("label", "")).casefold(): arm
        for arm in _dicts(_dig(protocol, "armsInterventionsModule", "armGroups"))
    }
    outcomes = [
        item
        for item in _dicts(
            _dig(study, "resultsSection", "outcomeMeasuresModule", "outcomeMeasures")
        )
        if item.get("type") == "PRIMARY"
        and item.get("paramType") == "COUNT_OF_PARTICIPANTS"
    ]
    base = TrialMetrics(nct_id, condition=condition, n=n, funding=funding)
    if not outcomes:
        missing.append("no primary count-of-participants outcome in posted results")
        return _with(base, missing)
    if len(outcomes) > 1:
        missing.append("more than one primary count outcome; endpoint is ambiguous")
        return _with(base, missing)
    outcome = outcomes[0]
    endpoint = str(outcome.get("title") or "").strip() or None
    units = str(outcome.get("unitOfMeasure") or "").strip() or None

    groups = _dicts(outcome.get("groups"))
    roles: dict[str, tuple[str, str]] = {}  # group id -> (role, arm label)
    treatment: str | None = None
    for group in groups:
        arm = arms.get(str(group.get("title", "")).casefold())
        if arm is None:
            continue
        kind = str(arm.get("type") or "")
        if kind == "EXPERIMENTAL":
            roles[str(group.get("id"))] = ("treatment", str(arm.get("label")))
            names = [
                _intervention(str(name)) for name in arm.get("interventionNames") or []
            ]
            treatment = " + ".join(name for name in names if name) or None
        elif kind in _COMPARATOR_ARMS:
            roles[str(group.get("id"))] = ("comparator", str(arm.get("label")))
    by_role = {role: group_id for group_id, (role, _label) in roles.items()}
    if (
        len(groups) != 2
        or len(roles) != 2
        or set(by_role) != {"treatment", "comparator"}
    ):
        missing.append(
            "primary outcome does not compare one experimental arm with one comparator"
        )
        return _with(_replace(base, endpoint=endpoint, units=units), missing)
    comparator = roles[by_role["comparator"]][1]
    if treatment is None:
        missing.append("experimental arm names no intervention")

    denoms = {
        str(item.get("groupId")): _count(item.get("value"))
        for denom in _dicts(outcome.get("denoms"))
        if str(denom.get("units", "")).casefold() == "participants"
        for item in _dicts(denom.get("counts"))
    }
    classes = _dicts(outcome.get("classes"))
    categories = _dicts(classes[0].get("categories")) if len(classes) == 1 else []
    events = (
        {
            str(item.get("groupId")): _count(item.get("value"))
            for item in _dicts(categories[0].get("measurements"))
        }
        if len(categories) == 1
        else {}
    )
    t_id, c_id = by_role["treatment"], by_role["comparator"]
    t_events, t_total = events.get(t_id), denoms.get(t_id)
    c_events, c_total = events.get(c_id), denoms.get(c_id)
    counted = _replace(
        base,
        treatment=treatment,
        comparator=comparator,
        endpoint=endpoint,
        units=units,
        treatment_events=t_events,
        treatment_total=t_total,
        comparator_events=c_events,
        comparator_total=c_total,
    )
    if (
        t_events is None
        or t_total is None
        or c_events is None
        or c_total is None
        or t_total == 0
        or c_total == 0
        or t_events > t_total
        or c_events > c_total
    ):
        missing.append("event counts and denominators not reported for both arms")
        return _with(counted, missing)

    # Reported counts only: comparator event rate minus experimental event rate.
    arr: float | None = (c_events / c_total - t_events / t_total) * 100
    if endpoint is None or not endpoint_is_harm(endpoint):
        missing.append("endpoint direction not established; ARR unavailable")
        arr = None
    elif arr is not None and arr <= 0:
        missing.append(
            "no absolute risk reduction: experimental arm did not have fewer events"
        )
        arr = None

    p_values = {
        str(item.get("pValue")).strip()
        for item in _dicts(outcome.get("analyses"))
        if {str(group) for group in item.get("groupIds") or []} == {t_id, c_id}
        and str(item.get("pValue") or "").strip()
    }
    p_value = next(iter(p_values)) if len(p_values) == 1 else None
    if not p_values:
        missing.append("p-value not reported")
    elif p_value is None:
        missing.append("more than one p-value reported for the comparison")
    elif not p_value_significant(p_value):
        missing.append(f"reported p-value {p_value} is not below {SIGNIFICANCE}")
    return _with(_replace(counted, arr_percent=arr, p_value=p_value), missing)


def _replace(metrics: TrialMetrics, **changes: Any) -> TrialMetrics:
    values = {name: getattr(metrics, name) for name in TrialMetrics.__slots__}
    return TrialMetrics(**(values | changes))


def _with(metrics: TrialMetrics, missing: list[str]) -> TrialMetrics:
    return _replace(metrics, missing=(*metrics.missing, *missing))


@dataclass(frozen=True, slots=True)
class TrialAssessment:
    """Eligibility of the article and completeness of its registry results."""

    pubmed: PubMedRecord | None
    ineligible: tuple[str, ...]
    metrics: TrialMetrics | None = None

    @property
    def qualified(self) -> bool:
        return (
            self.pubmed is not None
            and not self.ineligible
            and self.metrics is not None
            and self.metrics.complete
        )

    @property
    def nct_id(self) -> str | None:
        return self.metrics.nct_id if self.metrics else None

    def reasons(self) -> list[str]:
        missing = list(self.metrics.missing) if self.metrics else []
        if self.metrics is None and not self.ineligible:
            missing = ["no registry results were read"]
        return [*self.ineligible, *missing]


def registry_pmids(study: Any) -> tuple[str, ...]:
    """PMIDs a registry record names as its results publications.

    Sponsor-entered RESULT references are preferred; the NLM's automatic
    DERIVED links count only when there are none. BACKGROUND references never do.
    """
    references = _dicts(
        _dig(study, "protocolSection", "referencesModule", "references")
    )
    for kind in ("RESULT", "DERIVED"):
        pmids = tuple(
            dict.fromkeys(
                str(item.get("pmid")).strip()
                for item in references
                if item.get("type") == kind
                and str(item.get("pmid") or "").strip().isdigit()
            )
        )
        if pmids:
            return pmids
    return ()


def linked_registration(
    pubmed: PubMedRecord, nct_hint: str | None = None
) -> tuple[str | None, str | None]:
    """(NCT number, None), or (None, why no single registration is linked).

    The article may name its registration (PubMed's DataBank list), or the
    registry may name the article (nct_hint, from its references; confirmed
    against the registry record by assess_trial).
    """
    if len(pubmed.nct_ids) > 1:
        return None, "more than one ClinicalTrials.gov registration linked"
    if nct_hint is not None:
        if pubmed.nct_ids and pubmed.nct_ids[0] != nct_hint:
            return None, "PubMed and the registry link different registrations"
        return nct_hint, None
    if pubmed.nct_ids:
        return pubmed.nct_ids[0], None
    return None, "no ClinicalTrials.gov registration linked"


def assess_trial(
    pubmed: PubMedRecord | None,
    study: Any = None,
    pmid: str | None = None,
    nct_hint: str | None = None,
) -> TrialAssessment:
    """Decide whether a PubMed article and its registry results qualify.

    A trial can be found by its PMID or by its NCT number, but either way it
    qualifies only as a peer-reviewed PubMed article linked to one registration.
    """
    if pubmed is None or (pmid is not None and pubmed.pmid != pmid):
        return TrialAssessment(None, ("PubMed record missing or malformed",))
    ineligible = pubmed_ineligibility(pubmed)
    nct_id, unlinked = linked_registration(pubmed, nct_hint)
    if unlinked:
        ineligible.append(unlinked)
    if pubmed.published is None:
        ineligible.append("no full publication date")
    if ineligible or nct_id is None:
        return TrialAssessment(pubmed, tuple(ineligible))
    if study is None:
        return TrialAssessment(
            pubmed,
            (),
            TrialMetrics(nct_id, missing=("registry record unavailable",)),
        )
    if nct_id not in pubmed.nct_ids and pubmed.pmid not in registry_pmids(study):
        return TrialAssessment(
            pubmed, ("registry record does not name this article as its results",)
        )
    return TrialAssessment(
        pubmed, (), trial_metrics(nct_id, study, pubmed.grant_agencies)
    )


def format_percent(value: float) -> str:
    """3.8 -> '3.8', 0.04 -> '0.04'. Rounded for display; the receipt keeps more."""
    text = f"{value:.1f}" if value >= 0.1 else f"{value:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def trial_public_text(assessment: TrialAssessment) -> tuple[str, str]:
    """The trial question and reason. Raises ValueError unless every part holds."""
    metrics = assessment.metrics
    if not assessment.qualified or metrics is None:
        raise ValueError("trial does not qualify for the public trial statement")
    assert metrics.n is not None and metrics.arr_percent is not None
    assert metrics.p_value is not None
    question = (
        f"What did the clinical trial for {metrics.treatment} demonstrate in human "
        f"patients with {metrics.condition}?"
    )
    reason = (
        f"Peer-reviewed human trial (N={metrics.n}) demonstrated an absolute risk "
        f"reduction of {format_percent(metrics.arr_percent)}% "
        f"({p_value_phrase(metrics.p_value)}); funded by {metrics.funding}."
    )
    return question, reason


# --- openFDA approvals --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FdaApproval:
    """An original NDA or BLA approval, as Drugs@FDA records it."""

    application_number: str
    submission_type: str
    submission_number: str
    approved_on: date
    sponsor: str
    brand_names: tuple[str, ...]
    generic_names: tuple[str, ...]

    @property
    def key(self) -> str:
        return (
            f"{self.application_number}:{self.submission_type}{self.submission_number}"
        )

    @property
    def drug(self) -> str:
        brand = " / ".join(self.brand_names)
        generic = ", ".join(self.generic_names)
        if brand and generic:
            return f"{brand} ({generic})"
        return brand or generic or self.application_number


def fda_application_url(application_number: str) -> str:
    query = {"search": f'application_number:"{application_number}"', "limit": "1"}
    return f"{OPENFDA_DRUGSFDA}?{urlencode(query)}"


def _yyyymmdd(value: Any) -> date | None:
    text = str(value or "")
    if not re.fullmatch(r"\d{8}", text):
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError:
        return None


def fda_approvals(application: Any) -> list[FdaApproval]:
    """Original NDA or BLA approvals in one Drugs@FDA application record."""
    if not isinstance(application, dict):
        return []
    number = str(application.get("application_number") or "")
    if not re.fullmatch(r"(?:NDA|BLA)\d{6}", number):
        return []
    products = _dicts(application.get("products"))
    brands = tuple(
        dict.fromkeys(
            str(item.get("brand_name")).strip()
            for item in products
            if str(item.get("brand_name") or "").strip()
        )
    )
    names = _dig(application, "openfda", "generic_name")
    generics = tuple(
        dict.fromkeys(
            str(name).strip()
            for name in (names if isinstance(names, list) else [])
            if str(name).strip()
        )
    )
    approvals = []
    for submission in _dicts(application.get("submissions")):
        approved = _yyyymmdd(submission.get("submission_status_date"))
        if (
            submission.get("submission_type") != "ORIG"
            or submission.get("submission_status") != "AP"
            or approved is None
            or not str(submission.get("submission_number") or "").strip()
        ):
            continue
        approvals.append(
            FdaApproval(
                application_number=number,
                submission_type="ORIG",
                submission_number=str(submission["submission_number"]).strip(),
                approved_on=approved,
                sponsor=str(application.get("sponsor_name") or "").strip(),
                brand_names=brands,
                generic_names=generics,
            )
        )
    return approvals


# --- Adapter ------------------------------------------------------------------


class KleinBiomedAdapter(LazLongAdapter):
    adapter_name = "klein_biomed"
    subject_domain = "science_health"

    def __init__(self, *, lookback_days: int = LOOKBACK_DAYS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        self._lookback_days = lookback_days

    def poll_requests(self) -> list[PollRequest]:
        today = self._clock().astimezone(UTC).date()
        start = today - timedelta(days=self._lookback_days)
        return [
            PollRequest(
                PUBMED_REQUEST,
                ESEARCH,
                {
                    "db": "pubmed",
                    "term": PUBMED_TERM,
                    "datetype": "mhda",
                    "mindate": start.strftime("%Y/%m/%d"),
                    "maxdate": today.strftime("%Y/%m/%d"),
                    "retmode": "json",
                    "retmax": str(PAGE_SIZE),
                    "retstart": "0",
                },
            ),
            PollRequest(
                CTGOV_REQUEST,
                CTGOV_SEARCH,
                {
                    "filter.advanced": (
                        "AREA[StudyType]INTERVENTIONAL AND "
                        "AREA[DesignAllocation]RANDOMIZED AND "
                        f"AREA[ResultsFirstPostDate]RANGE[{start.isoformat()},"
                        f"{today.isoformat()}]"
                    ),
                    "pageSize": str(PAGE_SIZE),
                    "countTotal": "true",
                    "format": "json",
                },
            ),
            PollRequest(
                FDA_REQUEST,
                OPENFDA_DRUGSFDA,
                {
                    "search": (
                        'submissions.submission_type:"ORIG" AND '
                        'submissions.submission_status:"AP" AND '
                        "submissions.submission_status_date:"
                        f"[{start:%Y%m%d} TO {today:%Y%m%d}]"
                    ),
                    "limit": str(PAGE_SIZE),
                    "skip": "0",
                },
            ),
        ]

    def extract_items(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if request.name == PUBMED_REQUEST:
            ids = _dig(payload, "esearchresult", "idlist")
            if not isinstance(ids, list):
                return []
            return [{"pmid": str(item)} for item in ids if str(item).isdigit()]
        if request.name == FDA_REQUEST:
            return _dicts(payload.get("results"))
        if request.name == CTGOV_REQUEST:
            rows = []
            for study in _dicts(payload.get("studies")):
                nct_id = _dig(study, "protocolSection", "identificationModule", "nctId")
                if isinstance(nct_id, str) and _NCT.match(nct_id):
                    rows.append(
                        {"nct_id": nct_id, "pmids": list(registry_pmids(study))}
                    )
            return rows
        return []

    def next_request(
        self, request: PollRequest, payload: dict[str, Any]
    ) -> PollRequest | None:
        if request.name == CTGOV_REQUEST:
            token = payload.get("nextPageToken")
            if not isinstance(token, str) or not token:
                return None
            return PollRequest(
                request.name, request.url, {**request.params, "pageToken": token}
            )
        if request.name == PUBMED_REQUEST:
            offset_key, total = (
                "retstart",
                _count(_dig(payload, "esearchresult", "count")),
            )
        elif request.name == FDA_REQUEST:
            offset_key, total = (
                "skip",
                _count(_dig(payload, "meta", "results", "total")),
            )
        else:
            return None
        offset = (_count(request.params.get(offset_key)) or 0) + PAGE_SIZE
        if total is None or offset >= min(total, MAX_RESULTS):
            return None
        return PollRequest(
            request.name, request.url, {**request.params, offset_key: str(offset)}
        )

    def assemble_records(
        self, items: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        # One record per trial publication, whichever identifier found it.
        trials: dict[str, dict[str, Any]] = {}
        for row in items.get(PUBMED_REQUEST, []):
            pmid = str(row.get("pmid", ""))
            if pmid.isdigit():
                trials.setdefault(f"pubmed:{pmid}", {"kind": "trial", "pmid": pmid})
        for row in items.get(CTGOV_REQUEST, []):
            nct_id = str(row.get("nct_id", ""))
            pmids = [
                str(item) for item in row.get("pmids") or [] if str(item).isdigit()
            ]
            if not _NCT.match(nct_id):
                continue
            if len(pmids) == 1:
                record = trials.setdefault(
                    f"pubmed:{pmids[0]}", {"kind": "trial", "pmid": pmids[0]}
                )
                if record.get("nct_id", nct_id) != nct_id:
                    record["nct_conflict"] = True
                record["nct_id"] = nct_id
            else:
                # Without one results article the trial cannot be called
                # peer-reviewed; it is kept so the reason is logged.
                trials.setdefault(
                    f"ctgov:{nct_id}",
                    {"kind": "trial", "pmid": None, "nct_id": nct_id, "pmids": pmids},
                )
        records: list[dict[str, Any]] = list(trials.values())
        today = self._clock().astimezone(UTC).date()
        start = today - timedelta(days=self._lookback_days)
        approvals: dict[str, FdaApproval] = {}
        for row in items.get(FDA_REQUEST, []):
            for approval in fda_approvals(row):
                if start <= approval.approved_on <= today:
                    approvals.setdefault(approval.key, approval)
        records.extend(
            {"kind": "fda_approval", "approval": approval}
            for approval in approvals.values()
        )
        return records

    def enrich_record(self, raw_record: dict[str, Any]) -> dict[str, Any]:
        """Read the PubMed record, and the registry results if it is eligible."""
        if raw_record.get("kind") != "trial":
            return raw_record
        if raw_record.get("pmid") is None:
            pmids = raw_record.get("pmids") or []
            reason = (
                "registration names no PubMed results publication"
                if not pmids
                else "registration names more than one PubMed results publication"
            )
            logger.info(
                "%s: %s does not qualify: %s",
                self.adapter_name,
                raw_record.get("nct_id"),
                reason,
            )
            return {**raw_record, "assessment": TrialAssessment(None, (reason,))}
        if raw_record.get("nct_conflict"):
            reason = "more than one registration names this article"
            return {**raw_record, "assessment": TrialAssessment(None, (reason,))}
        pmid = str(raw_record["pmid"])
        nct_hint = raw_record.get("nct_id")
        pubmed = parse_pubmed_xml(self._get(efetch_url(pmid)).content)
        assessment = assess_trial(pubmed, pmid=pmid, nct_hint=nct_hint)
        if assessment.metrics is not None:  # eligible, with one NCT number
            nct_id = assessment.metrics.nct_id
            try:
                study = self._get(ctgov_url(nct_id)).json()
            except ValueError as error:
                raise PollError(
                    f"bad registry JSON: {error}", retryable=True
                ) from error
            except PollError as error:
                if error.retryable:
                    raise
                study = None  # not registered or withdrawn: results unavailable
            assessment = assess_trial(pubmed, study, pmid=pmid, nct_hint=nct_hint)
        if not assessment.qualified:
            logger.info(
                "%s: PMID %s does not qualify: %s",
                self.adapter_name,
                pmid,
                "; ".join(assessment.reasons()),
            )
        return {**raw_record, "assessment": assessment}

    def record_key(self, raw_record: dict[str, Any]) -> str:
        if raw_record.get("kind") == "trial":
            if raw_record.get("pmid") is None:
                return f"ctgov:{raw_record['nct_id']}"
            return f"pubmed:{raw_record['pmid']}"
        if raw_record.get("kind") == "fda_approval":
            approval: FdaApproval = raw_record["approval"]
            return f"fda:{approval.key}"
        raise ValueError(f"unknown record kind: {raw_record.get('kind')!r}")

    def record_timestamp(self, raw_record: dict[str, Any]) -> datetime:
        if raw_record.get("kind") == "fda_approval":
            approval: FdaApproval = raw_record["approval"]
            day = approval.approved_on
            return datetime(day.year, day.month, day.day, tzinfo=UTC)
        assessment: TrialAssessment | None = raw_record.get("assessment")
        if assessment is None or assessment.pubmed is None:
            raise ValueError("trial record has not been read from PubMed")
        if assessment.pubmed.published is None:
            raise ValueError("PubMed record has no full publication date")
        return assessment.pubmed.published

    def compute_signal_score(self, raw_record: dict[str, Any]) -> float:
        if raw_record.get("kind") == "fda_approval":
            return FDA_SCORE
        assessment: TrialAssessment | None = raw_record.get("assessment")
        if assessment is None or not assessment.qualified or assessment.metrics is None:
            return 0.0
        large = (assessment.metrics.n or 0) >= LARGE_TRIAL_N
        return TRIAL_SCORE + (LARGE_TRIAL_BONUS if large else 0.0)

    def evidence_targets(self, raw_record: dict[str, Any]) -> list[EvidenceTarget]:
        if raw_record.get("kind") == "fda_approval":
            approval: FdaApproval = raw_record["approval"]
            return [
                EvidenceTarget(
                    url=fda_application_url(approval.application_number),
                    tier=PRIMARY_TIER,
                    kind=KIND_FDA,
                    excerpt=approval.application_number,
                    metadata={
                        "application_number": approval.application_number,
                        "submission_type": approval.submission_type,
                        "submission_number": approval.submission_number,
                        "approved_on": approval.approved_on.isoformat(),
                    },
                )
            ]
        assessment: TrialAssessment | None = raw_record.get("assessment")
        if assessment is None or not assessment.qualified:
            return []
        assert assessment.pubmed is not None and assessment.metrics is not None
        pubmed, metrics = assessment.pubmed, assessment.metrics
        return [
            EvidenceTarget(
                url=efetch_url(pubmed.pmid),
                tier=PRIMARY_TIER,
                kind=KIND_PUBMED,
                excerpt=pubmed.pmid,
                metadata={"pmid": pubmed.pmid, "nct_id": metrics.nct_id},
            ),
            EvidenceTarget(
                url=ctgov_url(metrics.nct_id),
                tier=PRIMARY_TIER,
                kind=KIND_CTGOV,
                excerpt=metrics.endpoint,
                metadata={
                    "pmid": pubmed.pmid,
                    "nct_id": metrics.nct_id,
                    "sample_size_n": metrics.n,
                    "absolute_risk_reduction_percent": metrics.arr_percent,
                    "p_value": metrics.p_value,
                },
            ),
        ]

    def verify_evidence_bar(self, artifacts: list[SourceArtifact]) -> bool:
        """Repeat the whole assessment on the sealed payloads themselves."""
        fda = [item for item in artifacts if item.metadata.get("kind") == KIND_FDA]
        if fda:
            return all(_sealed_approval_holds(item) for item in fda)
        pubmed = [
            item for item in artifacts if item.metadata.get("kind") == KIND_PUBMED
        ]
        ctgov = [item for item in artifacts if item.metadata.get("kind") == KIND_CTGOV]
        if len(pubmed) != 1 or len(ctgov) != 1:
            return False
        return _sealed_trial_holds(pubmed[0], ctgov[0])

    def public_interest_for(self, record: dict[str, Any]) -> tuple[str, str]:
        """Only this adapter's own wording, never text carried in on a record."""
        return (
            self.default_public_question(record),
            self.default_public_interest_reason(record),
        )

    def default_public_question(self, record: dict[str, Any]) -> str:
        if record.get("kind") == "fda_approval":
            approval: FdaApproval = record["approval"]
            return (
                f"What did the FDA approve on {approval.approved_on.isoformat()} "
                f"in application {approval.application_number} for {approval.drug}, "
                "and what trial evidence supports it?"
            )
        return trial_public_text(record["assessment"])[0]

    def default_public_interest_reason(self, record: dict[str, Any]) -> str:
        if record.get("kind") == "fda_approval":
            approval: FdaApproval = record["approval"]
            return (
                f"The FDA approved {approval.drug} on "
                f"{approval.approved_on.isoformat()} (Drugs@FDA application "
                f"{approval.application_number}); no peer-reviewed human trial with "
                "reported absolute risk reduction is linked to this record, so none "
                "is claimed."
            )
        return trial_public_text(record["assessment"])[1]

    def receipt_details(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("kind") == "fda_approval":
            approval: FdaApproval = record["approval"]
            return {
                "detector": self.adapter_name,
                "record_type": "fda_approval",
                "application_number": approval.application_number,
                "submission": f"{approval.submission_type} {approval.submission_number}",
                "approved_on": approval.approved_on.isoformat(),
                "sponsor": approval.sponsor or UNAVAILABLE,
                "drug": approval.drug,
                "trial_evidence": UNAVAILABLE,
                "sample_size_n": UNAVAILABLE,
                "absolute_risk_reduction_percent": UNAVAILABLE,
                "p_value": UNAVAILABLE,
                "funding": UNAVAILABLE,
            }
        assessment: TrialAssessment = record["assessment"]
        pubmed = assessment.pubmed
        return {
            "detector": self.adapter_name,
            "record_type": "human_rct",
            "pmid": pubmed.pmid if pubmed else UNAVAILABLE,
            "journal": pubmed.journal if pubmed else UNAVAILABLE,
            "medline_status": pubmed.medline_status if pubmed else UNAVAILABLE,
            "publication_types": [name for _ui, name in pubmed.publication_types]
            if pubmed
            else [],
            "metrics": assessment.metrics.as_dict() if assessment.metrics else {},
            "not_qualified_because": assessment.reasons(),
            "arr_definition": (
                "absolute risk reduction in percentage points: comparator event rate "
                "minus experimental event rate, from registry counts; not relative"
            ),
        }

    def build_paste_ready_prompt(
        self, record: dict[str, Any], artifacts: list[SourceArtifact]
    ) -> str:
        if record.get("kind") == "fda_approval":
            approval: FdaApproval = record["approval"]
            return render_prompt(
                f"FDA approval: {approval.drug}",
                {
                    "Application": approval.application_number,
                    "Submission": f"{approval.submission_type} {approval.submission_number}",
                    "Approved on": approval.approved_on.isoformat(),
                    "Sponsor": approval.sponsor or UNAVAILABLE,
                    "Trial evidence": f"{UNAVAILABLE}: no eligible trial is linked",
                },
                artifacts,
                "From the sealed FDA record alone, say what was approved, when, and "
                "for whom. Do not describe trial results, efficacy, or risk "
                "reduction: none is in the sealed sources. This is not medical advice.",
            )
        assessment: TrialAssessment = record["assessment"]
        metrics = assessment.metrics
        assert metrics is not None and metrics.arr_percent is not None
        return render_prompt(
            f"Human randomized controlled trial: {metrics.treatment} for {metrics.condition}",
            {
                "Endpoint (primary)": f"{metrics.endpoint} ({metrics.units})",
                "Comparator": metrics.comparator,
                "N (actual enrollment)": metrics.n,
                "Events": (
                    f"experimental {metrics.treatment_events}/{metrics.treatment_total}; "
                    f"comparator {metrics.comparator_events}/{metrics.comparator_total}"
                ),
                "Absolute risk reduction": (
                    f"{format_percent(metrics.arr_percent)} percentage points "
                    "(absolute, not relative)"
                ),
                "p-value (as reported)": metrics.p_value,
                "Funding": metrics.funding,
            },
            artifacts,
            "From the sealed sources alone, state the absolute risk reduction with its "
            "comparator, endpoint, and N in the first paragraph, and the funding. "
            "Never give a relative risk without the baseline rates. This is not "
            "medical advice.",
        )


# --- Evidence bar on sealed payloads ------------------------------------------


def _sealed_bytes(artifact: SourceArtifact) -> bytes | None:
    try:
        return Path(artifact.raw_payload_path).read_bytes()
    except OSError:
        return None


def _sealed_trial_holds(pubmed: SourceArtifact, ctgov: SourceArtifact) -> bool:
    if not (
        pubmed.tier == PRIMARY_TIER
        and ctgov.tier == PRIMARY_TIER
        and host_of(pubmed.origin_url) == EUTILS_HOST
        and host_of(ctgov.origin_url) == CTGOV_HOST
    ):
        return False
    pubmed_raw, ctgov_raw = _sealed_bytes(pubmed), _sealed_bytes(ctgov)
    if pubmed_raw is None or ctgov_raw is None:
        return False
    try:
        study = json.loads(ctgov_raw)
    except ValueError:
        return False
    pmid = str(pubmed.metadata.get("pmid", ""))
    nct_id = ctgov.metadata.get("nct_id")
    assessment = assess_trial(
        parse_pubmed_xml(pubmed_raw),
        study,
        pmid=pmid,
        nct_hint=nct_id if isinstance(nct_id, str) else None,
    )
    metrics = assessment.metrics
    return (
        assessment.qualified
        and metrics is not None
        and metrics.nct_id
        == pubmed.metadata.get("nct_id")
        == ctgov.metadata.get("nct_id")
        and pmid == ctgov.metadata.get("pmid")
        and metrics.n == ctgov.metadata.get("sample_size_n")
        and metrics.arr_percent == ctgov.metadata.get("absolute_risk_reduction_percent")
        and metrics.p_value == ctgov.metadata.get("p_value")
    )


def _sealed_approval_holds(artifact: SourceArtifact) -> bool:
    if artifact.tier != PRIMARY_TIER or host_of(artifact.origin_url) != OPENFDA_HOST:
        return False
    raw = _sealed_bytes(artifact)
    if raw is None:
        return False
    try:
        payload = json.loads(raw)
    except ValueError:
        return False
    results = _dicts(payload.get("results")) if isinstance(payload, dict) else []
    wanted = (
        artifact.metadata.get("application_number"),
        artifact.metadata.get("submission_type"),
        artifact.metadata.get("submission_number"),
        artifact.metadata.get("approved_on"),
    )
    return any(
        (
            approval.application_number,
            approval.submission_type,
            approval.submission_number,
            approval.approved_on.isoformat(),
        )
        == wanted
        for row in results
        for approval in fda_approvals(row)
    )


__all__ = [
    "FdaApproval",
    "KleinBiomedAdapter",
    "PubMedRecord",
    "TrialAssessment",
    "TrialMetrics",
    "assess_trial",
    "endpoint_is_harm",
    "fda_approvals",
    "format_percent",
    "p_value_phrase",
    "p_value_significant",
    "parse_pubmed_xml",
    "pubmed_ineligibility",
    "trial_metrics",
    "trial_public_text",
]
