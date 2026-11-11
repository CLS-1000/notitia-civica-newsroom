"""Importer for CLS-1000/spec-1 intelligence records and their human verdicts.

Reads spec-1's intelligence store (spec1_intelligence.jsonl), one record per
line as spec-1's cycle writes it (spec1_core.app.cycle.build_record_dict plus
the store's written_at), and optionally a cls_verdicts verdicts.jsonl. Each
usable record's signal URL is sealed into custody and emitted as a
Spec1LeadPacket for newsroom triage.

Fields read from a record:

    record_id, signal_url, signal_source, outcome_classification
    (or classification), confidence, analysis_tier, signal_excerpt,
    hypothesis, llm_reasoning, run_id, written_at

Only analysis_tier='llm_verified' records are imported: spec-1 treats
rule_based output as triage, never analysis. A record whose latest human
verdict is 'incorrect' is left out; any other verdict travels with the lead.

gatepost's `gatepost brief` Markdown is not imported: its signal lines name a
source but carry no URL, so there is nothing to seal and no lead could pass
the evidence bar.

The newsroom packet is still analysis_tier='rule_based'. spec-1's model
verdict is carried as context, not as the newsroom's own verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
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
from spec1_bridge.pdx1_importer import classify_source_tier as classify_civic_tier

logger = logging.getLogger(__name__)

ADAPTER_NAME: Final = "spec1_importer"
SUBJECT_DOMAIN: Final = "spec1_osint"
ARTIFACT_KIND: Final = "spec1_signal_source"
LLM_VERIFIED: Final = "llm_verified"
VALID_VERDICTS: Final = frozenset({"correct", "incorrect", "partial", "unclear"})
EXCERPT_CHARS: Final = 300

# Federal hosts that publish filed public records, on top of the Oregon and
# Washington hosts pdx1_importer already treats as primary.
FEDERAL_PRIMARY_HOSTS: Final = (
    "congress.gov",
    "efile.fara.gov",
    "fara.gov",
    "federalregister.gov",
    "govinfo.gov",
    "sec.gov",
    "uscourts.gov",
)


class Spec1RecordError(ValueError):
    """A store line is not a usable spec-1 intelligence record."""


@dataclass(frozen=True, slots=True)
class Spec1Record:
    record_id: str
    run_id: str
    signal_url: str | None
    signal_source: str
    classification: str
    confidence: float
    analysis_tier: str
    signal_excerpt: str
    hypothesis: str
    llm_reasoning: str
    written_at: datetime
    public_question: str = ""
    public_interest_reason: str = ""


@dataclass(frozen=True, slots=True)
class Verdict:
    record_id: str
    verdict: str
    reviewer: str
    reviewed_at: str
    notes: str


@dataclass(slots=True)
class Spec1Import:
    """Outcome of one import. skipped counts records by why they were left out."""

    packets: list[Spec1LeadPacket] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)


def _text(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    return value if isinstance(value, str) else ""


def parse_record(line: str) -> Spec1Record:
    """Validate one JSONL line against the fields spec-1's cycle writes."""
    try:
        raw = json.loads(line)
    except ValueError as error:
        raise Spec1RecordError(f"not JSON: {error}") from error
    if not isinstance(raw, dict):
        raise Spec1RecordError("not a JSON object")
    record_id = _text(raw, "record_id")
    if not record_id:
        raise Spec1RecordError("record_id missing")
    confidence = raw.get("confidence")
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        raise Spec1RecordError("confidence missing or not a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise Spec1RecordError(f"confidence {confidence} is outside 0..1")
    written = _text(raw, "written_at")
    try:
        written_at = datetime.fromisoformat(written)
    except ValueError as error:
        raise Spec1RecordError(f"bad written_at: {written!r}") from error
    if written_at.tzinfo is None:
        written_at = written_at.replace(tzinfo=UTC)
    url = _text(raw, "signal_url")
    return Spec1Record(
        record_id=record_id,
        run_id=_text(raw, "run_id"),
        signal_url=url or None,
        signal_source=_text(raw, "signal_source"),
        classification=_text(raw, "outcome_classification")
        or _text(raw, "classification"),
        confidence=float(confidence),
        analysis_tier=_text(raw, "analysis_tier"),
        signal_excerpt=_text(raw, "signal_excerpt"),
        hypothesis=_text(raw, "hypothesis"),
        llm_reasoning=_text(raw, "llm_reasoning"),
        written_at=written_at,
        public_question=_text(raw, "public_question"),
        public_interest_reason=_text(raw, "public_interest_reason"),
    )


def load_verdicts(path: Path | str | None) -> tuple[dict[str, Verdict], list[str]]:
    """Latest verdict per record_id, by reviewed_at, then file order. Returns (verdicts, errors)."""
    if path is None:
        return {}, []
    file_path = Path(path)
    if not file_path.exists():
        return {}, [f"verdicts file not found: {file_path}"]
    latest: dict[str, tuple[str, int, Verdict]] = {}
    errors: list[str] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except ValueError:
                errors.append(f"verdicts line {number}: not JSON")
                continue
            if not isinstance(raw, dict):
                errors.append(f"verdicts line {number}: not a JSON object")
                continue
            verdict = Verdict(
                record_id=_text(raw, "record_id"),
                verdict=_text(raw, "verdict"),
                reviewer=_text(raw, "reviewer") or "anonymous",
                reviewed_at=_text(raw, "reviewed_at"),
                notes=_text(raw, "notes"),
            )
            if not verdict.record_id or verdict.verdict not in VALID_VERDICTS:
                errors.append(
                    f"verdicts line {number}: missing record_id or unknown verdict"
                )
                continue
            key = (verdict.reviewed_at, number)
            current = latest.get(verdict.record_id)
            if current is None or key >= current[:2]:
                latest[verdict.record_id] = (verdict.reviewed_at, number, verdict)
    return {record_id: entry[2] for record_id, entry in latest.items()}, errors


def classify_source_tier(url: str) -> str:
    """tier_1_primary for filed-record hosts (federal, Oregon, Washington), else tier_3."""
    host = (httpx.URL(url).host or "").lower()
    if any(host == h or host.endswith("." + h) for h in FEDERAL_PRIMARY_HOSTS):
        return "tier_1_primary"
    return classify_civic_tier(url)


def lead_id_for(record: Spec1Record) -> str:
    """Stable lead ID from the spec-1 record ID.

    This hash only names the lead, so it is marked usedforsecurity=False.
    Custody's integrity hashes are separate and stay full SHA-256.
    """
    digest = hashlib.sha256(
        f"spec1:{record.record_id}".encode(), usedforsecurity=False
    ).digest()
    return str(uuid.UUID(bytes=digest[:16], version=5))


def score_record(record: Spec1Record) -> float:
    """spec-1's final record confidence on a 0-100 scale. A rule-based screen."""
    return clamp_score(record.confidence * 100.0)


def pick_excerpt(record: Spec1Record, text: str) -> str:
    """spec-1's own excerpt if the source states it verbatim, else the source's opening."""
    if record.signal_excerpt and excerpt_is_verbatim(record.signal_excerpt, text):
        return record.signal_excerpt
    return normalize_whitespace(text)[:EXCERPT_CHARS].strip()


def public_interest_for(record: Spec1Record) -> tuple[str, str]:
    """Return the record's own question and reason, else neutral defaults.

    spec-1's hypothesis is unverified, so the default asks whether the source
    supports it rather than repeating it as a claim.
    """
    source = record.signal_source or "this signal"
    return public_interest(
        record.public_question,
        record.public_interest_reason,
        f"Does the sealed source from {source} support what spec-1 flagged?",
        "If the lead holds up, the public needs it reported accurately; if it "
        "does not, it should not spread unchecked.",
    )


def build_prompt(
    record: Spec1Record, artifact: SourceArtifact, verdict: Verdict | None
) -> str:
    facts: dict[str, Any] = {
        "spec-1 record": f"{record.record_id} (run {record.run_id or '?'})",
        "Source": record.signal_source or "(unnamed)",
        "spec-1 classification / confidence": f"{record.classification or '?'} / {record.confidence:.2f}",
        "spec-1 analysis tier": record.analysis_tier,
        "Hypothesis (spec-1, unverified)": record.hypothesis or "(none)",
        "Model reasoning (spec-1, unverified)": record.llm_reasoning or "(none)",
        "Human verdict": f"{verdict.verdict} by {verdict.reviewer}"
        if verdict
        else "none filed",
    }
    return render_prompt(
        f"OSINT LEAD: spec-1 {record.record_id}",
        facts,
        [artifact],
        "Check spec-1's hypothesis against the sealed source and say what an editor must verify first.",
    )


def import_spec1_records(
    store_path: Path | str,
    allocate_source_id: Callable[[], str],
    *,
    verdicts_path: Path | str | None = None,
    is_known: Callable[[str], bool] = lambda _lead_id: False,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    transport: httpx.BaseTransport | None = None,
) -> Spec1Import:
    """Read spec1_intelligence.jsonl and emit leads for model-verified records.

    Sources are fetched only for new leads that clear every check above, so a
    re-run over the same store fetches nothing.
    """
    result = Spec1Import()
    verdicts, verdict_errors = load_verdicts(verdicts_path)
    result.errors.extend(verdict_errors)
    file_path = Path(store_path)
    if not file_path.exists():
        result.errors.append(f"store not found: {file_path}")
        return result

    seen: set[str] = set()
    with file_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = parse_record(line)
            except Spec1RecordError as error:
                result.skipped["malformed"] += 1
                result.errors.append(f"line {number}: {error}")
                continue
            if record.analysis_tier != LLM_VERIFIED:
                result.skipped["not_llm_verified"] += 1
                continue
            verdict = verdicts.get(record.record_id)
            if verdict is not None and verdict.verdict == "incorrect":
                result.skipped["verdict_incorrect"] += 1
                continue
            if record.signal_url is None:
                result.skipped["no_url"] += 1
                continue
            lead_id = lead_id_for(record)
            if lead_id in seen or is_known(lead_id):
                result.skipped["known"] += 1
                continue
            seen.add(lead_id)

            tier = classify_source_tier(record.signal_url)
            try:
                receipt = fetch_and_seal_source(
                    record.signal_url,
                    allocate_source_id(),
                    tier,
                    archive_dir=archive_dir,
                    transport=transport,
                )
            except (httpx.HTTPError, CustodyError, ValueError, OSError) as error:
                result.skipped["seal_failed"] += 1
                result.errors.append(
                    f"{record.record_id}: could not seal {record.signal_url}: {error}"
                )
                continue

            archive_path = Path(receipt["archive_path"])
            text = read_artifact_text(
                archive_path.read_bytes(), receipt["content_type"]
            )
            if not text or not text.strip():
                result.skipped["no_source_text"] += 1
                continue

            artifact = SourceArtifact(
                source_id=receipt["source_id"],
                tier=tier,
                origin_url=record.signal_url,
                sha256_hash=receipt["raw_sha256"],
                raw_payload_path=str(archive_path),
                captured_at=datetime.fromisoformat(receipt["retrieved_at"]),
                metadata={
                    "kind": ARTIFACT_KIND,
                    "content_type": receipt["content_type"],
                    "engine": "spec-1",
                    "spec1_record_id": record.record_id,
                    "spec1_run_id": record.run_id,
                    "signal_source": record.signal_source,
                    "spec1_classification": record.classification,
                    "human_verdict": verdict.verdict if verdict else None,
                },
            )
            excerpt = pick_excerpt(record, text)
            question, reason = public_interest_for(record)
            result.packets.append(
                Spec1LeadPacket(
                    lead_id=lead_id,
                    timestamp=record.written_at,
                    adapter_name=ADAPTER_NAME,
                    anomaly_score=score_record(record),
                    evidence_bar_passed=bool(excerpt),
                    source_artifacts=[artifact],
                    verbatim_excerpts={artifact.source_id: excerpt},
                    paste_ready_prompt=build_prompt(record, artifact, verdict),
                    analysis_tier="rule_based",
                    public_question=question,
                    public_interest_reason=reason,
                )
            )

    logger.info(
        "spec-1: %d leads, skipped %s, %d errors",
        len(result.packets),
        dict(result.skipped),
        len(result.errors),
    )
    return result
