"""Data contracts shared by every SPEC-1 adapter and the triage worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List  # noqa: UP035


@dataclass(frozen=True)
class SourceArtifact:
    source_id: str  # Format: S-XXXX
    tier: str  # 'tier_1_primary', 'tier_2_institutional', etc.
    origin_url: str
    sha256_hash: str
    raw_payload_path: str
    captured_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)  # noqa: UP006


@dataclass
class Spec1LeadPacket:
    lead_id: str
    timestamp: datetime
    adapter_name: str
    anomaly_score: float  # 0.0 to 100.0
    evidence_bar_passed: bool
    source_artifacts: List[SourceArtifact]  # noqa: UP006
    verbatim_excerpts: Dict[str, str]  # {source_id: text_span}  # noqa: UP006
    paste_ready_prompt: str
    analysis_tier: str = "rule_based"
    # Five-Mask Standards: the question a story would answer for the public, and
    # why the public has a stake in it. Producers fill both, using what the raw
    # record says when it says anything, through public_interest().
    public_question: str = ""
    public_interest_reason: str = ""
    # Detector findings kept in the triage record's decomposition_receipt, such
    # as the claim matcher's priority and shingle evidence. Never used by a gate.
    receipt_details: Dict[str, Any] = field(default_factory=dict)  # noqa: UP006


def _clean(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def public_interest(
    raw_question: object,
    raw_reason: object,
    default_question: str,
    default_reason: str,
) -> tuple[str, str]:
    """Return (question, reason), preferring non-blank text from the raw record.

    Each value falls back to its default on its own, so a record that states
    only a question still gets the default reason.
    """
    return (
        _clean(raw_question) or _clean(default_question),
        _clean(raw_reason) or _clean(default_reason),
    )
