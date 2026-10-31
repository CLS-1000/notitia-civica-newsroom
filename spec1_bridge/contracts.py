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
