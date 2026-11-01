"""Importer for CLS-1000/pdx-1i signals.

Reads pdx1_signals.jsonl, extracts ORESTAR, SEI, and civic records,
seals primary sources into radiant_ledger's source_registry, and emits
Spec1LeadPacket objects for newsroom triage.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import uuid

from newsroom.harlan_eternity.custody import fetch_and_seal_source
from spec1_bridge.contracts import SourceArtifact, Spec1LeadPacket

logger = logging.getLogger(__name__)

PDX1_NAMESPACE = uuid.UUID("a3b89f21-7c42-4f1b-8e2a-9e123456789a")

PRIMARY_DOMAINS = (
    "sos.oregon.gov",
    "oregonlegislature.gov",
    "portland.gov",
    "multco.us",
    "clackamas.us",
    "washingtoncountyor.gov",
    "oregon.gov",
)


def classify_source_tier(url: str) -> str:
    """Classify sourcing tier based on origin domain."""
    clean_url = url.lower()
    for domain in PRIMARY_DOMAINS:
        if domain in clean_url:
            return "tier_1_primary"
    return "tier_3_secondary"


def build_pdx1_prompt(record: dict[str, Any], excerpt: str) -> str:
    """Format a paste-ready prompt preserving role-based seat designations."""
    record_id = record.get("record_id", "UNKNOWN")
    pattern = record.get("pattern", "Civic / Campaign Finance Activity")
    priority = record.get("priority", "NORMAL")
    anomaly = record.get("anomaly", {})
    entity_ids = ", ".join(record.get("entity_ids", [])) or "None specified"

    return (
        f"INVESTIGATIVE LEAD: PDX-1i Signal [{record_id}]\n"
        f"Pattern: {pattern} (Priority: {priority})\n"
        f"Associated Entities/Seats: {entity_ids}\n"
        f"Anomaly Details: {json.dumps(anomaly, default=str)}\n"
        f"Primary Verbatim Excerpt:\n"
        f'"{excerpt}"\n\n'
        f"TASK: Draft a nonpartisan factual brief evaluating this civic transaction. "
        f"Cite all assertions strictly using the provided primary source ID."
    )


def import_pdx1_signals(
    store_path: Path | str,
    priority_threshold: str = "NORMAL",
) -> list[Spec1LeadPacket]:
    """Parse pdx1_signals.jsonl and emit newsroom-compatible Spec1LeadPackets."""
    file_path = Path(store_path)
    if not file_path.exists():
        logger.warning("PDX-1i signals file not found: %s", file_path)
        return []

    priority_ranks = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "CRITICAL": 3}
    min_rank = priority_ranks.get(priority_threshold.upper(), 1)

    lead_packets: list[Spec1LeadPacket] = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.error("Skipping malformed JSON line in %s", file_path)
                continue

            record_priority = record.get("priority", "NORMAL").upper()
            if priority_ranks.get(record_priority, 1) < min_rank:
                continue

            source_url = record.get("url")
            if not source_url:
                continue

            engine_record_id = str(record.get("record_id", uuid.uuid4()))
            lead_id = str(uuid.uuid5(PDX1_NAMESPACE, engine_record_id))

            tier = classify_source_tier(source_url)

            try:
                custody_receipt = fetch_and_seal_source(
                    url=source_url,
                    source_id=f"S-{engine_record_id[:8].upper()}",
                    tier=tier,
                )
            except Exception as e:
                logger.error("Failed to seal source %s: %s", source_url, e)
                continue

            archive_path = Path(custody_receipt["archive_path"])
            verbatim_text = ""
            if archive_path.exists():
                try:
                    verbatim_text = archive_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    verbatim_text = ""

            sample_excerpt = verbatim_text[:300].strip() if verbatim_text else "Primary docket record sealed."
            source_id = custody_receipt["source_id"]

            artifact = SourceArtifact(
                source_id=source_id,
                tier=tier,
                origin_url=source_url,
                sha256_hash=custody_receipt["raw_sha256"],
                raw_payload_path=str(archive_path),
                captured_at=datetime.now(UTC),
                metadata={
                    "engine": "pdx-1i",
                    "source_type": record.get("source_type", "civic_filing"),
                    "entity_ids": record.get("entity_ids", []),
                },
            )

            raw_score = float(record.get("anomaly", {}).get("score", 50.0))
            normalized_score = (raw_score * 100.0) if raw_score <= 1.0 else min(raw_score, 100.0)

            prompt = build_pdx1_prompt(record, sample_excerpt)

            packet = Spec1LeadPacket(
                lead_id=lead_id,
                timestamp=datetime.now(UTC),
                adapter_name="pdx1_importer",
                anomaly_score=normalized_score,
                evidence_bar_passed=(tier == "tier_1_primary"),
                source_artifacts=[artifact],
                verbatim_excerpts={source_id: sample_excerpt},
                paste_ready_prompt=prompt,
                analysis_tier="rule_based",
            )
            lead_packets.append(packet)

    return lead_packets
