"""Individual WA PDC contributors are not named in the triage prompt.

pdx-1i's WA PDC pattern names the contributor ("The contributor is <NAME> of
<city>, <ST>, ..."). build_prompt replaces a private person's name with "an
individual contributor from <city>, <state>", deciding who is a person from
the sealed data.wa.gov row's contributor_category. Committees, PACs and
businesses keep their names. The record and contributor data below are
invented.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spec1_bridge.contracts import SourceArtifact
from spec1_bridge.pdx1_importer import build_prompt, parse_record, redact_wa_contributor

URL = "https://my.pdc.wa.gov/public/registrations/campaign-finance-report/110400001"
EVIDENCE = "https://data.wa.gov/resource/kv7h-kjye.json?id=29990001"

INDIVIDUAL_PATTERN = (
    "Washington PDC receipt 29990001 reports a contribution of $250.00 to Example "
    "County Democrats (not stated) in Various, Washington, for election cycle 2026. "
    "The contributor is QUIMBY ROSALIND of Tacoma, WA, recorded as Cash."
)
PAC_PATTERN = (
    "Washington PDC receipt 29990002 reports a contribution of $1,000.00 to Example "
    "County Democrats (not stated) in Various, Washington, for election cycle 2026. "
    "The contributor is HARBOR TRADES PAC of Seattle, WA, recorded as Cash."
)


def _record(pattern: str, record_id: str, **overrides: Any) -> Any:
    raw: dict[str, Any] = {
        "record_id": record_id,
        "source_type": "WA_PDC",
        "pattern": pattern,
        "outcome": "ESCALATE",
        "priority": "ELEVATED",
        "tier": "HARD_RECORD",
        "anomaly": None,
        "entity_ids": [],
        "url": URL,
        "evidence_url": EVIDENCE,
        "published_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "origin": "live",
    }
    raw.update(overrides)
    return parse_record(json.dumps(raw))


def _artifact(tmp_path: Path, row: dict[str, Any] | None) -> SourceArtifact:
    path = tmp_path / "S-1.blob"
    path.write_text(json.dumps([row] if row is not None else []), encoding="utf-8")
    return SourceArtifact(
        source_id="S-1",
        tier="tier_1_primary",
        origin_url=EVIDENCE,
        sha256_hash="0" * 64,
        raw_payload_path=str(path),
        captured_at=datetime.now(UTC),
        metadata={"content_type": "application/json"},
    )


def _row(category: str) -> dict[str, Any]:
    return {
        "id": "29990001",
        "filer_name": "Example County Democrats",
        "amount": "250.00",
        "contributor_category": category,
        "contributor_state": "WA",
    }


def test_individual_contributor_is_not_named(tmp_path: Path) -> None:
    record = _record(INDIVIDUAL_PATTERN, "rec_individual")
    prompt = build_prompt(record, _artifact(tmp_path, _row("Individual")))

    assert "QUIMBY" not in prompt
    assert "ROSALIND" not in prompt
    assert (
        "The contributor is an individual contributor from Tacoma, WA, recorded as "
        "Cash." in prompt
    )
    # The committee the money went to is still named.
    assert "Example County Democrats" in prompt


def test_pac_contributor_keeps_its_name(tmp_path: Path) -> None:
    record = _record(PAC_PATTERN, "rec_pac")
    prompt = build_prompt(
        record, _artifact(tmp_path, _row("Political Action Committee"))
    )

    assert "The contributor is HARBOR TRADES PAC of Seattle, WA" in prompt
    assert "an individual contributor" not in prompt


def test_cut_off_pattern_uses_the_sealed_state(tmp_path: Path) -> None:
    """pdx-1i often cuts the pattern off after "of"; the state comes from the row."""
    pattern = INDIVIDUAL_PATTERN.replace("of Tacoma, WA, recorded as Cash.", "of...")
    record = _record(pattern, "rec_cut")
    redacted = redact_wa_contributor(record, _artifact(tmp_path, _row("Individual")))

    assert "QUIMBY" not in redacted
    assert redacted.endswith("The contributor is an individual contributor from WA...")


def test_city_only_and_not_stated_locations(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, {"contributor_category": "Individual"})
    city_only = INDIVIDUAL_PATTERN.replace(
        "of Tacoma, WA, recorded as Cash.", "of Puyallup,..."
    )
    assert redact_wa_contributor(_record(city_only, "rec_city"), artifact).endswith(
        "an individual contributor from Puyallup..."
    )
    unstated = INDIVIDUAL_PATTERN.replace("Tacoma, WA", "not stated, not stated")
    assert redact_wa_contributor(_record(unstated, "rec_unstated"), artifact).endswith(
        "an individual contributor (location not stated), recorded as Cash."
    )


def test_unknown_category_is_treated_as_a_person(tmp_path: Path) -> None:
    """With no contributor_category in the sealed source, the name is removed."""
    record = _record(INDIVIDUAL_PATTERN, "rec_unknown")
    redacted = redact_wa_contributor(record, _artifact(tmp_path, None))
    assert "QUIMBY" not in redacted


def test_other_source_types_are_unchanged(tmp_path: Path) -> None:
    record = _record(INDIVIDUAL_PATTERN, "rec_or", source_type="ORESTAR")
    artifact = _artifact(tmp_path, _row("Individual"))
    assert redact_wa_contributor(record, artifact) == INDIVIDUAL_PATTERN
