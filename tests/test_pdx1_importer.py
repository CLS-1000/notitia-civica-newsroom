"""Tests for spec1_bridge/pdx1_importer.py."""

import json
from pathlib import Path
from unittest.mock import patch

from spec1_bridge.pdx1_importer import (
    classify_source_tier,
    import_pdx1_signals,
)


def test_classify_source_tier() -> None:
    assert classify_source_tier("https://sos.oregon.gov/orestar/tran.html") == "tier_1_primary"
    assert classify_source_tier("https://www.oregonlegislature.gov/bills") == "tier_1_primary"
    assert classify_source_tier("https://www.oregonlive.com/news/article.html") == "tier_3_secondary"


def test_import_pdx1_signals_end_to_end(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "pdx1_signals.jsonl"
    fake_record = {
        "record_id": "pdx-test-101",
        "url": "https://sos.oregon.gov/orestar/tran/99812",
        "source_type": "orestar_campaign_finance",
        "pattern": "Major PAC Contribution Discrepancy",
        "priority": "HIGH",
        "entity_ids": ["PAC-Clean-Energy", "Seat-Commission-03"],
        "anomaly": {"score": 0.85},
    }
    jsonl_file.write_text(json.dumps(fake_record) + "\n", encoding="utf-8")

    dummy_receipt = {
        "source_id": "S-PDX-TEST",
        "raw_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "archive_path": str(tmp_path / "dummy.blob"),
    }
    Path(dummy_receipt["archive_path"]).write_text("ORESTAR Certified Filing Details...", encoding="utf-8")

    with patch("spec1_bridge.pdx1_importer.fetch_and_seal_source", return_value=dummy_receipt):
        packets = import_pdx1_signals(jsonl_file)

    assert len(packets) == 1
    p = packets[0]
    assert p.adapter_name == "pdx1_importer"
    assert p.anomaly_score == 85.0
    assert p.analysis_tier == "rule_based"
    assert p.evidence_bar_passed is True
    assert "S-PDX-TEST" in p.verbatim_excerpts
