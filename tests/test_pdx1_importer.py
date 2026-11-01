"""Tests for spec1_bridge/pdx1_importer.py.

Fixture records follow pdx-1i's IntelligenceRecord.model_dump_json() exactly
(src/pdx1/models.py at ffa5f58): anomaly is null unless the record is a sigma
anomaly, url may be null, priority is ELEVATED | STANDARD | MONITOR. Sealing
runs for real through custody, against an httpx.MockTransport.
"""

from __future__ import annotations

import io
import itertools
import json
import uuid
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest

from spec1_bridge import mycroft_triage, pdx1_importer
from spec1_bridge.adapters.base_lazlong import excerpt_is_verbatim, read_artifact_text
from spec1_bridge.mycroft_triage import (
    Enrichment,
    OllamaEnricher,
    TriageReport,
    triage_packets,
)
from spec1_bridge.pdx1_importer import (
    Pdx1RecordError,
    classify_source_tier,
    import_pdx1_signals,
    parse_record,
    score_record,
)

ORESTAR_URL = "https://secure.sos.state.or.us/orestar/gotoPublicTransactionSearch.do?id=1"
ORESTAR_PAGE = (
    "<html><body><h1>ORESTAR Transaction</h1>"
    "<p>Contribution $25,000 to Committee X</p></body></html>"
)
PRESS_URL = "https://www.oregonlive.com/politics/2026/10/story.html"
PRESS_PAGE = "<html><body><p>Commissioner seat 3 said the vote was close.</p></body></html>"
SEI_URL = "https://apps.oregon.gov/sei/filing/42"
SEI_PAGE = "<html><body><p>Statement of Economic Interest, filed 2026.</p></body></html>"
PDF_URL = "https://www.portland.gov/sites/default/files/record.bin"


def _record(**overrides: Any) -> dict[str, Any]:
    published = (datetime.now(UTC) - timedelta(days=5)).replace(microsecond=0)
    record: dict[str, Any] = {
        "record_id": "rec_a",
        "run_id": "pdx1_2026_1024_191904",
        "source": "ORESTAR",
        "source_type": "ORESTAR",
        "pattern": "Contribution $25,000 to Committee X",
        "outcome": "ESCALATE",
        "priority": "STANDARD",
        "confidence": 0.9,
        "tier": "HARD_RECORD",
        "gates": {"credibility": True, "volume": True, "velocity": True, "novelty": True, "detail": {}},
        "anomaly": None,
        "entity_ids": ["seat-metro-d2"],
        "signal_id": "sig_a",
        "dedup_hash": "",
        "url": ORESTAR_URL,
        "published_at": published.isoformat().replace("+00:00", "Z"),
        "created_at": "2026-11-01T14:41:18.943198Z",
    }
    record.update(overrides)
    return record


ANOMALY = {
    "observed": 25000.0,
    "baseline_mean": 1200.0,
    "baseline_stddev": 900.0,
    "sigma": 25.3,
    "tier": "TIER_1",
    "window_days": 90,
    "sample_size": 40,
}


class Web:
    """Mock web: serves fixed pages and counts requests."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.pages: dict[str, tuple[int, str, bytes]] = {
            ORESTAR_URL: (200, "text/html; charset=utf-8", ORESTAR_PAGE.encode()),
            PRESS_URL: (200, "text/html; charset=utf-8", PRESS_PAGE.encode()),
            SEI_URL: (200, "text/html; charset=utf-8", SEI_PAGE.encode()),
            PDF_URL: (200, "application/octet-stream", b"\x00\x01binary"),
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        status, content_type, body = self.pages.get(url, (404, "text/plain", b"not found"))
        return httpx.Response(status, headers={"content-type": content_type}, content=body)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _write_store(path: Path, records: list[dict[str, Any] | str]) -> Path:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _allocator() -> Callable[[], str]:
    counter = itertools.count(1)
    return lambda: f"S-{next(counter)}"


def _by_record(packets: list[Any]) -> dict[str, Any]:
    return {p.source_artifacts[0].metadata["pdx1_record_id"]: p for p in packets}


def test_classify_source_tier_uses_real_hosts() -> None:
    assert classify_source_tier(ORESTAR_URL) == "tier_1_primary"
    assert classify_source_tier("https://data.wa.gov/resource/kv7h-kjye.json") == "tier_1_primary"
    assert classify_source_tier("https://olis.oregonlegislature.gov/liz/2025R1") == "tier_1_primary"
    assert classify_source_tier(SEI_URL) == "tier_1_primary"
    assert classify_source_tier(PRESS_URL) == "tier_3_secondary"
    assert classify_source_tier("https://sos.oregon.gov.example.com/x") == "tier_3_secondary"


def test_parse_record_accepts_null_anomaly_and_null_url() -> None:
    record = parse_record(json.dumps(_record(url=None)))
    assert record.sigma is None
    assert record.anomaly is None
    assert record.url is None
    assert record.published_at.tzinfo is not None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"priority": "HIGH"}, "unknown priority"),
        ({"tier": "TIER_1"}, "unknown tier"),
        ({"anomaly": {"score": 0.8}}, "numeric sigma"),
        ({"published_at": "yesterday"}, "bad published_at"),
        ({"record_id": None}, "record_id"),
    ],
)
def test_parse_record_rejects_what_pdx1_never_writes(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(Pdx1RecordError, match=message):
        parse_record(json.dumps(_record(**overrides)))


def test_score_is_priority_floor_raised_by_sigma() -> None:
    assert score_record(parse_record(json.dumps(_record(priority="MONITOR")))) == 30.0
    assert score_record(parse_record(json.dumps(_record(priority="STANDARD")))) == 45.0
    assert score_record(parse_record(json.dumps(_record(priority="ELEVATED")))) == 60.0
    three_sigma = dict(ANOMALY, sigma=3.0)
    assert score_record(parse_record(json.dumps(_record(priority="MONITOR", anomaly=three_sigma)))) == 60.0
    assert score_record(parse_record(json.dumps(_record(priority="ELEVATED", anomaly=ANOMALY)))) == 100.0


def test_import_end_to_end_seals_for_real(tmp_path: Path) -> None:
    web = Web()
    store = _write_store(
        tmp_path / "pdx1_signals.jsonl",
        [
            _record(record_id="rec_anomaly", priority="ELEVATED", anomaly=ANOMALY),
            _record(record_id="rec_plain"),
            _record(record_id="rec_monitor", priority="MONITOR"),
            _record(record_id="rec_no_url", url=None),
            "{not json",
            _record(record_id="rec_inferred", tier="INFERRED", url=SEI_URL, pattern="Seat 2 linked to PAC"),
            _record(record_id="rec_press", tier="REPORTED", url=PRESS_URL, pattern="not in page"),
        ],
    )
    result = import_pdx1_signals(
        store, _allocator(), archive_dir=tmp_path / "archive", transport=web.transport
    )
    leads = _by_record(result.packets)

    assert set(leads) == {"rec_anomaly", "rec_plain", "rec_inferred", "rec_press"}
    assert result.skipped == {"below_priority": 1, "no_url": 1, "malformed": 1}
    assert any("line 5" in error for error in result.errors)

    anomaly = leads["rec_anomaly"]
    artifact = anomaly.source_artifacts[0]
    assert anomaly.anomaly_score == 100.0
    assert anomaly.evidence_bar_passed is True
    assert anomaly.analysis_tier == "rule_based"
    assert artifact.tier == "tier_1_primary"
    assert artifact.source_id.startswith("S-") and artifact.source_id[2:].isdigit()
    assert anomaly.lead_id == str(uuid.uuid5(pdx1_importer.PDX1_NAMESPACE, "rec_anomaly"))
    assert anomaly.verbatim_excerpts == {artifact.source_id: "Contribution $25,000 to Committee X"}

    assert leads["rec_inferred"].evidence_bar_passed is False
    assert leads["rec_press"].source_artifacts[0].tier == "tier_3_secondary"

    for packet in result.packets:
        sealed = packet.source_artifacts[0]
        text = read_artifact_text(Path(sealed.raw_payload_path).read_bytes(), sealed.metadata["content_type"])
        assert text is not None
        for excerpt in packet.verbatim_excerpts.values():
            assert excerpt and excerpt_is_verbatim(excerpt, text)


def test_duplicate_record_ids_in_one_store_count_as_known(tmp_path: Path) -> None:
    web = Web()
    store = _write_store(tmp_path / "s.jsonl", [_record(), _record()])
    result = import_pdx1_signals(store, _allocator(), archive_dir=tmp_path / "a", transport=web.transport)
    assert len(result.packets) == 1
    assert result.skipped == {"known": 1}
    assert len(web.requests) == 1


def test_seal_failure_and_unreadable_source_are_skipped(tmp_path: Path) -> None:
    web = Web()
    store = _write_store(
        tmp_path / "s.jsonl",
        [
            _record(record_id="rec_404", url="https://secure.sos.state.or.us/missing"),
            _record(record_id="rec_binary", url=PDF_URL),
            _record(record_id="rec_ok"),
        ],
    )
    result = import_pdx1_signals(store, _allocator(), archive_dir=tmp_path / "a", transport=web.transport)
    assert list(_by_record(result.packets)) == ["rec_ok"]
    assert result.skipped == {"seal_failed": 1, "no_source_text": 1}
    assert any("rec_404" in error for error in result.errors)


def test_known_leads_are_not_fetched_again(tmp_path: Path) -> None:
    store = _write_store(tmp_path / "s.jsonl", [_record(record_id="rec_a"), _record(record_id="rec_b")])
    first = import_pdx1_signals(store, _allocator(), archive_dir=tmp_path / "a", transport=Web().transport)
    known = {p.lead_id for p in first.packets}

    web = Web()
    calls: list[str] = []

    def allocate() -> str:
        calls.append("x")
        return "S-1"

    second = import_pdx1_signals(
        store, allocate, is_known=known.__contains__, archive_dir=tmp_path / "b", transport=web.transport
    )
    assert second.packets == []
    assert second.skipped == {"known": 2}
    assert web.requests == []
    assert calls == []


def test_missing_store_and_bad_priority(tmp_path: Path) -> None:
    result = import_pdx1_signals(tmp_path / "missing.jsonl", _allocator())
    assert result.packets == []
    assert result.errors and "store not found" in result.errors[0]
    with pytest.raises(ValueError, match="min_priority"):
        import_pdx1_signals(tmp_path / "missing.jsonl", _allocator(), min_priority="HIGH")


class _OfflineEnricher(OllamaEnricher):
    def enrich(self, packet: Any) -> Enrichment:
        return Enrichment("unavailable", "test", error="offline")


def test_only_hard_primary_records_clear_triage(
    tmp_path: Path, db: psycopg.Connection[tuple[Any, ...]]
) -> None:
    store = _write_store(
        tmp_path / "s.jsonl",
        [
            _record(record_id="rec_hard", priority="ELEVATED", anomaly=ANOMALY),
            _record(record_id="rec_inferred", tier="INFERRED", url=SEI_URL, pattern="x"),
            _record(record_id="rec_press", tier="REPORTED", url=PRESS_URL, pattern="x"),
        ],
    )
    result = import_pdx1_signals(
        store,
        lambda: mycroft_triage.allocate_source_id(db),
        archive_dir=tmp_path / "a",
        transport=Web().transport,
    )
    report = triage_packets(result.packets, "oregon_civic", db, _OfflineEnricher())

    assert report.written == [str(uuid.uuid5(pdx1_importer.PDX1_NAMESPACE, "rec_hard"))]
    assert {(r.gate, r.reason) for r in report.rejected} == {
        ("authority", "evidence bar not passed"),
        ("authority", "no tier 1 or tier 2 source"),
    }
    row = db.execute(
        "SELECT adapter_name, analysis_tier::text, anomaly_score FROM triage_records"
    ).fetchone()
    assert row == ("pdx1_importer", "rule_based", 100.0)


def test_cli_pdx1_store_writes_then_dedupes(
    tmp_path: Path, ledger_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mycroft_triage, "OllamaEnricher", _OfflineEnricher)
    monkeypatch.setattr(mycroft_triage, "run_adapter", lambda *_args: TriageReport())
    web = Web()
    real_seal = pdx1_importer.fetch_and_seal_source

    def offline_seal(*args: Any, **kwargs: Any) -> Any:
        return real_seal(*args, **{**kwargs, "transport": web.transport})

    monkeypatch.setattr(pdx1_importer, "fetch_and_seal_source", offline_seal)
    store = _write_store(tmp_path / "s.jsonl", [_record(priority="ELEVATED"), _record(record_id="rec_m", priority="MONITOR")])

    def run(*extra: str) -> dict[str, Any]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            assert mycroft_triage.main(["--dsn", ledger_dsn, "--pdx1-store", str(store), *extra]) == 0
        return json.loads(buffer.getvalue())

    assert run()["pdx1_importer"] == {"written": 1, "rejected": 0}
    assert run()["pdx1_importer"] == {"written": 0, "rejected": 0}
    assert len(web.requests) == 1
    assert run("--pdx1-min-priority", "MONITOR")["pdx1_importer"] == {"written": 0, "rejected": 1}
