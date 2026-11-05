"""Tests for spec1_bridge/spec1_importer.py.

Fixture records copy the shape of a real llm_verified line from spec-1's
`data` branch (data/spec1_intelligence.jsonl): signal_url, outcome_classification,
confidence, analysis_tier, written_at, and signal_excerpt that may be null.
Sealing runs for real through custody, against an httpx.MockTransport.
"""

from __future__ import annotations

import hashlib
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

from spec1_bridge import mycroft_triage, spec1_importer
from spec1_bridge.adapters.base_lazlong import excerpt_is_verbatim, read_artifact_text
from spec1_bridge.mycroft_triage import (
    Enrichment,
    OllamaEnricher,
    TriageReport,
    triage_packets,
)
from spec1_bridge.spec1_importer import (
    Spec1RecordError,
    import_spec1_records,
    lead_id_for,
    load_verdicts,
    parse_record,
)

GOV_URL = "https://www.congress.gov/bill/119th-congress/house-bill/1"
GOV_PAGE = (
    "<html><body><h1>H.R.1</h1>"
    "<p>The committee reported the bill with an amendment on October 20.</p></body></html>"
)
NEWS_URL = "https://warontherocks.com/cogs-of-war/the-pentagons-ai-edge"
NEWS_PAGE = (
    "<html><body><p>The Pentagon's AI edge is being distilled away.</p></body></html>"
)


def _record(**overrides: Any) -> dict[str, Any]:
    written = (datetime.now(UTC) - timedelta(days=3)).isoformat()
    record: dict[str, Any] = {
        "id": "7ab89c7995b341eb3d03883a2784cbc5",
        "run_id": "run-a66c1983582f",
        "created_at": "2026-10-20 02:58:11",
        "source_type": "pipeline",
        "schema_version": 1,
        "record_id": "rec-af668b065ac3",
        "signal_id": "38ca3f1b6589672e",
        "signal_source": "congress_gov",
        "signal_url": GOV_URL,
        "opportunity_id": "opp-3fd0ef54b86e",
        "opportunity_score": 0.71,
        "opportunity_priority": "STANDARD",
        "gate_results": '{"credibility": true, "volume": true, "velocity": true, "novelty": true}',
        "investigation_id": "inv-18f3ec740f1e",
        "hypothesis": "The committee advanced H.R.1 with an amendment.",
        "outcome_classification": "INVESTIGATE",
        "outcome_confidence": 0.8,
        "pattern": "[STANDARD] H.R.1 reported",
        "classification": "INVESTIGATE",
        "confidence": 0.8165,
        "source_weight": 0.85,
        "analyst_weight": 0.8933,
        "environment": "production",
        "written_at": written,
        "signal_excerpt": "The committee reported the bill with an amendment on October 20.",
        "llm_reasoning": "Committee action on a filed bill.",
        "analysis_tier": "llm_verified",
    }
    record.update(overrides)
    return record


class Web:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self.pages = {
            GOV_URL: (200, "text/html; charset=utf-8", GOV_PAGE.encode()),
            NEWS_URL: (200, "text/html; charset=utf-8", NEWS_PAGE.encode()),
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        status, content_type, body = self.pages.get(
            url, (404, "text/plain", b"not found")
        )
        return httpx.Response(
            status, headers={"content-type": content_type}, content=body
        )

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _write(path: Path, rows: list[dict[str, Any] | str]) -> Path:
    path.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _allocator() -> Callable[[], str]:
    counter = itertools.count(1)
    return lambda: f"S-{next(counter)}"


def _by_record(packets: list[Any]) -> dict[str, Any]:
    return {p.source_artifacts[0].metadata["spec1_record_id"]: p for p in packets}


def test_parse_record_accepts_real_shape_with_null_excerpt() -> None:
    record = parse_record(json.dumps(_record(signal_excerpt=None)))
    assert record.signal_excerpt == ""
    assert record.classification == "INVESTIGATE"
    assert record.written_at.tzinfo is not None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"record_id": None}, "record_id"),
        ({"confidence": "high"}, "confidence"),
        ({"confidence": 1.5}, "outside 0..1"),
        ({"written_at": "soon"}, "written_at"),
    ],
)
def test_parse_record_rejects_bad_lines(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(Spec1RecordError, match=message):
        parse_record(json.dumps(_record(**overrides)))


def test_lead_id_is_deterministic_and_not_a_security_hash() -> None:
    record = parse_record(json.dumps(_record()))
    expected = uuid.UUID(
        bytes=hashlib.sha256(b"spec1:rec-af668b065ac3", usedforsecurity=False).digest()[
            :16
        ],
        version=5,
    )
    assert lead_id_for(record) == str(expected)
    assert lead_id_for(parse_record(json.dumps(_record()))) == lead_id_for(record)


def test_lead_id_hash_is_marked_usedforsecurity_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    real = hashlib.sha256

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(spec1_importer.hashlib, "sha256", spy)
    lead_id_for(parse_record(json.dumps(_record())))
    assert calls == [{"usedforsecurity": False}]


def test_load_verdicts_keeps_the_latest(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "verdicts.jsonl",
        [
            {
                "record_id": "rec-1",
                "verdict": "correct",
                "reviewer": "a",
                "reviewed_at": "2026-10-01T00:00:00+00:00",
            },
            {
                "record_id": "rec-1",
                "verdict": "incorrect",
                "reviewer": "b",
                "reviewed_at": "2026-10-02T00:00:00+00:00",
            },
            {"record_id": "rec-2", "verdict": "maybe"},
            "{bad",
        ],
    )
    verdicts, errors = load_verdicts(path)
    assert verdicts["rec-1"].verdict == "incorrect"
    assert "rec-2" not in verdicts
    assert len(errors) == 2


def test_import_end_to_end_offline(tmp_path: Path) -> None:
    web = Web()
    store = _write(
        tmp_path / "spec1_intelligence.jsonl",
        [
            _record(record_id="rec-gov"),
            _record(record_id="rec-news", signal_url=NEWS_URL, signal_excerpt=None),
            _record(record_id="rec-rule", analysis_tier="rule_based"),
            _record(record_id="rec-old", analysis_tier=None),
            _record(record_id="rec-wrong"),
            _record(record_id="rec-nourl", signal_url=None),
            _record(record_id="rec-gov"),
            "{not json",
        ],
    )
    verdicts = _write(
        tmp_path / "verdicts.jsonl",
        [
            {
                "record_id": "rec-wrong",
                "verdict": "incorrect",
                "reviewed_at": "2026-10-30T00:00:00+00:00",
            },
            {
                "record_id": "rec-gov",
                "verdict": "partial",
                "reviewer": "matt",
                "reviewed_at": "2026-10-30T00:00:00+00:00",
            },
        ],
    )
    result = import_spec1_records(
        store,
        _allocator(),
        verdicts_path=verdicts,
        archive_dir=tmp_path / "a",
        transport=web.transport,
    )
    leads = _by_record(result.packets)

    assert set(leads) == {"rec-gov", "rec-news"}
    assert result.skipped == {
        "not_llm_verified": 2,
        "verdict_incorrect": 1,
        "no_url": 1,
        "known": 1,
        "malformed": 1,
    }
    gov = leads["rec-gov"]
    artifact = gov.source_artifacts[0]
    assert gov.anomaly_score == pytest.approx(81.65)
    assert gov.analysis_tier == "rule_based"
    assert artifact.source_id[2:].isdigit()
    assert artifact.metadata["human_verdict"] == "partial"
    assert gov.verbatim_excerpts == {
        artifact.source_id: "The committee reported the bill with an amendment on October 20."
    }
    assert "partial by matt" in gov.paste_ready_prompt
    assert leads["rec-news"].source_artifacts[0].tier == "tier_3_secondary"
    for packet in result.packets:
        sealed = packet.source_artifacts[0]
        text = read_artifact_text(
            Path(sealed.raw_payload_path).read_bytes(), sealed.metadata["content_type"]
        )
        assert text is not None
        for excerpt in packet.verbatim_excerpts.values():
            assert excerpt and excerpt_is_verbatim(excerpt, text)


def test_offline_fixture_is_deterministic(tmp_path: Path) -> None:
    store = _write(
        tmp_path / "s.jsonl",
        [
            _record(record_id="rec-gov"),
            _record(record_id="rec-news", signal_url=NEWS_URL),
        ],
    )
    runs = [
        import_spec1_records(
            store, _allocator(), archive_dir=tmp_path / name, transport=Web().transport
        )
        for name in ("a", "b")
    ]
    for first, second in zip(runs[0].packets, runs[1].packets, strict=True):
        assert first.lead_id == second.lead_id
        assert first.anomaly_score == second.anomaly_score
        assert first.verbatim_excerpts == second.verbatim_excerpts
        assert first.paste_ready_prompt == second.paste_ready_prompt
        assert (
            first.source_artifacts[0].sha256_hash
            == second.source_artifacts[0].sha256_hash
        )


def test_known_leads_are_not_fetched_and_failures_are_reported(tmp_path: Path) -> None:
    store = _write(
        tmp_path / "s.jsonl",
        [
            _record(record_id="rec-gov"),
            _record(record_id="rec-404", signal_url="https://www.congress.gov/missing"),
        ],
    )
    web = Web()
    known = {lead_id_for(parse_record(json.dumps(_record(record_id="rec-gov"))))}
    result = import_spec1_records(
        store,
        _allocator(),
        is_known=known.__contains__,
        archive_dir=tmp_path / "a",
        transport=web.transport,
    )
    assert result.packets == []
    assert result.skipped == {"known": 1, "seal_failed": 1}
    assert web.requests == ["https://www.congress.gov/missing"]
    assert any("rec-404" in e for e in result.errors)


def test_missing_store_is_reported(tmp_path: Path) -> None:
    result = import_spec1_records(
        tmp_path / "none.jsonl", _allocator(), verdicts_path=tmp_path / "nov.jsonl"
    )
    assert result.packets == []
    assert any("verdicts file not found" in e for e in result.errors)
    assert any("store not found" in e for e in result.errors)


class _OfflineEnricher(OllamaEnricher):
    def enrich(self, packet: Any) -> Enrichment:
        return Enrichment("unavailable", "test", error="offline")


def test_primary_source_clears_triage_and_news_alone_does_not(
    tmp_path: Path, db: psycopg.Connection[tuple[Any, ...]]
) -> None:
    store = _write(
        tmp_path / "s.jsonl",
        [
            _record(record_id="rec-gov"),
            _record(record_id="rec-news", signal_url=NEWS_URL),
        ],
    )
    result = import_spec1_records(
        store,
        lambda: mycroft_triage.allocate_source_id(db),
        archive_dir=tmp_path / "a",
        transport=Web().transport,
    )
    report = triage_packets(result.packets, "spec1_osint", db, _OfflineEnricher())
    assert len(report.written) == 1
    assert [(r.gate, r.reason) for r in report.rejected] == [
        ("authority", "no tier 1 or tier 2 source")
    ]
    row = db.execute(
        "SELECT adapter_name, subject_domain, analysis_tier::text FROM triage_records"
    ).fetchone()
    assert row == ("spec1_importer", "spec1_osint", "rule_based")


def test_cli_spec1_jsonl_writes_then_dedupes(
    tmp_path: Path, ledger_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mycroft_triage, "OllamaEnricher", _OfflineEnricher)
    monkeypatch.setattr(mycroft_triage, "run_adapter", lambda *_args: TriageReport())
    web = Web()
    real_seal = spec1_importer.fetch_and_seal_source

    def offline_seal(*args: Any, **kwargs: Any) -> Any:
        return real_seal(*args, **{**kwargs, "transport": web.transport})

    monkeypatch.setattr(spec1_importer, "fetch_and_seal_source", offline_seal)
    store = _write(tmp_path / "s.jsonl", [_record(record_id="rec-gov")])

    def run() -> dict[str, Any]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            assert (
                mycroft_triage.main(["--dsn", ledger_dsn, "--spec1-jsonl", str(store)])
                == 0
            )
        return json.loads(buffer.getvalue())

    assert run()["spec1_importer"] == {"written": 1, "rejected": 0}
    assert run()["spec1_importer"] == {"written": 0, "rejected": 0}
    assert len(web.requests) == 1


def test_source_tiers_cover_federal_and_civic_records() -> None:
    from spec1_bridge.spec1_importer import classify_source_tier

    assert classify_source_tier(GOV_URL) == "tier_1_primary"
    assert classify_source_tier("https://efile.fara.gov/docs/1.pdf") == "tier_1_primary"
    assert (
        classify_source_tier("https://secure.sos.state.or.us/orestar/x")
        == "tier_1_primary"
    )
    assert classify_source_tier(NEWS_URL) == "tier_3_secondary"
    assert (
        classify_source_tier("https://congress.gov.example.net/x") == "tier_3_secondary"
    )
