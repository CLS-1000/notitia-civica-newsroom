"""Tests for spec1_bridge/proc_track_importer.py and its mycroft_triage wiring.

The proc_track checkout is a fixture: tests/fixtures/proc_track/replay_harness.py
has the real module's interface with a small rule table, and each test writes a
session cache. Dates are relative to now so leads clear the freshness gate.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import shutil
import stat
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest

from spec1_bridge import mycroft_triage
from spec1_bridge.adapters.base_lazlong import excerpt_is_verbatim, read_artifact_text
from spec1_bridge.mycroft_triage import Enrichment, OllamaEnricher, triage_packets
from spec1_bridge.proc_track_importer import (
    ADAPTER_NAME,
    ProcTrackError,
    history_url,
    import_proc_track,
    load_replay_harness,
    parse_action_date,
)

FIXTURE_HARNESS = (
    Path(__file__).parent / "fixtures" / "proc_track" / "replay_harness.py"
)
SESSION = "2026R1"


def _stamp(days_ago: float, hour: int = 10) -> str:
    moment = datetime.now(UTC) - timedelta(days=days_ago)
    return moment.replace(hour=hour, minute=0, second=0, microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


def _rows(
    prefix: str,
    number: int,
    actions: list[str],
    *,
    start_id: int,
    session: str = SESSION,
) -> list[dict[str, Any]]:
    return [
        {
            "MeasureHistoryId": start_id + index,
            "SessionKey": session,
            "MeasurePrefix": prefix,
            "MeasureNumber": number,
            "Chamber": "S",
            "ActionDate": _stamp(30 - index, hour=9 + index),
            "ActionText": text,
            "VoteText": None,
        }
        for index, text in enumerate(actions)
    ]


def _session_rows() -> list[dict[str, Any]]:
    intro, passed = "Introduction and first reading.", "Third reading. Passed."
    return [
        *_rows("SB", 1, [intro, passed, "Governor vetoed."], start_id=100),
        *_rows(
            "SB",
            2,
            [intro, passed, "Governor vetoed.", "Veto overridden by two-thirds vote."],
            start_id=200,
        ),
        *_rows(
            "HB",
            3,
            [intro, passed, "Governor signed with line-item veto."],
            start_id=300,
        ),
        *_rows(
            "HB",
            4,
            [intro, passed, "Filed without Governor's signature."],
            start_id=400,
        ),
        *_rows("HB", 5, [intro, passed, "Governor signed."], start_id=500),
        *_rows(
            "HB", 6, [intro, "Rule that proc_track has no mapping for."], start_id=600
        ),
        *_rows(
            "SB", 7, [intro, passed, "Governor vetoed."], start_id=700, session="2025R1"
        ),
    ]


@pytest.fixture
def proc_track(tmp_path: Path) -> Path:
    root = tmp_path / "proc_track"
    (root / "cache").mkdir(parents=True)
    shutil.copy(FIXTURE_HARNESS, root / "replay_harness.py")
    (root / "cache" / f"{SESSION}.json").write_text(
        json.dumps(_session_rows()), encoding="utf-8"
    )
    return root


def _allocator() -> Callable[[], str]:
    counter = itertools.count(1)
    return lambda: f"S-{next(counter)}"


def _by_measure(packets: list[Any]) -> dict[str, Any]:
    return {p.source_artifacts[0].metadata["measure"]: p for p in packets}


def test_missing_checkout_and_cache_raise(tmp_path: Path) -> None:
    with pytest.raises(ProcTrackError, match="replay_harness.py"):
        load_replay_harness(tmp_path)
    shutil.copy(FIXTURE_HARNESS, tmp_path / "replay_harness.py")
    with pytest.raises(ProcTrackError, match="fetch_session.py 2026R1"):
        import_proc_track(tmp_path, SESSION, _allocator(), archive_dir=tmp_path / "a")


def test_notable_outcomes_become_leads(proc_track: Path, tmp_path: Path) -> None:
    result = import_proc_track(
        proc_track, SESSION, _allocator(), archive_dir=tmp_path / "a"
    )
    leads = _by_measure(result.packets)

    assert set(leads) == {"SB1", "SB2", "HB3", "HB4"}
    assert {m: p.source_artifacts[0].metadata["outcome"] for m, p in leads.items()} == {
        "SB1": "veto",
        "SB2": "veto_override",
        "HB3": "line_item_veto",
        "HB4": "enacted_without_signature",
    }
    assert leads["SB2"].anomaly_score == 85.0
    assert leads["SB2"].verbatim_excerpts == {
        leads["SB2"]
        .source_artifacts[0]
        .source_id: "Veto overridden by two-thirds vote."
    }
    assert result.measures_replayed == 6
    assert set(result.halts) == {"HB6"}
    for packet in result.packets:
        assert packet.adapter_name == ADAPTER_NAME
        assert packet.analysis_tier == "rule_based"
        assert packet.evidence_bar_passed is True
        assert packet.source_artifacts[0].tier == "tier_1_primary"


def test_halted_and_other_session_measures_are_not_leads(
    proc_track: Path, tmp_path: Path
) -> None:
    result = import_proc_track(
        proc_track, SESSION, _allocator(), archive_dir=tmp_path / "a"
    )
    measures = _by_measure(result.packets)
    assert "HB5" not in measures
    assert "HB6" not in measures
    assert "SB7" not in measures
    assert "NO RULE MATCHED" in result.halts["HB6"]


def test_sealed_blob_matches_hash_and_holds_the_excerpt(
    proc_track: Path, tmp_path: Path
) -> None:
    result = import_proc_track(
        proc_track, SESSION, _allocator(), archive_dir=tmp_path / "a"
    )
    for packet in result.packets:
        artifact = packet.source_artifacts[0]
        blob = Path(artifact.raw_payload_path)
        raw = blob.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == artifact.sha256_hash
        assert not blob.stat().st_mode & stat.S_IWUSR
        text = read_artifact_text(raw, artifact.metadata["content_type"])
        assert text is not None
        for excerpt in packet.verbatim_excerpts.values():
            assert excerpt_is_verbatim(excerpt, text)
        rows = json.loads(raw)
        assert {row["MeasureHistoryId"] for row in rows} <= set(range(100, 500))


def test_reimport_is_deterministic_and_skips_known_leads(
    proc_track: Path, tmp_path: Path
) -> None:
    first = import_proc_track(
        proc_track, SESSION, _allocator(), archive_dir=tmp_path / "a"
    )
    second = import_proc_track(
        proc_track, SESSION, _allocator(), archive_dir=tmp_path / "b"
    )
    assert [p.lead_id for p in first.packets] == [p.lead_id for p in second.packets]
    assert [p.source_artifacts[0].sha256_hash for p in first.packets] == [
        p.source_artifacts[0].sha256_hash for p in second.packets
    ]

    known = {p.lead_id for p in first.packets}
    calls: list[str] = []

    def allocate() -> str:
        calls.append("x")
        return "S-1"

    third = import_proc_track(
        proc_track,
        SESSION,
        allocate,
        is_known=known.__contains__,
        archive_dir=tmp_path / "c",
    )
    assert third.packets == []
    assert third.skipped_known == 4
    assert calls == []
    assert not (tmp_path / "c").exists()


def test_measure_filter(proc_track: Path, tmp_path: Path) -> None:
    result = import_proc_track(
        proc_track, SESSION, _allocator(), archive_dir=tmp_path / "a", measures={"SB1"}
    )
    assert list(_by_measure(result.packets)) == ["SB1"]
    assert result.measures_replayed == 1


def test_action_date_is_salem_local_time() -> None:
    assert parse_action_date("2025-06-20T16:11:22") == datetime(
        2025, 6, 20, 23, 11, 22, tzinfo=UTC
    )
    assert parse_action_date("2025-01-13T08:01:32") == datetime(
        2025, 1, 13, 16, 1, 32, tzinfo=UTC
    )


def test_history_url_targets_one_measure() -> None:
    url = history_url("2025R1", "SB", 976)
    assert url.startswith(
        "https://api.oregonlegislature.gov/odata/ODataService.svc/MeasureHistoryActions?"
    )
    assert "SessionKey%20eq%20%272025R1%27" in url
    assert "MeasureNumber%20eq%20976" in url


class _OfflineEnricher(OllamaEnricher):
    def enrich(self, packet: Any) -> Enrichment:
        return Enrichment("unavailable", "test", error="offline")


def test_leads_clear_triage_and_land_in_the_ledger(
    proc_track: Path, tmp_path: Path, db: psycopg.Connection[tuple[Any, ...]]
) -> None:
    result = import_proc_track(
        proc_track,
        SESSION,
        lambda: mycroft_triage.allocate_source_id(db),
        archive_dir=tmp_path / "a",
    )
    report = triage_packets(result.packets, "oregon_civic", db, _OfflineEnricher())
    assert len(report.written) == 4
    assert report.rejected == []

    rows = db.execute(
        "SELECT adapter_name, analysis_tier::text, source_ids FROM triage_records"
    ).fetchall()
    assert {row[0] for row in rows} == {ADAPTER_NAME}
    assert {row[1] for row in rows} == {"rule_based"}
    registered = db.execute(
        "SELECT count(*) FROM source_registry WHERE domain = 'api.oregonlegislature.gov'"
    ).fetchone()
    assert registered is not None and registered[0] == 4


def _run_cli(argv: list[str]) -> tuple[int, dict[str, Any]]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = mycroft_triage.main(argv)
    out = buffer.getvalue().strip()
    return code, json.loads(out) if out else {}


def test_cli_proc_track_only_writes_then_dedupes(
    proc_track: Path,
    tmp_path: Path,
    ledger_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mycroft_triage, "OllamaEnricher", _OfflineEnricher)
    argv = [
        "--dsn",
        ledger_dsn,
        "--proc-track-path",
        str(proc_track),
        "--proc-track-session",
        SESSION,
        "--proc-track-only",
    ]

    code, summary = _run_cli(argv)
    assert code == 0
    assert summary == {"proc_track_importer": {"written": 4, "rejected": 0}}
    assert len(list((tmp_path / "archive" / "warc").glob("*.blob"))) == 4

    code, summary = _run_cli(argv)
    assert code == 0
    assert summary == {"proc_track_importer": {"written": 0, "rejected": 0}}
    assert len(list((tmp_path / "archive" / "warc").glob("*.blob"))) == 4


def test_cli_session_falls_back_to_session_key(
    proc_track: Path, tmp_path: Path, ledger_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mycroft_triage, "OllamaEnricher", _OfflineEnricher)
    code, summary = _run_cli(
        [
            "--dsn",
            ledger_dsn,
            "--session-key",
            SESSION,
            "--proc-track-path",
            str(proc_track),
            "--proc-track-only",
        ]
    )
    assert code == 0
    assert summary["proc_track_importer"]["written"] == 4


def test_cli_rejects_bad_proc_track_arguments(
    tmp_path: Path, ledger_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OLIS_SESSION_KEY", raising=False)
    monkeypatch.delenv("PROC_TRACK_SESSION", raising=False)
    with pytest.raises(SystemExit) as no_session:
        mycroft_triage.main(["--dsn", ledger_dsn, "--proc-track-path", str(tmp_path)])
    assert no_session.value.code == 2
    with pytest.raises(SystemExit) as no_path:
        mycroft_triage.main(["--dsn", ledger_dsn, "--proc-track-only"])
    assert no_path.value.code == 2

    code, _summary = _run_cli(
        [
            "--dsn",
            ledger_dsn,
            "--proc-track-path",
            str(tmp_path / "missing"),
            "--proc-track-session",
            SESSION,
            "--proc-track-only",
        ]
    )
    assert code == 1
