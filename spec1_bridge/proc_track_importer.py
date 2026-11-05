"""Importer for CLS-1000/proc_track: Oregon Legislature procedural outcomes.

proc_track replays each measure's official OLIS action history
(MeasureHistoryActions) through its rule table and state machine. This importer
reads a proc_track checkout offline: the session cache (cache/<session>.json)
and replay_harness.py. Nothing is fetched from the network.

A measure becomes a lead only when its replay completes and records a notable
procedural outcome: a veto override, a veto, a line-item veto, or a law that
took effect without the Governor's signature. A replay that halts is a rule gap
in proc_track, not news, so it is counted and logged but never emitted.

The evidence is the measure's own OLIS rows, exactly as proc_track cached them,
sealed as one JSON blob. The lead's excerpt is the verbatim ActionText of the
row that produced the outcome, so the corroboration gate can find it in the
sealed payload. Every packet is analysis_tier='rule_based'.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Final
from urllib.parse import quote
from zoneinfo import ZoneInfo

from newsroom.harlan_eternity.custody import (
    DEFAULT_ARCHIVE_DIR,
    validate_source_id,
)
from spec1_bridge.adapters.base_lazlong import render_prompt
from spec1_bridge.contracts import SourceArtifact, Spec1LeadPacket

logger = logging.getLogger(__name__)

ADAPTER_NAME: Final = "proc_track_importer"
SUBJECT_DOMAIN: Final = "oregon_civic"
SOURCE_TIER: Final = "tier_1_primary"
ARTIFACT_KIND: Final = "olis_measure_history"
CONTENT_TYPE: Final = "application/json; charset=utf-8"
ODATA_HISTORY_URL: Final = (
    "https://api.oregonlegislature.gov/odata/ODataService.svc/MeasureHistoryActions"
)
# OLIS ActionDate values carry no offset; they are Salem local time.
OLIS_TIMEZONE: Final = ZoneInfo("America/Los_Angeles")

PROC_TRACK_NAMESPACE: Final = uuid.UUID("5d2c8e41-9b7a-5f3e-8c16-0a4b7e29d351")


@dataclass(frozen=True, slots=True)
class Outcome:
    kind: str
    label: str
    score: float


# Keyed by proc_track rule_id (replay_harness.RULES). When a measure has several,
# the highest score is the lead.
NOTABLE_RULES: Final[dict[str, Outcome]] = {
    "veto_overridden": Outcome(
        "veto_override", "Legislature overrode the Governor's veto", 85.0
    ),
    "repass_notwithstanding": Outcome(
        "veto_override",
        "Legislature repassed the measure over the Governor's veto",
        85.0,
    ),
    "governor_vetoed": Outcome("veto", "Vetoed by the Governor", 70.0),
    "line_item_veto": Outcome(
        "veto",
        "Governor purported to sign with line-item veto (recorded as a veto)",
        70.0,
    ),
    "governor_signed_line_item": Outcome(
        "line_item_veto",
        "Signed into law with part of it vetoed (line-item veto)",
        60.0,
    ),
    "filed_without_signature": Outcome(
        "enacted_without_signature", "Became law without the Governor's signature", 55.0
    ),
    "filed_sos_without_signature": Outcome(
        "enacted_without_signature",
        "Filed with the Secretary of State without the Governor's signature",
        55.0,
    ),
}


class ProcTrackError(RuntimeError):
    """The proc_track checkout or its session cache cannot be used."""


@dataclass(slots=True)
class ProcTrackImport:
    """Outcome of one import. halts are measures proc_track could not replay."""

    packets: list[Spec1LeadPacket] = field(default_factory=list)
    halts: dict[str, str] = field(default_factory=dict)
    skipped_known: int = 0
    measures_replayed: int = 0


def load_replay_harness(proc_track_path: Path | str) -> ModuleType:
    """Load replay_harness.py from a proc_track checkout without touching sys.path."""
    harness_file = Path(proc_track_path) / "replay_harness.py"
    if not harness_file.is_file():
        raise ProcTrackError(f"no replay_harness.py in {proc_track_path}")
    spec = importlib.util.spec_from_file_location(
        "_proc_track_replay_harness", harness_file
    )
    if spec is None or spec.loader is None:
        raise ProcTrackError(f"cannot load {harness_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("replay", "parse_row", "apply_corrections"):
        if not callable(getattr(module, name, None)):
            raise ProcTrackError(f"{harness_file} has no {name}()")
    return module


def proc_track_commit(proc_track_path: Path | str) -> str | None:
    """Return the checkout's HEAD commit, read from .git, or None."""
    git_dir = Path(proc_track_path) / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head or None
        ref = head.removeprefix("ref: ")
        ref_file = git_dir / ref
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip() or None
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                sha, _, name = line.partition(" ")
                if name == ref:
                    return sha
    except OSError:
        return None
    return None


def load_session_rows(
    proc_track_path: Path | str, session: str
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Group a session cache by measure, in cache order. Returns (rows, cache sha256)."""
    cache_file = Path(proc_track_path) / "cache" / f"{session}.json"
    try:
        raw = cache_file.read_bytes()
    except OSError as error:
        raise ProcTrackError(
            f"no session cache {cache_file}; run scripts/fetch_session.py {session} "
            "in the proc_track checkout"
        ) from error
    try:
        rows = json.loads(raw)
    except ValueError as error:
        raise ProcTrackError(f"{cache_file} is not valid JSON: {error}") from error
    if not isinstance(rows, list):
        raise ProcTrackError(f"{cache_file} is not a list of history rows")
    by_measure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or row.get("SessionKey") != session:
            continue
        by_measure[
            f"{row.get('MeasurePrefix', '')}{row.get('MeasureNumber', '')}"
        ].append(row)
    return dict(by_measure), hashlib.sha256(raw).hexdigest()


def history_url(session: str, prefix: str, number: Any) -> str:
    """The OLIS OData query that returns exactly this measure's history rows."""
    query = (
        f"SessionKey eq '{session}' and MeasurePrefix eq '{prefix}' "
        f"and MeasureNumber eq {number}"
    )
    return f"{ODATA_HISTORY_URL}?$filter={quote(query, safe='')}"


def parse_action_date(value: str) -> datetime:
    """OLIS ActionDate (Salem local, no offset) as an aware UTC datetime."""
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=OLIS_TIMEZONE)
    return moment.astimezone(UTC)


def find_outcome(
    harness: ModuleType, rows: list[dict[str, Any]], history: list[Any]
) -> tuple[Outcome, dict[str, Any]] | None:
    """Return the most notable outcome the replay recorded and the row behind it.

    A rule counts only if the completed replay applied it on that row's date, so
    a row the state machine treated as a no-op cannot become a lead.
    """
    applied = {(entry[0], entry[4]) for entry in history if len(entry) >= 5}
    best: tuple[Outcome, dict[str, Any]] | None = None
    for row in rows:
        emits, _gaps = harness.parse_row(row.get("ActionText") or "")
        for emit in emits:
            outcome = NOTABLE_RULES.get(emit.rule_id)
            if outcome is None or (row.get("ActionDate"), emit.rule_id) not in applied:
                continue
            if best is None or outcome.score > best[0].score:
                best = (outcome, row)
    return best


def seal_rows(
    rows: list[dict[str, Any]],
    source_id: str,
    *,
    archive_dir: Path,
    captured_at: datetime,
) -> tuple[Path, str]:
    """Write the rows as one read-only JSON blob, named like custody's blobs.

    The bytes are deterministic for the same rows, so the hash is reproducible.
    """
    validate_source_id(source_id)
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = captured_at.strftime("%Y%m%dT%H%M%S%fZ")
    final_path = archive_dir / f"{source_id}_{stamp}_{digest[:12]}.blob"
    descriptor, temp_name = tempfile.mkstemp(dir=archive_dir, suffix=".partial")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.chmod(0o444)
        temp_path.replace(final_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return final_path, digest


def lead_id_for(session: str, measure: str, kind: str, row: dict[str, Any]) -> str:
    """Stable per outcome row: re-importing the same cache yields the same lead."""
    key = f"{session}:{measure}:{kind}:{row.get('MeasureHistoryId')}:{row.get('ActionDate')}"
    return str(uuid.uuid5(PROC_TRACK_NAMESPACE, key))


def build_prompt(
    session: str,
    measure: str,
    outcome: Outcome,
    row: dict[str, Any],
    states: dict[str, Any],
    artifact: SourceArtifact,
) -> str:
    facts = {
        "Session": session,
        "Measure": measure,
        "Outcome": outcome.label,
        "Decisive OLIS action": f"{row.get('ActionDate')} ({row.get('Chamber')}): "
        f"{row.get('ActionText')}",
        "Final chamber states (proc_track replay)": json.dumps(states, sort_keys=True),
    }
    return render_prompt(
        f"PROCEDURAL OUTCOME: {measure} ({session})",
        facts,
        [artifact],
        (
            "Draft a nonpartisan, factual account of this measure's procedural outcome "
            "from the sealed OLIS history. State the date and chamber of each step you "
            "rely on."
        ),
    )


def import_proc_track(
    proc_track_path: Path | str,
    session: str,
    allocate_source_id: Callable[[], str],
    *,
    is_known: Callable[[str], bool] = lambda _lead_id: False,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    measures: set[str] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ProcTrackImport:
    """Replay a session from a proc_track checkout and emit leads for notable outcomes.

    allocate_source_id is called only for leads that are new, so re-running over
    an unchanged cache allocates no IDs and seals nothing.
    """
    harness = load_replay_harness(proc_track_path)
    by_measure, cache_sha256 = load_session_rows(proc_track_path, session)
    commit = proc_track_commit(proc_track_path)
    result = ProcTrackImport()

    for measure in sorted(by_measure):
        if measures is not None and measure not in measures:
            continue
        cached = by_measure[measure]
        # proc_track rule: ActionDate is the only defensible sort key.
        ordered = sorted(
            harness.apply_corrections(cached),
            key=lambda r: (r.get("ActionDate") or "", r.get("MeasureHistoryId") or 0),
        )
        item, halt, _gaps = harness.replay(measure, ordered)
        result.measures_replayed += 1
        if halt:
            result.halts[measure] = halt
            logger.info("proc_track halted on %s %s: %s", session, measure, halt)
            continue

        found = find_outcome(harness, ordered, list(item.history))
        if found is None:
            continue
        outcome, row = found
        lead_id = lead_id_for(session, measure, outcome.kind, row)
        if is_known(lead_id):
            result.skipped_known += 1
            continue

        excerpt = row.get("ActionText") or ""
        captured_at = clock()
        source_id = allocate_source_id()
        evidence_rows = sorted(
            cached,
            key=lambda r: (r.get("ActionDate") or "", r.get("MeasureHistoryId") or 0),
        )
        blob_path, digest = seal_rows(
            evidence_rows, source_id, archive_dir=archive_dir, captured_at=captured_at
        )
        states = dict(item.states)
        artifact = SourceArtifact(
            source_id=source_id,
            tier=SOURCE_TIER,
            origin_url=history_url(
                session,
                str(cached[0].get("MeasurePrefix", "")),
                cached[0].get("MeasureNumber"),
            ),
            sha256_hash=digest,
            raw_payload_path=str(blob_path),
            captured_at=captured_at,
            metadata={
                "kind": ARTIFACT_KIND,
                "content_type": CONTENT_TYPE,
                "engine": "proc_track",
                "session": session,
                "measure": measure,
                "outcome": outcome.kind,
                "final_states": states,
                "proc_track_commit": commit,
                "cache_sha256": cache_sha256,
                "data_corrections": item.payload.get("data_corrections", []),
            },
        )
        result.packets.append(
            Spec1LeadPacket(
                lead_id=lead_id,
                timestamp=parse_action_date(row["ActionDate"]),
                adapter_name=ADAPTER_NAME,
                anomaly_score=outcome.score,
                evidence_bar_passed=bool(excerpt),
                source_artifacts=[artifact],
                verbatim_excerpts={source_id: excerpt} if excerpt else {},
                paste_ready_prompt=build_prompt(
                    session, measure, outcome, row, states, artifact
                ),
                analysis_tier="rule_based",
            )
        )

    logger.info(
        "proc_track %s: %d replayed, %d halted, %d leads, %d already known",
        session,
        result.measures_replayed,
        len(result.halts),
        len(result.packets),
        result.skipped_known,
    )
    return result
