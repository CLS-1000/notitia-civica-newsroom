"""Fleet Control: the subject engines, anomaly detectors, and importers.

Each engine is listed once in ENGINES. An engine the desk can run on its own has
a runner; one that needs editor-supplied inputs (survey areas, movements, witness
reports) or has no harvest entry point yet says so instead, and its button stays
disabled rather than pretending to run.

A run happens on a background thread so the desk stays responsive, and its
outcome is appended to a JSON Lines run log. The log is the only record of
polls: the ledger records leads, not attempts, so last-poll time, pass rate and
ERROR status all come from it.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import psycopg
from psycopg.rows import dict_row

Ledger = Literal["triage", "telemetry"]
Group = Literal["engine", "importer", "adapter"]


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """What one harvest did: leads written, leads rejected by triage, errors."""

    written: int
    rejected: int
    errors: tuple[str, ...] = ()


Runner = Callable[[psycopg.Connection[Any]], RunOutcome]
RunnerFactory = Callable[[Mapping[str, str]], Runner]


@dataclass(frozen=True, slots=True)
class Engine:
    key: str
    label: str
    group: Group
    family: str
    ledger: Ledger = "triage"
    number: int | None = None
    requires: tuple[str, ...] = ()
    factory: RunnerFactory | None = None
    note: str = ""

    def missing(self, env: Mapping[str, str]) -> list[str]:
        return [name for name in self.requires if not env.get(name)]

    def blocker(self, env: Mapping[str, str]) -> str | None:
        """Why this engine cannot run from the desk now, or None if it can."""
        if self.factory is None:
            return self.note or "No harvest entry point yet."
        missing = self.missing(env)
        if missing:
            return "Set " + ", ".join(missing) + " in the desk's environment."
        return None


# --- Runners ------------------------------------------------------------------
# Imports are deferred so the desk starts without loading every adapter's
# scientific stack (skyfield, scipy, pyswisseph).


def _report(report: Any) -> RunOutcome:
    return RunOutcome(len(report.written), len(report.rejected))


def _enricher() -> Any:
    from spec1_bridge.mycroft_triage import DEFAULT_OLLAMA_MODEL, OllamaEnricher

    return OllamaEnricher(model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))


def _adapter_runner(build: Callable[[Mapping[str, str]], Any]) -> RunnerFactory:
    def factory(env: Mapping[str, str]) -> Runner:
        def run(conn: psycopg.Connection[Any]) -> RunOutcome:
            from spec1_bridge.mycroft_triage import run_adapter

            return _report(run_adapter(build(env), conn, _enricher()))

        return run

    return factory


def _ubik(env: Mapping[str, str]) -> Any:
    from spec1_bridge.adapters.ubik_oregon import UbikOregonAdapter

    return UbikOregonAdapter(env["OLIS_SESSION_KEY"])


def _seldon(_env: Mapping[str, str]) -> Any:
    from spec1_bridge.adapters.seldon_civic import SeldonCivicAdapter

    return SeldonCivicAdapter()


def _delos(_env: Mapping[str, str]) -> Any:
    from spec1_bridge.adapters.delos_econ import DelosEconAdapter

    return DelosEconAdapter()


def _klein(_env: Mapping[str, str]) -> Any:
    from spec1_bridge.adapters.klein_biomed import KleinBiomedAdapter

    return KleinBiomedAdapter()


def _mazer(_env: Mapping[str, str]) -> Any:
    from spec1_bridge.adapters.mazer_defense import MazerDefenseAdapter

    return MazerDefenseAdapter()


def _ubik_spectral(env: Mapping[str, str]) -> Any:
    from spec1_bridge.adapters.ubik_spectral import (
        UbikSpectralAdapter,
        load_reports_jsonl,
    )

    return UbikSpectralAdapter(load_reports_jsonl(Path(env["UBIK_SPECTRAL_REPORTS"])))


def _travel(_env: Mapping[str, str]) -> Any:
    from spec1_bridge.adapters.travel_border import TravelBorderAdapter

    return TravelBorderAdapter()


def _nature(_env: Mapping[str, str]) -> Any:
    from spec1_bridge.adapters.nature_hydro import NatureHydroAdapter

    return NatureHydroAdapter()


def _claim_matcher(env: Mapping[str, str]) -> Runner:
    def run(conn: psycopg.Connection[Any]) -> RunOutcome:
        from spec1_bridge.adapters.claim_matcher import DEFAULT_CACHE_PATH
        from spec1_bridge.mycroft_triage import run_claim_matcher

        cache = env.get("CLAIM_CACHE_PATH") or str(DEFAULT_CACHE_PATH)
        key = env.get("GOOGLE_FACTCHECK_API_KEY") or None
        return _report(run_claim_matcher(cache, conn, _enricher(), api_key=key))

    return run


def _pdx1(env: Mapping[str, str]) -> Runner:
    def run(conn: psycopg.Connection[Any]) -> RunOutcome:
        from spec1_bridge.mycroft_triage import run_pdx1

        priority = env.get("PDX1_MIN_PRIORITY") or "STANDARD"
        return _report(run_pdx1(env["PDX1_STORE_PATH"], priority, conn, _enricher()))

    return run


def _spec1(env: Mapping[str, str]) -> Runner:
    def run(conn: psycopg.Connection[Any]) -> RunOutcome:
        from spec1_bridge.mycroft_triage import run_spec1

        verdicts = env.get("SPEC1_VERDICTS_PATH") or None
        return _report(run_spec1(env["SPEC1_STORE_PATH"], verdicts, conn, _enricher()))

    return run


def _proc_track(env: Mapping[str, str]) -> Runner:
    def run(conn: psycopg.Connection[Any]) -> RunOutcome:
        from spec1_bridge.mycroft_triage import run_proc_track

        session = env.get("PROC_TRACK_SESSION") or env["OLIS_SESSION_KEY"]
        return _report(
            run_proc_track(env["PROC_TRACK_PATH"], session, conn, _enricher())
        )

    return run


ENGINES: Final[tuple[Engine, ...]] = (
    Engine(
        "ubik_oregon",
        "Ubik",
        "engine",
        "Civic",
        number=1,
        requires=("OLIS_SESSION_KEY",),
        factory=_adapter_runner(_ubik),
    ),
    Engine(
        "seldon_civic",
        "Seldon",
        "engine",
        "Civic",
        number=2,
        factory=_adapter_runner(_seldon),
    ),
    Engine(
        "delos_econ",
        "Delos",
        "engine",
        "Economy",
        number=3,
        factory=_adapter_runner(_delos),
    ),
    Engine(
        "klein_biomed",
        "Klein",
        "engine",
        "Health",
        number=4,
        factory=_adapter_runner(_klein),
    ),
    Engine(
        "mazer_defense",
        "Mazer",
        "engine",
        "Defense",
        number=6,
        factory=_adapter_runner(_mazer),
        note="Runs USAspending only; GAO dockets are editor-supplied.",
    ),
    Engine(
        "valis_uap",
        "Valis",
        "engine",
        "UAP",
        ledger="telemetry",
        number=7,
        note="Needs NUFORC witness reports and orbital elements supplied by "
        "the editor; no harvest entry point yet.",
    ),
    Engine(
        "trantor_antiquity",
        "Trantor",
        "engine",
        "Antiquity",
        number=8,
        note="Needs editor-defined survey areas; locations stay Editor Eyes Only.",
    ),
    Engine(
        "ubik_spectral",
        "Ubik Spectral",
        "engine",
        "Folklore",
        number=9,
        requires=("UBIK_SPECTRAL_REPORTS",),
        factory=_adapter_runner(_ubik_spectral),
    ),
    Engine(
        "eschaton_prophecy",
        "Eschaton",
        "engine",
        "Disinfo",
        ledger="telemetry",
        number=10,
        note="A computed calendar, read by the detectors; it harvests nothing.",
    ),
    Engine(
        "gnosis_mystic",
        "Gnosis",
        "engine",
        "Esoteric",
        number=11,
        note="Needs editor-supplied movements and their own public pages.",
    ),
    Engine(
        "deckard_ct_mapper",
        "Deckard",
        "engine",
        "Disinfo",
        ledger="telemetry",
        note="Reads a moby-dork harvest manifest; run it after a Moby batch.",
    ),
    Engine(
        "pdx1_importer",
        "pdx-1i",
        "importer",
        "Civic",
        requires=("PDX1_STORE_PATH",),
        factory=_pdx1,
    ),
    Engine(
        "proc_track_importer",
        "proc_track",
        "importer",
        "Civic",
        requires=("PROC_TRACK_PATH", "PROC_TRACK_SESSION"),
        factory=_proc_track,
    ),
    Engine(
        "spec1_importer",
        "spec-1",
        "importer",
        "Disinfo",
        requires=("SPEC1_STORE_PATH",),
        factory=_spec1,
    ),
    Engine(
        "claim_matcher", "Claim Matcher", "adapter", "Disinfo", factory=_claim_matcher
    ),
    Engine(
        "travel_border", "Travel", "adapter", "Travel", factory=_adapter_runner(_travel)
    ),
    Engine(
        "nature_hydro", "Nature", "adapter", "Nature", factory=_adapter_runner(_nature)
    ),
)

ENGINE_BY_KEY: Final = {engine.key: engine for engine in ENGINES}

# subject_domain values as adapters and importers write them, by desk family.
DOMAIN_FAMILIES: Final[Mapping[str, str]] = {
    "oregon_civic": "Civic",
    "washington_civic": "Civic",
    "oregon_legislature": "Civic",
    "federal_legislation": "Civic",
    "supply_chain": "Economy",
    "science_health": "Health",
    "security_defense": "Defense",
    "uap": "UAP",
    "archaeological_antiquity": "Antiquity",
    "anomalous_lore": "Folklore",
    "esoteric_philosophy": "Esoteric",
    "narrative_integrity": "Disinfo",
    "border_crossings": "Travel",
    "hydrology": "Nature",
}
FAMILIES: Final = (
    "Civic",
    "Economy",
    "Health",
    "Defense",
    "UAP",
    "Antiquity",
    "Folklore",
    "Esoteric",
    "Disinfo",
    "Travel",
    "Nature",
)


def family_of(subject_domain: str) -> str:
    return DOMAIN_FAMILIES.get(subject_domain, subject_domain.replace("_", " ").title())


def domains_in(family: str) -> list[str]:
    return [domain for domain, name in DOMAIN_FAMILIES.items() if name == family]


# --- Run log and controller ---------------------------------------------------


@dataclass(slots=True)
class RunRecord:
    engine: str
    started_at: str
    finished_at: str
    status: Literal["ok", "error"]
    written: int = 0
    rejected: int = 0
    errors: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Fleet:
    """Starts runs on background threads and remembers how they went."""

    def __init__(
        self,
        dsn: str,
        run_log: Path,
        *,
        env: Mapping[str, str] | None = None,
        runners: Mapping[str, Runner] | None = None,
        spawn: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self._dsn = dsn
        self._log = run_log
        self._env = env
        self._runners = dict(runners or {})
        self._spawn = spawn or _spawn_thread
        self._active: set[str] = set()
        self._lock = threading.Lock()

    @property
    def env(self) -> Mapping[str, str]:
        return self._env if self._env is not None else os.environ

    def runner_for(self, engine: Engine) -> Runner | None:
        if engine.key in self._runners:
            return self._runners[engine.key]
        if engine.blocker(self.env) is not None or engine.factory is None:
            return None
        return engine.factory(self.env)

    def blocker(self, engine: Engine) -> str | None:
        if engine.key in self._runners:
            return None
        return engine.blocker(self.env)

    def is_active(self, key: str) -> bool:
        with self._lock:
            return key in self._active

    def start(self, engine: Engine) -> bool:
        """Start a run; False if one is already going or the engine cannot run."""
        runner = self.runner_for(engine)
        if runner is None:
            return False
        with self._lock:
            if engine.key in self._active:
                return False
            self._active.add(engine.key)
        self._spawn(lambda: self._run(engine.key, runner))
        return True

    def _run(self, key: str, runner: Runner) -> None:
        started = _now()
        try:
            with psycopg.connect(self._dsn, autocommit=True) as conn:
                outcome = runner(conn)
            record = RunRecord(
                key,
                started,
                _now(),
                "ok",
                outcome.written,
                outcome.rejected,
                list(outcome.errors[:5]),
            )
        except Exception as error:  # noqa: BLE001 - every failure must be logged
            last = traceback.format_exception_only(error)[-1].strip()
            record = RunRecord(key, started, _now(), "error", errors=[last])
        finally:
            with self._lock:
                self._active.discard(key)
        self._append(record)

    def _append(self, record: RunRecord) -> None:
        self._log.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")

    def last_runs(self) -> dict[str, RunRecord]:
        runs: dict[str, RunRecord] = {}
        try:
            lines = self._log.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return runs
        for line in lines:
            try:
                item = json.loads(line)
                runs[item["engine"]] = RunRecord(**item)
            except (ValueError, TypeError, KeyError):
                continue
        return runs


def _spawn_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, daemon=True, name="fleet-run").start()


# --- Card view ----------------------------------------------------------------


def ledger_counts(conn: psycopg.Connection[Any]) -> dict[str, dict[str, Any]]:
    """Leads and latest lead time per engine, from both ledgers."""
    with conn.cursor(row_factory=dict_row) as cursor:
        rows = cursor.execute(
            """
            SELECT adapter_name AS key, count(*) AS leads,
                   max(created_at) AS latest
            FROM triage_records GROUP BY adapter_name
            UNION ALL
            SELECT metrics->'receipt'->>'detector', count(*), max(observed_at)
            FROM sociology_telemetry.narrative_observations
            WHERE metrics->'receipt'->>'detector' IS NOT NULL
            GROUP BY 1
            """
        ).fetchall()
    return {row["key"]: row for row in rows}


def card(
    engine: Engine, fleet: Fleet, counts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    run = fleet.last_runs().get(engine.key)
    stats = counts.get(engine.key, {})
    total = (run.written + run.rejected) if run else 0
    if fleet.is_active(engine.key):
        status = "ACTIVE"
    elif run is not None and run.status == "error":
        status = "ERROR"
    else:
        status = "IDLE"
    latest = stats.get("latest")
    return {
        "engine": engine,
        "status": status,
        "leads": stats.get("leads", 0),
        "last_poll": run.finished_at if run else None,
        "latest_lead": latest.isoformat(timespec="seconds") if latest else None,
        "pass_rate": (f"{100 * run.written / total:.0f}%" if run and total else None),
        "run": run,
        "blocker": fleet.blocker(engine),
    }
