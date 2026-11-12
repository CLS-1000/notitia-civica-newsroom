"""Master Control Cockpit: the top bar, Fleet, Triage, Vault, and Publish tabs.

Runs against the real PostgreSQL fixtures. Harvests and deploys never touch the
network or a real repository: runners and the command runner are injected, and
background work runs synchronously so each test sees its outcome.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import psycopg
import pytest
from conftest import Conn
from fastapi.testclient import TestClient
from test_phase2_workstation import BASE_URL, SLUG, FakeAnthropic, drafter, seed, sign

from newsroom.penfield_desk.app import create_app, steelman_paragraphs
from newsroom.penfield_desk.cockpit import raw_view, status_badge
from newsroom.penfield_desk.deploy import DeployConfig, Deployer
from newsroom.penfield_desk.fleet import (
    ENGINE_BY_KEY,
    ENGINES,
    Fleet,
    RunOutcome,
    family_of,
)

NAMED_ENGINES = (
    "Oregon Civic", "Federal Civic", "Supply Chain", "Health and Biomed", "Defense Contracts", "UAP Telemetry", "Archaeology LiDAR",
    "Living Folklore", "Prophecy Calendar", "Esoteric Philosophy", "CT Mapper",
)  # fmt: skip
TABS = (
    "[FLEET CONTROL]",
    "[TRIAGE QUEUE]",
    "[EDITORIAL DESK]",
    "[EVIDENCE VAULT]",
    "[PUBLISH &amp; DEPLOY]",
)


def run_now(work: Callable[[], None]) -> None:
    work()


def never_run(_work: Callable[[], None]) -> None:
    """Leave the run 'in progress' so ACTIVE state can be observed."""


class FakeCommands:
    """Stands in for subprocess.run and records each command."""

    def __init__(self, fail: str | None = None, staged: bool = False) -> None:
        self.calls: list[tuple[list[str], Path]] = []
        self.fail = fail
        self.staged = staged

    def __call__(
        self, argv: Sequence[str], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        self.calls.append((args, cwd))
        code = 0
        out = ""
        if self.fail and self.fail in " ".join(args):
            code = 1
            out = "simulated failure"
        elif args[:3] == ["git", "diff", "--cached"]:
            # Before any add, report what was configured; after, report changes.
            added = any(call[0][:2] == ["git", "add"] for call in self.calls)
            code = 1 if (added or self.staged) else 0
        elif args[:2] == ["git", "rev-parse"]:
            out = "abc1234\n"
        return subprocess.CompletedProcess(args, code, out, "")

    def ran(self, *prefix: str) -> bool:
        return any(call[0][: len(prefix)] == list(prefix) for call in self.calls)


@pytest.fixture
def site(tmp_path: Path) -> tuple[Path, Path]:
    web = tmp_path / "web_public"
    (web / "dist" / "_astro").mkdir(parents=True)
    (web / "package.json").write_text("{}")
    (web / "dist" / "index.html").write_text("<html>new</html>")
    (web / "dist" / "_astro" / "a.css").write_text("body{}")
    target = tmp_path / "pages"
    (target / ".git").mkdir(parents=True)
    (target / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (target / "index.html").write_text("<html>old</html>")
    (target / "owc").mkdir()
    (target / "owc" / "legacy.html").write_text("legacy")
    return web, target


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "archive" / "warc").mkdir(parents=True)
    return root


def make_client(
    dsn: str,
    tmp_path: Path,
    archive: Path,
    *,
    runners: dict[str, Callable[[psycopg.Connection[Any]], RunOutcome]] | None = None,
    spawn: Callable[[Callable[[], None]], None] = run_now,
    deployer: Deployer | None = None,
    env: dict[str, str] | None = None,
) -> TestClient:
    fleet = Fleet(
        dsn,
        tmp_path / "runs.jsonl",
        env=env or {},
        runners=runners,
        spawn=spawn,
    )
    app = create_app(
        dsn,
        lambda: drafter(FakeAnthropic()),
        fleet=fleet,
        deployer=deployer
        or Deployer(DeployConfig(dsn=dsn), command=FakeCommands(), spawn=run_now),
        archive_root=archive,
    )
    return TestClient(app, base_url=BASE_URL)


@pytest.fixture
def cockpit(
    ledger_dsn: str, db: Conn, tmp_path: Path, archive: Path
) -> Iterator[TestClient]:
    with make_client(ledger_dsn, tmp_path, archive) as client:
        yield client


def add_source(
    conn: Conn,
    source_id: str,
    payload: bytes,
    archive: Path,
    *,
    domain: str = "example.gov",
    tier: str = "tier_1_primary",
    write: bool = True,
    sha: str | None = None,
) -> None:
    path = f"archive/warc/{source_id}.blob"
    if write:
        (archive / path).write_bytes(payload)
    conn.execute(
        """
        INSERT INTO source_registry
            (source_id, url, domain, tier, raw_sha256, archive_path,
             verbatim_text, retrieved_at)
        VALUES (%s, %s, %s, %s, %s, %s, NULL, now())
        """,
        (
            source_id,
            f"https://{domain}/{source_id}",
            domain,
            tier,
            sha or hashlib.sha256(payload).hexdigest(),
            path,
        ),
    )


def add_lead(
    conn: Conn,
    *,
    adapter: str = "delos_econ",
    domain: str = "supply_chain",
    score: float = 82.0,
    question: str | None = "Why are three drugs short at once?",
    sources: tuple[str, ...] = ("S-1",),
) -> uuid.UUID:
    record_id = uuid.uuid4()
    conn.execute(
        """
        INSERT INTO triage_records
            (record_id, adapter_name, subject_domain, anomaly_score,
             decomposition_receipt, paste_ready_prompt, source_ids,
             public_question, public_interest_reason)
        VALUES (%s, %s, %s, %s, %s, 'Assess this.', %s, %s, %s)
        """,
        (
            record_id,
            adapter,
            domain,
            score,
            json.dumps(
                {
                    "gates": [
                        {"gate": "authority", "passed": True, "reason": "fda.gov"}
                    ],
                    "finding": "Two approved applications remain.",
                }
            ),
            list(sources),
            question,
            "Patients may lose access." if question else None,
        ),
    )
    return record_id


# --- Top bar ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "current"),
    [
        ("/fleet", "[FLEET CONTROL]"),
        ("/triage", "[TRIAGE QUEUE]"),
        ("/", "[EDITORIAL DESK]"),
        ("/vault", "[EVIDENCE VAULT]"),
        ("/publish", "[PUBLISH &amp; DEPLOY]"),
    ],
)
def test_every_tab_has_the_app_bar(
    cockpit: TestClient, path: str, current: str
) -> None:
    page = cockpit.get(path)
    assert page.status_code == 200
    for tab in TABS:
        assert tab in page.text
    assert f'aria-current="page">{current}</a>' in page.text
    assert "DB ONLINE" in page.text
    assert 'hx-get="/status"' in page.text


def test_status_badge_counts_articles_and_last_harvest(
    cockpit: TestClient, db: Conn
) -> None:
    seed(db)
    add_lead(db)
    badge = cockpit.get("/status").text
    assert "DB ONLINE" in badge
    assert "<b>1</b> articles" in badge
    assert "Last harvest <b>never</b>" not in badge


def test_status_badge_reports_a_down_database() -> None:
    status = status_badge("host=127.0.0.1 port=1 dbname=none connect_timeout=1")
    assert status["db_ok"] is False
    assert status["db_error"]


def test_palette_tokens_match_the_design_system(cockpit: TestClient) -> None:
    tokens = cockpit.get("/static/tokens.css").text.lower()
    for color in ("#0e1116", "#161b22", "#1d4e56"):
        assert color in tokens
    assert cockpit.get("/static/cockpit.css").status_code == 200
    assert "showModal" in cockpit.get("/static/cockpit.js").text


# --- Fleet Control ------------------------------------------------------------


def test_fleet_lists_every_engine_importer_and_adapter(cockpit: TestClient) -> None:
    page = cockpit.get("/fleet").text
    for name in NAMED_ENGINES:
        assert f"</span>{name}</span>" in page or f">{name}</span>" in page
    for name in ("pdx-1i", "proc_track", "spec-1", "Travel", "Nature"):
        assert f">{name}</span>" in page
    assert page.count('class="card ') == len(ENGINES)
    assert sum(1 for engine in ENGINES if engine.group == "engine") == 11


def test_fleet_counts_leads_per_engine(cockpit: TestClient, db: Conn) -> None:
    add_lead(db)
    add_lead(db, score=90)
    card = cockpit.get("/fleet/delos_econ").text
    assert "<dt>Leads produced</dt><dd>2" in card


def test_engines_needing_inputs_cannot_be_run(cockpit: TestClient) -> None:
    card = cockpit.get("/fleet/trantor_antiquity").text
    assert "survey areas" in card
    assert "disabled>Run Harvest Now" in card
    refused = cockpit.post("/fleet/trantor_antiquity/run")
    assert refused.status_code == 409


def test_missing_environment_is_named(cockpit: TestClient) -> None:
    card = cockpit.get("/fleet/pdx1_importer").text
    assert "Set PDX1_STORE_PATH" in card
    assert cockpit.post("/fleet/pdx1_importer/run").status_code == 409


def test_unknown_engine_is_404(cockpit: TestClient) -> None:
    assert cockpit.get("/fleet/nope").status_code == 404
    assert cockpit.post("/fleet/nope/run").status_code == 404


def test_run_harvest_now_records_the_outcome(
    ledger_dsn: str, db: Conn, tmp_path: Path, archive: Path
) -> None:
    seen: list[bool] = []

    def runner(conn: psycopg.Connection[Any]) -> RunOutcome:
        seen.append(conn.execute("SELECT 1").fetchone() == (1,))
        return RunOutcome(written=3, rejected=1, errors=("fda.gov: timeout",))

    with make_client(
        ledger_dsn, tmp_path, archive, runners={"delos_econ": runner}
    ) as client:
        response = client.post("/fleet/delos_econ/run")
        assert response.status_code == 200
        assert seen == [True]
        assert "IDLE" in response.text
        assert "75%" in response.text
        assert "(3/4 last run)" in response.text
        assert "1 source error: fda.gov: timeout" in response.text

    record = json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[-1])
    assert record["engine"] == "delos_econ"
    assert record["status"] == "ok"
    assert record["written"] == 3


def test_failed_run_shows_error_status(
    ledger_dsn: str, db: Conn, tmp_path: Path, archive: Path
) -> None:
    def runner(_conn: psycopg.Connection[Any]) -> RunOutcome:
        raise RuntimeError("upstream 503")

    with make_client(
        ledger_dsn, tmp_path, archive, runners={"seldon_civic": runner}
    ) as client:
        card = client.post("/fleet/seldon_civic/run").text
    assert "ERROR" in card
    assert "RuntimeError: upstream 503" in card


def test_active_run_polls_and_refuses_a_second_start(
    ledger_dsn: str, db: Conn, tmp_path: Path, archive: Path
) -> None:
    with make_client(
        ledger_dsn,
        tmp_path,
        archive,
        runners={"klein_biomed": lambda _conn: RunOutcome(0, 0)},
        spawn=never_run,
    ) as client:
        first = client.post("/fleet/klein_biomed/run").text
        assert "ACTIVE" in first
        assert 'hx-trigger="every 2s"' in first
        assert "Harvesting…" in first
        second = client.post("/fleet/klein_biomed/run").text
        assert "already in progress" in second


def test_environment_unlocks_an_importer(tmp_path: Path) -> None:
    fleet = Fleet("", tmp_path / "runs.jsonl", env={"PDX1_STORE_PATH": "x.jsonl"})
    assert fleet.blocker(ENGINE_BY_KEY["pdx1_importer"]) is None
    assert fleet.runner_for(ENGINE_BY_KEY["pdx1_importer"]) is not None


def test_run_log_skips_corrupt_lines(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    good = {
        "engine": "delos_econ",
        "started_at": "a",
        "finished_at": "b",
        "status": "ok",
        "written": 1,
        "rejected": 0,
        "errors": [],
    }
    log.write_text("not json\n" + json.dumps(good) + "\n{}\n")
    runs = Fleet("", log).last_runs()
    assert list(runs) == ["delos_econ"]


def test_run_harvest_buttons_are_origin_guarded(cockpit: TestClient) -> None:
    refused = cockpit.post(
        "/fleet/delos_econ/run", headers={"Origin": "https://evil.example"}
    )
    assert refused.status_code == 403
    deploy = cockpit.post("/publish/deploy", headers={"Sec-Fetch-Site": "cross-site"})
    assert deploy.status_code == 403


# --- Triage Queue -------------------------------------------------------------


def test_triage_filters_by_domain_and_score(cockpit: TestClient, db: Conn) -> None:
    add_lead(db, adapter="delos_econ", domain="supply_chain", score=82)
    add_lead(db, adapter="klein_biomed", domain="science_health", score=40)
    add_lead(db, adapter="mazer_defense", domain="security_defense", score=95)

    everything = cockpit.get("/triage").text
    assert everything.count('class="lead-row"') == 3
    assert ">Economy</span>" in everything

    economy = cockpit.get("/triage?family=Economy").text
    assert economy.count('class="lead-row"') == 1
    assert "delos_econ" in economy

    high = cockpit.get("/triage?min_score=80").text
    assert high.count('class="lead-row"') == 2
    assert "klein_biomed" not in high


def test_triage_partial_swaps_only_the_rows(cockpit: TestClient, db: Conn) -> None:
    add_lead(db)
    partial = cockpit.get(
        "/triage?min_score=50",
        headers={"HX-Request": "true", "HX-Target": "triage-rows"},
    ).text
    assert partial.startswith('<div id="triage-rows">')
    assert "[FLEET CONTROL]" not in partial


@pytest.mark.parametrize("query", ["min_score=nan", "min_score=abc", "family=Bogus"])
def test_triage_ignores_bad_filters(cockpit: TestClient, db: Conn, query: str) -> None:
    add_lead(db)
    page = cockpit.get(f"/triage?{query}")
    assert page.status_code == 200
    assert page.text.count('class="lead-row"') == 1


def test_lead_detail_shows_receipt_and_promote(cockpit: TestClient, db: Conn) -> None:
    record_id = add_lead(db)
    detail = cockpit.get(f"/triage/{record_id}").text
    assert "Why are three drugs short at once?" in detail
    assert "Patients may lose access." in detail
    assert "Plain-English Decomposition Receipt" in detail
    assert "Two approved applications remain." in detail
    assert f'hx-post="/leads/{record_id}/draft"' in detail
    assert "Promote to Desk" in detail


def test_promote_to_desk_drafts_and_opens_the_desk(
    cockpit: TestClient, db: Conn
) -> None:
    seed(db)  # the fake draft cites S-1 and S-2
    record_id = add_lead(db, sources=("S-1", "S-2"))
    response = cockpit.post(f"/leads/{record_id}/draft", follow_redirects=False)
    assert response.status_code == 303, response.text
    target = response.headers["HX-Redirect"]
    assert target.startswith("/?article=")
    desk = cockpit.get(target).text
    assert "[EDITORIAL DESK]</a>" in desk
    detail = cockpit.get(f"/triage/{record_id}").text
    assert "Open on Desk" in detail


def test_unknown_lead_is_404(cockpit: TestClient) -> None:
    assert cockpit.get(f"/triage/{uuid.uuid4()}").status_code == 404


# --- Editorial Desk -----------------------------------------------------------


def test_desk_keeps_three_panes_inside_the_cockpit(
    cockpit: TestClient, db: Conn
) -> None:
    seed(db)
    page = cockpit.get(f"/?article={SLUG}").text
    for text in (
        "SPEC-1 Lead Queue",
        "Primary Source Documents",
        "Sign &amp; Publish",
        "dors_venabili Gates",
        "[EVIDENCE VAULT]",
    ):
        assert text in page
    assert 'class="epi verified"' in page
    assert "The audit period is not stated." in page


def test_live_preview_renders_epistemic_callouts(cockpit: TestClient, db: Conn) -> None:
    seed(db)
    body = (
        "County records list 14 closed sites. [S-2]\n\n"
        "Steelman: The county says closures saved $2 million. [S-1]"
    )
    live = cockpit.post(
        f"/articles/{SLUG}/validate",
        data={
            "body_markdown": body,
            "uncertainty_disclosures": "Whether sites reopen is unknown.",
        },
    ).text
    assert "Verified" in live
    assert "Whether sites reopen is unknown." in live
    assert 'class="epi steelman"' in live
    assert "The county says closures saved $2 million." in live


def test_steelman_marker_needs_the_colon() -> None:
    assert steelman_paragraphs("**Steelman:** A case.\n\nSteelmanning is hard.") == [
        "A case."
    ]


# --- Evidence Vault -----------------------------------------------------------


def test_vault_lists_sources_with_tier_and_fingerprint(
    cockpit: TestClient, db: Conn, archive: Path
) -> None:
    add_source(db, "S-10", b"<html>record</html>", archive, domain="oregon.gov")
    add_source(db, "S-11", b"x", archive, tier="tier_4_unverified")
    page = cockpit.get("/vault").text
    assert "S-10" in page and "S-11" in page
    assert "Tier 1 · Primary" in page
    assert "Tier 4 · Unverified" in page
    assert hashlib.sha256(b"<html>record</html>").hexdigest() in page
    assert "2 sealed sources" in page


def test_vault_search_and_flagged_filter(
    cockpit: TestClient, db: Conn, archive: Path
) -> None:
    add_source(db, "S-10", b"a", archive, domain="oregon.gov")
    add_source(db, "S-11", b"b", archive, domain="fda.gov")
    db.execute(
        "SELECT flag_sources_for_tier_review(ARRAY['S-11'], 'factual correction')"
    )
    found = cockpit.get("/vault?q=oregon", headers={"HX-Target": "vault-results"}).text
    assert found.startswith('<div id="vault-results">')
    assert "S-10" in found and "S-11" not in found

    flagged = cockpit.get("/vault?flagged=on").text
    assert "S-11" in flagged and "S-10" not in flagged
    assert 'class="vault-row flagged"' in flagged
    assert "⚑ factual correction" in flagged


def test_vault_search_treats_wildcards_literally(
    cockpit: TestClient, db: Conn, archive: Path
) -> None:
    add_source(db, "S-10", b"a", archive)
    assert "S-10" not in cockpit.get("/vault?q=%25").text.split("<tbody>")[1]


def test_vault_paginates(cockpit: TestClient, db: Conn, archive: Path) -> None:
    for number in range(55):
        add_source(db, f"S-{100 + number}", str(number).encode(), archive)
    first = cockpit.get("/vault").text
    assert "page 1 of 2" in first
    assert first.count('class="vault-row') == 50
    second = cockpit.get("/vault?page=2").text
    assert second.count('class="vault-row') == 5


def test_vault_marks_editor_eyes_only(
    cockpit: TestClient, db: Conn, archive: Path
) -> None:
    add_source(db, "S-12", b"site", archive)
    db.execute(
        "INSERT INTO restricted_sources (source_id, reason) VALUES (%s, %s)",
        ("S-12", "ARPA site location"),
    )
    assert "EDITOR EYES ONLY" in cockpit.get("/vault").text
    raw = cockpit.get("/vault/S-12/raw").text
    assert "EDITOR EYES ONLY: ARPA site location" in raw


def test_raw_modal_shows_bytes_and_rechecks_the_seal(
    cockpit: TestClient, db: Conn, archive: Path
) -> None:
    add_source(db, "S-13", b"WARC/1.1\r\n<p>Budget & audit</p>", archive)
    raw = cockpit.get("/vault/S-13/raw").text
    assert "Seal verified" in raw
    assert "&lt;p&gt;Budget &amp; audit&lt;/p&gt;" in raw
    assert "data-close-modal" in raw


def test_raw_modal_flags_a_seal_mismatch(
    cockpit: TestClient, db: Conn, archive: Path
) -> None:
    add_source(db, "S-14", b"tampered", archive, sha="0" * 64)
    assert "SEAL MISMATCH" in cockpit.get("/vault/S-14/raw").text


def test_raw_modal_reports_a_missing_blob(
    cockpit: TestClient, db: Conn, archive: Path
) -> None:
    add_source(db, "S-15", b"gone", archive, write=False)
    assert "Blob not found on disk" in cockpit.get("/vault/S-15/raw").text


def test_raw_modal_shows_binary_as_hex(tmp_path: Path) -> None:
    blob = tmp_path / "b.blob"
    payload = b"%PDF-1.7\n\xff\xfe\x00\x01" * 4
    blob.write_bytes(payload)
    view = raw_view(blob, hashlib.sha256(payload).hexdigest())
    assert view["text"] is None
    assert view["seal_ok"]
    assert view["hex_lines"][0].startswith("00000000  25 50 44 46")


def test_raw_modal_refuses_paths_outside_the_archive(
    cockpit: TestClient, db: Conn
) -> None:
    db.execute(
        """
        INSERT INTO source_registry
            (source_id, url, domain, tier, raw_sha256, archive_path, retrieved_at)
        VALUES ('S-16', NULL, 'x.gov', 'tier_1_primary', %s, '../../etc/passwd',
                now())
        """,
        ("0" * 64,),
    )
    assert cockpit.get("/vault/S-16/raw").status_code == 403
    assert cockpit.get("/vault/S-99/raw").status_code == 404


# --- Publish & Deploy ---------------------------------------------------------


def test_publish_lists_only_signed_verified_articles(
    cockpit: TestClient, db: Conn
) -> None:
    seed(db)
    before = cockpit.get("/publish").text
    assert "No signed, llm_verified articles yet." in before
    sign(db)
    after = cockpit.get("/publish").text
    assert "Department reports shortfall" in after
    assert "https://cls-1000.github.io/notitia-civica-ux/" in after
    assert "Build &amp; Sync to Public Site" in after
    assert "hx-confirm=" in after


def test_deploy_exports_builds_copies_and_pushes(
    ledger_dsn: str, db: Conn, tmp_path: Path, archive: Path, site: tuple[Path, Path]
) -> None:
    web, target = site
    commands = FakeCommands()
    deployer = Deployer(
        DeployConfig(dsn=ledger_dsn, web_public=web, target=target, python="py"),
        command=commands,
        spawn=run_now,
    )
    with make_client(ledger_dsn, tmp_path, archive, deployer=deployer) as client:
        log = client.post("/publish/deploy").text
        assert log.count("OK</span>") == 4
        assert "Deployed as abc1234" in log
        assert "https://cls-1000.github.io/notitia-civica-ux/" in log
        assert client.get("/publish/deploy/log").text.count("OK</span>") == 4

    export = commands.calls[1][0]
    assert export[:2] == ["py", str(web / "scripts" / "export_verified.py")]
    assert export[export.index("--out") + 1] == str(web / "src" / "data")
    assert export[export.index("--dsn") + 1] == ledger_dsn
    assert (["npx", "astro", "build", "--base", "/notitia-civica-ux"], web) in (
        commands.calls
    )
    assert (target / "index.html").read_text() == "<html>new</html>"
    assert (target / "_astro" / "a.css").exists()
    assert (target / "owc" / "legacy.html").read_text() == "legacy"
    assert (target / ".git" / "HEAD").read_text() == "ref: refs/heads/main\n"
    add = next(call[0] for call in commands.calls if call[0][:2] == ["git", "add"])
    assert sorted(add[3:]) == ["_astro/a.css", "index.html"]
    assert commands.ran("git", "push", "origin", "HEAD")


def test_deploy_stops_when_export_fails(
    ledger_dsn: str, tmp_path: Path, site: tuple[Path, Path]
) -> None:
    web, target = site
    commands = FakeCommands(fail="export_verified.py")
    deployer = Deployer(
        DeployConfig(dsn=ledger_dsn, web_public=web, target=target),
        command=commands,
        spawn=run_now,
    )
    assert deployer.start()
    run = deployer.current
    assert run is not None and not run.succeeded
    assert [step.state for step in run.steps] == [
        "failed", "skipped", "skipped", "skipped",
    ]  # fmt: skip
    assert "simulated failure" in "\n".join(run.steps[0].lines)
    assert (target / "index.html").read_text() == "<html>old</html>"
    assert not commands.ran("npx")


def test_deploy_refuses_a_checkout_with_staged_changes(
    ledger_dsn: str, site: tuple[Path, Path]
) -> None:
    web, target = site
    commands = FakeCommands(staged=True)
    deployer = Deployer(
        DeployConfig(dsn=ledger_dsn, web_public=web, target=target),
        command=commands,
        spawn=run_now,
    )
    deployer.start()
    assert deployer.current is not None
    assert "already has staged changes" in "\n".join(deployer.current.steps[0].lines)
    assert not commands.ran("git", "push")


def test_deploy_refuses_a_target_without_git(
    ledger_dsn: str, tmp_path: Path, site: tuple[Path, Path]
) -> None:
    web, _target = site
    bare = tmp_path / "bare"
    bare.mkdir()
    deployer = Deployer(
        DeployConfig(dsn=ledger_dsn, web_public=web, target=bare),
        command=FakeCommands(),
        spawn=run_now,
    )
    deployer.start()
    assert deployer.current is not None
    assert "is not a git checkout" in deployer.current.steps[0].lines[-1]


def test_deploy_with_nothing_changed_does_not_commit(
    ledger_dsn: str, site: tuple[Path, Path]
) -> None:
    web, target = site

    class Unchanged(FakeCommands):
        def __call__(
            self, argv: Sequence[str], cwd: Path
        ) -> subprocess.CompletedProcess[str]:
            result = super().__call__(argv, cwd)
            if list(argv)[:3] == ["git", "diff", "--cached"]:
                return subprocess.CompletedProcess(list(argv), 0, "", "")
            return result

    commands = Unchanged()
    deployer = Deployer(
        DeployConfig(dsn=ledger_dsn, web_public=web, target=target),
        command=commands,
        spawn=run_now,
    )
    deployer.start()
    assert deployer.current is not None and deployer.current.succeeded
    assert not commands.ran("git", "commit")
    assert not commands.ran("git", "push")


def test_only_one_deploy_runs_at_a_time(
    ledger_dsn: str, db: Conn, tmp_path: Path, archive: Path
) -> None:
    deployer = Deployer(
        DeployConfig(dsn=ledger_dsn), command=FakeCommands(), spawn=never_run
    )
    with make_client(ledger_dsn, tmp_path, archive, deployer=deployer) as client:
        first = client.post("/publish/deploy").text
        assert 'hx-trigger="every 1s"' in first
        second = client.post("/publish/deploy").text
        assert "A deploy is already running." in second


def test_family_mapping_covers_every_adapter_domain() -> None:
    assert family_of("supply_chain") == "Economy"
    assert family_of("narrative_integrity") == "Disinfo"
    assert family_of("brand_new_domain") == "Brand New Domain"
