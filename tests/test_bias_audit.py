"""Monthly bias and sourcing scorecard: baseline, arithmetic, and ledger reads."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from conftest import REPO_ROOT, Conn

from newsroom.dors_venabili import bias_audit
from newsroom.dors_venabili.bias_audit import (
    BASELINE_ENV,
    DEFAULT_BASELINE_PATH,
    DOMAIN_MAP,
    ROLE_NOTE,
    Baseline,
    BaselineError,
    compute_audit,
    load_baseline,
    previous_period,
    run_audit,
    window_for,
)

GENERATED = datetime(2026, 11, 1, 6, 0, tzinfo=UTC)
WINDOW = window_for("2026-10")
BASELINE = Baseline(
    "Test baseline", {"government_law": 50.0, "science_health": 50.0}, 2.5
)


def article(domain: str, **extra: Any) -> dict[str, Any]:
    return {
        "article_id": uuid.uuid4(),
        "section": domain,
        "subject_domain": None,
        "status": "published",
        "source_ids": [],
        "source_roles": {},
    } | extra


def audit(
    articles: list[dict[str, Any]],
    corrections: list[dict[str, Any]] | None = None,
    baseline: Baseline = BASELINE,
) -> dict[str, Any]:
    return compute_audit(WINDOW, articles, corrections or [], baseline, GENERATED)


def coverage(report: dict[str, Any], domain: str) -> dict[str, Any]:
    return next(row for row in report["domain_coverage"] if row["domain"] == domain)


# --- Baseline -----------------------------------------------------------------


def test_default_baseline_is_the_documented_file() -> None:
    assert DEFAULT_BASELINE_PATH == (
        REPO_ROOT / "newsroom" / "dors_venabili" / "civic_baselines.json"
    )
    baseline = load_baseline(DEFAULT_BASELINE_PATH)
    assert baseline.reference_source == (
        "BLS Consumer Expenditure Survey & Oregon State Budget (2025-2027)"
    )
    assert baseline.max_sigma == 2.5
    assert baseline.weights == {
        "economy_supply_chain": 20.0,
        "government_law": 18.0,
        "science_health": 16.0,
        "infrastructure": 12.0,
        "energy_environment": 10.0,
        "labor_trades": 8.0,
        "security_defense": 6.0,
        "technology_cyber": 5.0,
        "culture_anomalies": 5.0,
    }


def test_every_mapped_domain_is_in_the_default_baseline() -> None:
    weights = load_baseline(DEFAULT_BASELINE_PATH).weights
    assert set(DOMAIN_MAP.values()) <= set(weights)


def test_env_var_overrides_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "custom.json"
    custom.write_text(
        json.dumps(
            {
                "reference_source": "Custom",
                "domain_target_weights": {"government_law": 100},
                "max_allowed_divergence_sigma": 3,
            }
        )
    )
    monkeypatch.setenv(BASELINE_ENV, str(custom))
    assert load_baseline() == Baseline("Custom", {"government_law": 100.0}, 3.0)
    monkeypatch.delenv(BASELINE_ENV)
    assert load_baseline().reference_source.startswith("BLS")


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        ([], "JSON object"),
        (
            {"domain_target_weights": {"a": 100}, "max_allowed_divergence_sigma": 2},
            "reference",
        ),
        ({"reference_source": " ", "domain_target_weights": {"a": 100}}, "reference"),
        ({"reference_source": "x", "domain_target_weights": {}}, "non-empty"),
        (
            {"reference_source": "x", "domain_target_weights": {"a": "50"}},
            "not a number",
        ),
        (
            {"reference_source": "x", "domain_target_weights": {"a": True}},
            "not a number",
        ),
        (
            {"reference_source": "x", "domain_target_weights": {"a": -1, "b": 101}},
            "zero",
        ),
        ({"reference_source": "x", "domain_target_weights": {"a": 60, "b": 30}}, "sum"),
        (
            {
                "reference_source": "x",
                "domain_target_weights": {"a": 100},
                "max_allowed_divergence_sigma": 0,
            },
            "positive",
        ),
    ],
)
def test_bad_baselines_are_refused(tmp_path: Path, data: Any, problem: str) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data))
    with pytest.raises(BaselineError, match=problem):
        load_baseline(path)


def test_missing_baseline_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BaselineError, match="cannot read"):
        load_baseline(tmp_path / "absent.json")


# --- Window -------------------------------------------------------------------


def test_window_is_the_30_days_ending_with_the_period() -> None:
    assert WINDOW.end == datetime(2026, 11, 1, tzinfo=UTC)
    assert WINDOW.start == datetime(2026, 10, 2, tzinfo=UTC)
    assert window_for("2026-12").end == datetime(2027, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("period", ["2026-13", "2026/10", "October", ""])
def test_bad_periods_are_refused(period: str) -> None:
    with pytest.raises(ValueError, match="YYYY-MM"):
        window_for(period)


def test_default_period_is_the_previous_month() -> None:
    assert previous_period(datetime(2026, 11, 11, tzinfo=UTC)) == "2026-10"
    assert previous_period(datetime(2027, 1, 3, tzinfo=UTC)) == "2026-12"


# --- Coverage -----------------------------------------------------------------


def test_coverage_matches_its_target_without_a_flag() -> None:
    report = audit([article("oregon_legislature"), article("science_health")])
    row = coverage(report, "government_law")
    assert row == {
        "domain": "government_law",
        "articles": 1,
        "share_pct": 50.0,
        "target_pct": 50.0,
        "gap_pct_points": 0.0,
        "sigma": 0.0,
        "flagged": False,
    }


def test_lopsided_coverage_is_flagged_by_binomial_sigma() -> None:
    report = audit([article("federal_legislation") for _ in range(30)])
    row = coverage(report, "government_law")
    # (30 - 15) / sqrt(30 * 0.5 * 0.5) = 5.48
    assert row["sigma"] == 5.48 and row["flagged"] is True
    assert coverage(report, "science_health")["sigma"] == -5.48


def test_small_deviations_stay_inside_the_limit() -> None:
    report = audit([article("federal_legislation")] * 3 + [article("science_health")])
    assert coverage(report, "government_law")["sigma"] == 1.0
    assert not any(row["flagged"] for row in report["domain_coverage"])


def test_lead_subject_domain_wins_over_section() -> None:
    report = audit([article("misc", subject_domain="science_health")])
    assert coverage(report, "science_health")["articles"] == 1
    assert report["unmapped_domains"] == []


def test_baseline_domain_names_are_accepted_as_sections() -> None:
    report = audit(
        [article("science_health")], baseline=load_baseline(DEFAULT_BASELINE_PATH)
    )
    assert coverage(report, "science_health")["articles"] == 1


def test_unknown_domains_are_listed_not_forced() -> None:
    report = audit([article("astrology"), article(""), article("science_health")])
    assert report["unmapped_domains"] == [
        {"value": "(none)", "articles": 1},
        {"value": "astrology", "articles": 1},
    ]
    assert report["articles"]["total"] == 3
    assert coverage(report, "science_health")["share_pct"] == 33.33
    assert any("unmapped" in note for note in report["notes"])


def test_empty_window_reports_nothing_as_flagged() -> None:
    report = audit([])
    assert all(row["sigma"] is None for row in report["domain_coverage"])
    assert all(row["share_pct"] is None for row in report["domain_coverage"])
    assert not any(row["flagged"] for row in report["domain_coverage"])
    assert "No articles were published in this window." in report["notes"]


# --- Source roles -------------------------------------------------------------


def test_citations_are_grouped_by_desk_role() -> None:
    report = audit(
        [
            article(
                "science_health",
                source_ids=["S-1", "S-2", "S-3", "S-4"],
                source_roles={
                    "S-1": "government_record",
                    "S-2": "official",
                    "S-3": "independent_expert",
                    "S-4": "affected_party",
                },
            ),
            article("science_health", source_ids=["S-5"], source_roles={}),
        ]
    )
    roles = report["source_roles"]
    assert roles["citations"] == 5
    groups = {row["group"]: row for row in roles["groups"]}
    assert groups["government"]["citations"] == 2
    assert groups["government"]["share_pct"] == 40.0
    assert groups["academic_expert"]["share_pct"] == 20.0
    assert groups["frontline"]["share_pct"] == 20.0
    assert groups["unassigned"]["citations"] == 1
    assert "other" not in groups
    assert ROLE_NOTE in report["notes"]


def test_unknown_roles_are_grouped_as_other() -> None:
    report = audit(
        [article("science_health", source_ids=["S-1"], source_roles={"S-1": "pundit"})]
    )
    groups = {row["group"]: row for row in report["source_roles"]["groups"]}
    assert groups["other"] == {
        "group": "other",
        "roles": ["pundit"],
        "citations": 1,
        "share_pct": 100.0,
    }


# --- Corrections --------------------------------------------------------------


def test_correction_rates_are_per_article() -> None:
    items = [article("science_health") for _ in range(4)]
    corrections = [
        {"article_id": items[0]["article_id"], "correction_type": "factual_correction"},
        {"article_id": items[0]["article_id"], "correction_type": "clarification"},
        {"article_id": items[1]["article_id"], "correction_type": "clarification"},
        {"article_id": items[2]["article_id"], "correction_type": "methodology_update"},
    ]
    result = audit(items, corrections)["corrections"]
    assert result == {
        "factual_corrections": 1,
        "clarifications": 2,
        "methodology_updates": 1,
        "articles_corrected_or_clarified": 2,
        "factual_correction_rate_pct": 25.0,
        "clarification_rate_pct": 50.0,
    }


def test_report_shape_for_the_reader_site() -> None:
    report = audit([article("science_health", status="retracted")])
    assert report["period"] == "2026-10"
    assert report["window"] == {
        "start": "2026-10-02T00:00:00+00:00",
        "end": "2026-11-01T00:00:00+00:00",
    }
    assert report["generated_at"] == "2026-11-01T06:00:00+00:00"
    assert report["articles"] == {"total": 1, "retracted": 1}
    assert report["baseline"] == {
        "reference_source": "Test baseline",
        "max_allowed_divergence_sigma": 2.5,
    }
    json.dumps(report)  # serializable as written


# --- Ledger -------------------------------------------------------------------

PROOF = "a" * 64


def insert_article(
    conn: Conn,
    slug: str,
    section: str,
    published_at: datetime,
    *,
    verified: bool = True,
    roles: dict[str, str] | None = None,
    triage_domain: str | None = None,
) -> uuid.UUID:
    record_id = None
    if triage_domain:
        record_id = uuid.uuid4()
        conn.execute(
            """
            INSERT INTO triage_records
                (record_id, adapter_name, subject_domain, anomaly_score,
                 paste_ready_prompt)
            VALUES (%s, 'test', %s, 60, 'Assess this.')
            """,
            (record_id, triage_domain),
        )
    article_id = uuid.uuid4()
    conn.execute(
        """
        INSERT INTO articles
            (article_id, slug, section, badge_type, headline, body_markdown,
             analysis_tier, status, human_approved, editor_name,
             editorial_proof_hash, published_at, source_ids, source_roles,
             triage_record_id)
        VALUES (%s, %s, %s, 'ANALYSIS', 'Headline', 'Body [S-1]',
                %s, %s, %s, 'Desk Editor', %s, %s, %s, %s, %s)
        """,
        (
            article_id,
            slug,
            section,
            "llm_verified" if verified else "rule_based",
            "published" if verified else "approved",
            verified,
            PROOF if verified else None,
            published_at,
            list((roles or {}).keys()),
            json.dumps(roles or {}),
            record_id,
        ),
    )
    return article_id


def correct(conn: Conn, article_id: uuid.UUID, kind: str, when: datetime) -> None:
    conn.execute(
        """
        INSERT INTO article_corrections
            (article_id, correction_type, original_text, corrected_text,
             explanation, editor_signature, corrected_at)
        VALUES (%s, %s, 'old', 'new', 'Fixed a figure.', 'Desk Editor', %s)
        """,
        (article_id, kind, when),
    )


def counts(conn: Conn) -> tuple[Any, ...]:
    row = conn.execute(
        "SELECT (SELECT count(*) FROM articles), (SELECT count(*) FROM article_corrections)"
    ).fetchone()
    assert row is not None
    return row


def test_run_audit_reads_only_the_windows_public_articles(
    db: Conn, ledger_dsn: str, tmp_path: Path
) -> None:
    inside = WINDOW.start + timedelta(days=3)
    first = insert_article(
        db,
        "inside",
        "oregon_legislature",
        inside,
        roles={"S-1": "official", "S-2": "affected_party"},
    )
    insert_article(db, "lead", "misc", inside, triage_domain="science_health")
    insert_article(db, "before", "science_health", WINDOW.start - timedelta(seconds=1))
    insert_article(db, "at-end", "science_health", WINDOW.end)
    insert_article(db, "unverified", "science_health", inside, verified=False)
    correct(db, first, "factual_correction", inside + timedelta(days=1))
    correct(db, first, "clarification", WINDOW.end + timedelta(days=1))  # after window
    before = counts(db)

    path = run_audit(ledger_dsn, "2026-10", tmp_path, clock=lambda: GENERATED)

    assert path == tmp_path / "2026-10.json"
    report = json.loads(path.read_text())
    assert report["articles"]["total"] == 2
    domains = {row["domain"]: row["articles"] for row in report["domain_coverage"]}
    assert domains["government_law"] == 1 and domains["science_health"] == 1
    assert report["source_roles"]["citations"] == 2
    assert report["corrections"]["factual_corrections"] == 1
    assert report["corrections"]["clarifications"] == 0
    assert counts(db) == before  # the audit wrote nothing to the ledger


def test_audit_reads_in_a_read_only_transaction(
    db: Conn, ledger_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    insert_article(db, "inside", "science_health", WINDOW.start + timedelta(days=1))
    before = counts(db)
    writing = "DELETE FROM articles RETURNING article_id"
    monkeypatch.setattr(bias_audit, "_ARTICLES", writing)
    with (
        psycopg.connect(ledger_dsn) as conn,
        pytest.raises(psycopg.errors.ReadOnlySqlTransaction),
    ):
        bias_audit.read_window(conn, WINDOW)
    assert counts(db) == before


def test_main_reports_a_bad_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(BASELINE_ENV, str(tmp_path / "absent.json"))
    code = bias_audit.main(["--period", "2026-10", "--out-dir", str(tmp_path)])
    assert code == 2
    assert "cannot read baseline" in capsys.readouterr().err
    assert not (tmp_path / "2026-10.json").exists()
