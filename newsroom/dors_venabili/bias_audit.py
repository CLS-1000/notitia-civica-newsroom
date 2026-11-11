"""Monthly public bias and sourcing scorecard.

Reads the trailing 30 days of published reporting from radiant_ledger and writes
one JSON report per period for the reader site
(web_public/src/pages/trust/audits/[period].astro). It reports three things:

1. Domain coverage: each article's share of the window by civic domain, against
   the target weights in a baseline file. An article's domain is its lead's
   subject_domain, or its section when it has no lead, mapped onto the
   baseline's domains by DOMAIN_MAP. A domain is flagged when its article count
   is further from its target than max_allowed_divergence_sigma binomial
   standard deviations: z = (count - n*p) / sqrt(n*p*(1-p)), where n is the
   window's article count and p the domain's target weight. Values DOMAIN_MAP
   does not know are listed as unmapped, never forced into a domain.
2. Source roles: every citation (an article's source_id) counted by the role the
   desk assigned it, and grouped into government, academic/expert, frontline
   (affected parties), industry, and advocacy. The desk's roles do not separate
   tradespeople from other affected parties, so the report says so rather than
   claiming a trades share.
3. Corrections: factual corrections and clarifications recorded against the
   window's articles up to the end of the window, as rates per article.

Only articles the public export would publish are read: status published or
retracted, analysis_tier llm_verified, and human_approved. The query runs in a
read-only transaction.

The baseline is newsroom/dors_venabili/civic_baselines.json unless the
CIVIC_BASELINE_PATH environment variable names another file. It must hold a
non-empty reference_source, domain_target_weights summing to 100, and a
positive max_allowed_divergence_sigma.

Usage:
    python -m newsroom.dors_venabili.bias_audit --period 2026-10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import psycopg
from psycopg.rows import dict_row

DEFAULT_DSN: Final = "dbname=radiant_ledger"
DEFAULT_BASELINE_PATH: Final = Path(__file__).resolve().parent / "civic_baselines.json"
BASELINE_ENV: Final = "CIVIC_BASELINE_PATH"
DEFAULT_OUT_DIR: Final = (
    Path(__file__).resolve().parents[2] / "web_public" / "src" / "data" / "audits"
)
WINDOW: Final = timedelta(days=30)
WEIGHT_TOTAL: Final = 100.0
WEIGHT_TOLERANCE: Final = 0.01
UNASSIGNED: Final = "unassigned"

# Lead subject domains and desk sections, onto the baseline's civic domains.
DOMAIN_MAP: Final[Mapping[str, str]] = {
    "supply_chain": "economy_supply_chain",
    "oregon_legislature": "government_law",
    "federal_legislation": "government_law",
    "science_health": "science_health",
    "border_crossings": "infrastructure",
    "hydrology": "energy_environment",
    "security_defense": "security_defense",
    "narrative_integrity": "technology_cyber",
    "anomalous_lore": "culture_anomalies",
    "esoteric_philosophy": "culture_anomalies",
    "archaeological_antiquity": "culture_anomalies",
}

# The desk's source roles (newsroom/penfield_desk/app.py), grouped for readers.
ROLE_GROUPS: Final[Mapping[str, tuple[str, ...]]] = {
    "government": ("government_record", "official"),
    "academic_expert": ("independent_expert",),
    "frontline": ("affected_party",),
    "industry": ("industry",),
    "advocacy": ("advocacy",),
    UNASSIGNED: (UNASSIGNED,),
}

ROLE_NOTE: Final = (
    "Frontline counts sources the desk recorded as affected parties; the desk's "
    "roles do not separate tradespeople from other affected parties."
)

_ARTICLES: Final = """
    SELECT a.article_id, a.slug, a.section, t.subject_domain, a.status,
           a.published_at, a.source_ids, a.source_roles
    FROM articles a
    LEFT JOIN triage_records t ON t.record_id = a.triage_record_id
    WHERE a.status IN ('published', 'retracted')
      AND a.analysis_tier = 'llm_verified'
      AND a.human_approved = TRUE
      AND a.published_at >= %(start)s
      AND a.published_at < %(end)s
    ORDER BY a.published_at, a.slug
"""

_CORRECTIONS: Final = """
    SELECT article_id, correction_type
    FROM article_corrections
    WHERE article_id = ANY(%(ids)s)
      AND corrected_at < %(end)s
"""


class BaselineError(ValueError):
    """The baseline file is missing or does not hold what the audit needs."""


@dataclass(frozen=True, slots=True)
class Baseline:
    reference_source: str
    weights: dict[str, float]
    max_sigma: float


@dataclass(frozen=True, slots=True)
class Window:
    period: str
    start: datetime
    end: datetime


def baseline_path(environ: Mapping[str, str] | None = None) -> Path:
    override = (environ if environ is not None else os.environ).get(BASELINE_ENV)
    return Path(override) if override else DEFAULT_BASELINE_PATH


def load_baseline(path: Path | None = None) -> Baseline:
    """Read and check a baseline file (default: baseline_path())."""
    target = path if path is not None else baseline_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BaselineError(f"cannot read baseline {target}: {error}") from error
    if not isinstance(data, dict):
        raise BaselineError("baseline must be a JSON object")
    source = data.get("reference_source")
    if not isinstance(source, str) or not source.strip():
        raise BaselineError("reference_source must be a non-empty string")
    raw_weights = data.get("domain_target_weights")
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise BaselineError("domain_target_weights must be a non-empty object")
    weights: dict[str, float] = {}
    for domain, weight in raw_weights.items():
        if isinstance(weight, bool) or not isinstance(weight, int | float):
            raise BaselineError(f"weight for {domain!r} is not a number")
        if not math.isfinite(weight) or weight < 0:
            raise BaselineError(f"weight for {domain!r} must be zero or more")
        weights[str(domain)] = float(weight)
    total = sum(weights.values())
    if abs(total - WEIGHT_TOTAL) > WEIGHT_TOLERANCE:
        raise BaselineError(f"domain_target_weights sum to {total:g}, not 100")
    max_sigma = data.get("max_allowed_divergence_sigma")
    if (
        isinstance(max_sigma, bool)
        or not isinstance(max_sigma, int | float)
        or not math.isfinite(max_sigma)
        or max_sigma <= 0
    ):
        raise BaselineError("max_allowed_divergence_sigma must be a positive number")
    return Baseline(source.strip(), weights, float(max_sigma))


def window_for(period: str) -> Window:
    """The 30 days ending at the close of a YYYY-MM period (00:00 UTC on the 1st)."""
    try:
        month_start = datetime.strptime(period, "%Y-%m").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"period must be YYYY-MM, got {period!r}") from error
    year, month = divmod(month_start.month, 12)
    end = month_start.replace(year=month_start.year + year, month=month + 1)
    return Window(period, end - WINDOW, end)


def previous_period(now: datetime) -> str:
    first = now.astimezone(UTC).replace(day=1)
    return f"{first - timedelta(days=1):%Y-%m}"


def article_domain(article: Mapping[str, Any]) -> str:
    """The raw domain value an article is filed under."""
    value = article.get("subject_domain") or article.get("section") or ""
    return str(value).strip()


def _percent(part: float, whole: float) -> float | None:
    return None if whole <= 0 else round(100.0 * part / whole, 2)


def domain_coverage(
    articles: Sequence[Mapping[str, Any]], baseline: Baseline
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(per-domain coverage against target, unmapped values with counts)."""
    counts: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    for article in articles:
        raw = article_domain(article)
        domain = raw if raw in baseline.weights else DOMAIN_MAP.get(raw)
        if domain is None or domain not in baseline.weights:
            unmapped[raw or "(none)"] += 1
        else:
            counts[domain] += 1
    total = len(articles)
    rows: list[dict[str, Any]] = []
    for domain, weight in baseline.weights.items():
        count = counts[domain]
        p = weight / WEIGHT_TOTAL
        spread = math.sqrt(total * p * (1 - p))
        sigma = (count - total * p) / spread if spread > 0 else None
        share = _percent(count, total)
        rows.append(
            {
                "domain": domain,
                "articles": count,
                "share_pct": share,
                "target_pct": weight,
                "gap_pct_points": None if share is None else round(share - weight, 2),
                "sigma": None if sigma is None else round(sigma, 2),
                "flagged": sigma is not None and abs(sigma) > baseline.max_sigma,
            }
        )
    return rows, [
        {"value": value, "articles": count} for value, count in sorted(unmapped.items())
    ]


def source_roles(articles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    roles: Counter[str] = Counter()
    for article in articles:
        assigned = article.get("source_roles") or {}
        for source_id in article.get("source_ids") or []:
            role = assigned.get(source_id) if isinstance(assigned, dict) else None
            roles[str(role) if role else UNASSIGNED] += 1
    total = sum(roles.values())
    known = {role for members in ROLE_GROUPS.values() for role in members}
    groups = [
        {
            "group": group,
            "roles": list(members),
            "citations": sum(roles[role] for role in members),
            "share_pct": _percent(sum(roles[role] for role in members), total),
        }
        for group, members in ROLE_GROUPS.items()
    ]
    other = sum(count for role, count in roles.items() if role not in known)
    if other:
        groups.append(
            {
                "group": "other",
                "roles": sorted(role for role in roles if role not in known),
                "citations": other,
                "share_pct": _percent(other, total),
            }
        )
    return {
        "citations": total,
        "roles": [
            {"role": role, "citations": count, "share_pct": _percent(count, total)}
            for role, count in sorted(
                roles.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "groups": groups,
    }


def correction_rates(
    articles: Sequence[Mapping[str, Any]], corrections: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    kinds = Counter(str(row["correction_type"]) for row in corrections)
    corrected = {
        row["article_id"]
        for row in corrections
        if row["correction_type"] in ("factual_correction", "clarification")
    }
    total = len(articles)
    return {
        "factual_corrections": kinds["factual_correction"],
        "clarifications": kinds["clarification"],
        "methodology_updates": kinds["methodology_update"],
        "articles_corrected_or_clarified": len(corrected),
        "factual_correction_rate_pct": _percent(kinds["factual_correction"], total),
        "clarification_rate_pct": _percent(kinds["clarification"], total),
    }


def compute_audit(
    window: Window,
    articles: Sequence[Mapping[str, Any]],
    corrections: Sequence[Mapping[str, Any]],
    baseline: Baseline,
    generated_at: datetime,
) -> dict[str, Any]:
    """Build the report from rows already read. Pure, so it is testable offline."""
    coverage, unmapped = domain_coverage(articles, baseline)
    notes = [ROLE_NOTE]
    if not articles:
        notes.append("No articles were published in this window.")
    if unmapped:
        notes.append(
            "Some articles are filed under domains the baseline does not cover; "
            "they are listed as unmapped and counted in the total."
        )
    return {
        "period": window.period,
        "window": {"start": window.start.isoformat(), "end": window.end.isoformat()},
        "generated_at": generated_at.astimezone(UTC).isoformat(),
        "articles": {
            "total": len(articles),
            "retracted": sum(1 for row in articles if row.get("status") == "retracted"),
        },
        "baseline": {
            "reference_source": baseline.reference_source,
            "max_allowed_divergence_sigma": baseline.max_sigma,
        },
        "domain_coverage": coverage,
        "unmapped_domains": unmapped,
        "source_roles": source_roles(articles),
        "corrections": correction_rates(articles, corrections),
        "notes": notes,
    }


def read_window(
    conn: psycopg.Connection[Any], window: Window
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the window's public articles and their corrections, read-only."""
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute(_ARTICLES, {"start": window.start, "end": window.end})
        articles = cursor.fetchall()
        cursor.execute(
            _CORRECTIONS,
            {"ids": [row["article_id"] for row in articles], "end": window.end},
        )
        corrections = cursor.fetchall()
    return articles, corrections


def write_report(out_dir: Path, report: Mapping[str, Any]) -> Path:
    """Write <out_dir>/<period>.json atomically."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['period']}.json"
    descriptor, name = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temp.chmod(0o644)
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return path


def run_audit(
    dsn: str,
    period: str,
    out_dir: Path,
    *,
    baseline: Baseline | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    window = window_for(period)
    checked = baseline if baseline is not None else load_baseline()
    with psycopg.connect(dsn) as conn:
        articles, corrections = read_window(conn, window)
    report = compute_audit(window, articles, corrections, checked, clock())
    return write_report(out_dir, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL", DEFAULT_DSN))
    parser.add_argument(
        "--period", help="YYYY-MM; the window is the 30 days ending with it"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    period = args.period or previous_period(datetime.now(UTC))
    try:
        path = run_audit(args.dsn, period, args.out_dir)
    except (BaselineError, ValueError) as error:
        print(f"bias audit failed: {error}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
