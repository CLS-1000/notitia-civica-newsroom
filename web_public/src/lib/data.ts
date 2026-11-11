// Loads the JSON that scripts/export_verified.py writes. NC_DATA_DIR points a build
// at the labelled sample fixtures, and every page then shows a sample-data banner.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

export interface Source {
  source_id: string;
  url: string | null;
  domain: string;
  tier: string;
  raw_sha256: string;
  archive_url: string | null;
  retrieved_at: string;
}

export interface VerifiedClaim {
  claim: string;
  source_ids: string[];
}

export interface Steelman {
  for_label: string;
  for_text: string;
  against_label: string;
  against_text: string;
}

export interface Quarantine {
  domain: 5 | 6;
  label: string;
  text: string;
}

export interface Retraction {
  reason: string;
  retracted_at: string;
}

// One logged right-of-reply request. statement is the Five-Mask sentence the
// exporter writes, so every page words it the same way.
export interface Reply {
  subject_name: string;
  subject_type: "individual" | "institution";
  contact_method: string;
  requested_at: string;
  deadline_at: string;
  response_status:
    | "responded"
    | "no_response_by_deadline"
    | "declined_to_comment"
    | "pending";
  response_summary: string | null;
  statement: string;
}

export interface Article {
  slug: string;
  status: "published" | "retracted";
  retraction: Retraction | null;
  section: string;
  badge_type: string;
  headline: string;
  deck: string | null;
  body_markdown: string;
  editor_name: string;
  editorial_proof_hash: string;
  signed_at: string;
  published_at: string;
  verified_claims: VerifiedClaim[];
  uncertainty_disclosures: string[];
  evidence_bar: number;
  sources: Source[];
  replies: Reply[];
  steelman?: Steelman;
  quarantine?: Quarantine;
}

export type CorrectionType = "clarification" | "factual_correction" | "methodology_update";

// Written by scripts/export_verified.py from the append-only article_corrections
// table. A clarification or methodology update may have no before or after text.
export interface Correction {
  correction_id: string;
  slug: string;
  headline: string;
  correction_type: CorrectionType;
  corrected_at: string;
  reason: string;
  before: string | null;
  after: string | null;
  editor_name: string;
}

const CORRECTION_LABELS: Record<CorrectionType, string> = {
  clarification: "Clarification",
  factual_correction: "Correction",
  methodology_update: "Methodology update",
};

export function correctionLabel(correction: Correction): string {
  return CORRECTION_LABELS[correction.correction_type] ?? "Correction";
}

export interface Measure {
  measure: string;
  session: string;
  docket: string;
  headline: string;
  slug: string;
  published_at: string;
  primary_sources: number;
}

const DATA_DIR = process.env.NC_DATA_DIR ?? "src/data";
export const IS_SAMPLE_BUILD = process.env.NC_DATA_DIR !== undefined;

function load<T>(name: string): T {
  return JSON.parse(readFileSync(join(DATA_DIR, name), "utf8")) as T;
}

export const articles: Article[] = load<Article[]>("articles.json").toSorted(
  (a, b) => b.published_at.localeCompare(a.published_at),
);

// Retracted articles keep their pages, but never appear as current reporting.
export const liveArticles = articles.filter((item) => item.status === "published");
export const retractedArticles = articles
  .filter((item) => item.status === "retracted")
  .toSorted((a, b) =>
    (b.retraction?.retracted_at ?? "").localeCompare(a.retraction?.retracted_at ?? ""),
  );

export const corrections: Correction[] = load<Correction[]>(
  "corrections.json",
).toSorted((a, b) => b.corrected_at.localeCompare(a.corrected_at));

export function correctionsFor(slug: string): Correction[] {
  return corrections.filter((item) => item.slug === slug);
}

// Written by newsroom/dors_venabili/bias_audit.py, one file per YYYY-MM period.
export interface DomainCoverage {
  domain: string;
  articles: number;
  share_pct: number | null;
  target_pct: number;
  gap_pct_points: number | null;
  sigma: number | null;
  flagged: boolean;
}

export interface RoleGroup {
  group: string;
  roles: string[];
  citations: number;
  share_pct: number | null;
}

export interface BiasAudit {
  period: string;
  window: { start: string; end: string };
  generated_at: string;
  articles: { total: number; retracted: number };
  baseline: { reference_source: string; max_allowed_divergence_sigma: number };
  domain_coverage: DomainCoverage[];
  unmapped_domains: { value: string; articles: number }[];
  source_roles: {
    citations: number;
    roles: { role: string; citations: number; share_pct: number | null }[];
    groups: RoleGroup[];
  };
  corrections: {
    factual_corrections: number;
    clarifications: number;
    methodology_updates: number;
    articles_corrected_or_clarified: number;
    factual_correction_rate_pct: number | null;
    clarification_rate_pct: number | null;
  };
  notes: string[];
}

const AUDIT_PERIOD = /^\d{4}-\d{2}\.json$/;

function loadAudits(): BiasAudit[] {
  const dir = join(DATA_DIR, "audits");
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((name) => AUDIT_PERIOD.test(name))
    .map((name) => load<BiasAudit>(join("audits", name)))
    .toSorted((a, b) => b.period.localeCompare(a.period));
}

export const audits: BiasAudit[] = loadAudits();

export function auditUrl(period: string): string {
  return `/trust/audits/${period}/`;
}

export function formatPercent(value: number | null): string {
  return value === null ? "n/a" : `${value.toFixed(1)}%`;
}

export function articleUrl(slug: string): string {
  return `/citizens-cognisance/${slug}/`;
}

export function formatDate(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "long",
    timeZone: "UTC",
  }).format(new Date(value));
}

export function formatDateTime(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  }).format(new Date(value));
}

const TIER_LABELS: Record<string, string> = {
  tier_1_primary: "Tier 1 primary",
  tier_2_institutional: "Tier 2 institutional",
  tier_3_secondary: "Tier 3 secondary",
  tier_4_unverified: "Tier 4 unverified",
};

export function tierLabel(tier: string): string {
  return TIER_LABELS[tier] ?? tier;
}

export function tierClass(tier: string): string {
  return `tier-${tier.match(/\d/)?.[0] ?? "4"}`;
}

export function primaryCount(article: Article): number {
  return article.sources.filter((item) => item.tier === "tier_1_primary").length;
}

const OLIS_DOCUMENT =
  /olis\.oregonlegislature\.gov\/liz\/(\d{4}[A-Z]\d)\/Downloads\/MeasureDocument\/([A-Z]+\d+)\b/;

// Measures come only from published articles, through their OLIS document sources.
export function trackedMeasures(items: Article[]): Measure[] {
  return items.flatMap((article) => {
    for (const source of article.sources) {
      const match = source.url?.match(OLIS_DOCUMENT);
      if (match?.[1] && match[2]) {
        const [, session, measure] = match;
        return [
          {
            measure: measure.replace(/^([A-Z]+)(\d+)$/, "$1 $2"),
            session,
            docket: `https://olis.oregonlegislature.gov/liz/${session}/Measures/Overview/${measure}`,
            headline: article.headline,
            slug: article.slug,
            published_at: article.published_at,
            primary_sources: primaryCount(article),
          },
        ];
      }
    }
    return [];
  });
}
