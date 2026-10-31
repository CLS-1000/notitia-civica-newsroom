-- Notitia Civica, Phase 0: evidentiary ledger.
-- Target: PostgreSQL 18 with pgvector. Requires a role allowed to CREATE EXTENSION.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE analysis_tier_enum AS ENUM ('rule_based', 'llm_verified');

CREATE TYPE source_tier_enum AS ENUM (
    'tier_1_primary',
    'tier_2_institutional',
    'tier_3_secondary',
    'tier_4_unverified'
);

CREATE TYPE pub_status_enum AS ENUM (
    'harvested',
    'triaged',
    'drafting',
    'compliance_hold',
    'approved',
    'published',
    'retracted'
);

CREATE TABLE source_registry (
    source_id     VARCHAR(32)      PRIMARY KEY
                                   CHECK (source_id ~ '^S-[0-9]+$'),
    url           TEXT,            -- NULL when a harvest records only the domain
    domain        TEXT             NOT NULL,
    tier          source_tier_enum NOT NULL,
    raw_sha256    CHAR(64)         NOT NULL
                                   CHECK (raw_sha256 ~ '^[0-9a-f]{64}$'),
    archive_path  TEXT             NOT NULL,
    verbatim_text TEXT,            -- NULL for payloads with no text form (PDF, images)
    retrieved_at  TIMESTAMPTZ      NOT NULL
);

CREATE INDEX source_registry_domain_idx ON source_registry (domain);
CREATE INDEX source_registry_raw_sha256_idx ON source_registry (raw_sha256);

-- Append-only: a sealed source is evidence, so corrections are new rows.
CREATE OR REPLACE FUNCTION block_source_mutation()
RETURNS TRIGGER AS $$ BEGIN RAISE EXCEPTION 'CRITICAL_AUDIT_VIOLATION: source_registry is an append-only ledger; updates and deletions are prohibited.'; END; $$ LANGUAGE plpgsql;

CREATE TRIGGER trg_immutable_source_registry
BEFORE UPDATE OR DELETE ON source_registry
FOR EACH ROW EXECUTE FUNCTION block_source_mutation();

-- Allocates S-<n> ids for Moby harvests, whose records carry no source_id. Starts
-- high to stay clear of ids assigned by hand through fetch_and_seal_source().
CREATE SEQUENCE source_registry_harvest_id_seq START WITH 1000000;

CREATE TABLE triage_records (
    record_id             UUID               PRIMARY KEY DEFAULT gen_random_uuid(),
    adapter_name          TEXT               NOT NULL,
    subject_domain        TEXT               NOT NULL,
    anomaly_score         DOUBLE PRECISION   NOT NULL,
    decomposition_receipt JSONB              NOT NULL DEFAULT '{}'::jsonb,
    paste_ready_prompt    TEXT               NOT NULL,
    source_ids            VARCHAR(32)[]      NOT NULL DEFAULT '{}',
    analysis_tier         analysis_tier_enum NOT NULL DEFAULT 'rule_based',
    created_at            TIMESTAMPTZ        NOT NULL DEFAULT now()
);

CREATE INDEX triage_records_source_ids_idx ON triage_records USING GIN (source_ids);
CREATE INDEX triage_records_anomaly_score_idx ON triage_records (anomaly_score DESC);

CREATE TABLE articles (
    article_id           UUID               PRIMARY KEY DEFAULT gen_random_uuid(),
    slug                 TEXT               NOT NULL UNIQUE,
    section              TEXT               NOT NULL,
    badge_type           TEXT               NOT NULL,
    headline             TEXT               NOT NULL,
    deck                 TEXT,
    body_markdown        TEXT               NOT NULL,
    analysis_tier        analysis_tier_enum NOT NULL DEFAULT 'rule_based',
    status               pub_status_enum    NOT NULL DEFAULT 'drafting',
    human_approved       BOOLEAN            NOT NULL DEFAULT FALSE,
    editor_name          TEXT,
    -- A hex SHA-256 digest; without this format rule '' would satisfy the NULL gate.
    editorial_proof_hash CHAR(64)
                         CHECK (editorial_proof_hash ~ '^[0-9a-f]{64}$'),
    published_at         TIMESTAMPTZ
);

CREATE INDEX articles_status_idx ON articles (status);

-- Fires on every write that leaves a row published, not only on the transition
-- into 'published', so a live article cannot have its approval or proof removed.
CREATE FUNCTION enforce_publication_gates() RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    failed_gates TEXT[] := '{}';
BEGIN
    IF NEW.status IS DISTINCT FROM 'published' THEN
        RETURN NEW;
    END IF;

    -- Fail closed: anything other than the verified tier is refused.
    IF NEW.analysis_tier IS DISTINCT FROM 'llm_verified' THEN
        failed_gates := failed_gates || 'analysis_tier must be llm_verified'::TEXT;
    END IF;

    IF NEW.human_approved IS NOT TRUE THEN
        failed_gates := failed_gates || 'human_approved must be TRUE'::TEXT;
    END IF;

    IF NEW.editorial_proof_hash IS NULL THEN
        failed_gates := failed_gates || 'editorial_proof_hash must be set'::TEXT;
    END IF;

    IF cardinality(failed_gates) > 0 THEN
        RAISE EXCEPTION 'publication blocked for article %: %',
            NEW.article_id, array_to_string(failed_gates, '; ')
            USING ERRCODE = 'check_violation',
                  CONSTRAINT = 'enforce_publication_gates';
    END IF;

    NEW.published_at := COALESCE(NEW.published_at, now());
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_enforce_publication_gates
    BEFORE INSERT OR UPDATE ON articles
    FOR EACH ROW
    EXECUTE FUNCTION enforce_publication_gates();

-- Quarantine: narrative telemetry for Domains 5 and 6 stays out of the editorial
-- ledger. Nothing here references, or is referenced by, the public tables, and
-- access must be granted explicitly.
CREATE SCHEMA sociology_telemetry;
REVOKE ALL ON SCHEMA sociology_telemetry FROM PUBLIC;

CREATE TABLE sociology_telemetry.narrative_observations (
    observation_id  UUID             PRIMARY KEY DEFAULT gen_random_uuid(),
    domain          SMALLINT         NOT NULL CHECK (domain IN (5, 6)),
    narrative_label TEXT             NOT NULL,
    observed_text   TEXT             NOT NULL,
    source_ids      VARCHAR(32)[]    NOT NULL DEFAULT '{}',
    metrics         JSONB            NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector,          -- Dimension is left open until a model is chosen.
    observed_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX narrative_observations_domain_idx
    ON sociology_telemetry.narrative_observations (domain, observed_at DESC);

COMMIT;
