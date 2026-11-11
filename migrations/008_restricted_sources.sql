-- Notitia Civica: sources kept for Editor Eyes Only.
--
-- Some sealed sources carry a location that the law keeps from the public:
-- archaeological site coordinates (ARPA, 16 U.S.C. 470hh; NHPA section 304)
-- and records exempt from disclosure (ORS 192.355). A restricted source stays in
-- source_registry, so an article can still cite it and readers still see its
-- domain, tier, and SHA-256, but the public export leaves out its URL and its
-- archive link. Editors see everything in the Penfield Desk.
--
-- A restriction is one-way: rows cannot be changed or deleted, so a location
-- once withheld is never published by accident.

BEGIN;

CREATE TABLE IF NOT EXISTS restricted_sources (
    -- No foreign key: like articles.source_ids, this refers to source_registry
    -- without one, so TRUNCATE source_registry still meets its audit trigger.
    source_id     VARCHAR(32) PRIMARY KEY CHECK (source_id ~ '^S-[0-9]+$'),
    reason        TEXT        NOT NULL CHECK (btrim(reason) <> ''),
    restricted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION block_restricted_source_mutation() RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'CRITICAL_AUDIT_VIOLATION: restricted_sources is append-only; '
        'a restriction cannot be changed or lifted.';
END;
$$;

DROP TRIGGER IF EXISTS trg_restricted_sources_append_only ON restricted_sources;
CREATE TRIGGER trg_restricted_sources_append_only
    BEFORE UPDATE OR DELETE ON restricted_sources
    FOR EACH ROW EXECUTE FUNCTION block_restricted_source_mutation();

DROP TRIGGER IF EXISTS trg_restricted_sources_no_truncate ON restricted_sources;
CREATE TRIGGER trg_restricted_sources_no_truncate
    BEFORE TRUNCATE ON restricted_sources
    FOR EACH STATEMENT EXECUTE FUNCTION block_restricted_source_mutation();

COMMIT;
