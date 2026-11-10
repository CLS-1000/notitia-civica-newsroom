-- Notitia Civica, Five-Mask Standards: the source reliability learning loop.
--
-- When a factual correction or a retraction shows that an article went wrong,
-- every source it cited is flagged for a tier review, and later drafts citing
-- those sources show a warning. The flag lives on source_registry, so this
-- narrows that table's append-only rule: an UPDATE may now touch the three review
-- columns and nothing else. Every evidence column stays immutable, and DELETE
-- and TRUNCATE stay refused.
--
-- A flag is one-way. flagged_at keeps the first flag's time, and a later reason
-- is appended to the earlier ones, never written over them. Clearing a flag
-- needs a recorded tier review, which this migration does not provide.

BEGIN;

ALTER TABLE source_registry
    ADD COLUMN IF NOT EXISTS tier_review_flagged BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS tier_review_reason  TEXT,
    ADD COLUMN IF NOT EXISTS flagged_at          TIMESTAMPTZ;

ALTER TABLE source_registry
    DROP CONSTRAINT IF EXISTS tier_review_is_complete,
    ADD CONSTRAINT tier_review_is_complete CHECK (
        tier_review_flagged = (flagged_at IS NOT NULL)
        AND tier_review_flagged = (btrim(COALESCE(tier_review_reason, '')) <> '')
    );

CREATE OR REPLACE FUNCTION guard_source_registry_update() RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    review_columns CONSTANT TEXT[] :=
        ARRAY['tier_review_flagged', 'tier_review_reason', 'flagged_at'];
BEGIN
    -- Comparing whole rows minus the review columns keeps any column added
    -- later immutable by default.
    IF (to_jsonb(NEW) - review_columns) IS DISTINCT FROM (to_jsonb(OLD) - review_columns)
       OR (OLD.tier_review_flagged AND NOT NEW.tier_review_flagged)
       OR (OLD.flagged_at IS NOT NULL
           AND NEW.flagged_at IS DISTINCT FROM OLD.flagged_at)
       OR (OLD.tier_review_reason IS NOT NULL
           AND left(COALESCE(NEW.tier_review_reason, ''), length(OLD.tier_review_reason))
               IS DISTINCT FROM OLD.tier_review_reason) THEN
        RAISE EXCEPTION 'CRITICAL_AUDIT_VIOLATION: source_registry is an append-only '
            'ledger; only a tier-review flag may be added to a source.';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_immutable_source_registry ON source_registry;
CREATE TRIGGER trg_immutable_source_registry
    BEFORE DELETE ON source_registry
    FOR EACH ROW EXECUTE FUNCTION block_source_mutation();

DROP TRIGGER IF EXISTS trg_source_registry_review_only ON source_registry;
CREATE TRIGGER trg_source_registry_review_only
    BEFORE UPDATE ON source_registry
    FOR EACH ROW EXECUTE FUNCTION guard_source_registry_update();

CREATE OR REPLACE FUNCTION flag_sources_for_tier_review(
    source_ids TEXT[], reason TEXT
) RETURNS void
LANGUAGE sql
AS $$
    UPDATE source_registry
    SET tier_review_flagged = TRUE,
        flagged_at = COALESCE(flagged_at, now()),
        tier_review_reason = CASE
            WHEN tier_review_reason IS NULL THEN reason
            ELSE tier_review_reason || '; ' || reason
        END
    WHERE source_id = ANY(source_ids);
$$;

-- In the database, so a correction or retraction written by any path flags.
CREATE OR REPLACE FUNCTION flag_sources_after_factual_correction() RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    article RECORD;
BEGIN
    SELECT slug, source_ids INTO article FROM articles
    WHERE article_id = NEW.article_id;
    PERFORM flag_sources_for_tier_review(
        article.source_ids::TEXT[],
        format('factual correction to %s on %s: %s', article.slug,
               to_char(NEW.corrected_at AT TIME ZONE 'UTC', 'YYYY-MM-DD'),
               btrim(NEW.explanation))
    );
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS trg_flag_sources_after_factual_correction
    ON article_corrections;
CREATE TRIGGER trg_flag_sources_after_factual_correction
    AFTER INSERT ON article_corrections
    FOR EACH ROW WHEN (NEW.correction_type = 'factual_correction')
    EXECUTE FUNCTION flag_sources_after_factual_correction();

CREATE OR REPLACE FUNCTION flag_sources_after_retraction() RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM flag_sources_for_tier_review(
        NEW.source_ids::TEXT[],
        format('retraction of %s on %s: %s', NEW.slug,
               to_char(COALESCE(NEW.retracted_at, now()) AT TIME ZONE 'UTC',
                       'YYYY-MM-DD'),
               btrim(COALESCE(NEW.retraction_reason, 'no reason recorded')))
    );
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS trg_flag_sources_after_retraction ON articles;
CREATE TRIGGER trg_flag_sources_after_retraction
    AFTER UPDATE OF status ON articles
    FOR EACH ROW
    WHEN (NEW.status = 'retracted' AND OLD.status IS DISTINCT FROM 'retracted')
    EXECUTE FUNCTION flag_sources_after_retraction();

COMMIT;
