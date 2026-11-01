-- Notitia Civica, Phase 3: a retraction is a public record, so it stays in the
-- ledger with its reason. The proof hash is untouched: it still covers the text
-- that was signed.

BEGIN;

ALTER TABLE articles
    ADD COLUMN retraction_reason TEXT,
    ADD COLUMN retracted_at      TIMESTAMPTZ,
    ADD CONSTRAINT retraction_is_complete CHECK (
        (retraction_reason IS NULL) = (retracted_at IS NULL)
        AND (retraction_reason IS NULL OR btrim(retraction_reason) <> '')
    ),
    ADD CONSTRAINT retraction_needs_retracted_status CHECK (
        retracted_at IS NULL OR status = 'retracted'
    );

-- Once recorded, a retraction cannot be reworded, reversed, or deleted.
CREATE FUNCTION protect_retraction_record() RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.retracted_at IS NULL THEN
        RETURN COALESCE(NEW, OLD);
    END IF;

    IF TG_OP = 'DELETE'
       OR NEW.retracted_at IS DISTINCT FROM OLD.retracted_at
       OR NEW.retraction_reason IS DISTINCT FROM OLD.retraction_reason
       OR NEW.status IS DISTINCT FROM OLD.status THEN
        RAISE EXCEPTION 'retraction of article % is permanent', OLD.slug
            USING ERRCODE = 'restrict_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER articles_protect_retraction
    BEFORE UPDATE OR DELETE ON articles
    FOR EACH ROW EXECUTE FUNCTION protect_retraction_record();

COMMIT;
