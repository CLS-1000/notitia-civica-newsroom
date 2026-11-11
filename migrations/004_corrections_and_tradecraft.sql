-- Notitia Civica, Five-Mask Standards: corrections and analytic tradecraft.
--
-- uncertainty_disclosures already exists (002). This adds the editor's
-- affirmation that a draft has no open questions, recorded at sign-off, and an
-- append-only corrections log. A correction never edits the signed article, so
-- the editorial proof hash keeps covering the text that was signed.

BEGIN;

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS no_open_questions_confirmed BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS article_corrections (
    correction_id    UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id       UUID         NOT NULL
                                  REFERENCES articles (article_id) ON DELETE RESTRICT,
    correction_type  VARCHAR(32)  NOT NULL,
    original_text    TEXT,
    corrected_text   TEXT,
    explanation      TEXT         NOT NULL,
    editor_signature VARCHAR(128) NOT NULL,
    corrected_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT correction_type_is_known CHECK (
        correction_type IN ('clarification', 'factual_correction', 'methodology_update')
    ),
    CONSTRAINT correction_is_explained CHECK (btrim(explanation) <> ''),
    CONSTRAINT correction_is_signed CHECK (btrim(editor_signature) <> ''),
    -- A factual correction must show readers what changed.
    CONSTRAINT factual_correction_shows_change CHECK (
        correction_type <> 'factual_correction'
        OR (
            btrim(COALESCE(original_text, '')) <> ''
            AND btrim(COALESCE(corrected_text, '')) <> ''
        )
    )
);

CREATE INDEX IF NOT EXISTS article_corrections_article_idx
    ON article_corrections (article_id, corrected_at);

-- WORM: a published correction is part of the public record. A mistaken
-- correction is fixed by appending another one, never by editing this one.
CREATE OR REPLACE FUNCTION block_correction_mutation() RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'article_corrections is append-only; % is prohibited', TG_OP
        USING ERRCODE = 'restrict_violation';
END;
$$;

DROP TRIGGER IF EXISTS trg_immutable_article_corrections ON article_corrections;
CREATE TRIGGER trg_immutable_article_corrections
    BEFORE UPDATE OR DELETE ON article_corrections
    FOR EACH ROW EXECUTE FUNCTION block_correction_mutation();

DROP TRIGGER IF EXISTS trg_immutable_article_corrections_truncate
    ON article_corrections;
CREATE TRIGGER trg_immutable_article_corrections_truncate
    BEFORE TRUNCATE ON article_corrections
    FOR EACH STATEMENT EXECUTE FUNCTION block_correction_mutation();

COMMIT;
