-- Notitia Civica, Five-Mask Standards: the right-of-reply log.
--
-- allegation_subjects lists who a draft makes allegations against; the editor
-- declares them on the desk. Each needs a logged reply request before sign-off,
-- and none may still be inside its reply window. Without a reply record the
-- editor must attest the piece makes no adverse findings or allegations
-- (no_allegations_attested), which the gate refuses for an INVESTIGATION or a
-- body with adverse-finding terms. The log freezes when the
-- article is published, so the public reply statement always matches what was
-- signed. A response that arrives later is published as a correction.

BEGIN;

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS allegation_subjects JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS no_allegations_attested BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS article_reply_records (
    reply_id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id       UUID         NOT NULL
                                  REFERENCES articles (article_id) ON DELETE CASCADE,
    subject_name     VARCHAR(256) NOT NULL,
    subject_type     VARCHAR(32)  NOT NULL,
    contact_method   VARCHAR(64)  NOT NULL,
    requested_at     TIMESTAMPTZ  NOT NULL,
    deadline_at      TIMESTAMPTZ  NOT NULL,
    response_status  VARCHAR(32)  NOT NULL,
    response_summary TEXT,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT reply_subject_is_named CHECK (btrim(subject_name) <> ''),
    CONSTRAINT reply_subject_type_is_known CHECK (
        subject_type IN ('individual', 'institution')
    ),
    CONSTRAINT reply_contact_method_is_known CHECK (
        contact_method IN ('email', 'phone', 'formal_letter', 'spokesperson')
    ),
    CONSTRAINT reply_status_is_known CHECK (
        response_status IN (
            'responded', 'no_response_by_deadline', 'declined_to_comment', 'pending'
        )
    ),
    CONSTRAINT reply_deadline_follows_request CHECK (deadline_at > requested_at),
    -- Readers are told what the subject said, so a response needs a summary.
    CONSTRAINT reply_response_is_summarized CHECK (
        response_status <> 'responded' OR btrim(COALESCE(response_summary, '')) <> ''
    )
);

CREATE INDEX IF NOT EXISTS article_reply_records_status_idx
    ON article_reply_records (article_id, response_status);

CREATE OR REPLACE FUNCTION freeze_published_reply_log() RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    article_status pub_status_enum;
BEGIN
    SELECT status INTO article_status FROM articles
    WHERE article_id = COALESCE(NEW.article_id, OLD.article_id);
    -- An article being deleted is already gone, so its cascade passes.
    IF article_status IN ('published', 'retracted')
       OR (TG_OP = 'UPDATE' AND NEW.article_id IS DISTINCT FROM OLD.article_id) THEN
        RAISE EXCEPTION 'reply log of article % is frozen after publication',
            COALESCE(NEW.article_id, OLD.article_id)
            USING ERRCODE = 'restrict_violation';
    END IF;
    RETURN COALESCE(NEW, OLD);
END;
$$;

DROP TRIGGER IF EXISTS trg_freeze_published_reply_log ON article_reply_records;
CREATE TRIGGER trg_freeze_published_reply_log
    BEFORE INSERT OR UPDATE OR DELETE ON article_reply_records
    FOR EACH ROW EXECUTE FUNCTION freeze_published_reply_log();

COMMIT;
