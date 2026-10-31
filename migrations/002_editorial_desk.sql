-- Notitia Civica, Phase 2: what the editorial desk must keep to prove a sign-off.
-- signed_at and source_ids are inputs to editorial_proof_hash, so the hash cannot
-- be recomputed from a row without them.

BEGIN;

ALTER TABLE articles
    ADD COLUMN triage_record_id        UUID REFERENCES triage_records (record_id),
    ADD COLUMN source_ids              VARCHAR(32)[] NOT NULL DEFAULT '{}',
    ADD COLUMN source_roles            JSONB         NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN verified_claims         JSONB         NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN uncertainty_disclosures JSONB         NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN is_private_person       BOOLEAN       NOT NULL DEFAULT FALSE,
    ADD COLUMN right_of_reply_logged   BOOLEAN       NOT NULL DEFAULT FALSE,
    ADD COLUMN sign_off_checks         JSONB,
    ADD COLUMN signed_at               TEXT;

CREATE INDEX articles_triage_record_idx ON articles (triage_record_id);

COMMIT;
