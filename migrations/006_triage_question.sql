-- Notitia Civica, Five-Mask Standards: the public-interest question.
--
-- Every lead states the question a story would answer and why the public has a
-- stake in it. The desk shows both on the lead, and a draft copies them from its
-- lead so the article keeps them if the lead is ever re-scored.

BEGIN;

ALTER TABLE triage_records
    ADD COLUMN IF NOT EXISTS public_question        TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS public_interest_reason TEXT NOT NULL DEFAULT '';

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS public_question        TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS public_interest_reason TEXT NOT NULL DEFAULT '';

COMMIT;
