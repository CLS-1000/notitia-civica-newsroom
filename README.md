# Notitia Civica newsroom

The engine behind Notitia Civica, a nonpartisan civic-records publication. It takes
leads from public-record sources, seals every source it relies on, runs drafts through
deterministic gates, and publishes only what a human editor signs off. The public
reader site is built from the signed-off articles alone.

## How it fits together

1. **Intake** (`spec1_bridge/`). `mycroft_triage` runs leads from the adapters
   (`spec1_bridge/adapters/`) and from the spec-1, pdx-1i and proc_track importers
   through four gates (deduplication, authority, corroboration, freshness/anomaly) and
   writes the ones that pass to `triage_records`. A local Ollama model adds notes for
   the editor but never decides whether a lead passes. `moby_runner` registers a
   spec-1 harvest after checking each file against disk.
2. **Custody** (`newsroom/harlan_eternity/custody.py`). Each source is fetched once and
   its exact bytes are sealed under `archive/warc/`, with a SHA-256 hash in
   `source_registry`.
3. **Gates** (`newsroom/dors_venabili/gatekeeper.py`). Headline, loaded-language,
   quote and grounding, PII and defamation, and source-diversity checks. They call no
   model, so the same draft always gets the same verdict, and any issue blocks
   publication.
4. **Penfield Desk** (`newsroom/penfield_desk/`). A local editorial workstation on
   `127.0.0.1:8000` for one editor on one machine. It can ask Claude for a first draft
   from the sealed sources; the draft decides nothing.
5. **Sign-off** (`newsroom/harlan_eternity/signoff.py`). The only code path that
   publishes. It re-runs every check inside the publishing transaction, and a database
   trigger is the final barrier.
6. **Public site** (`web_public/`). An Astro site. `web_public/scripts/export_verified.py`
   is the only bridge from the ledger: it exports published or retracted articles that
   are `llm_verified` and human-approved, recomputes each proof hash, and fails the run
   if an article changed after sign-off.

The ledger schema is in `migrations/` (PostgreSQL 18 with pgvector).

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- PostgreSQL 18 with the `vector` and `pgcrypto` extensions
- Node 24 for `web_public/`
- Optional: a local [Ollama](https://ollama.com/) server for triage notes, and an
  Anthropic API key for the desk's drafter

## Run it

```bash
uv sync
for f in migrations/*.sql; do psql -d radiant_ledger -f "$f"; done

# Triage leads into the ledger (DSN defaults to $DATABASE_URL, else radiant_ledger)
uv run python -m spec1_bridge.mycroft_triage --help
uv run python -m spec1_bridge.mycroft_triage --pdx1-store path/to/pdx1_signals.jsonl

# Open the editorial desk at http://127.0.0.1:8000
uv run python -m newsroom.penfield_desk.app

# Export signed-off articles and build the public site
uv run python web_public/scripts/export_verified.py --out web_public/src/data
cd web_public && npm ci && npm run build
```

`npm run build:sample` builds the site from the labelled sample fixtures in
`web_public/fixtures/` instead of the ledger.

## Tests

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

The tests start a throwaway PostgreSQL cluster, so they need the PostgreSQL 18 server
binaries and pgvector. If pgvector isn't installed system-wide, run
`scripts/bootstrap_test_pgvector.sh` once, or point `NOTITIA_TEST_DSN` at a server that
has it. CI (`.github/workflows/ci_gates.yml`) checks the public site: CSS lint, type
check, both builds, the 80 KB page budget, and pa11y.

## Ownership and rights

Notitia Civica is published by EVASTARARCANA LLC, a commercial entity, as stated on the
site's [ownership page](web_public/src/pages/trust/ownership.astro).

© EVASTARARCANA LLC. All rights reserved. This is not open source. Issues and data
corrections are welcome; outside pull requests are not accepted.
