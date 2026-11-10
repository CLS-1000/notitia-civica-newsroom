# Ecosystem Exchange Serialization Standard, v1

Every record that one CLS-1000 component hands to another travels in an
**ExchangeEnvelope**. The envelope says who made the record, what kind of
record it is, its ID, and when it was made. The record itself goes in
`payload`.

- Python model: `spec1_bridge/exchange.py` (`ExchangeEnvelope`)
- JSON Schema: [`schemas/exchange-v1.schema.json`](../schemas/exchange-v1.schema.json)
  (JSON Schema 2020-12, generated from the model)
- Component map: [`schemas/ecosystem.json`](../schemas/ecosystem.json)

> **Status.** Version 1 is defined, but no component emits or consumes it yet.
> Today each producer still writes its own format, and this repo's importers
> read those formats directly. `schemas/ecosystem.json` lists both the current
> format and the envelope's `record_type` for each flow, so components can
> adopt the envelope one at a time.

## Wire format

One envelope is one JSON object, encoded as UTF-8. A stream of envelopes is
JSON Lines: one envelope per line, with no blank lines between them.

```json
{"schema_version":"1","producer":"pdx-1i","record_type":"campaign_finance","record_id":"rec_a","created_at":"2026-10-15T12:00:00Z","payload":{"source_type":"ORESTAR","url":"https://example.gov/filing/123"}}
```

| Field | Type | Rule |
|---|---|---|
| `schema_version` | string | Always `"1"`. If it's missing, it's read as `"1"`. Any other value is refused. |
| `producer` | string | One of `spec-1`, `pdx-1i`, `notitia-civica`, `newsroom`, `gatepost`, `proc_track`. |
| `record_type` | string | snake_case, 1–64 characters, e.g. `intelligence`, `campaign_finance`, `chamber_flow`, `brief`. |
| `record_id` | string | 1–256 characters. The producer's own stable ID for the record. |
| `created_at` | string | RFC 3339 date-time **with an offset**. Times with no offset are refused. Other offsets are converted to UTC. Python writes UTC as `Z`. |
| `payload` | object | A JSON object. Its keys and shape belong to the producer and `record_type`. |

No other fields are allowed. An envelope with an extra field is invalid, not
silently trimmed, so a typo such as `record_typ` fails loudly.

In Python:

```python
from spec1_bridge.exchange import ExchangeEnvelope

line = envelope.to_json()  # one JSON Lines row
envelope = ExchangeEnvelope.from_json(line)  # raises pydantic.ValidationError
```

Envelopes are frozen once built. `payload` is a plain dict, so treat it as
read-only too.

## Producer responsibilities

1. **Use a stable `record_id`.** Re-sending the same record uses the same ID.
   Consumers treat `(producer, record_id)` as the record's identity, so a new
   ID makes a duplicate.
2. **Set `created_at` to when you made the record, not when you sent it.**
   Always include an offset.
3. **Use one `record_type` per payload shape.** If the shape changes in a way
   that breaks readers, use a new `record_type` (for example
   `campaign_finance_v2`) rather than changing the old one.
4. **Payloads hold JSON only.** No binary data. Link to sealed or public files
   by URL instead of embedding them.
5. **Include the evidence a consumer needs.** The newsroom seals a record's
   source before it can become a lead, so a record with no source URL is never
   imported. This is why `gatepost brief` Markdown is not imported today.
6. **Never put secrets in a payload:** no API keys, tokens or credentials.
7. **Mark what is unverified.** A model's hypothesis or classification goes in
   a field named for what it is (spec-1 uses `hypothesis` and
   `analysis_tier`), never in a field a consumer would read as fact.

## Consumer responsibilities

1. Validate every envelope before reading `payload`. In Python, use
   `ExchangeEnvelope.from_json`. Elsewhere, use the JSON Schema.
2. Refuse any `schema_version` you don't know. Don't guess.
3. Refuse a `record_type` you don't handle, rather than reading a payload
   whose shape you don't know.
4. De-duplicate on `(producer, record_id)`.

## Versioning

Version 1 is fixed. Any change to the envelope itself, including adding an
optional field, becomes version 2: a new model, a new
`schemas/exchange-v2.schema.json`, and a period when consumers accept both
versions. Changes to a payload are versioned through `record_type`, not
through the envelope.

## Keeping the schema in sync

`schemas/exchange-v1.schema.json` is generated from the model:

```sh
uv run python -m spec1_bridge.exchange          # rewrite it
uv run python -m spec1_bridge.exchange --check  # exit 1 if it is stale
```

`tests/test_exchange_envelope.py` fails if the committed file is out of date.
