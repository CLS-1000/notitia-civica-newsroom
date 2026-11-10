"""Ecosystem Exchange Serialization Standard, schema version 1."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from conftest import REPO_ROOT
from pydantic import ValidationError

from spec1_bridge import exchange
from spec1_bridge.exchange import PRODUCERS, ExchangeEnvelope

CREATED = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
ECOSYSTEM = REPO_ROOT / "schemas" / "ecosystem.json"


def fields(**overrides: Any) -> dict[str, Any]:
    return {
        "producer": "pdx-1i",
        "record_type": "campaign_finance",
        "record_id": "rec_a",
        "created_at": CREATED,
        "payload": {"source_type": "ORESTAR", "amount": 25000, "tags": ["a", None]},
    } | overrides


def envelope(**overrides: Any) -> ExchangeEnvelope:
    return ExchangeEnvelope(**fields(**overrides))


# --- Serialization -----------------------------------------------------------


def test_serializes_to_one_json_line() -> None:
    line = envelope().to_json()
    assert "\n" not in line
    assert json.loads(line) == {
        "schema_version": "1",
        "producer": "pdx-1i",
        "record_type": "campaign_finance",
        "record_id": "rec_a",
        "created_at": "2026-10-15T12:00:00Z",
        "payload": {"source_type": "ORESTAR", "amount": 25000, "tags": ["a", None]},
    }


def test_round_trips() -> None:
    original = envelope()
    assert ExchangeEnvelope.from_json(original.to_json()) == original
    assert ExchangeEnvelope.from_json(original.to_json().encode()) == original


def test_jsonl_stream_round_trips() -> None:
    records = [envelope(record_id=f"rec_{n}") for n in range(3)]
    stream = "".join(item.to_json() + "\n" for item in records)
    parsed = [ExchangeEnvelope.from_json(line) for line in stream.splitlines()]
    assert parsed == records


def test_other_offsets_are_normalized_to_utc() -> None:
    pacific = timezone(timedelta(hours=-7))
    item = envelope(created_at=datetime(2026, 10, 15, 5, 0, tzinfo=pacific))
    assert item.created_at == CREATED
    assert item.created_at.utcoffset() == timedelta(0)
    assert json.loads(item.to_json())["created_at"] == "2026-10-15T12:00:00Z"


def test_offset_strings_are_accepted() -> None:
    line = (
        envelope()
        .to_json()
        .replace("2026-10-15T12:00:00Z", "2026-10-15T05:00:00-07:00")
    )
    assert ExchangeEnvelope.from_json(line).created_at == CREATED


def test_unicode_payload_survives() -> None:
    item = envelope(payload={"quote": "“Café” résumé — 東京"})
    assert ExchangeEnvelope.from_json(item.to_json()).payload == item.payload


# --- Validation --------------------------------------------------------------


def test_schema_version_defaults_to_one() -> None:
    assert envelope().schema_version == "1"
    line = json.dumps(json.loads(envelope().to_json()) | {"schema_version": "1"})
    assert ExchangeEnvelope.from_json(line).schema_version == "1"


@pytest.mark.parametrize("version", ["2", "1.0", 1, ""])
def test_unknown_schema_versions_are_refused(version: object) -> None:
    with pytest.raises(ValidationError):
        envelope(schema_version=version)


@pytest.mark.parametrize("producer", PRODUCERS)
def test_every_listed_producer_is_accepted(producer: str) -> None:
    assert envelope(producer=producer).producer == producer


@pytest.mark.parametrize("producer", ["spec1", "SPEC-1", "pdx1", "unknown", ""])
def test_unlisted_producers_are_refused(producer: str) -> None:
    with pytest.raises(ValidationError, match="producer"):
        envelope(producer=producer)


@pytest.mark.parametrize(
    "record_type", ["intelligence", "campaign_finance", "chamber_flow", "brief", "a1"]
)
def test_snake_case_record_types_are_accepted(record_type: str) -> None:
    assert envelope(record_type=record_type).record_type == record_type


@pytest.mark.parametrize(
    "record_type", ["", "Brief", "chamber-flow", "1brief", "has space", "x" * 65]
)
def test_malformed_record_types_are_refused(record_type: str) -> None:
    with pytest.raises(ValidationError, match="record_type"):
        envelope(record_type=record_type)


@pytest.mark.parametrize("record_id", ["", "x" * 257])
def test_record_id_length_is_bounded(record_id: str) -> None:
    with pytest.raises(ValidationError, match="record_id"):
        envelope(record_id=record_id)


def test_naive_timestamps_are_refused() -> None:
    with pytest.raises(ValidationError, match="created_at"):
        envelope(created_at=datetime(2026, 10, 15, 12, 0))
    line = envelope().to_json().replace("2026-10-15T12:00:00Z", "2026-10-15T12:00:00")
    with pytest.raises(ValidationError, match="created_at"):
        ExchangeEnvelope.from_json(line)


@pytest.mark.parametrize(
    "payload", [[1, 2], "text", None, 3], ids=["list", "string", "null", "number"]
)
def test_payload_must_be_an_object(payload: object) -> None:
    with pytest.raises(ValidationError, match="payload"):
        envelope(payload=payload)


@pytest.mark.parametrize(
    "value", [b"bytes", object(), {1, 2}], ids=["bytes", "object", "set"]
)
def test_payload_must_hold_json_values(value: object) -> None:
    with pytest.raises(ValidationError, match="payload"):
        envelope(payload={"bad": value})


@pytest.mark.parametrize(
    "field", ["producer", "record_type", "record_id", "created_at", "payload"]
)
def test_every_field_but_the_version_is_required(field: str) -> None:
    values = fields()
    del values[field]
    with pytest.raises(ValidationError, match=field):
        ExchangeEnvelope(**values)


def test_extra_fields_are_refused() -> None:
    with pytest.raises(ValidationError, match="record_typ"):
        ExchangeEnvelope(**fields(record_typ="brief"))
    line = json.dumps(json.loads(envelope().to_json()) | {"signature": "x"})
    with pytest.raises(ValidationError, match="signature"):
        ExchangeEnvelope.from_json(line)


def test_envelopes_are_frozen() -> None:
    item = envelope()
    with pytest.raises(ValidationError, match="frozen"):
        item.record_id = "rec_b"
    with pytest.raises(ValidationError, match="frozen"):
        item.schema_version = "2"  # type: ignore[assignment]


def test_malformed_json_is_a_validation_error() -> None:
    with pytest.raises(ValidationError):
        ExchangeEnvelope.from_json("{not json")


# --- Schema generation -------------------------------------------------------


def test_committed_schema_matches_the_model() -> None:
    committed = exchange.SCHEMA_PATH.read_text(encoding="utf-8")
    assert committed == exchange.render_schema(), (
        "run: uv run python -m spec1_bridge.exchange"
    )


def test_schema_states_the_contract() -> None:
    schema = exchange.json_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "ExchangeEnvelope"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "producer",
        "record_type",
        "record_id",
        "created_at",
        "payload",
    ]
    properties = schema["properties"]
    assert set(properties) == {
        "schema_version",
        "producer",
        "record_type",
        "record_id",
        "created_at",
        "payload",
    }
    assert properties["schema_version"]["const"] == "1"
    assert properties["producer"]["enum"] == list(PRODUCERS)
    assert properties["record_type"]["pattern"] == "^[a-z][a-z0-9_]*$"
    assert properties["created_at"]["format"] == "date-time"
    assert properties["payload"]["type"] == "object"


def test_schema_cli_writes_and_checks(tmp_path: Path) -> None:
    target = tmp_path / "out" / "schema.json"
    assert exchange.main(["--path", str(target), "--check"]) == 1
    assert exchange.main(["--path", str(target)]) == 0
    assert target.read_text(encoding="utf-8") == exchange.render_schema()
    assert exchange.main(["--path", str(target), "--check"]) == 0
    target.write_text("{}\n", encoding="utf-8")
    assert exchange.main(["--path", str(target), "--check"]) == 1


def test_committed_schema_check_passes() -> None:
    assert exchange.main(["--check"]) == 0


# --- Ecosystem map -----------------------------------------------------------


def ecosystem() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(ECOSYSTEM.read_text(encoding="utf-8"))
    return data


def test_ecosystem_map_uses_only_known_producers() -> None:
    data = ecosystem()
    assert data["exchange_schema_version"] == "1"
    assert (REPO_ROOT / data["exchange_schema"]).exists()

    ids = [item["id"] for item in data["components"]]
    assert sorted(ids) == sorted(PRODUCERS)
    emits = {item["id"]: item["emits"] for item in data["components"]}
    for flow in data["flows"]:
        assert flow["from"] in PRODUCERS and flow["to"] in PRODUCERS
        if flow.get("envelope_applies", True):
            envelope(producer=flow["from"], record_type=flow["record_type"])
            assert flow["record_type"] in emits[flow["from"]]
        else:
            assert flow["record_type"] is None


def test_ecosystem_is_multi_repo_with_every_repository_named() -> None:
    data = ecosystem()
    assert data["architecture"] == "multi-repo"
    repositories = {item["id"]: item["repository"] for item in data["components"]}
    assert repositories == {
        "spec-1": "CLS-1000/spec-1",
        "pdx-1i": "CLS-1000/pdx-1i",
        "proc_track": "CLS-1000/proc_track",
        "gatepost": "CLS-1000/gatepost",
        "newsroom": "CLS-1000/notitia-civica-newsroom",
        "notitia-civica": "CLS-1000/notitia-civica",
    }


def test_proc_track_code_dependency_is_recorded_while_it_exists() -> None:
    """Fails when the exception is removed, so the map gets updated with it."""
    importer = (REPO_ROOT / "spec1_bridge" / "proc_track_importer.py").read_text(
        encoding="utf-8"
    )
    runs_proc_track_code = "importlib" in importer and "replay_harness" in importer
    recorded = [
        item
        for item in ecosystem()["code_dependencies"]
        if (item["from"], item["on"]) == ("newsroom", "proc_track")
    ]
    assert runs_proc_track_code
    assert len(recorded) == 1
    assert recorded[0]["where"] == "spec1_bridge/proc_track_importer.py"
    assert recorded[0]["status"] == "exception to remove"


def test_ecosystem_map_points_at_real_files() -> None:
    data = ecosystem()
    for flow in data["flows"]:
        for key in ("reader", "writer"):
            if flow.get(key):
                assert (REPO_ROOT / flow[key]).exists(), flow[key]
    for component in data["components"]:
        for key in ("paths", "source_in_newsroom"):
            for path in component.get(key, []):
                assert (REPO_ROOT / path).exists(), path
    for dependency in data["code_dependencies"]:
        assert (REPO_ROOT / dependency["where"]).exists()
