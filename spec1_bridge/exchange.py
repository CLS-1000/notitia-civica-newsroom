"""Ecosystem Exchange Serialization Standard, schema version 1.

Every record one CLS-1000 component hands to another travels in an
ExchangeEnvelope: who produced it, what kind of record it is, its ID, when it
was made, and the record itself as a JSON object. The envelope is frozen and
rejects unknown fields, so a consumer can trust that what it validated is what
it reads.

docs/ecosystem-exchange.md describes the wire format and what each producer
must do. schemas/exchange-v1.schema.json is the same contract as JSON Schema,
for components that are not written in Python. It is generated from this
module, and a test fails if the committed file drifts:

    uv run python -m spec1_bridge.exchange            # rewrite the schema file
    uv run python -m spec1_bridge.exchange --check    # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, get_args

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
)

SCHEMA_VERSION: Final = "1"
SCHEMA_PATH: Final = (
    Path(__file__).resolve().parent.parent / "schemas" / "exchange-v1.schema.json"
)
JSON_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"

Producer = Literal[
    "spec-1", "pdx-1i", "notitia-civica", "newsroom", "gatepost", "proc_track"
]
PRODUCERS: Final[tuple[str, ...]] = get_args(Producer)


class ExchangeEnvelope(BaseModel):
    """One record on the wire between ecosystem components."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1"] = Field(
        default=SCHEMA_VERSION,
        description="Envelope format version. Only '1' exists; a consumer must "
        "refuse any other value.",
    )
    producer: Producer = Field(description="The component that made the record.")
    record_type: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="What the payload is, in snake_case, such as 'intelligence', "
        "'campaign_finance', 'chamber_flow' or 'brief'.",
    )
    record_id: str = Field(
        min_length=1,
        max_length=256,
        description="The producer's own stable ID for the record. A producer that "
        "re-sends a record sends the same ID.",
    )
    created_at: AwareDatetime = Field(
        description="When the producer made the record, with a UTC offset. Offsets "
        "other than UTC are accepted and normalized to UTC.",
    )
    payload: dict[str, JsonValue] = Field(
        description="The record itself. Its shape belongs to the producer and the "
        "record_type, not to this envelope.",
    )

    @field_validator("created_at")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    def to_json(self) -> str:
        """Serialize to one line of JSON, suitable for a JSON Lines stream."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, text: str | bytes) -> ExchangeEnvelope:
        """Parse and validate one envelope. Raises pydantic.ValidationError."""
        return cls.model_validate_json(text)


def json_schema() -> dict[str, Any]:
    """Return the JSON Schema for ExchangeEnvelope, with its dialect declared."""
    return {"$schema": JSON_SCHEMA_DIALECT, **ExchangeEnvelope.model_json_schema()}


def render_schema() -> str:
    return json.dumps(json_schema(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the exchange v1 JSON Schema.")
    parser.add_argument("--path", type=Path, default=SCHEMA_PATH)
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the file is out of date"
    )
    args = parser.parse_args(argv)

    rendered = render_schema()
    if args.check:
        current = args.path.read_text(encoding="utf-8") if args.path.exists() else ""
        if current != rendered:
            print(f"{args.path} is out of date; run python -m spec1_bridge.exchange")
            return 1
        return 0
    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
