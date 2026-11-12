"""Manual drafting: no API key, Claude by hand.

When ANTHROPIC_API_KEY is unset or still a placeholder, the desk cannot call the
API. ManualDrafter writes the exact request ClaudeDrafter would have sent (system
prompt, output schema, sealed sources) to a prompt file for the editor to give
Claude. The editor saves Claude's JSON reply beside it and clicks Draft again.
The reply then faces the same checks as an API draft: schema validation and the
invented-ID check here, then the gates and a human sign-off.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from newsroom.penfield_desk.claude_drafter import (
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    ClaudeDrafter,
    DraftError,
    DraftResult,
    DraftSource,
    _invented_ids,
    build_user_message,
)

MANUAL_DIR: Final = Path(os.environ.get("MANUAL_DRAFT_DIR", "data/manual_drafts"))
_FENCE: Final = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def has_api_key() -> bool:
    """A real key, not empty and not a redacted placeholder like sk-ant-..."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return bool(key) and not key.endswith("...")


def request_id(lead_prompt: str, sources: list[DraftSource]) -> str:
    material = build_user_message(lead_prompt, sources)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def render_prompt(lead_prompt: str, sources: list[DraftSource]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Reply with only one JSON object matching this schema: no prose before or "
        "after it, and no code fence.\n\n"
        f"{json.dumps(OUTPUT_SCHEMA, indent=2)}\n\n"
        f"{build_user_message(lead_prompt, sources)}\n"
    )


class ManualDrafter(ClaudeDrafter):
    def __init__(self, root: Path = MANUAL_DIR) -> None:
        super().__init__()
        self._root = root

    def draft(self, lead_prompt: str, sources: list[DraftSource]) -> DraftResult:
        """Return the editor-supplied draft, or write the prompt and raise DraftError."""
        if not sources:
            raise DraftError("a draft needs at least one source")
        rid = request_id(lead_prompt, sources)
        self._root.mkdir(parents=True, exist_ok=True)
        prompt_path = (self._root / f"{rid}.prompt.md").resolve()
        reply_path = (self._root / f"{rid}.reply.json").resolve()

        raw = reply_path.read_text(encoding="utf-8").strip() if reply_path.exists() else ""
        if not raw:
            prompt_path.write_text(render_prompt(lead_prompt, sources), encoding="utf-8")
            print(f"[manual draft] prompt ready: {prompt_path}", flush=True)
            raise DraftError(
                f"Manual mode: give {prompt_path} to Claude, save its JSON reply as "
                f"{reply_path}, then click Draft again."
            )

        try:
            draft = DraftResult.model_validate_json(_FENCE.sub("", raw))
        except ValidationError as error:
            raise DraftError(f"reply in {reply_path.name} did not match the schema: {error}") from error

        invented = _invented_ids(draft, {source.source_id for source in sources})
        if invented:
            raise DraftError(f"reply cited unknown sources: {', '.join(invented)}")
        return draft


def default_drafter() -> ClaudeDrafter:
    """The API drafter when a real key is set, the manual one otherwise."""
    return ClaudeDrafter() if has_api_key() else ManualDrafter()
