"""Minimal stand-in for CLS-1000/proc_track's replay_harness.py, for tests.

Same interface as the real module (parse_row, replay, apply_corrections) with a
small rule table, so the importer can be tested offline and deterministically.
"""

import re
from dataclasses import dataclass, field

RULES = [
    ("introduction", r"Introduction and first reading", "introduced", {}),
    ("passed", r"Third reading\. Passed", "passed", {}),
    ("governor_signed_line_item", r"Governor signed with line-item veto", "enacted_line_item_veto", {}),
    ("governor_signed", r"Governor signed", "enacted", {}),
    ("governor_vetoed", r"Governor vetoed", "vetoed", {}),
    ("veto_overridden", r"Veto overridden[^.]*", "veto_overridden", {}),
    ("filed_without_signature", r"Filed without Governor.s signature", "enacted", {}),
]
COMPILED = [(rid, re.compile(pat), st, pl) for rid, pat, st, pl in RULES]

FLOW = {
    None: {"introduced"},
    "introduced": {"passed"},
    "passed": {"enacted", "enacted_line_item_veto", "vetoed"},
    "vetoed": {"veto_overridden"},
}

DATA_CORRECTIONS = {}


@dataclass
class Emit:
    state: object
    payload: dict
    rule_id: str
    span: tuple
    text: str


@dataclass
class Item:
    id: str
    chamber: object = None
    states: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)
    history: list = field(default_factory=list)


def apply_corrections(rows):
    return [dict(r) for r in rows]


def parse_row(action_text):
    hits = []
    for rid, rx, state, payload in COMPILED:
        m = rx.search(action_text)
        if m:
            hits.append(Emit(state, dict(payload), rid, (m.start(), m.end()), m.group(0)))
            break
    return hits, []


def replay(measure_id, rows):
    item = Item(measure_id)
    for r in rows:
        emits, _ = parse_row(r["ActionText"])
        if not emits:
            return item, f"NO RULE MATCHED: {r['ActionText'][:70]}", []
        for e in emits:
            frm = item.states.get(r["Chamber"])
            if e.state not in FLOW.get(frm, set()):
                return item, f"{r['Chamber']}: {frm} -> {e.state} not allowed", []
            item.states[r["Chamber"]] = e.state
            item.chamber = r["Chamber"]
            item.history.append((r["ActionDate"], r["Chamber"], frm, e.state, e.rule_id))
    return item, None, []
