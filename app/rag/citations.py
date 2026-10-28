"""Citation parsing and grounding.

A claim with no citation into the retrieved set is dropped. Invented chunk
ids are removed. If nothing cited remains, the answer becomes an abstention.
"""

from __future__ import annotations

import re

CITE_RE = re.compile(r"\[(doc_[a-z0-9]+_[a-z]\d+)\]", re.I)
ANSWER_RE = re.compile(r"^\s*answer:\s*(.*)$", re.I)


def cited_ids(text: str, allowed: set[str]) -> list[str]:
    found: list[str] = []
    allowed_lower = {item.lower(): item for item in allowed}
    for match in CITE_RE.finditer(text):
        raw = match.group(1)
        canonical = allowed_lower.get(raw.lower())
        if canonical and canonical not in found:
            found.append(canonical)
    return found


def short_answer(text: str) -> str:
    for line in text.splitlines():
        match = ANSWER_RE.match(line)
        if match:
            return match.group(1).strip()
    stripped = text.strip()
    return stripped.split("\n", 1)[0][:300] if stripped else ""


def is_abstention(short: str) -> bool:
    lowered = short.lower().strip()
    return lowered.startswith("abstain") or lowered in {"insufficient", "unknown", "not enough evidence"}


def is_conflict_marker(short: str) -> bool:
    return short.lower().strip().startswith("conflict")


def enforce_grounding(answer: str, allowed: set[str]) -> tuple[str, list[str], bool]:
    """Return grounded answer text, valid citation ids, and whether it abstains."""
    if not answer.strip():
        return (
            "Answer: ABSTAIN\nThe model returned an empty response, so no claim is shown.",
            [],
            True,
        )
    ids = cited_ids(answer, allowed)
    kept_lines: list[str] = []
    answer_line = ""
    for line in answer.splitlines():
        if ANSWER_RE.match(line):
            answer_line = line.strip()
            continue
        line_ids = cited_ids(line, allowed)
        if line_ids:
            kept_lines.append(line.strip())
    if not answer_line:
        answer_line = "Answer: " + short_answer(answer)
    header = ANSWER_RE.match(answer_line)
    header_text = header.group(1).strip() if header else answer_line
    abstain = is_abstention(header_text)
    if not ids and not abstain:
        abstain = True
        header_text = "ABSTAIN"
        kept_lines = [
            "No cited evidence supported a factual claim, so this question is left unanswered."
        ]
    grounded = "Answer: " + header_text
    if kept_lines:
        grounded += "\n" + "\n".join(kept_lines)
    elif abstain:
        grounded += "\nThe corpus does not contain enough evidence to answer."
    return grounded, ids, abstain
