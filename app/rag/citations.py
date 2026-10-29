"""Citation parsing and grounding.

A claim with no citation into the retrieved set is dropped. Invented chunk
ids are removed. If nothing cited remains, the answer becomes an abstention.
A factual draft that forgot the bracket id is attached to the retrieved chunk
that actually contains the claim, instead of being thrown away.
"""

from __future__ import annotations

import re

from app.textutil import content_tokens

CITE_RE = re.compile(r"\[(doc_[a-z0-9]+_[a-z]\d+)\]", re.I)
ANSWER_RE = re.compile(r"^\s*answer:\s*(.*)$", re.I)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
IDENT_RE = re.compile(r"\b[A-Za-z]+-\d+\b|\b[a-z0-9]+(?:-[a-z0-9]+)+\b", re.I)
REFUSAL_RE = re.compile(
    r"\b(not specified|not stated|does not|do not|did not|cannot|can't|no evidence|"
    r"not enough|unknown|not named|never stated|not contain|doesnt|doesn't)\b",
    re.I,
)


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


def is_refusal(short: str) -> bool:
    """True when the draft declines instead of stating a fact."""
    if is_abstention(short) or is_conflict_marker(short):
        return is_abstention(short)
    return bool(REFUSAL_RE.search(short or ""))


def bind_uncited_claim(answer: str, hits: list[dict]) -> str:
    """Attach a supporting chunk id when the draft states a fact and forgets the bracket.

    Refusals are rewritten to ABSTAIN. A claim is kept only when every number
    and identifier in it appears in a retrieved chunk.
    """
    allowed = {hit["id"] for hit in hits}
    if cited_ids(answer, allowed):
        return answer
    claim = short_answer(answer)
    if not claim or is_conflict_marker(claim):
        return answer
    if is_abstention(claim) or is_refusal(claim):
        return f"Answer: ABSTAIN\n{claim}"
    support = _supporting_chunk(claim, hits)
    if support is None:
        return answer
    return answer.rstrip() + f" [{support}]"


def _supporting_chunk(claim: str, hits: list[dict]) -> str | None:
    numbers = set(NUMBER_RE.findall(claim))
    idents = {match.group(0).lower() for match in IDENT_RE.finditer(claim)}
    wanted = set(content_tokens(claim))
    if not wanted:
        return None
    best_id = None
    best = 0.0
    for hit in hits:
        text = hit.get("text") or ""
        lowered = text.lower()
        if any(number not in text for number in numbers):
            continue
        if any(ident not in lowered for ident in idents):
            continue
        recall = len(wanted & set(content_tokens(text))) / len(wanted)
        if recall > best:
            best = recall
            best_id = hit["id"]
    if best_id is None or best < 0.45:
        return None
    return best_id


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
