"""Lightweight contradiction detector over retrieved evidence.

Two chunks from different documents conflict when they talk about the same
entity and disagree in one of three ways:

- polarity: one describes harm or failure, the other describes safety or no change
- cause: both state a cause for the same event, and the causes differ
- number: they share enough wording to be the same claim, but the numbers differ

This is intentionally conservative. A disagreement the detector misses is
still visible to the reader in the cited excerpts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.textutil import content_tokens, jaccard, split_sentences

HARM = frozenset(
    """
    delay delayed delays timeout timeouts backlog degraded outage failed failure
    failures rollback incident unreachable error errors dropped drop loss
    """.split()
)
SAFETY = frozenset(
    """
    healthy unchanged unrelated remained nominal unaffected successful successfully
    """.split()
)
SAFETY_PHRASES = ("within slo", "no impact", "not modified", "was not", "were not", "did not change")
CAUSE_RE = re.compile(r"caused by ([^.]{3,80})", re.I)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


@dataclass(frozen=True)
class Conflict:
    left_id: str
    right_id: str
    left_title: str
    right_title: str
    left_excerpt: str
    right_excerpt: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_title": self.left_title,
            "right_title": self.right_title,
            "left_excerpt": self.left_excerpt,
            "right_excerpt": self.right_excerpt,
            "reason": self.reason,
        }


def _polarity(text: str) -> str | None:
    tokens = set(content_tokens(text))
    lowered = text.lower()
    harm = bool(tokens & HARM)
    safety = bool(tokens & SAFETY) or any(phrase in lowered for phrase in SAFETY_PHRASES)
    if harm and not safety:
        return "harm"
    if safety and not harm:
        return "safe"
    return None


def _entities(text: str) -> set[str]:
    entities = {match.group(0).lower() for match in re.finditer(r"\b[a-z0-9]+(?:-[a-z0-9]+)+\b", text, re.I)}
    entities.update(token for token in content_tokens(text) if len(token) >= 6)
    return entities


def _cause(text: str) -> str | None:
    match = CAUSE_RE.search(text)
    if not match:
        return None
    return " ".join(content_tokens(match.group(1)))[:80]


def _as_sentences(chunks: list[dict]) -> list[dict]:
    sentences = []
    for chunk in chunks:
        parts = split_sentences(chunk["text"]) or [chunk["text"]]
        for part in parts:
            sentences.append({**chunk, "text": part})
    return sentences


def detect_contradictions(chunks: list[dict]) -> list[Conflict]:
    """`chunks` items need id, doc_id, title, and text.

    Comparison is sentence by sentence. A whole chunk often mixes a claim
    with background, which hides a polarity clash.
    """
    sentences = _as_sentences(chunks)
    conflicts: list[Conflict] = []
    seen: set[tuple[str, str, str]] = set()
    for i, left in enumerate(sentences):
        for right in sentences[i + 1 :]:
            if left["doc_id"] == right["doc_id"]:
                continue
            pair = tuple(sorted((left["id"], right["id"])))
            shared = _entities(left["text"]) & _entities(right["text"])
            left_tokens = set(content_tokens(left["text"]))
            right_tokens = set(content_tokens(right["text"]))
            overlap = jaccard(left_tokens, right_tokens)
            reason = None
            if shared:
                left_polarity = _polarity(left["text"])
                right_polarity = _polarity(right["text"])
                if left_polarity and right_polarity and left_polarity != right_polarity:
                    named = sorted(shared, key=lambda item: (0 if "-" in item else 1, item))
                    reason = f"polarity clash on {named[0]}"
            if reason is None:
                left_cause = _cause(left["text"])
                right_cause = _cause(right["text"])
                if (
                    left_cause
                    and right_cause
                    and left_cause != right_cause
                    and overlap >= 0.15
                ):
                    reason = "competing causes"
            if reason is None and overlap >= 0.45:
                left_numbers = set(NUMBER_RE.findall(left["text"]))
                right_numbers = set(NUMBER_RE.findall(right["text"]))
                if left_numbers and right_numbers and left_numbers != right_numbers and shared:
                    reason = "same claim, different numbers"
            if reason is None:
                continue
            key = (pair[0], pair[1], reason)
            if key in seen:
                continue
            seen.add(key)
            conflicts.append(
                Conflict(
                    left_id=left["id"],
                    right_id=right["id"],
                    left_title=left["title"],
                    right_title=right["title"],
                    left_excerpt=_excerpt(left["text"]),
                    right_excerpt=_excerpt(right["text"]),
                    reason=reason,
                )
            )
            if len(conflicts) >= 5:
                return conflicts
    return conflicts


def _excerpt(text: str, limit: int = 280) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"
