"""Tokenization and hashing shared by indexing, dedup, and evaluation."""

from __future__ import annotations

import hashlib
import re
from collections import Counter

STOPWORDS = frozenset(
    """
    a an the of and or to in on for with from by at as is are was were be been being
    it this that these those into over after before about than then so if but
    we you they he she them his her its our your their who what when where which how
    also can may will just than via per
    """.split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|(?:\n+)")
HYPHEN_RE = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+)+\b", re.I)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    tokens = [token for token in tokens if len(token) > 1]
    if drop_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def content_tokens(text: str) -> list[str]:
    return tokenize(text, drop_stopwords=True)


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    parts = [part.strip() for part in SENTENCE_RE.split(cleaned)]
    return [part for part in parts if part]


def sha256_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def simhash(text: str, bits: int = 64) -> int:
    """64-bit Charikar simhash over token frequencies."""
    counts = Counter(tokenize(text))
    if not counts:
        return 0
    weights = [0] * bits
    for token, weight in counts.items():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        hashed = int(digest[:16], 16)
        for bit in range(bits):
            if hashed & (1 << bit):
                weights[bit] += weight
            else:
                weights[bit] -= weight
    value = 0
    for bit, weight in enumerate(weights):
        if weight > 0:
            value |= 1 << bit
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def coverage(query: str, documents: list[str]) -> float:
    needed = content_tokens(query)
    if not needed:
        return 0.0
    present: set[str] = set()
    for document in documents:
        present.update(content_tokens(document))
    found = sum(1 for token in needed if token in present)
    return found / len(needed)


def sentence_covered(chunk_text: str, sentence: str, threshold: float = 0.8) -> bool:
    wanted = content_tokens(sentence) or tokenize(sentence)
    if not wanted:
        return False
    have = set(tokenize(chunk_text))
    return sum(1 for token in wanted if token in have) / len(wanted) >= threshold
