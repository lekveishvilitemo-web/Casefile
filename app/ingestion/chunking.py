"""Two chunking strategies.

sentence_pack keeps sentence boundaries and packs them up to a character budget.
fixed_window cuts a sliding character window on whitespace. The evaluation
harness compares retrieval quality under both strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.textutil import normalize_text, split_sentences


@dataclass(frozen=True)
class RawChunk:
    ordinal: int
    text: str


def chunk_text(
    text: str,
    strategy: str = "sentence_pack",
    size: int = 700,
    overlap: int = 120,
) -> list[RawChunk]:
    if strategy == "fixed_window":
        return chunk_fixed_window(text, size=size, overlap=overlap)
    if strategy == "sentence_pack":
        return chunk_sentence_pack(text, size=size, overlap_sentences=1)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def chunk_fixed_window(text: str, size: int = 700, overlap: int = 120) -> list[RawChunk]:
    body = normalize_text(text)
    if not body:
        return []
    if len(body) <= size:
        return [RawChunk(0, body)]
    overlap = min(overlap, size // 2)
    chunks: list[RawChunk] = []
    start = 0
    while start < len(body):
        end = min(len(body), start + size)
        if end < len(body):
            space = body.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        piece = body[start:end].strip()
        if piece:
            chunks.append(RawChunk(len(chunks), piece))
        if end >= len(body):
            break
        next_start = max(end - overlap, start + 1)
        start = next_start
    return chunks


def chunk_sentence_pack(
    text: str, size: int = 700, overlap_sentences: int = 1
) -> list[RawChunk]:
    body = normalize_text(text)
    sentences = split_sentences(body)
    if not sentences:
        return []
    chunks: list[RawChunk] = []
    bucket: list[str] = []
    bucket_len = 0
    for sentence in sentences:
        extra = len(sentence) + (1 if bucket else 0)
        if bucket and bucket_len + extra > size:
            chunks.append(RawChunk(len(chunks), " ".join(bucket)))
            if overlap_sentences > 0:
                bucket = bucket[-overlap_sentences:]
                bucket_len = sum(len(part) + 1 for part in bucket)
            else:
                bucket = []
                bucket_len = 0
        bucket.append(sentence)
        bucket_len += len(sentence) + 1
    if bucket:
        packed = " ".join(bucket)
        if not chunks or packed != chunks[-1].text:
            chunks.append(RawChunk(len(chunks), packed))
    return chunks
