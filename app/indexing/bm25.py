"""Okapi BM25 implemented here, not delegated to a search library.

score(D, Q) = sum_i IDF(q_i) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |D| / avgdl))
IDF(q) = ln((N - n_q + 0.5) / (n_q + 0.5) + 1)

Postings use integer document ids. A dict of chunk-id strings per term does
not fit comfortably next to the vector index and a local model at 168k chunks.
"""

from __future__ import annotations

import math
from collections import Counter

from app.textutil import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.4, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._doc_len: list[int] = []
        self._strategy: list[str] = []
        self._terms: list[list[str]] = []
        self._rev: list[str] = []
        self._ids: dict[str, int] = {}
        self._total_len = 0
        self._alive = 0

    def __len__(self) -> int:
        return self._alive

    @property
    def avgdl(self) -> float:
        if self._alive == 0:
            return 0.0
        return self._total_len / self._alive

    def add(self, chunk_id: str, text: str, strategy: str) -> None:
        if chunk_id in self._ids:
            self.remove(chunk_id)
        counts = Counter(tokenize(text, drop_stopwords=True))
        length = sum(counts.values())
        index = len(self._rev)
        self._ids[chunk_id] = index
        self._rev.append(chunk_id)
        self._doc_len.append(length)
        self._strategy.append(strategy)
        terms: list[str] = []
        for term, freq in counts.items():
            posting = self._postings.get(term)
            if posting is None:
                self._postings[term] = [(index, freq)]
            else:
                posting.append((index, freq))
            terms.append(term)
        self._terms.append(terms)
        self._total_len += length
        self._alive += 1

    def remove(self, chunk_id: str) -> None:
        index = self._ids.pop(chunk_id, None)
        if index is None:
            return
        length = self._doc_len[index]
        self._doc_len[index] = -1
        self._total_len -= length
        self._alive -= 1
        for term in self._terms[index]:
            posting = self._postings.get(term)
            if not posting:
                continue
            kept = [item for item in posting if item[0] != index]
            if kept:
                self._postings[term] = kept
            else:
                del self._postings[term]
        self._terms[index] = []

    def search(
        self, query: str, k: int = 20, strategy: str | None = None
    ) -> list[tuple[str, float]]:
        terms = tokenize(query, drop_stopwords=True)
        if not terms or self._alive == 0:
            return []
        scores: dict[str, float] = {}
        n_docs = self._alive
        avgdl = self.avgdl or 1.0
        seen: set[str] = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            posting = self._postings.get(term)
            if not posting:
                continue
            n_q = len(posting)
            idf = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
            for index, freq in posting:
                doc_len = self._doc_len[index]
                if doc_len < 0:
                    continue
                if strategy is not None and self._strategy[index] != strategy:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                chunk_id = self._rev[index]
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (freq * (self.k1 + 1)) / (denom or 1.0)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(chunk_id, score) for chunk_id, score in ranked[:k] if score > 0]
