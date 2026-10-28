"""Okapi BM25 implemented here, not delegated to a search library.

score(D, Q) = sum_i IDF(q_i) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |D| / avgdl))
IDF(q) = ln((N - n_q + 0.5) / (n_q + 0.5) + 1)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from app.textutil import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.4, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_len: dict[str, int] = {}
        self._strategy: dict[str, str] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._doc_len)

    @property
    def avgdl(self) -> float:
        if not self._doc_len:
            return 0.0
        return self._total_len / len(self._doc_len)

    def add(self, chunk_id: str, text: str, strategy: str) -> None:
        if chunk_id in self._doc_len:
            self.remove(chunk_id)
        counts = Counter(tokenize(text, drop_stopwords=True))
        length = sum(counts.values())
        self._doc_len[chunk_id] = length
        self._strategy[chunk_id] = strategy
        self._total_len += length
        for term, freq in counts.items():
            self._postings[term][chunk_id] = freq

    def remove(self, chunk_id: str) -> None:
        length = self._doc_len.pop(chunk_id, None)
        self._strategy.pop(chunk_id, None)
        if length is None:
            return
        self._total_len -= length
        empty_terms: list[str] = []
        for term, posting in self._postings.items():
            if chunk_id in posting:
                del posting[chunk_id]
                if not posting:
                    empty_terms.append(term)
        for term in empty_terms:
            del self._postings[term]

    def search(
        self, query: str, k: int = 20, strategy: str | None = None
    ) -> list[tuple[str, float]]:
        terms = tokenize(query, drop_stopwords=True)
        if not terms or not self._doc_len:
            return []
        scores: dict[str, float] = defaultdict(float)
        n_docs = len(self._doc_len)
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
            for chunk_id, freq in posting.items():
                if strategy is not None and self._strategy.get(chunk_id) != strategy:
                    continue
                doc_len = self._doc_len.get(chunk_id, 0)
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                scores[chunk_id] += idf * (freq * (self.k1 + 1)) / (denom or 1.0)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(chunk_id, score) for chunk_id, score in ranked[:k] if score > 0]
