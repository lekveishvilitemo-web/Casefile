"""Evidence Coverage Rank Fusion.

This is the project's own retrieval ranker. Vector search and BM25 only
produce candidate lists. ECRF fuses those lists and then reorders them with
features the libraries do not apply:

1. Reciprocal rank fusion of the semantic list and the BM25 list.
2. A query-coverage bonus for candidates that actually contain the question's
   content words.
3. A redundancy penalty based on simhash, so near-duplicate chunks do not
   occupy multiple slots.
4. A source-diversity bonus so the evidence set prefers different documents
   when scores are close. Multi-document questions need that spread.

Semantic-only mode skips BM25 and the diversity pass and returns the vector
ranking unchanged, which is the baseline configuration in the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.textutil import content_tokens, hamming, jaccard


@dataclass
class Candidate:
    chunk_id: str
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    simhash: int = 0
    doc_id: str = ""
    text: str = ""
    fused: float = 0.0
    features: dict[str, float] = field(default_factory=dict)


def reciprocal_rank(rank: int | None, k: int = 60) -> float:
    if rank is None or rank < 1:
        return 0.0
    return 1.0 / (k + rank)


def fuse_candidates(
    semantic: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    *,
    rrf_k: int = 60,
) -> dict[str, Candidate]:
    fused: dict[str, Candidate] = {}
    for rank, (chunk_id, score) in enumerate(semantic, start=1):
        candidate = fused.setdefault(chunk_id, Candidate(chunk_id=chunk_id))
        candidate.semantic_rank = rank
        candidate.semantic_score = score
    for rank, (chunk_id, score) in enumerate(lexical, start=1):
        candidate = fused.setdefault(chunk_id, Candidate(chunk_id=chunk_id))
        candidate.bm25_rank = rank
        candidate.bm25_score = score
    for candidate in fused.values():
        candidate.fused = reciprocal_rank(candidate.semantic_rank, rrf_k) + reciprocal_rank(
            candidate.bm25_rank, rrf_k
        )
        candidate.features["rrf"] = candidate.fused
    return fused


def apply_coverage(candidates: dict[str, Candidate], query: str, weight: float = 0.18) -> None:
    query_terms = set(content_tokens(query))
    if not query_terms:
        return
    for candidate in candidates.values():
        overlap = jaccard(query_terms, set(content_tokens(candidate.text)))
        # Jaccard against a long chunk is harsh. Use containment of the query.
        have = set(content_tokens(candidate.text))
        containment = sum(1 for term in query_terms if term in have) / len(query_terms)
        candidate.features["coverage"] = containment
        candidate.features["jaccard"] = overlap
        candidate.fused += weight * containment


def select_diverse(
    candidates: list[Candidate],
    k: int,
    *,
    redundancy_hamming: int = 3,
    redundancy_penalty: float = 0.012,
    diversity_bonus: float = 0.004,
) -> list[Candidate]:
    """Greedy selection with a near-duplicate penalty and a new-document bonus."""
    remaining = list(candidates)
    chosen: list[Candidate] = []
    chosen_docs: set[str] = set()
    while remaining and len(chosen) < k:
        best_index = 0
        best_score = -1.0
        for index, candidate in enumerate(remaining):
            score = candidate.fused
            if candidate.doc_id and candidate.doc_id not in chosen_docs:
                score += diversity_bonus
            for previous in chosen:
                if hamming(candidate.simhash, previous.simhash) <= redundancy_hamming:
                    score -= redundancy_penalty
                    break
            if score > best_score:
                best_score = score
                best_index = index
        pick = remaining.pop(best_index)
        pick.features["selection_score"] = best_score
        chosen.append(pick)
        if pick.doc_id:
            chosen_docs.add(pick.doc_id)
    return chosen


def rank(
    query: str,
    semantic: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    lookup,
    k: int,
    *,
    hybrid: bool = True,
) -> list[Candidate]:
    """`lookup(chunk_id)` returns an object with text, doc_id, and simhash."""
    if not hybrid:
        ordered = []
        for chunk_id, score in semantic[:k]:
            meta = lookup(chunk_id)
            if meta is None:
                continue
            candidate = Candidate(
                chunk_id=chunk_id,
                semantic_rank=len(ordered) + 1,
                semantic_score=score,
                simhash=meta.simhash,
                doc_id=meta.doc_id,
                text=meta.text,
                fused=score,
            )
            candidate.features["semantic"] = score
            ordered.append(candidate)
        return ordered

    fused = fuse_candidates(semantic, lexical)
    for chunk_id, candidate in list(fused.items()):
        meta = lookup(chunk_id)
        if meta is None:
            del fused[chunk_id]
            continue
        candidate.text = meta.text
        candidate.doc_id = meta.doc_id
        candidate.simhash = meta.simhash
    apply_coverage(fused, query)
    ordered = sorted(fused.values(), key=lambda item: item.fused, reverse=True)
    # Consider a wider pool than k so diversity can swap in a second document.
    return select_diverse(ordered[: max(k * 4, k)], k)
