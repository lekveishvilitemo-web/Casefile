"""Wires storage, the custom ranker, and the investigation agent."""

from __future__ import annotations

import hashlib
import time

from app.config import Settings
from app.indexing.bm25 import BM25Index
from app.indexing.embeddings import Embedder, FastEmbedder
from app.indexing.hybrid import rank
from app.indexing.store import DocumentStore
from app.ingestion.loaders import LoadedDocument, load_bytes
from app.ingestion.pipeline import Ingestor
from app.rag.agent import InvestigationAgent
from app.rag.llm import LLMClient, build_llm


class Engine:
    def __init__(self, settings: Settings, embedder: Embedder, llm: LLMClient) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.embedder = embedder
        self.llm = llm
        self.store = DocumentStore(settings.db_path, settings.chroma_dir)
        self.bm25 = BM25Index()
        self.ingestor = Ingestor(self.store, embedder)
        self.agent = InvestigationAgent(
            self,
            llm,
            abstain_coverage=settings.abstain_coverage,
            abstain_semantic=settings.abstain_semantic,
            top_k=settings.top_k,
        )
        self._load_bm25()

    def _load_bm25(self) -> None:
        for chunk in self.store.iter_chunks():
            self.bm25.add(chunk.id, chunk.text, chunk.strategy)

    def search(self, query: str, *, strategy: str, retrieval: str, k: int) -> list[dict]:
        embedding = self.embedder.embed_query(query)
        semantic = self.store.vector_search(
            embedding, max(k, self.settings.candidate_k), strategy
        )
        lexical = []
        if retrieval == "hybrid":
            lexical = self.bm25.search(query, max(k, self.settings.candidate_k), strategy=strategy)
        candidates = rank(
            query,
            semantic,
            lexical,
            self.store.get_chunk,
            k,
            hybrid=retrieval == "hybrid",
        )
        hits = []
        for candidate in candidates:
            chunk = self.store.get_chunk(candidate.chunk_id)
            if chunk is None:
                continue
            hits.append(
                {
                    "id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "text": chunk.text,
                    "score": candidate.fused,
                    "semantic_score": candidate.semantic_score,
                    "bm25_score": candidate.bm25_score,
                    "features": candidate.features,
                }
            )
        return hits

    def read_chunk(self, chunk_id: str) -> dict | None:
        chunk = self.store.get_chunk(chunk_id)
        if chunk is None:
            return None
        return {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "source": chunk.source,
            "text": chunk.text,
        }

    def add_upload(self, filename: str, data: bytes, *, force: bool = False) -> list[dict]:
        loaded = load_bytes(filename, data)
        outcomes = []
        for item in loaded:
            source = item.metadata.get("source") or f"upload:{filename}"
            outcomes.append(self._add_loaded(item, source=str(source), force=force))
        return outcomes

    def add_text(self, title: str, text: str, *, source: str = "paste", fmt: str = "txt") -> dict:
        loaded = LoadedDocument(title=title or "Untitled", text=text, fmt=fmt, metadata={"source": source})
        return self._add_loaded(loaded, source=source, force=False)

    def _add_loaded(self, loaded: LoadedDocument, *, source: str, force: bool) -> dict:
        outcome = self.ingestor.add(loaded, source=source, check_near=not force, force=force)
        if outcome.created and outcome.document is not None:
            for chunk in self.store.chunks_for_document(outcome.document.id):
                self.bm25.add(chunk.id, chunk.text, chunk.strategy)
        return _outcome_dict(outcome)

    def update_upload(self, doc_id: str, filename: str, data: bytes) -> dict:
        loaded_list = load_bytes(filename, data)
        if len(loaded_list) != 1:
            raise ValueError("Updating a document requires a single-record file.")
        before = self.store.chunks_for_document(doc_id)
        outcome = self.ingestor.update(doc_id, loaded_list[0])
        for chunk in before:
            self.bm25.remove(chunk.id)
        if outcome.document is not None:
            for chunk in self.store.chunks_for_document(outcome.document.id):
                self.bm25.add(chunk.id, chunk.text, chunk.strategy)
        return _outcome_dict(outcome)

    def delete(self, doc_id: str) -> bool:
        before = self.store.chunks_for_document(doc_id)
        deleted = self.ingestor.delete(doc_id)
        if deleted:
            for chunk in before:
                self.bm25.remove(chunk.id)
        return deleted

    def investigate(self, question: str, *, strategy: str | None = None, retrieval: str = "hybrid") -> dict:
        strategy = strategy or self.settings.chunk_strategy
        if strategy not in {"sentence_pack", "fixed_window"}:
            raise ValueError("chunk_strategy must be sentence_pack or fixed_window.")
        if retrieval not in {"hybrid", "semantic"}:
            raise ValueError("retrieval must be hybrid or semantic.")
        cache_key = hashlib.sha256(
            f"{self.store.corpus_version()}|{strategy}|{retrieval}|{question}".encode("utf-8")
        ).hexdigest()
        cached = self.store.cache_get(cache_key)
        if cached is not None:
            cached["cached"] = True
            return cached
        started = time.perf_counter()
        result = self.agent.investigate(question, strategy=strategy, retrieval=retrieval)
        elapsed = int((time.perf_counter() - started) * 1000)
        result["latency_ms"] = elapsed
        result["cached"] = False
        self.store.cache_put(cache_key, result)
        self.store.record_event(
            "investigate",
            elapsed,
            {
                "abstained": result["abstained"],
                "conflict": result["conflict"],
                "citations": len(result["citations"]),
                "strategy": strategy,
                "retrieval": retrieval,
                "llm_error": (result.get("llm") or {}).get("error"),
            },
        )
        return result

    def stats(self) -> dict:
        return {
            "documents": self.store.count_documents(),
            "chunks": self.store.count_chunks(),
            "chunks_sentence_pack": self.store.count_chunks("sentence_pack"),
            "chunks_fixed_window": self.store.count_chunks("fixed_window"),
            "embedding_model": self.embedder.model_name,
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model,
            "events": self.store.event_summary(),
        }


def _outcome_dict(outcome) -> dict:
    document = outcome.document
    return {
        "created": outcome.created,
        "duplicate": outcome.duplicate,
        "reason": outcome.reason,
        "duplicate_of": outcome.duplicate_of,
        "document": document_dict(document) if document is not None else None,
    }


def document_dict(document, *, include_text: bool = False) -> dict:
    payload = {
        "id": document.id,
        "title": document.title,
        "source": document.source,
        "format": document.format,
        "sha256": document.sha256,
        "chunk_count": document.chunk_count,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "metadata": document.metadata,
    }
    if include_text:
        payload["text"] = document.text
    return payload


def build_engine(settings: Settings | None = None, *, embedder: Embedder | None = None, llm: LLMClient | None = None) -> Engine:
    settings = settings or Settings.from_env()
    embedder = embedder or FastEmbedder(settings.embedding_model)
    if llm is None:
        try:
            llm = build_llm(settings)
        except Exception:
            llm = LLMClient()
    return Engine(settings, embedder, llm)
