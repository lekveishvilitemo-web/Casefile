"""Add, update, and delete documents, including duplicate detection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.indexing.embeddings import Embedder
from app.indexing.store import ChunkRow, DocumentRow, DocumentStore
from app.ingestion.chunking import chunk_text
from app.ingestion.loaders import LoadedDocument
from app.textutil import sha256_text, simhash

STRATEGY_PARAMS = {
    "sentence_pack": {"size": 900, "overlap": 120},
    "fixed_window": {"size": 320, "overlap": 40},
}


@dataclass
class IngestOutcome:
    document: DocumentRow | None
    created: bool
    duplicate: bool
    reason: str | None = None
    duplicate_of: str | None = None


def make_doc_id(source: str, title: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}\n{title}\n{text}".encode("utf-8")).hexdigest()
    return "doc_" + digest[:12]


def build_chunks(
    doc_id: str,
    title: str,
    source: str,
    text: str,
    strategies: tuple[str, ...],
) -> list[ChunkRow]:
    rows: list[ChunkRow] = []
    for strategy in strategies:
        params = STRATEGY_PARAMS.get(strategy, {"size": 700, "overlap": 120})
        parts = chunk_text(text, strategy=strategy, size=params["size"], overlap=params["overlap"])
        for part in parts:
            rows.append(
                ChunkRow(
                    id=f"{doc_id}_{strategy[:1]}{part.ordinal}",
                    doc_id=doc_id,
                    ordinal=part.ordinal,
                    text=part.text,
                    strategy=strategy,
                    simhash=simhash(part.text),
                    title=title,
                    source=source,
                )
            )
    return rows


class Ingestor:
    def __init__(
        self,
        store: DocumentStore,
        embedder: Embedder,
        strategies: tuple[str, ...] = ("sentence_pack", "fixed_window"),
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.strategies = strategies

    def add(
        self,
        loaded: LoadedDocument,
        *,
        source: str,
        check_near: bool = True,
        force: bool = False,
    ) -> IngestOutcome:
        text = loaded.text.strip()
        if not text:
            raise ValueError("Document has no extractable text.")
        digest = sha256_text(text)
        if not force:
            existing = self.store.find_exact(digest)
            if existing is not None:
                return IngestOutcome(
                    document=existing,
                    created=False,
                    duplicate=True,
                    reason="exact_duplicate",
                    duplicate_of=existing.id,
                )
            if check_near:
                near = self.store.find_near_duplicate(simhash(text), max_distance=5)
                if near is not None:
                    return IngestOutcome(
                        document=near,
                        created=False,
                        duplicate=True,
                        reason="near_duplicate",
                        duplicate_of=near.id,
                    )
        doc_id = make_doc_id(source, loaded.title, text)
        if self.store.get_document(doc_id) is not None:
            existing = self.store.get_document(doc_id)
            return IngestOutcome(
                document=existing,
                created=False,
                duplicate=True,
                reason="exact_duplicate",
                duplicate_of=doc_id,
            )
        chunks = build_chunks(doc_id, loaded.title, source, text, self.strategies)
        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        metadata = dict(loaded.metadata)
        metadata["strategies"] = list(self.strategies)
        document = self.store.insert_document(
            doc_id=doc_id,
            title=loaded.title,
            source=source,
            fmt=loaded.fmt,
            digest=digest,
            simhash_value=simhash(text),
            text=text,
            metadata=metadata,
            chunks=chunks,
            embeddings=embeddings,
        )
        return IngestOutcome(document=document, created=True, duplicate=False)

    def update(self, doc_id: str, loaded: LoadedDocument, *, source: str | None = None) -> IngestOutcome:
        existing = self.store.get_document(doc_id)
        if existing is None:
            raise KeyError(doc_id)
        text = loaded.text.strip()
        if not text:
            raise ValueError("Document has no extractable text.")
        chunks = build_chunks(doc_id, loaded.title, source or existing.source, text, self.strategies)
        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        metadata = dict(existing.metadata)
        metadata.update(loaded.metadata)
        document = self.store.replace_document(
            doc_id=doc_id,
            title=loaded.title,
            source=source or existing.source,
            fmt=loaded.fmt,
            digest=sha256_text(text),
            simhash_value=simhash(text),
            text=text,
            metadata=metadata,
            chunks=chunks,
            embeddings=embeddings,
        )
        return IngestOutcome(document=document, created=False, duplicate=False)

    def delete(self, doc_id: str) -> bool:
        return self.store.delete_document(doc_id)
