"""SQLite is the source of truth. ChromaDB is the vector index.

Both live on disk under DATA_DIR, so a restart keeps documents, chunks, and
embeddings. User uploads are written into the same database.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.textutil import hamming


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pack_simhash(value: int) -> int:
    """SQLite integers are signed 64-bit. Fold the high bit into that range."""
    value &= (1 << 64) - 1
    if value >= (1 << 63):
        return value - (1 << 64)
    return value


def unpack_simhash(value: int) -> int:
    return int(value) & ((1 << 64) - 1)


@dataclass
class DocumentRow:
    id: str
    title: str
    source: str
    format: str
    sha256: str
    simhash: int
    text: str
    duplicate_of: str | None
    metadata: dict
    created_at: str
    updated_at: str
    chunk_count: int = 0


@dataclass
class ChunkRow:
    id: str
    doc_id: str
    ordinal: int
    text: str
    strategy: str
    simhash: int
    title: str
    source: str


class DocumentStore:
    def __init__(self, db_path: Path, chroma_dir: Path, collection_name: str = "evidence") -> None:
        self.db_path = db_path
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_sql()
        self._collection = None
        self._simhash_ids: list[str] = []
        self._simhash_values = np.zeros(0, dtype=np.uint64)
        self._refresh_simhash_cache()

    def close(self) -> None:
        self._conn.close()

    def _init_sql(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                format TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                simhash INTEGER NOT NULL,
                text TEXT NOT NULL,
                duplicate_of TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                strategy TEXT NOT NULL,
                simhash INTEGER NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy);

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                latency_ms INTEGER,
                detail_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def collection(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _refresh_simhash_cache(self) -> None:
        rows = self._conn.execute("SELECT id, simhash FROM documents WHERE duplicate_of IS NULL").fetchall()
        self._simhash_ids = [row["id"] for row in rows]
        if rows:
            self._simhash_values = np.array(
                [unpack_simhash(row["simhash"]) for row in rows], dtype=np.uint64
            )
        else:
            self._simhash_values = np.zeros(0, dtype=np.uint64)

    def find_exact(self, digest: str) -> DocumentRow | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE sha256 = ? AND duplicate_of IS NULL ORDER BY created_at LIMIT 1",
            (digest,),
        ).fetchone()
        return self._document_from_row(row) if row else None

    def find_near_duplicate(self, value: int, max_distance: int = 5) -> DocumentRow | None:
        if self._simhash_values.size == 0:
            return None
        target = np.uint64(value & ((1 << 64) - 1))
        # XOR then bit count. Do it in Python for the tail; 50k is still small.
        best_id = None
        best_distance = max_distance + 1
        for doc_id, current in zip(self._simhash_ids, self._simhash_values.tolist()):
            distance = hamming(int(current), int(target))
            if distance <= max_distance and distance < best_distance:
                best_distance = distance
                best_id = doc_id
        if best_id is None:
            return None
        return self.get_document(best_id)

    def insert_document(
        self,
        *,
        doc_id: str,
        title: str,
        source: str,
        fmt: str,
        digest: str,
        simhash_value: int,
        text: str,
        metadata: dict,
        chunks: list[ChunkRow],
        embeddings: list[list[float]],
    ) -> DocumentRow:
        now = utcnow()
        self._conn.execute(
            """
            INSERT INTO documents (
                id, title, source, format, sha256, simhash, text, duplicate_of,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
            """,
            (
                doc_id,
                title,
                source,
                fmt,
                digest,
                pack_simhash(simhash_value),
                text,
                json.dumps(metadata),
                now,
                now,
            ),
        )
        self._insert_chunks(chunks, embeddings)
        self._conn.commit()
        self.clear_cache()
        self._refresh_simhash_cache()
        row = self.get_document(doc_id)
        assert row is not None
        return row

    def replace_document(
        self,
        *,
        doc_id: str,
        title: str,
        source: str,
        fmt: str,
        digest: str,
        simhash_value: int,
        text: str,
        metadata: dict,
        chunks: list[ChunkRow],
        embeddings: list[list[float]],
    ) -> DocumentRow:
        existing = self.get_document(doc_id)
        if existing is None:
            raise KeyError(doc_id)
        self._delete_chunks(doc_id)
        now = utcnow()
        self._conn.execute(
            """
            UPDATE documents
            SET title = ?, source = ?, format = ?, sha256 = ?, simhash = ?, text = ?,
                metadata_json = ?, updated_at = ?, duplicate_of = NULL
            WHERE id = ?
            """,
            (
                title,
                source,
                fmt,
                digest,
                pack_simhash(simhash_value),
                text,
                json.dumps(metadata),
                now,
                doc_id,
            ),
        )
        self._insert_chunks(chunks, embeddings)
        self._conn.commit()
        self.clear_cache()
        self._refresh_simhash_cache()
        row = self.get_document(doc_id)
        assert row is not None
        return row

    def _insert_chunks(self, chunks: list[ChunkRow], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk needs one embedding")
        self._conn.executemany(
            """
            INSERT INTO chunks (id, doc_id, ordinal, text, strategy, simhash, title, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.id,
                    chunk.doc_id,
                    chunk.ordinal,
                    chunk.text,
                    chunk.strategy,
                    pack_simhash(chunk.simhash),
                    chunk.title,
                    chunk.source,
                )
                for chunk in chunks
            ],
        )
        self._add_vectors(chunks, embeddings)

    def _add_vectors(self, chunks: list[ChunkRow], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        batch = 256
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            vectors = embeddings[start : start + batch]
            self.collection().add(
                ids=[chunk.id for chunk in part],
                documents=[chunk.text for chunk in part],
                embeddings=vectors,
                metadatas=[
                    {
                        "doc_id": chunk.doc_id,
                        "title": (chunk.title or "untitled")[:200],
                        "source": (chunk.source or "upload")[:200],
                        "ordinal": chunk.ordinal,
                        "strategy": chunk.strategy,
                    }
                    for chunk in part
                ],
            )

    def known_digests(self) -> set[str]:
        rows = self._conn.execute("SELECT sha256 FROM documents").fetchall()
        return {row["sha256"] for row in rows}

    def insert_many(
        self,
        items: list[tuple[str, str, str, str, str, int, str, dict, list[ChunkRow], list[list[float]]]],
    ) -> int:
        """Bulk insert. Each item is document fields, chunks, and embeddings."""
        if not items:
            return 0
        now = utcnow()
        all_chunks: list[ChunkRow] = []
        all_embeddings: list[list[float]] = []
        for doc_id, title, source, fmt, digest, simhash_value, text, metadata, chunks, embeddings in items:
            self._conn.execute(
                """
                INSERT INTO documents (
                    id, title, source, format, sha256, simhash, text, duplicate_of,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    doc_id,
                    title,
                    source,
                    fmt,
                    digest,
                    pack_simhash(simhash_value),
                    text,
                    json.dumps(metadata),
                    now,
                    now,
                ),
            )
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
        self._conn.executemany(
            """
            INSERT INTO chunks (id, doc_id, ordinal, text, strategy, simhash, title, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.id,
                    chunk.doc_id,
                    chunk.ordinal,
                    chunk.text,
                    chunk.strategy,
                    pack_simhash(chunk.simhash),
                    chunk.title,
                    chunk.source,
                )
                for chunk in all_chunks
            ],
        )
        self._conn.commit()
        self._add_vectors(all_chunks, all_embeddings)
        self.clear_cache()
        self._refresh_simhash_cache()
        return len(items)

    def _delete_chunks(self, doc_id: str) -> None:
        rows = self._conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
        ids = [row["id"] for row in rows]
        if ids:
            self.collection().delete(ids=ids)
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def delete_document(self, doc_id: str) -> bool:
        existing = self.get_document(doc_id)
        if existing is None:
            return False
        self._delete_chunks(doc_id)
        self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._conn.commit()
        self.clear_cache()
        self._refresh_simhash_cache()
        return True

    def get_document(self, doc_id: str) -> DocumentRow | None:
        row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(
        self, limit: int = 50, offset: int = 0, query: str | None = None
    ) -> list[DocumentRow]:
        where = ""
        params: list = []
        if query:
            where = "WHERE d.title LIKE ? OR d.source LIKE ?"
            like = f"%{query}%"
            params.extend([like, like])
        params.extend([limit, offset])
        rows = self._conn.execute(
            f"""
            SELECT d.*, (
                SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id
            ) AS chunk_count
            FROM documents d
            {where}
            ORDER BY CASE WHEN d.source LIKE 'seed:%' OR d.source LIKE 'upload:%' OR d.source = 'paste' THEN 0 ELSE 1 END,
                     d.updated_at DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [self._document_from_row(row) for row in rows]

    def clear_cache(self) -> None:
        self._conn.execute("DELETE FROM response_cache")
        self._conn.commit()

    def count_documents(self, query: str | None = None) -> int:
        if not query:
            return int(self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"])
        like = f"%{query}%"
        return int(
            self._conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE title LIKE ? OR source LIKE ?",
                (like, like),
            ).fetchone()["n"]
        )

    def count_chunks(self, strategy: str | None = None) -> int:
        if strategy is None:
            return int(self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
        return int(
            self._conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE strategy = ?", (strategy,)
            ).fetchone()["n"]
        )

    def iter_chunk_texts(self, strategy: str | None = None):
        """Stream id, text, and strategy without holding every chunk row."""
        if strategy is None:
            cursor = self._conn.execute("SELECT id, text, strategy FROM chunks")
        else:
            cursor = self._conn.execute(
                "SELECT id, text, strategy FROM chunks WHERE strategy = ?",
                (strategy,),
            )
        while True:
            batch = cursor.fetchmany(2000)
            if not batch:
                break
            for row in batch:
                yield row["id"], row["text"], row["strategy"]

    def iter_chunks(self, strategy: str | None = None) -> list[ChunkRow]:
        if strategy is None:
            rows = self._conn.execute("SELECT * FROM chunks").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE strategy = ?", (strategy,)
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def chunks_for_document(self, doc_id: str) -> list[ChunkRow]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY strategy, ordinal", (doc_id,)
        ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> ChunkRow | None:
        row = self._conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return self._chunk_from_row(row) if row else None

    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRow]:
        found = []
        for chunk_id in chunk_ids:
            chunk = self.get_chunk(chunk_id)
            if chunk is not None:
                found.append(chunk)
        return found

    def vector_search(
        self, embedding: list[float], k: int, strategy: str
    ) -> list[tuple[str, float]]:
        collection = self.collection()
        total = collection.count()
        if total == 0 or k <= 0:
            return []
        try:
            result = collection.query(
                query_embeddings=[embedding],
                n_results=min(k, total),
                where={"strategy": strategy},
                include=["distances"],
            )
        except Exception:
            return []
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[tuple[str, float]] = []
        for chunk_id, distance in zip(ids, distances):
            # Chroma cosine space returns distance = 1 - cosine similarity.
            similarity = 1.0 - float(distance)
            hits.append((chunk_id, similarity))
        return hits

    def record_event(self, kind: str, latency_ms: int | None, detail: dict) -> None:
        self._conn.execute(
            "INSERT INTO events (ts, kind, latency_ms, detail_json) VALUES (?, ?, ?, ?)",
            (utcnow(), kind, latency_ms, json.dumps(detail)),
        )
        self._conn.commit()

    def event_summary(self) -> dict:
        rows = self._conn.execute(
            """
            SELECT kind, COUNT(*) AS n, AVG(latency_ms) AS avg_ms
            FROM events GROUP BY kind
            """
        ).fetchall()
        return {
            row["kind"]: {"count": row["n"], "avg_latency_ms": row["avg_ms"]}
            for row in rows
        }

    def cache_get(self, key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT payload_json FROM response_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def cache_put(self, key: str, payload: dict) -> None:
        self._conn.execute(
            """
            INSERT INTO response_cache (cache_key, created_at, payload_json)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET payload_json = excluded.payload_json,
                created_at = excluded.created_at
            """,
            (key, utcnow(), json.dumps(payload)),
        )
        self._conn.commit()

    def corpus_version(self) -> str:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, MAX(updated_at) AS updated FROM documents"
        ).fetchone()
        return f"{row['n']}:{row['updated'] or 'empty'}"

    def _document_from_row(self, row: sqlite3.Row) -> DocumentRow:
        keys = row.keys()
        chunk_count = int(row["chunk_count"]) if "chunk_count" in keys and row["chunk_count"] is not None else 0
        if "chunk_count" not in keys:
            chunk_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ?", (row["id"],)
                ).fetchone()["n"]
            )
        return DocumentRow(
            id=row["id"],
            title=row["title"],
            source=row["source"],
            format=row["format"],
            sha256=row["sha256"],
            simhash=unpack_simhash(row["simhash"]),
            text=row["text"],
            duplicate_of=row["duplicate_of"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            chunk_count=chunk_count,
        )

    def _chunk_from_row(self, row: sqlite3.Row) -> ChunkRow:
        return ChunkRow(
            id=row["id"],
            doc_id=row["doc_id"],
            ordinal=int(row["ordinal"]),
            text=row["text"],
            strategy=row["strategy"],
            simhash=unpack_simhash(row["simhash"]),
            title=row["title"],
            source=row["source"],
        )
