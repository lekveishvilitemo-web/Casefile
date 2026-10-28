"""Time hybrid search over the built corpus. This does not call the language model."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine import build_engine

QUERIES = [
    "ledger-service v2.4.1 retry count",
    "March payment delay OrbitalNet",
    "who wrote the novel about the whale",
    "capital city government formed in",
    "checkout-api latency incident",
    "scientific discovery published in",
    "which film won the award",
    "settlement-worker backlog",
]


def main() -> None:
    engine = build_engine()
    print(f"documents={engine.store.count_documents()} chunks={engine.store.count_chunks()}")
    timings = []
    for query in QUERIES:
        started = time.perf_counter()
        hits = engine.search(query, strategy="sentence_pack", retrieval="hybrid", k=6)
        elapsed = (time.perf_counter() - started) * 1000
        timings.append(elapsed)
        top = hits[0]["title"] if hits else "none"
        print(f"{elapsed:8.1f} ms  {query[:48]:48}  {top}")
    print(f"p50={statistics.median(timings):.1f} ms  max={max(timings):.1f} ms  n={len(timings)}")


if __name__ == "__main__":
    main()
