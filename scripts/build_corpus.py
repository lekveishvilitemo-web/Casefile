"""Build a reproducible corpus of at least 50,000 HotpotQA paragraphs plus the seed case.

HotpotQA distractor paragraphs are the public multi-hop source required by the
brief. The full training split is larger than this machine needs, so the build
stops at CORPUS_TARGET unique paragraphs (default 50,000) in dataset order.
That order is fixed for a given dataset revision, so the subset is repeatable.
Seed documents add contradictions, a missing-fact case, a prompt-injection
note, and a near-duplicate incident report.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.indexing.embeddings import FastEmbedder
from app.indexing.store import ChunkRow, DocumentStore
from app.ingestion.chunking import chunk_text
from app.ingestion.loaders import load_path
from app.ingestion.pipeline import STRATEGY_PARAMS, make_doc_id
from app.textutil import sha256_text, simhash

TARGET = int(os.environ.get("CORPUS_TARGET", "50000"))
STRATEGIES = ("sentence_pack", "fixed_window")


def wiki_url(title: str) -> str:
    slug = urllib.parse.quote(title.replace(" ", "_"))
    return f"https://en.wikipedia.org/wiki/{slug}"


def paragraphs(row: dict):
    context = row["context"]
    for title, sentences in zip(context["title"], context["sentences"]):
        text = " ".join(part.strip() for part in sentences if part and str(part).strip())
        if text:
            yield str(title), text


def gold_sentences(row: dict) -> list[dict] | None:
    context = row["context"]
    lookup: dict[str, list] = {}
    for title, sentences in zip(context["title"], context["sentences"]):
        lookup.setdefault(str(title), sentences)
    gold = []
    facts = row["supporting_facts"]
    for title, sent_id in zip(facts["title"], facts["sent_id"]):
        sentences = lookup.get(str(title))
        if sentences is None or sent_id >= len(sentences):
            return None
        sentence = str(sentences[sent_id]).strip()
        if not sentence:
            return None
        gold.append({"title": str(title), "sentence": sentence})
    return gold


def chunk_rows(doc_id: str, title: str, source: str, text: str) -> list[ChunkRow]:
    rows = []
    for strategy in STRATEGIES:
        params = STRATEGY_PARAMS[strategy]
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


def load_hotpot():
    from datasets import load_dataset

    for name in ("hotpotqa/hotpot_qa", "hotpot_qa"):
        try:
            return load_dataset(name, "distractor", split="train", streaming=True)
        except Exception as exc:
            print(f"Could not open {name}: {exc}")
    raise RuntimeError("HotpotQA could not be loaded.")


def collect_hotpot(jsonl_path: Path) -> list[dict]:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    seen.add(json.loads(line)["sha256"])
        print(f"Resuming from {len(seen)} paragraphs in {jsonl_path}")
    questions = []
    yes_no = 0
    if len(seen) >= TARGET:
        return questions
    dataset = load_hotpot()
    started = time.time()
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            added_texts = []
            for title, text in paragraphs(row):
                digest = sha256_text(text)
                if digest in seen:
                    continue
                seen.add(digest)
                doc_id = make_doc_id("hotpotqa", title, text)
                record = {
                    "id": doc_id,
                    "title": title,
                    "text": text,
                    "source": wiki_url(title),
                    "format": "txt",
                    "sha256": digest,
                    "metadata": {"dataset": "hotpotqa/hotpot_qa", "config": "distractor", "example_id": row.get("id")},
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                added_texts.append(text)
                if len(seen) >= TARGET:
                    break
            answer = str(row.get("answer") or "").strip()
            gold = gold_sentences(row)
            if gold and answer and len(questions) < 500:
                lowered = answer.lower()
                if lowered in {"yes", "no"}:
                    yes_no += 1
                    keep_yes_no = yes_no <= 4
                else:
                    keep_yes_no = True
                if keep_yes_no:
                    # Gold sentences live in the example's paragraphs, which are stored
                    # unless they were duplicates of earlier paragraphs. Duplicates are
                    # still in the corpus, so the sentence remains searchable.
                    kind = "simple" if len(gold) == 1 else "multihop"
                    questions.append(
                        {
                            "id": f"hotpot-{row.get('id')}",
                            "type": kind,
                            "question": row["question"],
                            "answer": answer,
                            "gold_sentences": [item["sentence"] for item in gold],
                            "gold_titles": sorted({item["title"] for item in gold}),
                            "dataset": "hotpotqa/hotpot_qa",
                        }
                    )
            if len(seen) % 1000 == 0:
                print(f"{len(seen)} unique paragraphs after {index + 1} examples ({time.time() - started:.0f}s)")
            if len(seen) >= TARGET:
                break
    print(f"Collected {len(seen)} unique paragraphs")
    return questions


def select_questions(candidates: list[dict]) -> list[dict]:
    simple = [item for item in candidates if item["type"] == "simple"]
    multi = [item for item in candidates if item["type"] == "multihop"]
    chosen = []
    seen_answers = set()
    for bucket, limit in ((simple, 14), (multi, 20)):
        kept = 0
        for item in bucket:
            key = item["answer"].strip().lower()
            if key in seen_answers:
                continue
            seen_answers.add(key)
            chosen.append(item)
            kept += 1
            if kept >= limit:
                break
    if len(chosen) < 32:
        for item in candidates:
            if item not in chosen:
                chosen.append(item)
            if len(chosen) >= 32:
                break
    return chosen


def index_jsonl(store: DocumentStore, embedder: FastEmbedder, jsonl_path: Path) -> int:
    known = store.known_digests()
    batch = []
    added = 0

    def flush():
        nonlocal added
        if not batch:
            return
        texts = []
        prepared = []
        for record in batch:
            rows = chunk_rows(record["id"], record["title"], record["source"], record["text"])
            prepared.append((record, rows))
            texts.extend(chunk.text for chunk in rows)
        vectors = embedder.embed_documents(texts)
        cursor = 0
        items = []
        for record, rows in prepared:
            chunk_vectors = vectors[cursor : cursor + len(rows)]
            cursor += len(rows)
            items.append(
                (
                    record["id"],
                    record["title"],
                    record["source"],
                    record["format"],
                    record["sha256"],
                    simhash(record["text"]),
                    record["text"],
                    record["metadata"],
                    rows,
                    chunk_vectors,
                )
            )
        store.insert_many(items)
        added += len(items)
        batch.clear()
        print(f"Indexed {added} hotpot paragraphs")

    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["sha256"] in known:
                continue
            known.add(record["sha256"])
            batch.append(record)
            if len(batch) >= 64:
                flush()
    flush()
    return added


def index_seed(store: DocumentStore, embedder: FastEmbedder) -> dict:
    from scripts.make_seed_docs import main as make_docs

    make_docs()
    seed_dir = ROOT / "data" / "seed"
    known = store.known_digests()
    added = 0
    skipped = []
    for path in sorted(seed_dir.iterdir()):
        if path.suffix.lower() not in {".txt", ".md", ".html", ".pdf", ".docx"}:
            continue
        for loaded in load_path(path):
            digest = sha256_text(loaded.text)
            if digest in known:
                skipped.append({"file": path.name, "reason": "exact_duplicate"})
                continue
            near = store.find_near_duplicate(simhash(loaded.text), max_distance=5)
            if near is not None:
                skipped.append({"file": path.name, "reason": "near_duplicate", "duplicate_of": near.id})
                continue
            source = f"seed:{path.name}"
            doc_id = make_doc_id(source, loaded.title, loaded.text)
            rows = chunk_rows(doc_id, loaded.title, source, loaded.text)
            vectors = embedder.embed_documents([chunk.text for chunk in rows])
            metadata = dict(loaded.metadata)
            metadata["seed"] = True
            store.insert_many(
                [
                    (
                        doc_id,
                        loaded.title,
                        source,
                        loaded.fmt,
                        digest,
                        simhash(loaded.text),
                        loaded.text,
                        metadata,
                        rows,
                        vectors,
                    )
                ]
            )
            known.add(digest)
            added += 1
    return {"added": added, "skipped": skipped}


def main() -> None:
    settings = Settings.from_env()
    settings.ensure_dirs()
    jsonl_path = settings.corpus_dir / "hotpot_paragraphs.jsonl"
    questions = collect_hotpot(jsonl_path)
    embedder = FastEmbedder(settings.embedding_model)
    store = DocumentStore(settings.db_path, settings.chroma_dir)
    seed_report = index_seed(store, embedder)
    hotpot_added = index_jsonl(store, embedder, jsonl_path)
    builtin_path = ROOT / "app" / "eval" / "builtin_questions.json"
    builtin = json.loads(builtin_path.read_text(encoding="utf-8"))
    selected = builtin + select_questions(questions)
    eval_path = settings.corpus_dir / "eval_questions.json"
    eval_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "documents": store.count_documents(),
        "chunks": store.count_chunks(),
        "hotpot_indexed_this_run": hotpot_added,
        "seed": seed_report,
        "eval_questions": len(selected),
        "target": TARGET,
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "train",
        "subset_rule": "First unique paragraphs in streaming train order until CORPUS_TARGET.",
    }
    (settings.corpus_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    store.close()


if __name__ == "__main__":
    main()
