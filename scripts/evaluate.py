"""Run the frozen question set against two retrieval configurations."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine import build_engine
from app.eval.metrics import (
    answer_correctness,
    citation_accuracy,
    mrr,
    ranked_evidence,
    recall_at_k,
    summarize,
)

CONFIGS = [
    {
        "name": "hybrid_sentence",
        "chunk_strategy": "sentence_pack",
        "retrieval": "hybrid",
        "description": "Sentence packing, BM25 + vector fusion (ECRF), and a bridge-entity second hop.",
    },
    {
        "name": "semantic_fixed",
        "chunk_strategy": "fixed_window",
        "retrieval": "semantic",
        "description": "Fixed windows and vector similarity only. No BM25 and no second hop.",
    },
]


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rss_mb() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return -1
    return -1


def _load_checkpoint(path: Path, config_name: str) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("config") == config_name:
            done[row["id"]] = row
    return done


def score_config(engine, questions: list[dict], config: dict, checkpoint: Path) -> dict:
    done = _load_checkpoint(checkpoint, config["name"])
    rows = []
    for index, question in enumerate(questions, start=1):
        cached = done.get(question["id"])
        if cached is not None:
            row = {key: value for key, value in cached.items() if key != "config"}
            rows.append(row)
            print(
                f"[{config['name']}] {index}/{len(questions)} {question['id']} checkpoint",
                flush=True,
            )
            continue
        started = time.perf_counter()
        result = engine.investigate(
            question["question"],
            strategy=config["chunk_strategy"],
            retrieval=config["retrieval"],
        )
        evidence = ranked_evidence(result)
        row = {
            "id": question["id"],
            "type": question["type"],
            "question": question["question"],
            "gold": question.get("answer"),
            "short_answer": result.get("short_answer"),
            "abstained": result.get("abstained"),
            "conflict": result.get("conflict"),
            "recall_at_5": recall_at_k(evidence, question, 5),
            "mrr": mrr(evidence, question),
            "answer_correctness": answer_correctness(result, question),
            "citation_accuracy": citation_accuracy(result, question),
        }
        rows.append(row)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"config": config["name"], **row}) + "\n")
        elapsed = time.perf_counter() - started
        recall = "n/a" if row["recall_at_5"] is None else f"{row['recall_at_5']:.2f}"
        print(
            f"[{config['name']}] {index}/{len(questions)} {question['id']} "
            f"answer={row['answer_correctness']:.2f} recall={recall} {elapsed:.1f}s rss={_rss_mb()}MB",
            flush=True,
        )
    metrics = summarize(rows)
    failures = sorted(rows, key=lambda item: item["answer_correctness"])[:8]
    return {
        "name": config["name"],
        "description": config["description"],
        "metrics": metrics,
        "failures": failures,
        "rows": rows,
    }


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Evaluation report",
        "",
        payload["summary"],
        "",
        "## Method",
        "",
        "Questions are frozen in `data/corpus/eval_questions.json`. Seed questions cover the Northline payments case: simple lookup, multi-document joins, contradictions, unanswerable facts, and a prompt-injection note. The rest are HotpotQA questions whose supporting sentences are inside the indexed subset.",
        "",
        "Two configurations share the same questions, the same embeddings, and the same model:",
        "",
    ]
    for config in payload["configurations"]:
        metrics = config["metrics"]
        lines.append(f"### {config['name']}")
        lines.append("")
        lines.append(config["description"])
        lines.append("")
        lines.append(f"- Questions: {metrics['n']}")
        lines.append(f"- Recall@5: {_fmt(metrics['recall_at_5'])}")
        lines.append(f"- MRR: {_fmt(metrics['mrr'])}")
        lines.append(f"- Answer correctness: {_fmt(metrics['answer_correctness'])}")
        lines.append(f"- Citation accuracy: {_fmt(metrics['citation_accuracy'])}")
        lines.append("")
        lines.append("| Type | n | Recall@5 | MRR | Answer | Citations |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for kind, values in metrics["by_type"].items():
            lines.append(
                f"| {kind} | {values['n']} | {_fmt(values['recall_at_5'])} | {_fmt(values['mrr'])} | "
                f"{_fmt(values['answer_correctness'])} | {_fmt(values['citation_accuracy'])} |"
            )
        lines.append("")
        lines.append("Lowest-scoring answers:")
        lines.append("")
        for failure in config["failures"]:
            lines.append(
                f"- `{failure['id']}` ({failure['type']}) gold={failure['gold']!r} "
                f"predicted={failure['short_answer']!r} correctness={failure['answer_correctness']:.2f}"
            )
        lines.append("")
    lines.extend(
        [
            "## Analysis",
            "",
            payload.get("analysis") or "",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_delta(left, right) -> str:
    if left is None or right is None:
        return "n/a"
    return f"{left - right:+.3f}"


def build_analysis(configurations: list[dict], model_label: str) -> str:
    hybrid = configurations[0]
    semantic = configurations[1]
    hm = hybrid["metrics"]
    sm = semantic["metrics"]
    failures = hybrid["failures"]
    lines = [
        f"The writer on this run was {model_label}. Both configurations saw the same questions and the same embedding model. hybrid_sentence adds BM25, Evidence Coverage Rank Fusion, and a bridge-entity second hop on sentence-packed chunks. semantic_fixed uses fixed windows and vector similarity only.",
        "",
        (
            f"hybrid_sentence Recall@5 was {_fmt(hm['recall_at_5'])} and MRR {_fmt(hm['mrr'])}, "
            f"against semantic_fixed Recall@5 {_fmt(sm['recall_at_5'])} ({_fmt_delta(hm['recall_at_5'], sm['recall_at_5'])}) "
            f"and MRR {_fmt(sm['mrr'])} ({_fmt_delta(hm['mrr'], sm['mrr'])}). "
            f"Answer correctness was {_fmt(hm['answer_correctness'])} versus {_fmt(sm['answer_correctness'])}. "
            f"Citation accuracy was {_fmt(hm['citation_accuracy'])} versus {_fmt(sm['citation_accuracy'])}."
        ),
        "",
    ]
    weak_types = []
    for kind, values in hm["by_type"].items():
        answer = values["answer_correctness"]
        recall = values["recall_at_5"]
        if answer is not None and answer < 0.5:
            weak_types.append(
                f"{kind} answer correctness {_fmt(answer)} on {values['n']} questions"
                + (f" (Recall@5 {_fmt(recall)})" if recall is not None else "")
            )
    if weak_types:
        lines.append("The weak hybrid slices were: " + "; ".join(weak_types) + ".")
        lines.append("")
    if failures:
        lines.append("Lowest hybrid answers, gold against what the engine actually returned:")
        lines.append("")
        for failure in failures[:5]:
            lines.append(
                f"- `{failure['id']}` ({failure['type']}) gold={failure['gold']!r} "
                f"predicted={failure['short_answer']!r} "
                f"correctness={failure['answer_correctness']:.2f} "
                f"recall@5={_fmt(failure['recall_at_5'])} abstained={failure['abstained']}."
            )
        lines.append("")
    recall_gap = None
    if hm["recall_at_5"] is not None and hm["answer_correctness"] is not None:
        recall_gap = hm["recall_at_5"] - hm["answer_correctness"]
    if recall_gap is not None and recall_gap > 0.15:
        lines.append(
            "Retrieval is ahead of the written answer. The 1.5B local model often sees a supporting sentence and still paraphrases past the gold span, abstains after grounding strips an uncited number, or cites a nearby chunk from the wrong title. The next change worth making is to keep this ranker and point `GROQ_API_KEY` at `llama-3.1-8b-instant` for the draft only. That does not change Recall@5, and it is the gap these numbers isolate."
        )
    elif hm["recall_at_5"] is not None and hm["recall_at_5"] < 0.45:
        lines.append(
            "Recall@5 is the binding failure. Supporting sentences are not in the first five chunks often enough for the writer to have a chance. The next change worth making is to raise the bridge hop from two title entities to a second search that uses the gold-style relation words already in the top chunk, and to score that hop inside this same harness before touching the model."
        )
    else:
        lines.append(
            "The two configurations are close. The next change worth making is a single ranker ablation: turn the near-duplicate penalty off and rerun hybrid_sentence, because a repeated Hotpot title can crowd out the second supporting page."
        )
    unanswerable = hm["by_type"].get("unanswerable") or {}
    if unanswerable.get("answer_correctness") is not None and unanswerable["answer_correctness"] < 1:
        lines.append("")
        lines.append(
            "Unanswerable items are not all abstentions. BM25 still hits the incident write-up, so the early abstain threshold does not fire, and the model has to refuse a question whose number was never written down. Grounding does catch some invented figures; the misses are fluent refusals that do not use the ABSTAIN marker."
        )
    return "\n".join(lines)


def main() -> None:
    started = time.perf_counter()
    engine = build_engine()
    model_label = f"{engine.llm.provider} ({Path(engine.llm.model).name})"
    print(
        f"engine ready in {time.perf_counter() - started:.1f}s "
        f"bm25={len(engine.bm25)} rss={_rss_mb()}MB model={model_label}",
        flush=True,
    )
    question_path = engine.settings.corpus_dir / "eval_questions.json"
    if not question_path.exists():
        raise SystemExit(f"Missing {question_path}. Run scripts/build_corpus.py first.")
    questions = load_questions(question_path)
    if len(questions) < 40:
        raise SystemExit(f"Expected at least 40 questions, found {len(questions)}.")
    checkpoint = ROOT / "data" / "evaluation_checkpoint.jsonl"
    configurations = [score_config(engine, questions, config, checkpoint) for config in CONFIGS]
    hybrid = configurations[0]["metrics"]
    semantic = configurations[1]["metrics"]
    summary = (
        f"Evaluated {len(questions)} questions with {model_label}. "
        f"hybrid_sentence answer correctness {_fmt(hybrid['answer_correctness'])}, "
        f"Recall@5 {_fmt(hybrid['recall_at_5'])}. "
        f"semantic_fixed answer correctness {_fmt(semantic['answer_correctness'])}, "
        f"Recall@5 {_fmt(semantic['recall_at_5'])}."
    )
    payload = {
        "summary": summary,
        "analysis": build_analysis(configurations, model_label),
        "configurations": configurations,
    }
    summary_path = ROOT / "data" / "evaluation_summary.json"
    # Drop per-question rows from the API summary; keep them in the full JSON.
    public = {
        "summary": summary,
        "configurations": [
            {"name": item["name"], "description": item["description"], "metrics": item["metrics"]}
            for item in configurations
        ],
    }
    summary_path.write_text(json.dumps(public, indent=2), encoding="utf-8")
    (ROOT / "data" / "evaluation_full.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(ROOT / "EVALUATION_REPORT.md", payload)
    print(summary)


if __name__ == "__main__":
    main()
