"""Run the frozen question set against two retrieval configurations."""

from __future__ import annotations

import json
import sys
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


def score_config(engine, questions: list[dict], config: dict) -> dict:
    rows = []
    for index, question in enumerate(questions, start=1):
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
        print(
            f"[{config['name']}] {index}/{len(questions)} {question['id']} "
            f"answer={row['answer_correctness']:.2f} recall={row['recall_at_5']}"
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
            payload.get("analysis")
            or "Fill this section after reading the per-type table. Say where retrieval missed, where the model abstained too early or too late, and which single change is worth trying next.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def main() -> None:
    engine = build_engine()
    question_path = engine.settings.corpus_dir / "eval_questions.json"
    if not question_path.exists():
        raise SystemExit(f"Missing {question_path}. Run scripts/build_corpus.py first.")
    questions = load_questions(question_path)
    if len(questions) < 40:
        raise SystemExit(f"Expected at least 40 questions, found {len(questions)}.")
    configurations = [score_config(engine, questions, config) for config in CONFIGS]
    hybrid = configurations[0]["metrics"]
    semantic = configurations[1]["metrics"]
    summary = (
        f"Evaluated {len(questions)} questions. "
        f"hybrid_sentence answer correctness {_fmt(hybrid['answer_correctness'])}, "
        f"Recall@5 {_fmt(hybrid['recall_at_5'])}. "
        f"semantic_fixed answer correctness {_fmt(semantic['answer_correctness'])}, "
        f"Recall@5 {_fmt(semantic['recall_at_5'])}."
    )
    payload = {"summary": summary, "configurations": configurations}
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
