# AI usage report

This project was built in Cursor by a cloud agent (Grok 4.7) from the Talent Growth Program brief, “Applied AI – AI Investigation Engine Task.”

## What the assistant did

- Read the brief and turned it into a Python investigation engine rather than a generic chat-with-PDF app.
- Chose the stack: FastAPI, Chroma, FastEmbed MiniLM, a local Qwen GGUF fallback, and HotpotQA plus a small contradictory seed case.
- Wrote the ingestion, the two chunkers, simhash dedup, Okapi BM25, Evidence Coverage Rank Fusion, the tool-calling agent, the citation and abstention checks, the HTTP API, and the interface.
- Wrote the corpus builder, the evaluation harness, the tests, the Dockerfile, and this documentation.
- Ran `scripts/evaluate.py` on the 50,009-document index with the local 1.5B model and wrote `EVALUATION_REPORT.md` from those scores. After the first pass abstained on facts that were already in the retrieved chunks, the assistant changed citation binding so an uncited claim is kept only when a retrieved chunk contains it. The author should still be able to explain that change.

## What was not delegated to a library

BM25, reciprocal-rank fusion, the coverage and diversity pass, simhash, and the contradiction rules are implemented in this repository. Chroma and FastEmbed are used for vector storage and embeddings. The language model writes the draft. It does not decide which passages exist: uncited claims are removed in code.

## What still has to be done by the author of the submission

The assistant cannot sit the oral defense. Before that conversation, the author should be able to:

- Walk the request path from `POST /api/investigate` through `InvestigationAgent`, `rank` in `app/indexing/hybrid.py`, and `enforce_grounding`.
- Explain why sentence packing is the default and what the fixed-window configuration gave up, using the numbers in `EVALUATION_REPORT.md`.
- Change a threshold (for example `ABSTAIN_COVERAGE`) or a tool schema and say what that does to abstention.
- Say which parts they rewrote after the assistant’s draft. If nothing was rewritten, say that plainly. The brief grades the author’s own understanding, and the practical portion of the interview does not allow an AI assistant.

## Tools

| Tool | Use |
| --- | --- |
| Cursor cloud agent (Grok 4.7) | Implementation and documentation |
| Hugging Face Datasets | HotpotQA distractor train, streaming |
| Hugging Face Hub | MiniLM ONNX weights and the Qwen2.5 1.5B GGUF file |

No paid API was required to build or run the system. A Groq, Gemini, or OpenAI key improves the writer and is optional.
