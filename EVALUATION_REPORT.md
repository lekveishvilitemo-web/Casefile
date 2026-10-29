# Evaluation report

Evaluated 44 questions with local (qwen2.5-1.5b-instruct-q4_k_m.gguf). hybrid_sentence answer correctness 0.283, Recall@5 0.673. semantic_fixed answer correctness 0.289, Recall@5 0.640.

## Method

Questions are frozen in `data/corpus/eval_questions.json`. Seed questions cover the Northline payments case: simple lookup, multi-document joins, contradictions, unanswerable facts, and a prompt-injection note. The rest are HotpotQA questions whose supporting sentences are inside the indexed subset.

Two configurations share the same questions, the same embeddings, and the same model:

### hybrid_sentence

Sentence packing, BM25 + vector fusion (ECRF), and a bridge-entity second hop.

- Questions: 44
- Recall@5: 0.673
- MRR: 0.774
- Answer correctness: 0.283
- Citation accuracy: 0.512

| Type | n | Recall@5 | MRR | Answer | Citations |
| --- | --- | --- | --- | --- | --- |
| contradiction | 2 | 1.000 | 1.000 | 1.000 | 0.833 |
| injection | 1 | 1.000 | 0.500 | 1.000 | 1.000 |
| multihop | 35 | 0.617 | 0.750 | 0.162 | 0.462 |
| simple | 3 | 1.000 | 1.000 | 0.586 | 0.567 |
| unanswerable | 3 | n/a | n/a | 0.667 | 0.667 |

Lowest-scoring answers:

- `seed-unknown-june` (unanswerable) gold=None predicted='The ledger-service v2.4.1 deploy on 2 March 2024 caused the payment delays, reducing retries from 5 to 1 and cutting the database connection pool from 40 to 8. [doc_44f3bb212d27_s0]' correctness=0.00
- `hotpot-5a7a06935542990198eaf050` (multihop) gold="Arthur's Magazine" predicted='First for Women was started first.' correctness=0.00
- `hotpot-5a8d7341554299441c6b9fe5` (multihop) gold='President Richard Nixon' predicted='Milhouse\'s middle name is "Mussolini."' correctness=0.00
- `hotpot-5a82171f5542990a1d231f4a` (multihop) gold='American' predicted='ABSTAIN' correctness=0.00
- `hotpot-5a84dd955542997b5ce3ff79` (multihop) gold='alcohol' predicted='Cadmium Chloride is slightly soluble in water and is also called cadmium chloride.' correctness=0.00
- `hotpot-5adf44985542993a75d2646d` (multihop) gold='Crambidae' predicted='Eumacaria contains only one species, Eumacaria madopata, the brown-bordered geometer moth, which is found in North America.' correctness=0.00
- `hotpot-5a89372855429951533612e6` (multihop) gold='6.213 km long' predicted='12 hours' correctness=0.00
- `hotpot-5abd90545542996e802b47d7` (multihop) gold='Jaime Meline' predicted='El-P' correctness=0.00

### semantic_fixed

Fixed windows and vector similarity only. No BM25 and no second hop.

- Questions: 44
- Recall@5: 0.640
- MRR: 0.755
- Answer correctness: 0.289
- Citation accuracy: 0.540

| Type | n | Recall@5 | MRR | Answer | Citations |
| --- | --- | --- | --- | --- | --- |
| contradiction | 2 | 1.000 | 1.000 | 0.750 | 0.500 |
| injection | 1 | 1.000 | 0.500 | 1.000 | 1.000 |
| multihop | 35 | 0.578 | 0.727 | 0.241 | 0.562 |
| simple | 3 | 1.000 | 1.000 | 0.253 | 0.361 |
| unanswerable | 3 | n/a | n/a | 0.333 | 0.333 |

Lowest-scoring answers:

- `seed-simple-incident-id` (simple) gold='INC-1842' predicted='CONFLICT' correctness=0.00
- `seed-unknown-oncall` (unanswerable) gold=None predicted='Charles Bourseul' correctness=0.00
- `seed-unknown-june` (unanswerable) gold=None predicted='The payments for Northline were affected by the rollback of the ledger-service to v2.3.9 on March 6, 2024, as documented in the postmortem INC-1842. [doc_a9d4a22cf267_f0]' correctness=0.00
- `seed-multi-notifier` (multihop) gold='merchant-notifier, rollback at 2024-03-06 11:40 UTC' predicted='ABSTAIN' correctness=0.00
- `hotpot-5a82171f5542990a1d231f4a` (multihop) gold='American' predicted='ABSTAIN' correctness=0.00
- `hotpot-5a84dd955542997b5ce3ff79` (multihop) gold='alcohol' predicted='Cadmium Chloride is slightly soluble in water and is also called cadmium chloride.' correctness=0.00
- `hotpot-5adf44985542993a75d2646d` (multihop) gold='Crambidae' predicted="Nepita conferta is the genus of moth in the world's seventh-largest country (India and Sri Lanka) that contains only one species." correctness=0.00
- `hotpot-5a7d0db955429909bec76924` (multihop) gold='2006' predicted='2011' correctness=0.00

## Analysis

The run used the on-disk index of 50,009 documents and 168,016 chunks, MiniLM embeddings, and the local Qwen2.5-1.5B-Instruct Q4 model at temperature 0. No API key was set. hybrid_sentence is sentence packing plus BM25, Evidence Coverage Rank Fusion, and a bridge-entity hop. semantic_fixed is fixed windows and vector similarity only.

hybrid_sentence Recall@5 is 0.673 and MRR is 0.774. semantic_fixed is 0.640 and 0.755. The fusion is a small retrieval gain, and it is concentrated where the wording matters: simple Northline lookups are Recall@5 1.000 in both, but answer correctness on those three questions is 0.586 with sentence packing and 0.253 with fixed windows. The fixed-window run answered "What incident id was assigned to the March 2024 payment delays?" with `CONFLICT`. The window had cut the id away from the question words, and a nearby disagreement was treated as the answer. Sentence packing kept "Incident INC-1842" in one chunk, and the hybrid run returned `INC-1842`.

Answer correctness does not follow that retrieval gain. On the 38 answerable questions, hybrid had Recall@5 of at least 0.5 on 32, and 16 of those 32 still scored 0 on the answer. Four questions had Recall@5 of 0. The 1.5B model is the binding constraint, in three repeating ways:

- It picks the wrong side of a comparison when both pages are present. "Which magazine was started first, Arthur's Magazine or First for Women?" retrieved one supporting sentence (Recall@5 0.500) and answered "First for Women was started first." The same pattern appears as `El-P` for `Jaime Meline`.
- It quotes a different fact from a page that does contain the answer. The Milhouse question had Recall@5 1.000 and the model answered with the middle name "Mussolini" instead of President Richard Nixon. "6.213 km long" came back as "12 hours" because the Bathurst 12 Hour page was in the evidence.
- It abstains when the page never arrives, which is the right behavior, and it also abstains on 7 answerable hybrid questions. `American` is the clean case: Recall@5 0.000 and the answer is `ABSTAIN`.

Unanswerable is 0.667 hybrid and 0.333 semantic. Revenue and the on-call name are refused in the hybrid run. "What happened to Northline payments in June 2024?" is not. The model copied the March ledger deploy, which is in the index, and never noticed that June is not. The semantic run is worse on this slice: the on-call question was answered "Charles Bourseul", a person from an unrelated Hotpot page that vector search ranked into the window. Keyword overlap with "INC-1842" is what kept the hybrid run on the incident file.

Contradictions are the strong slice. Both Northline clashes (OrbitalNet versus the ledger deploy, and checkout-api versus the finance note) were flagged in the hybrid run, with answer correctness 1.000 and citation accuracy 0.833. The injection note did not override the answer.

The next change worth making is to keep this ranker and swap only the draft model for `llama-3.1-8b-instant` via `GROQ_API_KEY`. Recall@5 is already ahead of the written answer, and 16 of the 32 retrieved hybrid questions failed in the writer. A scope check can sit in front of that: if the question names a month or year that the cited chunk does not contain, abstain. That one rule is the June miss, and it does not require a new index.
