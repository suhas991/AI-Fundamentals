# RAG: What, Why, and How to Build

---

## What Is RAG?

RAG (Retrieval-Augmented Generation) combines:
1. Retrieval: fetch relevant external knowledge.
2. Generation: answer using an LLM grounded on retrieved context.

It is a way to make LLM outputs more factual and up-to-date without retraining the base model.

---

## Why Use RAG?

- Knowledge freshness: use latest docs without model retraining.
- Domain grounding: use private company data.
- Better factuality: reduce unsupported answers.
- Lower cost than frequent fine-tuning for changing knowledge.

---

## End-to-End RAG Pipeline

```
Documents -> Chunking -> Embeddings -> Vector Index
                                     |
User Query -> Query Embedding -> Retrieval (top-k)
                                     |
                             (Optional) Re-ranker
                                     |
                      Prompt Assembly (context + instruction)
                                     |
                              LLM Generation
                                     |
                        Answer + Citations/Evidence
```

---

## How to Build (Practical Steps)

### 1. Data ingestion
- Collect high-quality sources (wiki, docs, tickets, policies).
- Keep metadata (source, timestamp, owner, access scope).

### 2. Chunking strategy
- Typical chunk size: 200 to 800 tokens.
- Use overlap to preserve context continuity.
- Prefer semantic boundaries (headings/paragraphs) over fixed windows when possible.

### 3. Embedding and indexing
- Choose embedding model aligned with language/domain.
- Store vectors + metadata in a vector store.
- Build filters for tenant/security/document type.

### 4. Retrieval design
- Dense retrieval for semantic matching.
- Hybrid retrieval (BM25 + vector) often improves recall.
- Tune top-k and similarity threshold.

### 5. Prompt construction
- Instruction to answer only from provided context.
- Include citations and uncertainty behavior.
- Guard against context overflow.

### 6. Evaluation and iteration
- Offline metrics: recall@k, MRR, nDCG.
- Output metrics: faithfulness, groundedness, answer relevance.
- Online metrics: latency, cost/query, user satisfaction.

---

## RAG vs Fine-Tuning (Interview Framing)

| Aspect | RAG | Fine-tuning |
|---|---|---|
| Updates knowledge quickly | Yes | No (requires retraining) |
| Teaches style/format behavior | Somewhat | Strong |
| Works with private docs | Yes | Yes |
| Cost for frequently changing facts | Usually lower | Usually higher |
| Hallucination reduction | Better when retrieval quality is high | Not guaranteed |

---

## Common Failure Modes and Fixes

1. Bad chunks -> irrelevant retrieval.
- Fix: improve chunk boundaries and metadata.

2. Low recall at retrieval.
- Fix: hybrid search, better embedding model, reranking.

3. LLM ignores context.
- Fix: stronger grounding prompt and citation requirements.

4. Security leakage through retrieval.
- Fix: strict ACL filters before retrieval results reach prompt.

---

## Key Takeaways

- RAG = retrieve first, generate second.
- Retrieval quality determines answer quality.
- Production RAG needs ranking, security, and evaluation loops.
- Use RAG for changing knowledge; use fine-tuning for durable behavior shifts.
