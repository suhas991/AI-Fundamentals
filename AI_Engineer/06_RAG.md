# 06 · RAG — Retrieval-Augmented Generation

---

## 1. Why RAG?

LLMs have a **knowledge cutoff** — they cannot answer questions about recent events or private/proprietary data.

**Problems RAG solves:**
- Stale knowledge (training data cutoff)
- Hallucinations on domain-specific facts
- No access to private enterprise documents
- Prohibitive cost of frequent fine-tuning

**Core idea:** At query time, *retrieve* relevant documents from an external store and *augment* the LLM prompt with that context.

```
User Query → Retrieve Docs → Augment Prompt → LLM → Answer
```

---

## 2. RAG Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│  INDEXING (Offline)                                  │
│  Documents → Chunk → Embed → Store in Vector DB      │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  RETRIEVAL & GENERATION (Online)                     │
│  Query → Embed Query → Vector Search → Top-k Chunks  │
│  → Inject into Prompt → LLM → Response               │
└─────────────────────────────────────────────────────┘
```

---

## 3. Document Ingestion & Chunking Strategies

### Why Chunk?
LLMs have a limited context window. Large documents must be split into smaller pieces for effective embedding and retrieval.

### Chunking Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Fixed-size** | Split every N tokens/characters | Simple, fast baseline |
| **Sliding window** | Fixed chunks with overlap (e.g., 512 tokens, 50-token overlap) | Preserves context across boundaries |
| **Recursive character** | Split by `\n\n`, `\n`, `.`, ` ` in order | General text documents |
| **Semantic chunking** | Split on sentence embedding similarity shifts | High-quality retrieval |
| **Document structure** | Split by headings, sections (Markdown/HTML aware) | Structured docs |

**Overlap recommendation:** 10–20% of chunk size to avoid cutting mid-sentence context.

---

## 4. Embedding Models

Embedding models convert text into dense vector representations that capture semantic meaning.

| Model | Dims | Notes |
|-------|------|-------|
| **text-embedding-ada-002** | 1536 | OpenAI; widely used baseline |
| **text-embedding-3-large** | 3072 | OpenAI; best quality |
| **BGE-M3** | 1024 | BAAI; multilingual; open-source |
| **E5-large** | 1024 | Strong cross-lingual |
| **Cohere Embed v3** | 1024 | Enterprise; supports int8 |
| **all-MiniLM-L6-v2** | 384 | Lightweight; fast |

**Key properties:**
- Higher dimensions ≠ always better (compute vs quality trade-off)
- Same model must be used for both indexing and querying
- Specialise for domain (legal, medical) if needed

---

## 5. Vector Databases

Databases optimised for storing and querying high-dimensional vectors.

| Database | Notes |
|----------|-------|
| **Pinecone** | Managed cloud; serverless option |
| **Weaviate** | Open-source + cloud; multi-modal |
| **ChromaDB** | Lightweight; embedded; great for prototyping |
| **Qdrant** | Rust-based; fast; filterable payloads |
| **pgvector** | PostgreSQL extension; SQL + vectors |
| **Milvus** | Scalable open-source |
| **FAISS** | Facebook; in-memory; research use |

---

## 6. Similarity Search

### Cosine Similarity
Measures the angle between two vectors — most common for text embeddings.

```
cos(A, B) = (A · B) / (|A| × |B|)
Range: -1 (opposite) to 1 (identical)
```

### ANN — Approximate Nearest Neighbour
Exact k-NN search is O(n·d) — too slow for large collections. ANN trades slight accuracy for massive speed gains.

| Algorithm | How It Works |
|-----------|-------------|
| **HNSW** (Hierarchical Navigable Small World) | Graph-based; very fast queries; high recall |
| **IVF** (Inverted File Index) | Cluster centroids; only search nearby clusters |
| **LSH** (Locality Sensitive Hashing) | Hash similar vectors to same bucket |

**HNSW is the most widely used** in production vector databases.

---

## 7. Naive RAG vs Advanced RAG

### Naive RAG
```
Query → Embed → Top-k Similarity → Concatenate → LLM
```
**Limitations:** Retrieves semantically similar but not always relevant chunks; no ranking; context stuffing.

### Advanced RAG Techniques

| Technique | Description |
|-----------|-------------|
| **Query rewriting** | LLM reformulates the query before retrieval |
| **HyDE** (Hypothetical Document Embeddings) | Generate a hypothetical answer, embed it, retrieve similar chunks |
| **Multi-query retrieval** | Generate N variants of the query, merge results |
| **Contextual compression** | Extract only the relevant sentences from retrieved chunks |
| **Self-RAG** | Model decides when to retrieve and critiques its own output |
| **Corrective RAG** | Evaluates retrieved docs, falls back to web search if poor quality |

---

## 8. Hybrid Search

Combines **semantic (vector) search** and **keyword (lexical) search** for better recall.

```
Hybrid Score = α × Vector Score + (1-α) × BM25 Score
```

### BM25 (Best Match 25)
A classic TF-IDF based ranking function — excellent for exact keyword matching.

**Why hybrid?**
- Semantic search fails on exact terms (product codes, names, acronyms)
- Keyword search misses paraphrasing and synonyms
- Hybrid gets the best of both

**RRF (Reciprocal Rank Fusion):** Merge ranked lists without needing score normalisation.

---

## 9. Re-Ranking

After initial retrieval, apply a **cross-encoder** to re-score and re-rank the top-k results.

```
Query + Document → Cross-Encoder → Relevance Score
```

**Bi-encoder (retrieval):** Encodes query and document independently — fast but less accurate.  
**Cross-encoder (re-ranking):** Encodes query and document together — slower but much more accurate.

**Popular re-rankers:** Cohere Rerank, BGE Reranker, FlashRank

**Typical pipeline:**
```
Retrieve top-100 → Re-rank → Keep top-5 → LLM
```

---

## 10. RAG Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| **Faithfulness** | Is the answer grounded in the retrieved context? | No hallucinations |
| **Answer Relevance** | Does the answer address the question? | On-topic |
| **Context Precision** | Are retrieved chunks actually useful? | Low noise |
| **Context Recall** | Did we retrieve all necessary information? | No missed facts |

**Frameworks:** RAGAS, TruLens, DeepEval

---

## Quick Reference Cheat Sheet

```
Problem         → Solution
─────────────────────────────────────────────
Knowledge cutoff → RAG
Poor retrieval   → Hybrid search + reranking
Wrong chunks     → Better chunking + overlap
Slow queries     → ANN (HNSW) + caching
Hallucinations   → Faithfulness eval + citations
Exact terms fail → BM25 / keyword search boost
```
