# 10 · Vector Databases & Embeddings

---

## 1. What Are Embeddings?

An **embedding** is a dense, fixed-size numerical vector that represents the **semantic meaning** of text (or images, audio, etc.).

```
"The cat sat on the mat"  →  [0.12, -0.54, 0.87, ..., 0.33]  (1536 numbers)
"A feline rested on a rug" →  [0.11, -0.52, 0.85, ..., 0.31]  (very similar!)
"Quantum computing basics"  →  [-0.88, 0.22, -0.43, ..., 0.67] (very different)
```

**Key property:** Semantically similar texts have vectors that are close together in vector space.

### How Embeddings Are Generated
Embedding models (e.g., BERT, text-embedding-3) are transformers that:
1. Tokenise input text
2. Pass through encoder layers
3. Pool hidden states → single vector

---

## 2. Embedding Dimensions

| Model | Dimensions | Notes |
|-------|-----------|-------|
| text-embedding-ada-002 | 1536 | OpenAI; widely used |
| text-embedding-3-small | 1536 | OpenAI; cheaper |
| text-embedding-3-large | 3072 | OpenAI; best quality |
| all-MiniLM-L6-v2 | 384 | Lightweight; fast |
| BGE-large-en | 1024 | Strong open-source |
| Cohere Embed v3 | 1024 | Enterprise |
| GTE-large | 1024 | Good multilingual |

**Higher dimensions:**
- Can represent more nuanced differences
- Require more memory and compute
- Don't always outperform lower-dimensional models

**Matryoshka Representation Learning (MRL):** Models like text-embedding-3 support **dimension truncation** — you can use 256, 512, or 1536 dims from the same model.

---

## 3. Vector Search — ANN Algorithms

Exact nearest-neighbour search is O(n·d) — too slow for millions of vectors.

**ANN (Approximate Nearest Neighbour)** trades slight recall loss for massive speed gains.

### HNSW (Hierarchical Navigable Small World)

The most popular algorithm used in production vector databases.

**How it works:**
- Build a multi-layer graph where each node has links to nearby neighbours
- Top layers = coarse, long-range connections
- Bottom layers = fine, short-range connections
- Query: Enter at top layer, greedily navigate toward query vector, descend

```
Layer 2: o——————————————o
Layer 1: o——o——————o——o
Layer 0: o-o-o-o-o-o-o-o  (densest)
              ↑ search here for fine results
```

**Pros:** Very fast queries, high recall, dynamic insertion  
**Cons:** High memory usage

### IVF (Inverted File Index)

- Cluster all vectors into `nlist` Voronoi cells during indexing
- Query: find the `nprobe` nearest centroids, then search only within those cells

**Pros:** Lower memory than HNSW  
**Cons:** Lower recall; full rebuild needed to add vectors

### HNSW vs IVF

| Aspect | HNSW | IVF |
|--------|------|-----|
| Speed | Very fast | Fast |
| Recall | High | Moderate |
| Memory | High | Lower |
| Dynamic updates | ✅ Easy | ❌ Needs rebuild |
| Best for | Real-time search | Batch / large static datasets |

---

## 4. Popular Vector Databases

| Database | Type | Highlights |
|----------|------|-----------|
| **Pinecone** | Managed cloud | Serverless; easy scaling; metadata filtering |
| **Weaviate** | Open-source + cloud | Multi-modal; GraphQL API; modules |
| **Qdrant** | Open-source + cloud | Rust-based; very fast; rich payload filtering |
| **ChromaDB** | Open-source embedded | Best for prototyping; zero setup |
| **pgvector** | PostgreSQL extension | SQL + vector search in one DB |
| **Milvus** | Open-source | Enterprise scale; cloud-native |
| **FAISS** | Library (Meta) | In-memory; research and offline use |
| **Azure AI Search** | Managed | Hybrid search; enterprise; Azure native |

### Choosing a Vector DB

```
Prototype / local dev → ChromaDB
Production (managed)  → Pinecone or Qdrant Cloud
Already using Postgres → pgvector
Enterprise / Azure    → Azure AI Search
Multi-modal           → Weaviate
High throughput       → Milvus or Qdrant
```

---

## 5. Chunking Strategies

| Strategy | Description | Chunk Size | Overlap |
|----------|-------------|-----------|---------|
| **Fixed-size** | Split every N tokens | 256–1024 | 10–20% |
| **Sliding window** | Fixed with overlap | 512 | 50–100 tokens |
| **Recursive character** | Try `\n\n`, `\n`, `.`, ` ` in order | 512 | 50 |
| **Sentence-based** | Split at sentence boundaries | Varies | 1–2 sentences |
| **Semantic** | Split on embedding similarity drop | Varies | None |
| **Document-aware** | Split by headings / sections | Varies | None |

### Rules of Thumb
- Shorter chunks → higher precision, may lose context
- Longer chunks → more context, lower retrieval precision
- Sweet spot: **256–512 tokens** for most use cases
- Always add **metadata** (source, page, section, date) to each chunk

---

## 6. Metadata Filtering

Combine vector similarity search with structured filters to narrow results.

```python
# Qdrant example: semantic search filtered by department and date
results = client.search(
    collection_name="docs",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="department", match=MatchValue(value="legal")),
            FieldCondition(key="year", range=Range(gte=2023))
        ]
    ),
    limit=10
)
```

**Common metadata fields:**
- `source` / `file_name`
- `page_number`
- `section` / `chapter`
- `date` / `created_at`
- `author`
- `language`
- `document_type` (policy, faq, contract)

**Pre-filtering vs Post-filtering:**
- Pre-filter (where supported): filter before ANN search — faster, fewer candidates
- Post-filter: retrieve more candidates, then filter — safer but slower

---

## Quick Reference

```
Embedding       = dense vector capturing semantic meaning
Similarity      = cosine similarity (dot product on normalised vectors)
ANN             = HNSW (fast, dynamic) or IVF (memory-efficient)
Chunking        = 256–512 tokens + 10% overlap as starting point
Metadata        = always attach source/date/section to chunks
DB choice       = ChromaDB (dev) → Pinecone/Qdrant (prod)
Same model      = must use same embedding model for index & query
```
