# 🔍 Embeddings & Vector Search

---

## What Are Embeddings?

An **embedding** is a numerical representation (a vector of floating-point numbers) that captures the **semantic meaning** of text (or images, audio, etc.) in a high-dimensional space.

> Think of embeddings as "coordinates in meaning space" — where similar meanings are located close together.

```
"dog"     → [0.21, -0.45, 0.87, 0.12, ...]  (768 or 1536 numbers)
"puppy"   → [0.22, -0.43, 0.85, 0.14, ...]  ← Very close!
"cat"     → [0.19, -0.40, 0.62, 0.09, ...]  ← Somewhat close
"table"   → [-0.55, 0.71, 0.03, 0.88, ...]  ← Far away
```

Two pieces of text with similar meaning will have vectors that **point in similar directions** in this high-dimensional space.

---

## Why Embeddings Matter

Before embeddings, text similarity required exact keyword matching:
- "automobile" ≠ "car" (different words, same meaning)
- Keyword search misses synonyms, paraphrases, related concepts

With embeddings:
- "automobile" ≈ "car" (similar vectors)
- "My stomach hurts" ≈ "I have abdominal pain" (semantic similarity captured)
- A search for "heart attack symptoms" finds documents about "myocardial infarction"

---

## How Are Embeddings Created?

Embeddings are produced by **encoder models** — neural networks trained to map text into vectors.

### The Training Objective
Models like **BERT**, **Sentence Transformers**, and **OpenAI's text-embedding-ada-002** are trained to produce:
- **Similar vectors** for semantically related text
- **Different vectors** for unrelated text

Using objectives like:
- **Contrastive learning** — pull similar pairs together, push dissimilar pairs apart
- **Masked Language Modeling** (BERT)
- **Next Sentence Prediction**

### Common Embedding Models

| Model | Provider | Dimensions | Notes |
|---|---|---|---|
| text-embedding-ada-002 | OpenAI | 1536 | Industry standard, widely used |
| text-embedding-3-small | OpenAI | 1536 | Cheaper, newer |
| text-embedding-3-large | OpenAI | 3072 | Higher quality |
| all-MiniLM-L6-v2 | HuggingFace | 384 | Fast, open-source |
| all-mpnet-base-v2 | HuggingFace | 768 | Better quality, open-source |
| embed-english-v3.0 | Cohere | 1024 | Commercial |
| BGE-large | BAAI | 1024 | Strong open-source performer |
| Nomic Embed | Nomic AI | 768 | Open, performant |

### Dimensionality
The number of dimensions is a design choice:
- **Higher dimensions** = more expressive, capture more nuance
- **Lower dimensions** = faster, cheaper, less storage

---

## Types of Embeddings

### 1. Word Embeddings (Early)
- **Word2Vec** (2013), **GloVe** (2014)
- One fixed vector per word
- Problem: No context — "bank" always has the same vector whether it means riverbank or financial bank

### 2. Contextual Embeddings (Modern)
- **BERT**, **Sentence Transformers**, **OpenAI embeddings**
- Each word gets a vector **based on its context**
- "bank" has different embeddings in "river bank" vs "bank account"

### 3. Sentence / Document Embeddings
- A single vector for an entire sentence or paragraph
- Produced by pooling token-level embeddings
- Used for semantic search, classification, clustering

### 4. Multimodal Embeddings
- **CLIP** (OpenAI) — same vector space for images and text
- "A dog playing in snow" → embedded close to 🐕❄️ photos
- Enables cross-modal search

---

## Measuring Similarity: Distance Metrics

Once you have embedding vectors, you measure similarity using distance metrics:

### Cosine Similarity (Most Common)
Measures the **angle** between two vectors (ignores magnitude):

```
Cosine Similarity = (A · B) / (|A| × |B|)

Range: -1 (opposite) to 1 (identical)

"dog" vs "puppy"    → 0.92  (very similar)
"dog" vs "cat"      → 0.75  (related)
"dog" vs "database" → 0.21  (unrelated)
```

### Euclidean Distance (L2)
Measures straight-line distance between vectors:
- Smaller = more similar

### Dot Product
Similar to cosine but also considers magnitude. Used in some retrieval systems.

### When to Use Which
| Metric | Best For |
|---|---|
| Cosine Similarity | Most text similarity tasks |
| Euclidean (L2) | When magnitude matters |
| Dot Product | Maximum inner product search (MIPS) |

---

## Vector Databases

To do fast similarity search over millions of embeddings, you need a **vector database** (or vector index).

### The Problem
Comparing a query vector against 10 million stored vectors one-by-one is too slow (O(n) search).

### The Solution: Approximate Nearest Neighbor (ANN) Search
Algorithms like **HNSW**, **IVF**, and **LSH** allow searching millions of vectors in milliseconds by trading a tiny bit of accuracy for massive speed gains.

### Popular Vector Databases

| Database | Type | Notes |
|---|---|---|
| **Pinecone** | Managed cloud | Easy to use, production-ready |
| **Weaviate** | Open-source / Cloud | Multi-modal support |
| **Qdrant** | Open-source / Cloud | Fast, Rust-based |
| **Chroma** | Open-source | Lightweight, great for prototyping |
| **Milvus** | Open-source | Enterprise-grade, large scale |
| **pgvector** | PostgreSQL extension | Add vectors to existing Postgres DB |
| **FAISS** | Library (Meta) | Fast, not a full DB, used in research |
| **Redis** | With vector extension | Familiar infrastructure |

### Vector DB Core Operations
```python
# Store an embedding
db.upsert(id="doc_1", vector=[0.21, -0.45, ...], metadata={"text": "..."})

# Search for similar vectors
results = db.query(vector=[0.22, -0.43, ...], top_k=5)
# Returns: [("doc_1", score=0.97), ("doc_3", score=0.89), ...]
```

---

## Semantic Search vs Keyword Search

| Feature | Keyword (BM25) | Semantic (Embeddings) |
|---|---|---|
| Matching | Exact words | Meaning/intent |
| Synonyms | ❌ Misses | ✅ Handles |
| Paraphrases | ❌ Misses | ✅ Handles |
| Typos | Partial | ✅ Often handles |
| Technical terms | ✅ Good | Depends on training |
| Speed | Very fast | Fast (with ANN) |
| Interpretability | High | Low (black box) |
| Multilingual | Needs translation | ✅ Native support |

### Hybrid Search (Best of Both)
Most production systems combine both approaches:
```
Query → [Keyword Search (BM25)] + [Semantic Search (Embeddings)]
                    ↓                         ↓
             keyword_results          semantic_results
                         ↓
              [Re-ranking / Fusion]
                         ↓
                  Final results
```

---

## End-to-End Embedding Pipeline

```
1. DOCUMENTS
   "Machine learning is a subset of AI..."
   "The Eiffel Tower was built in 1889..."
   "Python is a high-level programming language..."
        ↓
2. CHUNKING
   Split documents into manageable chunks (200-500 tokens each)
        ↓
3. EMBEDDING MODEL
   Each chunk → vector (e.g., [0.21, -0.45, 0.87, ...])
        ↓
4. VECTOR STORE
   Store vectors + metadata (original text, source, date, etc.)
        ↓
5. QUERY
   "What programming language was created by Guido van Rossum?"
        ↓
6. QUERY EMBEDDING
   Query → vector
        ↓
7. SIMILARITY SEARCH
   Compare query vector vs all stored vectors → top-k most similar
        ↓
8. RETRIEVE
   Return original text of top-k chunks
        ↓
9. USE IN LLM
   "Based on the following: [retrieved chunks], answer: [query]"
```

---

## Embeddings Beyond Text

| Modality | Application |
|---|---|
| **Images** | Image search, duplicate detection, visual similarity |
| **Audio** | Music recommendation, speaker identification |
| **Code** | Code search, bug detection, similar function detection |
| **Products** | E-commerce recommendation |
| **Graphs** | Node classification, link prediction |
| **User behavior** | Recommendation systems |

---

## Key Concepts Summary

```
EMBEDDING
├── A vector of numbers representing meaning
├── Similar meaning → similar vectors (high cosine similarity)
├── Dimensions: typically 384 – 3072
└── Created by: encoder models (BERT, OpenAI, etc.)

SIMILARITY METRICS
├── Cosine Similarity: angle-based, most common for text
├── Euclidean Distance: geometric distance
└── Dot Product: used in retrieval systems

VECTOR DATABASES
├── Store embeddings + metadata
├── Enable fast approximate nearest neighbor search
└── Examples: Pinecone, Chroma, Weaviate, pgvector

SEMANTIC SEARCH
├── Find relevant content by meaning, not exact words
├── Powers: Q&A systems, document search, RAG
└── Often combined with keyword search (hybrid)
```

---

*Previous: [06 — Hallucinations & Limitations](./06_Hallucinations_Limitations.md)*
