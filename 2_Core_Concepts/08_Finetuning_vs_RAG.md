# 🔧 Fine-tuning vs RAG (Retrieval-Augmented Generation)

---

## The Core Problem

Pre-trained LLMs are incredibly capable, but they have a critical limitation for real-world applications:

> **They only know what was in their training data — and that data has a cutoff date.**

If you're building a product assistant, internal knowledge base tool, or domain-specific application, the base model:
- Doesn't know your company's products
- Doesn't know your internal processes
- Doesn't know about events after its training cutoff
- May not follow your organization's specific tone or format

You have two primary strategies to solve this: **Fine-tuning** and **RAG**.

---

## Fine-Tuning

### What is Fine-Tuning?

**Fine-tuning** is the process of continuing to train a pre-trained LLM on a **custom dataset** specific to your task or domain. The model's weights are actually updated — new knowledge is baked into the model itself.

```
Pre-trained LLM (generic) + Your custom data → Fine-tuned LLM (specialized)
```

### How Fine-Tuning Works

1. **Prepare your dataset** — curate high-quality examples in prompt/completion format
2. **Choose a base model** — GPT-4, Llama 3, Mistral, etc.
3. **Training** — run gradient descent on your data, updating model weights
4. **Evaluate** — test on held-out data
5. **Deploy** — serve the fine-tuned model

### Types of Fine-Tuning

#### Full Fine-Tuning
Update **all** model weights on your data.
- Maximum expressiveness
- Very expensive (requires GPUs, significant compute)
- Risk of **catastrophic forgetting** (model forgets general capabilities)
- Practical mostly for small models or very large organizations

#### Parameter-Efficient Fine-Tuning (PEFT)
Update only a **small subset** of parameters:

**LoRA (Low-Rank Adaptation)**
The most popular PEFT method. Instead of updating full weight matrices, it trains small "adapter" matrices:
```
Original weight matrix W (frozen)
           +
Trainable: A × B  (low-rank decomposition, much smaller)
= W + A×B  (during inference, merged back)
```
- 99%+ of parameters frozen
- Only ~0.1% of parameters trained
- Performance close to full fine-tuning at a fraction of cost
- Can "swap" different LoRA adapters for different tasks

**QLoRA** — Quantized LoRA — fine-tune with even less memory (quantize base model to 4-bit).

#### Instruction Fine-Tuning
Train on `(instruction, response)` pairs to make the model better at following instructions. This is what makes a base model into a "chat model."

#### RLHF / DPO
Training with human preference data (covered in LLMs note).

### When Fine-Tuning Makes Sense

✅ **Good use cases:**
- Teaching a specific **style, tone, or format** (e.g., always respond in JSON, match company voice)
- Building a domain-specific model (legal, medical, finance) that needs deep expertise
- **Reducing inference costs** — a smaller fine-tuned model can match a larger base model
- Teaching **new reasoning patterns** or behaviors
- Removing / reducing specific behaviors (safety tuning)
- On-device deployment (requires a small, highly optimized model)

❌ **Poor use cases:**
- Injecting **frequently changing information** (product prices, live data)
- Providing access to **proprietary documents** (better with RAG)
- Tasks requiring **citations** to specific sources

### Fine-Tuning Data Requirements

| Quality Tier | Dataset Size |
|---|---|
| Minimal (style/format changes) | 50–500 examples |
| Task-specific (classification, extraction) | 500–5,000 examples |
| Domain specialization | 10,000–100,000 examples |
| Full domain expertise | 100,000+ examples |

**Data quality > quantity** — 100 perfect examples often beat 10,000 mediocre ones.

### Fine-Tuning Risks

- **Catastrophic forgetting** — model loses general capabilities
- **Overfitting** — memorizes training data instead of generalizing
- **Data poisoning** — bad training data degrades model quality
- **Drift from alignment** — fine-tuning can undermine safety tuning

---

## RAG (Retrieval-Augmented Generation)

### What is RAG?

**Retrieval-Augmented Generation** is an architecture where the LLM's context is **dynamically enriched** with relevant information retrieved from an external knowledge base at inference time.

Instead of changing the model, you change what information the model sees.

```
User Question
     ↓
[Retrieve relevant documents from knowledge base]
     ↓
[Stuff retrieved docs into the prompt context]
     ↓
LLM answers based on retrieved context
```

### RAG Architecture

```
┌─────────────────────────────────────────────────────┐
│                  INDEXING (Offline)                  │
│                                                      │
│  Documents → Chunking → Embeddings → Vector Store   │
└─────────────────────────────────────────────────────┘
                           ↓ (stored)
┌─────────────────────────────────────────────────────┐
│                  QUERYING (Online)                   │
│                                                      │
│  User Query                                          │
│      ↓                                               │
│  [Query Embedding]                                   │
│      ↓                                               │
│  [Vector Search → Top-K chunks]                      │
│      ↓                                               │
│  [Prompt Assembly]                                   │
│  "Answer based on: [chunk1] [chunk2] [chunk3]        │
│   Question: [user query]"                            │
│      ↓                                               │
│  [LLM generates answer]                              │
│      ↓                                               │
│  [Optional: cite sources in response]                │
└─────────────────────────────────────────────────────┘
```

### RAG Components

| Component | Description | Examples |
|---|---|---|
| **Document loader** | Ingest source documents | PDFs, DOCX, web pages, databases |
| **Text splitter** | Chunk documents into pieces | 500 tokens with 50-token overlap |
| **Embedding model** | Convert chunks to vectors | text-embedding-3-small |
| **Vector store** | Store and search vectors | Pinecone, Chroma, pgvector |
| **Retriever** | Find relevant chunks | Semantic, keyword, hybrid |
| **Re-ranker** | Re-order retrieved chunks | Cross-encoder models |
| **LLM** | Generate final answer | GPT-4, Claude, Llama |
| **Prompt template** | Structure the final prompt | Includes context + question |

### When RAG Makes Sense

✅ **Good use cases:**
- **Up-to-date information** — news, documentation, live databases
- **Large knowledge bases** — too much info to fit in context or fine-tune on
- **Verifiable answers** — need citations to source documents
- **Frequently changing data** — product catalogs, pricing, policies
- **Regulatory / compliance** — trace every response to a source
- **Multi-domain knowledge** — mix of different document types

❌ **Poor use cases:**
- Tasks requiring deep, integrated reasoning across all knowledge
- When the knowledge base is too small and can just go in the system prompt
- When latency is critical (retrieval adds milliseconds to seconds)
- When you need the model to **internalize** patterns, not just retrieve facts

### Advanced RAG Techniques

#### Chunking Strategies
How you split documents matters enormously:

```
Fixed size:       Split every N tokens
Sentence:         Split at sentence boundaries
Paragraph:        Split at paragraph breaks
Semantic:         Split when topic changes
Recursive:        Try multiple splitters in order
Document-aware:   Respect document structure (headers, sections)
```

#### Re-ranking
After retrieving top-K chunks by vector similarity, re-rank them with a more powerful (but slower) cross-encoder:

```
Retrieved: [chunk3 (0.91), chunk7 (0.88), chunk1 (0.85)]
After re-rank: [chunk7 (0.97), chunk3 (0.89), chunk1 (0.72)]
```

#### Hypothetical Document Embedding (HyDE)
1. Ask the LLM to generate a *hypothetical answer* to the query
2. Embed that hypothetical answer
3. Use it to search (often better than embedding the question itself)

#### Parent Document Retrieval
- Store small chunks for precise retrieval
- When retrieved, return their larger parent context for the LLM

---

## Fine-Tuning vs RAG: Direct Comparison

| Dimension | Fine-Tuning | RAG |
|---|---|---|
| **Changes the model?** | Yes (weights updated) | No (same model) |
| **Knowledge update** | Requires retraining | Just update the database |
| **Handles live data?** | ❌ No | ✅ Yes |
| **Provides citations?** | ❌ Hard | ✅ Natural |
| **Handles large knowledge base?** | ❌ Limited by training | ✅ Unlimited |
| **Cost** | High upfront, lower inference | Lower upfront, inference cost for retrieval |
| **Latency** | Fast (no retrieval step) | Slightly higher (retrieval adds time) |
| **Style / behavior tuning** | ✅ Excellent | ❌ Not suitable |
| **Confidential data security** | Risk during training | Data stays in your DB |
| **Reproducibility** | Fixed (after training) | Depends on retrieval |
| **Complexity** | High (training pipeline) | Medium (indexing + retrieval pipeline) |
| **Best for** | Format, tone, task specialization | Knowledge-intensive, dynamic info |

---

## Using Both Together

Fine-tuning and RAG are **not mutually exclusive** — the best production systems often combine them:

```
FINE-TUNING handles:
  ✓ Response format (always return JSON)
  ✓ Tone and persona (formal, brand-consistent)
  ✓ Domain reasoning (medical, legal expertise)
  ✓ Instruction following patterns

+

RAG handles:
  ✓ Specific factual knowledge (product specs, policies)
  ✓ Up-to-date information (recent events, live data)
  ✓ Large document corpora
  ✓ Citations and source tracing
```

### Example Stack
```
User: "What's the return policy for the Pro subscription?"
         ↓
[Fine-tuned model: trained to respond in company voice, 
 follow JSON schema, never make promises]
         ↓
[RAG retrieves: subscription_policy_v3.pdf, chunk 4]
         ↓
Model generates grounded, on-brand, cited response
```

---

## Quick Decision Framework

```
Ask yourself:

1. Do you need up-to-date or frequently changing info?
   YES → RAG

2. Do you need citations / traceability?
   YES → RAG

3. Do you have a very large knowledge base (100K+ docs)?
   YES → RAG

4. Do you want to change the model's style, format, or persona?
   YES → Fine-Tuning

5. Do you need domain-specific reasoning not in base model?
   YES → Fine-Tuning (or both)

6. Do you just want to provide context in each call?
   YES → Long context / system prompt (no fine-tuning or RAG needed)
```

---

## Summary

```
FINE-TUNING
├── Updates model weights on custom data
├── Bakes knowledge/behavior permanently into model
├── Best for: style, tone, format, specialized reasoning
├── Types: Full, LoRA, QLoRA, Instruction tuning
└── Requires: High-quality dataset + GPU compute

RAG
├── Retrieves relevant docs at query time
├── Injects them into the LLM's context
├── Best for: live data, large knowledge bases, citations
├── Components: Loader → Chunker → Embedder → VectorDB → LLM
└── Requires: Embedding model + vector database + retrieval logic

COMBINE BOTH for production-grade AI applications:
  Fine-tune for behavior + RAG for knowledge
```

---

*Previous: [07 — Embeddings & Vector Search](./07_Embeddings_Vector_Search.md)*

---

## Appendix: Useful Libraries & Frameworks

| Library | Purpose |
|---|---|
| **LangChain** | RAG pipelines, agents, chains |
| **LlamaIndex** | Advanced RAG, document ingestion |
| **Hugging Face PEFT** | LoRA, QLoRA, fine-tuning |
| **Unsloth** | Fast LoRA fine-tuning |
| **Axolotl** | Fine-tuning framework |
| **FAISS** | Vector similarity search |
| **Chroma** | Open-source vector DB |
| **Pinecone** | Managed vector DB |
| **Weaviate** | Open-source vector DB |
| **vLLM** | Fast LLM inference serving |
