# Natural Language Processing (NLP) — Study Notes
> For SDE-1 / AI Engineer Interview Prep | Suhas N H

---

## What is NLP?
Natural Language Processing is the branch of AI that enables machines to **understand, interpret, and generate human language** — text and speech.

```
Human Language (messy, ambiguous, contextual)
              ↓
           NLP Pipeline
              ↓
Machine-understandable representation → Task output
```

---

## 1. Tokenization, Embeddings & Vectors

### Tokenization — Breaking Text into Pieces

Before any model can process text, it must be split into **tokens** — the atomic units the model works with.

> A token is NOT always a word. It can be a word, part of a word, punctuation, or even a single character.

#### Types of Tokenization

**Word-level tokenization**
```
"I love deep learning" → ["I", "love", "deep", "learning"]
```
Problem: "running", "runner", "ran" are treated as completely different tokens — vocabulary explodes.

**Character-level tokenization**
```
"cat" → ["c", "a", "t"]
```
Problem: Sequences get very long, loses word-level meaning.

**Subword tokenization (used by modern LLMs)**
Best of both worlds — common words stay whole, rare words split into meaningful pieces.

```
"unhappiness" → ["un", "happiness"]
"tokenization" → ["token", "ization"]
"GPT"          → ["G", "PT"]
```

**Popular subword algorithms:**
| Algorithm | Used by |
|-----------|---------|
| BPE (Byte Pair Encoding) | GPT-2, GPT-4, LLaMA |
| WordPiece | BERT, DistilBERT |
| SentencePiece | T5, Gemini |

#### Tokens and Cost (Critical for AI Engineers)
```
"Hello, how are you?" ≈ 5 tokens
1,000 tokens          ≈ 750 words
GPT-4 charges per token — understanding this controls API cost
```

> **Interview line:** *"Tokenization is the very first step in any NLP pipeline. Modern LLMs use subword tokenization like BPE — it handles rare words and multiple languages efficiently without a massive vocabulary."*

---

### Embeddings — Giving Words Meaning as Numbers

Neural networks can't read words — they need numbers. **Embeddings convert tokens into dense vectors of floating point numbers** that capture semantic meaning.

```
"king"   → [0.21, -0.45, 0.87, 0.12, ...]   (300 or 1536 dimensions)
"queen"  → [0.19, -0.41, 0.85, 0.31, ...]
"apple"  → [-0.72, 0.33, -0.11, 0.94, ...]
```

Words with similar meanings end up with **similar vectors**.

#### The Magic of Embeddings — Semantic Arithmetic
```
king - man + woman ≈ queen

Paris - France + Italy ≈ Rome
```
This shows embeddings encode actual relationships — not just spellings.

#### One-Hot Encoding vs Embeddings

| | One-Hot | Embeddings |
|---|---|---|
| Representation | [0,0,1,0,0...] sparse | [0.21, -0.45...] dense |
| Dimensions | = vocabulary size (50,000+) | 300–1536 (fixed) |
| Captures meaning | ❌ No | ✅ Yes |
| Memory | Huge | Compact |

#### Popular Embedding Models
| Model | Dimensions | Best for |
|-------|-----------|---------|
| Word2Vec | 300 | Classic NLP |
| GloVe | 300 | Classic NLP |
| text-embedding-ada-002 | 1536 | OpenAI RAG |
| text-embedding-3-small | 1536 | OpenAI (cheaper) |
| BGE-large | 1024 | Open source RAG |
| sentence-transformers | 768 | Semantic search |

> **Interview line:** *"In AIPO, if I wanted semantic search over incident reports, I'd embed each report using a model like text-embedding-ada-002, store in a vector DB like ChromaDB, and retrieve the most relevant ones for any new incident query."*

---

### Vectors & Vector Space

An embedding IS a vector — a point in high-dimensional space.

```
2D simplified example:

            Technology
                ↑
    "GPU" •     |     • "CPU"
                |
    "Python" •  |  • "Java"
─────────────────────────────→ Programming
                |
    "Football"• |  • "Cricket"
                |
             Sports
```

Words in the same domain cluster together in vector space.

**Measuring similarity between vectors:**

**Cosine Similarity** — most common in NLP (measures angle, not distance)
```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)

Result: 1.0  = identical meaning
        0.0  = unrelated
       -1.0  = opposite meaning

"king" vs "queen"  → ~0.85 (very similar)
"king" vs "banana" → ~0.12 (unrelated)
```

**Euclidean Distance** — measures raw distance in space (less preferred for high dims)

> This is the core of **semantic search and RAG** — find vectors closest to the query vector.

---

---

## 2. Transformers Architecture & Attention Mechanism

### Why Transformers?

Before Transformers (2017), NLP used RNNs and LSTMs which processed text **sequentially** — word by word. Problems:
- Slow (can't parallelize)
- Forgot long-range dependencies (what was said 100 words ago?)

**Transformers** process the **entire sequence at once** using Attention — solving both problems.

> Paper: *"Attention Is All You Need"* — Vaswani et al., Google, 2017. Most influential AI paper of the decade.

---

### High-Level Transformer Architecture

```
INPUT TEXT → Tokenize → Token Embeddings + Positional Encoding
                                    ↓
                        ┌─────────────────────┐
                        │   ENCODER STACK     │  (understands input)
                        │                     │
                        │  Multi-Head         │
                        │  Self-Attention     │
                        │       +             │
                        │  Feed Forward       │
                        │       +             │
                        │  Layer Norm         │
                        │  (× N layers)       │
                        └─────────────────────┘
                                    ↓
                        ┌─────────────────────┐
                        │   DECODER STACK     │  (generates output)
                        │                     │
                        │  Masked Self-Attn   │
                        │       +             │
                        │  Cross-Attention    │
                        │       +             │
                        │  Feed Forward       │
                        │  (× N layers)       │
                        └─────────────────────┘
                                    ↓
                              OUTPUT TEXT
```

---

### Positional Encoding — Knowing Word Order

Transformers process all tokens simultaneously — so they need a way to know **position** of each token.

```
"Dog bites man" ≠ "Man bites dog"
```

Positional encodings are added to token embeddings to inject order information.

---

### The Attention Mechanism — The Core Idea

Attention answers: **"When processing this word, which other words in the sequence should I focus on?"**

```
Sentence: "The animal didn't cross the street because it was too tired"

When processing "it":
  Attention scores:
  "animal" → 0.85  ← HIGH attention (it refers to animal)
  "street" → 0.12
  "tired"  → 0.60  ← HIGH attention (why it didn't cross)
  "cross"  → 0.20
```

The model learns these relationships during training — not hardcoded.

#### Q, K, V — Query, Key, Value (The Mechanism)

Think of it like a **search engine**:
- **Query (Q)** — What am I looking for? (current word)
- **Key (K)** — What does each word advertise about itself?
- **Value (V)** — What information does each word actually contain?

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

Step 1: Score = Q × K^T       → how relevant is each token to the query?
Step 2: Scale = Score / √d_k  → prevent large values (d_k = key dimension)
Step 3: Softmax               → convert to probabilities (sum to 1)
Step 4: × V                   → weighted sum of values
```

You don't need to memorize the formula — understand the intuition.

> **Interview line:** *"Attention lets the model dynamically decide which parts of the input are relevant for processing each token. This is why Transformers handle long-range dependencies so much better than RNNs."*

---

### Multi-Head Attention

Instead of one attention, run **multiple attention heads in parallel** — each head learns different types of relationships.

```
Head 1 → focuses on syntactic relationships (subject-verb)
Head 2 → focuses on coreference (pronouns → nouns)
Head 3 → focuses on semantic similarity
...
Head 8 → ...

All heads concatenated → linear projection → rich representation
```

---

### Self-Attention vs Cross-Attention

| | Self-Attention | Cross-Attention |
|---|---|---|
| What attends to what | Token attends to other tokens in same sequence | Decoder attends to encoder output |
| Used in | Encoder, Decoder | Decoder only |
| Purpose | Understand context within sequence | Align input and output |

---

### Feed Forward Layer

After attention, each token passes through a **2-layer fully connected network** independently.

```
Attention output → Linear → ReLU → Linear → output
```

This is where a lot of the model's "knowledge" is stored — it acts like a key-value memory.

---

### Layer Normalization & Residual Connections

```
output = LayerNorm(x + SubLayer(x))
```

- **Residual connection** (+x) — lets gradients flow, prevents vanishing gradient
- **Layer norm** — stabilizes training by normalizing activations

---

---

## 3. BERT vs GPT — Two Transformer Families

The full Transformer has Encoder + Decoder. Modern models use only one half:

```
Full Transformer = Encoder + Decoder (original — used for translation)
BERT family     = Encoder only       (reads and understands)
GPT family      = Decoder only       (generates text)
```

---

### BERT — Bidirectional Encoder Representations from Transformers

**Google, 2018**

#### How it reads:
Sees the **entire sentence at once, in both directions** — left AND right context.

```
"The bank can guarantee [MASK] to this company"

BERT sees:
← "company" "this" "to" [MASK] "guarantee" "can" "bank" "The"
→ "The" "bank" "can" "guarantee" [MASK] "to" "this" "company"

Predicts: "deposits" (using full bidirectional context)
```

#### Training objectives:
- **Masked Language Model (MLM)** — randomly mask 15% of tokens, predict them
- **Next Sentence Prediction (NSP)** — predict if sentence B follows sentence A

#### What BERT is good at:
- Classification (spam, sentiment, intent)
- Named Entity Recognition (NER)
- Question Answering (extractive)
- Semantic similarity
- **NOT good at generation** — it's an encoder, not a generator

#### BERT variants:
| Model | Difference |
|-------|-----------|
| DistilBERT | 60% smaller, 97% performance |
| RoBERTa | Better training, no NSP |
| ALBERT | Parameter sharing, smaller |
| DeBERTa | Disentangled attention, stronger |

---

### GPT — Generative Pre-trained Transformer

**OpenAI, 2018 → GPT-4 (2023)**

#### How it reads:
**Left to right only** — autoregressive, predicts the next token based on previous tokens.

```
"The quick brown fox" → predicts → "jumps"
"The quick brown fox jumps" → predicts → "over"
```

It NEVER sees future tokens during training (masked self-attention).

#### Training objective:
- **Causal Language Model (CLM)** — predict the next token given all previous tokens
- Trained on massive text (internet, books, code)

#### What GPT is good at:
- Text generation
- Summarization
- Translation
- Code generation
- Conversational AI
- **Not natively good at** understanding/classification (needs prompt engineering)

#### GPT Evolution:
| Model | Params | Key milestone |
|-------|--------|--------------|
| GPT-1 | 117M | Proof of concept |
| GPT-2 | 1.5B | Too dangerous to release (was the claim) |
| GPT-3 | 175B | Few-shot learning, API era begins |
| InstructGPT | 175B | RLHF alignment |
| GPT-4 | ~1T (est.) | Multimodal, state of the art |

---

### BERT vs GPT — Side by Side

| | BERT | GPT |
|---|---|---|
| Architecture | Encoder only | Decoder only |
| Direction | Bidirectional | Left-to-right |
| Training task | Masked token prediction | Next token prediction |
| Best for | Understanding, classification | Generation, completion |
| Context | Full sequence at once | Previous tokens only |
| Fine-tune for | NER, QA, classification | Chat, summarization, code |
| Examples | BERT, RoBERTa, DeBERTa | GPT-4, Claude, LLaMA, Mistral |

> **Interview line:** *"BERT reads the whole sentence bidirectionally — great for understanding. GPT generates left-to-right — great for creation. It's the difference between a reader and a writer."*

---

### T5 & Encoder-Decoder Models (Bonus)

**T5 (Google)**, **BART (Meta)** — use both encoder and decoder.
- Frames every NLP task as text-to-text
- Input: "summarize: [article]" → Output: "summary text"
- Great for translation, summarization, question answering

---

---

## 4. Semantic Similarity

**Semantic similarity** measures how similar the *meaning* of two pieces of text is — not just surface word overlap.

```
"The car broke down"  vs  "The vehicle stopped working"
→ Different words, HIGH semantic similarity (~0.91)

"The car broke down"  vs  "I love pizza"
→ LOW semantic similarity (~0.05)
```

### Why It Matters
- Search engines (find relevant docs, not just keyword match)
- RAG systems (retrieve semantically relevant chunks)
- Duplicate detection
- Recommendation systems
- Clustering documents by topic

### How to Compute Semantic Similarity

**Step 1:** Embed both texts into vectors
```python
vec_A = embedding_model.encode("The car broke down")
vec_B = embedding_model.encode("The vehicle stopped working")
```

**Step 2:** Compute cosine similarity
```python
from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity([vec_A], [vec_B])[0][0]
# score = 0.91
```

### Approaches

| Approach | How | Quality |
|----------|-----|---------|
| Keyword matching (TF-IDF, BM25) | Exact word overlap | Low semantic understanding |
| Word embeddings (Word2Vec) | Average word vectors | Medium |
| Sentence Transformers (SBERT) | Encode full sentence | High |
| Cross-encoders | Compare pair directly | Highest (but slow) |

### Bi-encoder vs Cross-encoder

**Bi-encoder (used in retrieval)**
```
Text A → Encoder → Vector A ──┐
                               ├→ cosine_similarity → score
Text B → Encoder → Vector B ──┘
```
Fast — pre-compute vectors. Used in vector DBs.

**Cross-encoder (used in re-ranking)**
```
[Text A + Text B] → Encoder → single relevance score
```
More accurate but slow — can't pre-compute. Used to re-rank top-k results.

> **Interview line:** *"In AIPO, when a new incident comes in, I'd use a bi-encoder to quickly retrieve the top 10 similar past incidents from the vector DB, then a cross-encoder to re-rank and find the most relevant one for root cause analysis."*

---

---

## 5. Core NLP Tasks

### Text Classification

Assign a **label/category** to a piece of text.

```
Input:  "This product is absolutely terrible, don't buy it."
Output: Negative (sentiment) | Review (category) | Complaint (intent)
```

**Types:**
- **Sentiment analysis** — Positive / Negative / Neutral
- **Topic classification** — Sports / Tech / Politics
- **Intent detection** — Book flight / Check balance / Cancel order
- **Spam detection** — Spam / Not spam
- **Toxicity detection** — Safe / Toxic

**How it works (with BERT):**
```
Text → Tokenize → BERT encoder → [CLS] token representation → Linear layer → Softmax → Label
```

The `[CLS]` (classification) token in BERT aggregates the whole sequence meaning — perfect for classification.

**Metrics:**
- Accuracy, Precision, Recall, F1-score
- For imbalanced classes → use F1, not accuracy

---

### Text Summarization

Condense a long document into a **shorter version preserving key information**.

**Two types:**

**Extractive Summarization**
Picks actual sentences from the original text.
```
Original: [Sentence 1] [Sentence 2] [Sentence 3] [Sentence 4] [Sentence 5]
Summary:  [Sentence 2]              [Sentence 4]
```
- No new words generated
- Always factually accurate
- Can feel choppy

**Abstractive Summarization**
Generates new sentences that paraphrase the content — like a human would.
```
Original: "The revenue for Q3 2024 increased by 23% compared to Q3 2023,
           driven primarily by cloud services growth of 45%..."
Summary:  "Cloud services boosted Q3 revenue by 23% year-over-year."
```
- Can introduce errors (hallucination risk)
- More natural and concise
- GPT / T5 / BART based

**LLM-based summarization prompt pattern:**
```
System: You are a concise summarizer. Return only the summary.
User: Summarize the following in 3 bullet points:
      [DOCUMENT]
```

**Evaluation metric — ROUGE:**
- ROUGE-1: unigram overlap
- ROUGE-2: bigram overlap
- ROUGE-L: longest common subsequence
- Score 0–1; higher = more similar to reference summary

---

### Machine Translation

Convert text from one language to another while **preserving meaning**.

```
Input (English):  "The weather is beautiful today."
Output (French):  "Le temps est magnifique aujourd'hui."
```

**Evolution:**
```
Rule-based MT → Statistical MT → Neural MT → Transformer MT
(dictionaries)   (phrase tables)  (RNN seq2seq) (current SOTA)
```

**How Transformers do translation:**
```
English tokens → Encoder → Context vectors
                                  ↓
French tokens ← Decoder ← Cross-attention over encoder output
(generated one by one, left to right)
```

Cross-attention is what connects encoder (source) to decoder (target) — the model learns to align source and target tokens.

**Evaluation metric — BLEU score:**
- Measures n-gram overlap between generated and reference translation
- Score 0–100; >30 is generally good; >50 is excellent
- Limitation: doesn't capture fluency or meaning perfectly

**Modern translation APIs:**
- Google Cloud Translation, DeepL, Azure Translator
- For AI engineers — you'll mostly call these APIs, not train translation models

---

### Named Entity Recognition (NER)

Identify and classify **named entities** in text — people, places, organizations, dates, etc.

```
"Apple was founded by Steve Jobs in Cupertino in 1976."
         ORG               PERSON      LOCATION    DATE
```

**Entity types:**
- PER — Person
- ORG — Organization
- LOC — Location
- DATE — Date / Time
- MONEY — Monetary value
- PRODUCT — Product name

**Used in:** Information extraction, knowledge graphs, document processing

---

### Question Answering (QA)

**Extractive QA:** Find the answer span within a given context passage.
```
Context: "The Eiffel Tower is located in Paris, France. It was built in 1889."
Question: "When was the Eiffel Tower built?"
Answer:   "1889"  ← extracted directly from context
```

**Generative QA (RAG-based):** Retrieve relevant docs, then generate answer.
```
Question → Vector search → Relevant chunks → LLM → Generated answer
```

This is the foundation of **RAG systems** — what you built intuition for with AIPO.

---

---

## NLP Pipeline — End to End

```
Raw Text
   ↓
Preprocessing (lowercase, remove noise)
   ↓
Tokenization (BPE / WordPiece)
   ↓
Embedding (token → vector)
   ↓
Transformer Layers (attention + FFN)
   ↓
Task-specific Head
   ├── [CLS] → Linear → Softmax     (Classification)
   ├── Span extraction               (QA / NER)
   ├── Decoder → token generation    (Summarization / Translation)
   └── Pooling → cosine similarity   (Semantic Similarity)
   ↓
Output
```

---

## Quick Revision Cheatsheet

| Concept | One-liner |
|---------|-----------|
| Token | Smallest unit of text a model processes (word, subword, char) |
| BPE | Subword tokenization used by GPT — handles rare words |
| Embedding | Dense vector representation of a token capturing meaning |
| Cosine Similarity | Measures angle between vectors — standard for semantic similarity |
| Transformer | Architecture using attention to process all tokens in parallel |
| Attention | Mechanism to weigh how relevant each token is to every other token |
| Q, K, V | Query, Key, Value — the three components of attention |
| Multi-head Attention | Multiple attention heads running in parallel, each learning different patterns |
| Positional Encoding | Injects word order info since Transformers process all tokens simultaneously |
| BERT | Encoder-only, bidirectional — great for understanding/classification |
| GPT | Decoder-only, left-to-right — great for generation |
| MLM | BERT's training task — predict masked tokens |
| CLM | GPT's training task — predict next token |
| Semantic similarity | Meaning-based comparison via cosine similarity of embeddings |
| Extractive summary | Picks sentences from original text |
| Abstractive summary | Generates new sentences paraphrasing the content |
| ROUGE | Evaluation metric for summarization (n-gram overlap) |
| BLEU | Evaluation metric for translation (n-gram precision) |
| NER | Named Entity Recognition — tags people, places, orgs in text |
| Bi-encoder | Encode texts separately, compare vectors — fast retrieval |
| Cross-encoder | Encode text pair together — slow but more accurate, used in re-ranking |
