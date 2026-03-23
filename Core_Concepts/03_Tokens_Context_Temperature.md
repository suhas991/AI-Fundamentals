# 🔢 Tokens, Context Window & Temperature

---

## Part 1: Tokens

### What is a Token?

A **token** is the basic unit of text that an LLM processes. It is NOT the same as a word — tokens are chunks of characters determined by a **tokenizer** algorithm.

> **Rule of thumb:** 1 token ≈ 4 characters ≈ ¾ of a word (for English)

### Tokenization Examples

```
"Hello, world!"
→ ["Hello", ",", " world", "!"]  → 4 tokens

"The cat sat on the mat."
→ ["The", " cat", " sat", " on", " the", " mat", "."]  → 7 tokens

"Unbelievable"
→ ["Un", "bel", "iev", "able"]  → 4 tokens  (rare word = more tokens)

"supercalifragilisticexpialidocious"
→ Many small chunks  →  12+ tokens
```

### Why Tokens (Not Words)?

1. **Handles unknown words** — by breaking rare words into sub-word pieces
2. **Efficient vocabulary** — a fixed vocab of ~50,000 tokens can represent all English text
3. **Works across languages** — languages without spaces (Chinese, Japanese) are handled gracefully

### Common Tokenizers

| Tokenizer | Used By | Vocab Size |
|---|---|---|
| BPE (Byte-Pair Encoding) | GPT-2, GPT-3, GPT-4 | ~50,000 |
| WordPiece | BERT | ~30,000 |
| SentencePiece | LLaMA, T5 | ~32,000 |
| tiktoken (cl100k_base) | GPT-4, Claude-style | ~100,000 |

### Token Count by Language
Non-English text typically uses **more tokens** per word:

```
"Hello"         → 1 token  (English)
"Hola"          → 1 token  (Spanish)
"こんにちは"     → 3-5 tokens  (Japanese)
"مرحبا"         → 3-4 tokens  (Arabic)
```

### Why Token Count Matters

1. **Cost** — Most API pricing is per token (input + output tokens)
2. **Speed** — More tokens = slower inference
3. **Context limits** — Each model has a maximum token limit (the context window)

### Estimating Token Count
```
~750 words = ~1,000 tokens
1 page of text ≈ 500–700 tokens
A full novel (80,000 words) ≈ 107,000 tokens
```

### Token Count Tools
- OpenAI's Tokenizer: https://platform.openai.com/tokenizer
- `tiktoken` Python library: `import tiktoken`

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello, world!")
print(len(tokens))  # → 4
```

---

## Part 2: Context Window

### What is the Context Window?

The **context window** is the maximum amount of text (measured in tokens) that an LLM can "see" at one time. It includes:
- The system prompt
- The entire conversation history
- Any documents or data you've included
- The model's generated output

Think of it as the model's **working memory** — everything outside the context window is invisible to the model.

```
┌─────────────────────────────────────────┐
│             CONTEXT WINDOW              │
│                                         │
│  [System Prompt]                        │
│  [Previous Messages]                    │
│  [Current User Message]                 │
│  [Documents / RAG chunks]               │
│  [Model's response so far]              │
│                                         │
│  ← Total must be ≤ Max Context Tokens → │
└─────────────────────────────────────────┘
```

### Context Window Sizes (2024–2025)

| Model | Context Window |
|---|---|
| GPT-3.5 Turbo | 16K tokens |
| GPT-4o | 128K tokens |
| Claude 3.5 Sonnet | 200K tokens |
| Claude 3 Opus | 200K tokens |
| Gemini 1.5 Pro | 1M tokens |
| Gemini 1.5 Flash | 1M tokens |
| Llama 3.1 405B | 128K tokens |

### What Happens When You Exceed the Context Window?

- **Truncation** — older messages are dropped (you lose history)
- **Error** — some APIs return an error if you exceed the limit
- **Degraded performance** — even within limits, models often focus on the beginning and end (the "lost in the middle" problem)

### The "Lost in the Middle" Problem

Research shows that LLMs perform **worse on information placed in the middle** of a long context, even when it technically fits within the window:

```
[Important info here] ← Good recall
[--- Long middle section ---] ← Poor recall
[Important info here] ← Good recall
```

**Practical implication:** Put your most important instructions at the **start or end** of the context.

### Context Window vs Memory

| | Context Window | External Memory |
|---|---|---|
| Speed | Fast (in-weights) | Slower (retrieval) |
| Limit | Fixed (e.g., 200K) | Unlimited |
| Persistence | Lost after conversation | Stored externally |
| Accuracy | High for in-context info | Depends on retrieval |
| Mechanism | Attention | Vector DB / search |

---

## Part 3: Temperature

### What is Temperature?

**Temperature** is a parameter that controls the **randomness** of an LLM's output. It modifies the probability distribution over tokens before sampling.

- **Low temperature (→ 0):** More deterministic, focused, predictable
- **High temperature (→ 2):** More random, creative, surprising

### How It Works (Technically)

Before sampling the next token, the model produces **logits** — raw scores for every token in the vocabulary. These are converted to probabilities via **softmax**.

Temperature divides the logits before softmax:

```
adjusted_logit = logit / temperature
```

```
Temperature = 0.1 (sharp/focused):
  P("Paris") = 0.97
  P("Lyon")  = 0.02
  P("Rome")  = 0.01

Temperature = 1.0 (default):
  P("Paris") = 0.75
  P("Lyon")  = 0.15
  P("Rome")  = 0.10

Temperature = 1.5 (flatter/random):
  P("Paris") = 0.45
  P("Lyon")  = 0.32
  P("Rome")  = 0.23
```

### Temperature = 0

At temperature = 0, the model **always picks the highest-probability token** (greedy decoding). Output becomes fully deterministic — the same prompt will always produce the same response.

### Temperature Scale Guide

| Temperature | Behavior | Best For |
|---|---|---|
| 0.0 | Fully deterministic | Code, factual Q&A, data extraction |
| 0.1 – 0.3 | Very focused, consistent | Technical writing, summarization |
| 0.5 – 0.7 | Balanced | General-purpose chat, analysis |
| 0.8 – 1.0 | Creative and varied | Brainstorming, ideation |
| 1.1 – 1.5 | Very creative, sometimes incoherent | Poetry, experimental writing |
| > 1.5 | Often nonsensical | (Rarely useful) |

### Related Sampling Parameters

Temperature isn't the only way to control randomness:

#### Top-P (Nucleus Sampling)
Instead of sampling from all tokens, sample only from the **smallest set of tokens whose cumulative probability ≥ P**.

```
Top-P = 0.9:
Token probabilities: [0.5, 0.3, 0.15, 0.04, 0.01, ...]
Cumulative:          [0.5, 0.8, 0.95, ...]
→ Sample from only top 3 tokens (cumulative ≥ 0.9)
```

- **Low Top-P (0.1):** Very restricted vocabulary, safe outputs
- **High Top-P (0.95):** Wide vocabulary, diverse outputs

#### Top-K
Sample only from the **K most probable tokens**.
- `top_k=1` = greedy (same as temp=0)
- `top_k=50` = sample from top 50 tokens

#### Comparison Table

| Parameter | Controls | Default |
|---|---|---|
| `temperature` | Sharpness of probability distribution | 1.0 |
| `top_p` | Fraction of probability mass to sample from | 1.0 (no restriction) |
| `top_k` | Number of top tokens to consider | Varies |
| `max_tokens` | Maximum output length | Varies |
| `frequency_penalty` | Penalize repeating same tokens | 0.0 |
| `presence_penalty` | Encourage topic diversity | 0.0 |

### Practical Recommendations

```
Use temperature=0 for:
  - Code generation
  - Data extraction / parsing
  - Classification
  - Factual question answering

Use temperature=0.3–0.7 for:
  - Summarization
  - Translation
  - Standard chat assistants
  - Technical writing

Use temperature=0.8–1.0 for:
  - Creative writing
  - Story generation
  - Marketing copy
  - Brainstorming lists

Use temperature + top_p together:
  - Recommended: temperature=0.7, top_p=0.9 (standard creative balance)
  - Avoid setting both very high — compounds randomness
```

---

## Summary Cheatsheet

```
TOKENS
├── Basic unit of text for LLMs
├── ~4 chars / ~¾ word per token (English)
├── Affects: cost, speed, context limits
└── Tool: tiktoken, platform.openai.com/tokenizer

CONTEXT WINDOW
├── Max tokens the model can "see" at once
├── Includes: system prompt + history + input + output
├── Modern: 128K–1M tokens
└── Problem: "lost in the middle" for very long contexts

TEMPERATURE
├── Controls output randomness
├── 0.0 = deterministic; 2.0 = chaotic
├── Low for facts/code; High for creativity
└── Use with top_p/top_k for fine-grained control
```

---

*Previous: [02 — Large Language Models](./02_Large_Language_Models.md) | Next: [04 — Prompt Engineering](./04_Prompt_Engineering.md)*
