# 🧠 Large Language Models (LLMs)

---

## What is a Large Language Model?

A **Large Language Model (LLM)** is a type of deep learning model trained on massive amounts of text data to understand and generate human language. It is "large" in two senses:

1. **Large data** — trained on billions to trillions of tokens of text (books, web pages, code, etc.)
2. **Large model** — billions of parameters (weights) that store learned knowledge

> LLMs are the technology behind ChatGPT, Claude, Gemini, Llama, Mistral, and most modern AI assistants.

---

## The Foundation: The Transformer Architecture

All modern LLMs are built on the **Transformer** architecture, introduced in the landmark 2017 paper *"Attention Is All You Need"* by Google researchers.

### Key Innovation: Self-Attention
Before Transformers, models processed text sequentially (word by word). Transformers process the entire input **in parallel** and learn which words should "pay attention" to which other words.

```
Sentence: "The bank by the river was steep."
          ↑
"bank" attends to "river" → understands it means riverbank, not financial bank
```

### Architecture Overview
```
Input Text
    ↓
[Tokenization]
    ↓
[Token Embeddings] + [Positional Encodings]
    ↓
[Transformer Block × N]
    ├── Multi-Head Self-Attention
    ├── Add & Norm
    ├── Feed-Forward Network
    └── Add & Norm
    ↓
[Output Head]
    ↓
Next Token Prediction (probability distribution over vocabulary)
```

---

## How LLMs Are Trained

### Stage 1: Pre-training (Self-Supervised)
The model learns to **predict the next token** given all previous tokens. This is called **causal language modeling**.

```
Input:  "The capital of France is"
Target: "Paris"
```

This simple objective, applied billions of times across trillions of tokens, forces the model to compress world knowledge, grammar, reasoning patterns, and facts into its weights.

- **Data:** Common Crawl, Wikipedia, books, GitHub, StackOverflow, etc.
- **Compute:** Thousands of GPUs for weeks/months
- **Cost:** Millions to hundreds of millions of dollars

### Stage 2: Fine-tuning (Supervised)
After pre-training, the model is further trained on curated high-quality instruction-response pairs to make it more helpful and follow instructions.

### Stage 3: RLHF (Reinforcement Learning from Human Feedback)
Human raters rank model outputs. A **reward model** is trained on these rankings. The LLM is then fine-tuned using RL to maximize reward — making it more aligned, helpful, and less harmful.

```
LLM generates response
    ↓
Human rates it (better / worse)
    ↓
Reward model trained
    ↓
LLM fine-tuned via PPO/DPO to maximize reward
```

---

## Scale: Parameters and Why They Matter

| Model | Parameters | Organization | Year |
|---|---|---|---|
| GPT-2 | 1.5B | OpenAI | 2019 |
| GPT-3 | 175B | OpenAI | 2020 |
| PaLM | 540B | Google | 2022 |
| GPT-4 | ~1.8T (estimated) | OpenAI | 2023 |
| Llama 3.1 | 405B | Meta | 2024 |
| Claude 3 Opus | Unknown | Anthropic | 2024 |
| Gemini Ultra | Unknown | Google | 2024 |

**Parameters** are the numerical weights of the neural network. More parameters generally means:
- More world knowledge stored
- Better reasoning ability
- Better language fluency
- ... but also more compute required for inference

---

## Capabilities of LLMs

### What They Can Do
| Capability | Example |
|---|---|
| Text generation | Write essays, stories, reports |
| Summarization | Condense long documents |
| Translation | Translate between 100+ languages |
| Question answering | "What causes rain?" |
| Code generation | Write Python, SQL, JavaScript |
| Reasoning | Solve math problems, logic puzzles |
| Classification | Sentiment analysis, topic labeling |
| Instruction following | "Write a formal email to..."|
| Dialogue | Multi-turn conversations |
| Structured output | Generate JSON, tables |

### Emergent Abilities
Some capabilities **emerge unexpectedly** at scale — they don't exist in smaller models but appear suddenly as model size crosses a threshold:
- Multi-step arithmetic
- Chain-of-thought reasoning
- Analogical reasoning
- In-context learning (few-shot prompting)

---

## Key LLM Families (2024–2025)

### Closed-Source (API only)
| Family | Provider | Notable Models |
|---|---|---|
| GPT | OpenAI | GPT-4o, o1, o3 |
| Claude | Anthropic | Claude 3 Opus, Sonnet, Haiku; Claude 3.5/4 |
| Gemini | Google DeepMind | Gemini 1.5 Pro, Gemini 2.0 |

### Open-Source / Open-Weight
| Family | Provider | Notable Models |
|---|---|---|
| Llama | Meta | Llama 2, 3, 3.1 (8B–405B) |
| Mistral | Mistral AI | Mistral 7B, Mixtral 8x7B |
| Falcon | TII UAE | Falcon 40B, 180B |
| Phi | Microsoft | Phi-3 (small but capable) |
| Qwen | Alibaba | Qwen 2.5 |
| Gemma | Google | Gemma 2 (2B, 9B, 27B) |

---

## How LLMs Generate Text

LLMs are **next-token predictors**. At each step, they output a probability distribution over the vocabulary and sample from it:

```
"The Eiffel Tower is located in" 
→ P(Paris) = 0.87
→ P(France) = 0.06  
→ P(Europe) = 0.03
→ P(...) = 0.04
→ Sample → "Paris"

"The Eiffel Tower is located in Paris"
→ P(,) = 0.45
→ P(.) = 0.32
→ ...
```

This process repeats **autoregressively** — each new token is appended and fed back in — until the model generates a stop token or reaches the max length.

---

## Multimodal LLMs

Modern LLMs are no longer text-only. **Multimodal models** can process:
- 📷 Images (GPT-4V, Claude 3, Gemini)
- 🎵 Audio (Whisper, Gemini 1.5)
- 🎬 Video (Gemini 1.5 Pro)
- 📄 Documents/PDFs

This is achieved by adding **encoders** for other modalities that project them into the same embedding space as text tokens.

---

## LLM Limitations (Brief — detailed in separate note)

- **Knowledge cutoff** — no awareness of events after training data
- **Hallucination** — can confidently generate false information
- **Context window** — can only "see" a limited amount of text at once
- **No persistent memory** by default across conversations
- **Reasoning limits** — struggles with precise arithmetic, formal logic
- **Bias** — reflects biases present in training data

---

## Key Terms Quick Reference

| Term | Meaning |
|---|---|
| **Parameter** | A single learnable weight in the neural network |
| **Pre-training** | Initial training on massive unlabeled text data |
| **Fine-tuning** | Further training on specific, curated data |
| **RLHF** | Training technique using human preference feedback |
| **Inference** | Running a trained model to generate outputs |
| **Autoregressive** | Generating one token at a time, feeding back output |
| **Foundation model** | A large pre-trained model used as a base for many tasks |

---

*Previous: [01 — AI vs ML vs DL](./01_AI_ML_DL_Differences.md)*
