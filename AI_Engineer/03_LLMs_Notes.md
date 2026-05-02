# Large Language Models (LLMs) — Study Notes
> For SDE-1 / AI Engineer Interview Prep | Suhas N H

---

## What is an LLM?

A Large Language Model is a **Transformer-based model trained on massive text datasets**
to understand and generate human language at scale.

"Large" refers to two things:
- **Parameters** — billions of learned weights (GPT-3 = 175B, GPT-4 ~1T estimated)
- **Training data** — hundreds of billions to trillions of tokens (books, web, code)

```
Scale of parameters over time:

BERT (2018)      →   110M parameters
GPT-2 (2019)     →   1.5B parameters
GPT-3 (2020)     →   175B parameters
GPT-4 (2023)     →   ~1T parameters (estimated)
LLaMA 3 (2024)   →   8B / 70B / 405B variants
```

---

## 1. How LLMs Work — Next Token Prediction

### The Core Idea

LLMs are trained to do one thing: **predict the next token given all previous tokens.**
Everything else — reasoning, coding, answering questions — emerges from doing this at massive scale.

```
Input:  "The capital of France is"
Model:  What token comes next?

Candidates with probabilities:
  "Paris"    → 0.94  ← highest
  "Lyon"     → 0.02
  "London"   → 0.01
  "a"        → 0.01
  ...

Output: "Paris"
```

### Autoregressive Generation — Token by Token

LLMs generate text one token at a time, feeding each output back as input:

```
Step 1: "The capital of France is"              → "Paris"
Step 2: "The capital of France is Paris"        → "."
Step 3: "The capital of France is Paris."       → [END]

Final: "The capital of France is Paris."
```

This is called **autoregressive** generation — each token depends on all previous ones.

### Training — Causal Language Modelling (CLM)

During training, the model sees massive text and learns to predict every next token:

```
Training text: "Machine learning is a subset of artificial intelligence"

Input → Target pairs created automatically (self-supervised):
"Machine"                          → "learning"
"Machine learning"                 → "is"
"Machine learning is"              → "a"
"Machine learning is a"            → "subset"
...
```

No human labels needed — the text itself is the supervision signal.
This is why LLMs can train on the entire internet.

### What Happens Inside — The Full Flow

```
User Prompt (text)
       ↓
   Tokenizer  →  [token_ids: 1423, 892, 34, ...]
       ↓
Token Embeddings + Positional Encodings
       ↓
   N × Transformer Decoder Blocks
   ┌────────────────────────────────┐
   │  Masked Self-Attention         │ ← attends to previous tokens only
   │  +                             │
   │  Feed Forward Network          │ ← knowledge stored here
   │  +                             │
   │  Layer Norm + Residual         │
   └────────────────────────────────┘
       ↓
   Linear Layer (projects to vocab size)
       ↓
   Softmax → Probability distribution over entire vocabulary
       ↓
   Sample or Argmax → Next token
       ↓
   Repeat until [END] token or max length
```

### Emergent Abilities

At sufficient scale, LLMs develop capabilities that were NOT explicitly trained:
- Multi-step reasoning
- Code generation
- Analogical reasoning
- In-context learning (few-shot)

> These abilities "emerge" suddenly at scale thresholds — not gradual improvement.

> **Interview line:** *"LLMs are fundamentally next-token predictors trained at massive scale.
> The reasoning, coding, and chat abilities are emergent properties of doing
> this simple task on enough data."*

---

---

## 2. Context Window & Token Limits

### What is a Context Window?

The **maximum number of tokens an LLM can process in a single call** — input + output combined.

```
Context Window = System Prompt + Conversation History + User Input + Model Output

[    System Prompt    ][  History  ][ User Input ][ ← Model Output → ]
└──────────────────────────────────────────────────────────────────┘
                    Context Window Limit
```

Everything outside the context window is **invisible** to the model.
The model has NO memory beyond what's in the current context.

### Context Window Sizes (2024–2025)

| Model | Context Window |
|-------|---------------|
| GPT-3.5 Turbo | 16K tokens |
| GPT-4 Turbo | 128K tokens |
| GPT-4o | 128K tokens |
| Claude 3.5 Sonnet | 200K tokens |
| Claude 3 Opus | 200K tokens |
| Gemini 1.5 Pro | 1M tokens |
| Gemini 1.5 Flash | 1M tokens |
| LLaMA 3.1 | 128K tokens |
| Mistral Large | 128K tokens |

> 1M tokens ≈ ~700,000 words ≈ entire codebase or multiple books

### Why Context Window Matters for Engineers

**Token counting:** Every API call costs money per token.
```
Cost = (input_tokens + output_tokens) × price_per_token

GPT-4o: ~$5 per 1M input tokens, ~$15 per 1M output tokens
Claude Sonnet: ~$3 per 1M input, ~$15 per 1M output
```

**Context overflow problem:**
```
Long conversation → history grows → hits token limit → must truncate

Strategies to handle:
1. Sliding window    → keep only last N turns
2. Summarization     → compress old history into summary
3. RAG               → retrieve only relevant history chunks
4. Hierarchical      → summarize → then summarize the summaries
```

**The "Lost in the Middle" problem:**
Research shows LLMs perform best on information at the **start and end** of context.
Information buried in the middle gets less attention.

```
Context:  [A][B][C][D][E][F][G][H][I][J]
Recall:    ↑↑↑                        ↑↑↑
           High                       High
                    ↓↓↓↓↓↓
                     Low (lost in middle)
```

> **Interview line:** *"Context window is the model's working memory.
> In AIPO, if incident history is too long to fit, I'd use summarization
> or RAG to retrieve only the most relevant incidents rather than
> dumping everything into the prompt."*

---

---

## 3. Sampling Parameters — Temperature, Top-p, Top-k

These parameters **control how the model picks the next token** from its probability distribution.
They don't change what the model knows — they change how it samples.

### The Probability Distribution

At each step, the model produces probabilities for every token in the vocabulary:

```
After "The weather today is":
  "sunny"      → 0.45
  "cloudy"     → 0.25
  "rainy"      → 0.15
  "hot"        → 0.08
  "perfect"    → 0.04
  "terrible"   → 0.02
  ... (50,000 more tokens with tiny probs)
```

Now, which token do we pick?

---

### Temperature — Controls Randomness

Temperature **reshapes the probability distribution** — making it sharper or flatter.

```
Formula: adjusted_prob = prob^(1/temperature)
         then normalize to sum to 1
```

**Temperature = 0 (or near 0) → Deterministic / Greedy**
```
Always picks the highest probability token.
Same input → same output every time.

"sunny" → 0.99   ← always picked
"cloudy" → 0.01
```
Use for: Factual QA, code generation, structured outputs

**Temperature = 1.0 → Default / Balanced**
```
Samples proportionally to original probabilities.
"sunny" → 0.45  (picked 45% of the time)
"cloudy" → 0.25 (picked 25% of the time)
```
Use for: Chat, general tasks

**Temperature = 1.5–2.0 → High / Creative**
```
Flattens distribution — rare tokens get more chance.
"sunny"    → 0.25
"cloudy"   → 0.22
"mystical" → 0.18  ← rare token now has real chance
```
Use for: Creative writing, brainstorming, poetry

**Visual intuition:**
```
Low temp (0.2):     High temp (1.5):
████████░░          ████░░░░░░
█░░░░░░░░░          ███░░░░░░░
░░░░░░░░░░          ██░░░░░░░░
Sharp peak          Flat distribution
Predictable         Creative/Risky
```

---

### Top-k Sampling — Limit to k Candidates

**Only consider the k most probable tokens**, ignore the rest.

```
k = 3:
  "sunny"    → 0.45  ✅ in top-3
  "cloudy"   → 0.25  ✅ in top-3
  "rainy"    → 0.15  ✅ in top-3
  "hot"      → 0.08  ❌ excluded
  "perfect"  → 0.04  ❌ excluded
  ...all others      ❌ excluded

Sample only from: sunny(0.53), cloudy(0.29), rainy(0.18) [renormalized]
```

**Problem with fixed k:**
- Sometimes top-3 covers 99% of probability (safe situation)
- Sometimes top-3 covers only 30% (many equally likely tokens)
- Fixed k doesn't adapt to the distribution shape

---

### Top-p (Nucleus) Sampling — Dynamic Cutoff

**Pick the smallest set of tokens whose cumulative probability ≥ p**, then sample from them.

```
p = 0.9:

"sunny"    → 0.45  | cumulative: 0.45  ✅ include
"cloudy"   → 0.25  | cumulative: 0.70  ✅ include
"rainy"    → 0.15  | cumulative: 0.85  ✅ include
"hot"      → 0.08  | cumulative: 0.93  ✅ include (crosses 0.9 here)
"perfect"  → 0.04  | cumulative: 0.97  ❌ stop here

Sample from: {sunny, cloudy, rainy, hot} [renormalized]
```

Top-p adapts dynamically:
- Concentrated distribution → small nucleus (fewer tokens)
- Spread distribution → large nucleus (more tokens)

> Top-p is generally preferred over Top-k for this adaptability.

---

### Combining Parameters — Practical Settings

```
Use case              | Temperature | Top-p  | Top-k
─────────────────────────────────────────────────────
Code generation       |   0.1–0.3   |  0.95  |  —
Factual QA            |   0.0–0.2   |  0.9   |  —
Chatbot (general)     |   0.7–1.0   |  0.9   |  40
Creative writing      |   1.2–1.5   |  0.95  |  —
Structured JSON out   |   0.0       |  —     |  1
Brainstorming         |   1.0–1.3   |  0.95  |  —
```

> **Interview line:** *"Temperature controls creativity, Top-p controls vocabulary breadth.
> For code generation in AIPO, I'd use low temperature (~0.1) for deterministic,
> correct output. For incident summary generation, moderate temperature (~0.7)
> gives natural-sounding text."*

---

---

## 4. Hallucination & Mitigation

### What is Hallucination?

LLMs sometimes **confidently generate false, fabricated, or nonsensical information**
that sounds plausible but is factually wrong.

```
User:   "What papers did Einstein publish in 1932?"
LLM:    "In 1932, Einstein published 'Quantum Relativity and Field Unification'
         in the Journal of Theoretical Physics, arguing that..."

Reality: This paper doesn't exist. The LLM made it up — completely.
```

The model doesn't "know" it's wrong. It's predicting plausible tokens, not retrieving facts.

### Types of Hallucination

**Factual hallucination**
Generates incorrect facts — wrong dates, names, statistics, citations.
```
"The Eiffel Tower was built in 1901" (wrong — it was 1889)
```

**Fabricated references**
Invents books, papers, URLs, people that don't exist.
```
"According to Smith et al. (2019) published in Nature AI..."
→ Paper doesn't exist
```

**Intrinsic hallucination**
Output contradicts the provided context/document.
```
Document: "Revenue grew 15% in Q3"
Summary:  "Revenue declined in Q3" ← contradicts source
```

**Extrinsic hallucination**
Output can't be verified from provided context — adds unsupported information.

### Why Does Hallucination Happen?

```
1. Knowledge gaps       → Model wasn't trained on this fact
2. Knowledge cutoff     → Event happened after training data ended
3. Overconfidence       → Model always produces fluent output even when uncertain
4. Training artifacts   → Memorized plausible patterns, not ground truth
5. Long context         → Loses track of facts in long documents
```

### Mitigation Strategies

**1. RAG (Retrieval Augmented Generation)**
Ground the model's output in retrieved real documents.
```
Instead of: "Answer from memory"
Do:         "Here are 5 retrieved documents. Answer only from these."

Reduces hallucination dramatically for factual tasks.
```

**2. Prompt Engineering**
```
Bad:  "Tell me about the latest AI research"
Good: "Answer ONLY using the following documents. If the answer
       is not in the documents, say 'I don't know'.
       Documents: [...]"
```

**3. Temperature = 0**
```
Lower temperature → model picks highest-probability tokens
→ more conservative, less creative hallucinations
```

**4. Self-consistency / Chain-of-Thought**
```
Sample multiple reasoning paths → pick the most consistent answer
If 4/5 paths agree → higher confidence
```

**5. Verification layer**
```
LLM output → Fact-checking model → Verified output
           → Grounding score < threshold → Flag for review
```

**6. Citations / Source attribution**
```
Force the model to cite the exact sentence from source document.
If it can't cite → it doesn't say it.
```

**7. Fine-tuning on domain data**
```
Model trained on your specific domain data
→ less likely to confabulate domain-specific facts
```

**8. Confidence estimation**
```
Ask model: "How confident are you? (1-10)"
Or use log-probabilities of output tokens as confidence proxy
```

> **Interview line:** *"Hallucination is the #1 reliability problem with LLMs.
> In AIPO, I'd mitigate it by using RAG — grounding every AI response
> in retrieved incident documents and instructing the model to only
> use provided context. I'd also add a citation requirement so
> every claim maps to a source."*

---

---

## 5. Popular Models — GPT-4, Claude, Gemini, LLaMA, Mistral

### GPT-4 / GPT-4o — OpenAI

```
Type:           Closed source, API only
Parameters:     ~1T (estimated, MoE architecture)
Context:        128K tokens
Modalities:     Text, image, audio, video (GPT-4o)
Training:       Massive web data + RLHF + Constitutional AI
```

**GPT-4o ("omni")** — unified model handling text, image, audio natively in one model
(vs GPT-4 Vision which stitched modalities together)

**Strengths:**
- Best overall reasoning and instruction following
- Strong code generation
- Massive ecosystem (plugins, Assistants API)
- Function calling / tool use

**Weaknesses:**
- Expensive ($5–$15 per 1M tokens)
- Closed source — can't self-host
- Data sent to OpenAI

**Best for:** Complex reasoning, coding assistants, general-purpose apps

---

### Claude (Anthropic) — Claude 3.5 Sonnet / Claude 3 Opus

```
Type:           Closed source, API only
Context:        200K tokens (largest among major closed models)
Modalities:     Text, image
Training:       Constitutional AI + RLHF
```

**Model tiers:**
```
Claude 3 Haiku  → Fastest, cheapest (simple tasks)
Claude 3.5 Sonnet → Best balance (most used)
Claude 3 Opus   → Most capable (complex reasoning)
```

**Constitutional AI:** Anthropic's unique training approach —
model is trained with a set of principles (a "constitution") to be helpful, harmless, honest.

**Strengths:**
- Largest context window (200K) among closed models
- Excellent at long document understanding
- Strong at following nuanced instructions
- Very low hallucination rate relative to peers
- Best-in-class at coding and analysis

**Best for:** Long document processing, enterprise use cases, nuanced instruction following

---

### Gemini — Google DeepMind

```
Type:           Closed source (API) + some open weights
Context:        1M tokens (Gemini 1.5 Pro/Flash) — largest available
Modalities:     Text, image, audio, video, code natively
Training:       Trained on Google's massive data + TPUs
```

**Model tiers:**
```
Gemini 1.5 Flash  → Fast, cheap, 1M context
Gemini 1.5 Pro    → Most capable, 1M context
Gemini Ultra      → Highest capability (limited access)
```

**Strengths:**
- 1M token context — can process entire codebases
- Native multimodal (not bolted on)
- Deep Google ecosystem integration (Search, Workspace)
- Strong multilingual performance

**Best for:** Long context tasks, video understanding, Google Cloud integration

---

### LLaMA (Meta) — Open Weights

```
Type:           Open weights (downloadable, self-hostable)
Versions:       LLaMA 2 (2023), LLaMA 3 (2024), LLaMA 3.1 (2024)
Sizes:          8B, 70B, 405B parameter variants
Context:        128K tokens (LLaMA 3.1)
License:        Meta's custom license (mostly open, some restrictions)
```

**LLaMA 3.1 405B** competes with GPT-4 on many benchmarks — remarkable for open weights.

**Strengths:**
- Self-hostable — data never leaves your infrastructure
- No per-token API cost after hardware investment
- Massive community + fine-tuned variants (Alpaca, Vicuna, Mistral)
- Can be quantized to run on consumer hardware

**Fine-tuned variants built on LLaMA:**
```
Code LLaMA    → Specialized for code
LLaMA + RLHF  → Chat-optimized
Medical LLaMA → Healthcare domain
```

**Best for:** Privacy-sensitive data, self-hosted deployments, fine-tuning base

---

### Mistral — Mistral AI (France)

```
Type:           Mix — some open weights, some API-only
Flagship:       Mistral Large, Mistral 7B, Mixtral 8x7B
Context:        128K tokens
Architecture:   Uses Grouped Query Attention + Sliding Window Attention
```

**Mistral 7B** — punches far above its weight for its size.
Outperforms LLaMA 2 13B on most benchmarks despite being smaller.

**Mixtral 8x7B** — Mixture of Experts (MoE) architecture:
```
8 expert networks, each 7B params
For each token, 2 experts are activated
Effective params: ~13B active / 47B total
Performance of 70B model at cost of 13B
```

**Strengths:**
- Extremely efficient — great performance per parameter
- Open weights (Mistral 7B, Mixtral)
- Strong in European languages
- MoE architecture = fast inference

**Best for:** Edge deployment, cost-efficient self-hosting, European data regulations (GDPR)

---

### Quick Model Comparison

| Model | Open? | Context | Best at | Cost |
|-------|-------|---------|---------|------|
| GPT-4o | ❌ | 128K | Reasoning, general | High |
| Claude 3.5 Sonnet | ❌ | 200K | Long docs, coding | Medium |
| Gemini 1.5 Pro | ❌ | 1M | Long context, multimodal | Medium |
| LLaMA 3.1 405B | ✅ | 128K | Self-hosting, fine-tuning | Infra cost |
| Mistral 7B | ✅ | 32K | Efficient, edge | Very low |
| Mixtral 8x7B | ✅ | 32K | Efficient, MoE | Low |

---

---

## 6. Open Source vs Closed Source Models

### Closed Source (Proprietary)

Model weights are **not public** — accessed only via API.

```
Examples: GPT-4, Claude, Gemini, Cohere Command
```

| Pros | Cons |
|------|------|
| Best performance (generally) | Data sent to third party |
| No infrastructure management | Per-token cost at scale |
| Constantly updated | No customization of weights |
| Enterprise SLAs | Vendor lock-in |
| Multimodal out of the box | Rate limits |

**When to choose:**
- Rapid prototyping
- Best-in-class quality needed
- No sensitive data concerns
- Small to medium scale

---

### Open Source / Open Weights

Model weights are **publicly available** — download and run yourself.

```
Examples: LLaMA 3, Mistral, Mixtral, Falcon, Phi-3, Gemma
```

> Note: "Open source" in AI often means "open weights" — training code and data
> may not be fully open. True open source = weights + data + training code.

| Pros | Cons |
|------|------|
| Full data privacy (self-hosted) | Need GPU infrastructure |
| No per-token API cost | Performance gap vs closed (closing fast) |
| Fully customizable / fine-tunable | You manage reliability + scaling |
| No vendor lock-in | Model updates = your responsibility |
| GDPR / compliance friendly | Needs ML engineering expertise |

**When to choose:**
- Sensitive / regulated data (healthcare, finance, legal)
- High volume → API costs become prohibitive
- Need domain fine-tuning
- EU data sovereignty requirements
- Offline / air-gapped environments

### The Closing Gap

```
2023: GPT-4 >> Open source models
2024: LLaMA 3.1 405B ≈ GPT-4 on many benchmarks

The gap is closing fast. For specific domains with fine-tuning,
open models can match or exceed closed models.
```

### Hybrid Approach (Best Practice)

```
Routing strategy:
  Simple tasks      → Small open model (Mistral 7B)  → cheap, fast
  Complex tasks     → Closed model (GPT-4o)           → expensive, best quality
  Private data      → Self-hosted LLaMA              → secure
  Real-time needs   → Quantized local model           → no latency
```

> **Interview line:** *"For AIPO, since it handles enterprise incident data
> that may be sensitive, I'd evaluate a self-hosted LLaMA 3 70B fine-tuned
> on IT operations data vs Claude API with data processing agreements.
> The open model gives data privacy; the closed model gives better
> out-of-the-box performance."*

---

---

## 7. Model Benchmarks

Benchmarks are **standardized tests to compare LLM capabilities** across tasks.
They're the "unit tests" of the AI world.

### MMLU — Massive Multitask Language Understanding

```
What it tests:    Academic knowledge across 57 subjects
Format:           Multiple choice questions (4 options)
Subjects:         Math, science, law, medicine, history, coding...
Difficulty:       High school → Professional level
Score:            % correct (human expert ~89%)
```

**Example question:**
```
Which of the following best describes the phenomenon of quantum entanglement?

A) Two particles share classical information faster than light
B) Measurement of one particle instantly affects its entangled partner
C) Particles merge into a single quantum state permanently
D) Entangled particles must be in the same location

Answer: B
```

**Why it matters:** Tests breadth of world knowledge and reasoning — correlates well with real-world utility.

**Scores (approximate):**
```
GPT-4          → 86.4%
Claude 3 Opus  → 86.8%
Gemini Ultra   → 90.0%
LLaMA 3 70B   → 82.0%
Human expert   → ~89%
```

---

### HumanEval — Code Generation

```
What it tests:    Python code generation from docstrings
Created by:       OpenAI (2021)
Format:           164 programming problems
Metric:           pass@k — % of problems solved with k attempts
```

**Example problem:**
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """
    Check if in given list of numbers, are any two numbers
    closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    # Model must generate this:
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

**Scores (pass@1 — solved on first try):**
```
GPT-4          → 87%
Claude 3.5 Sonnet → 92%  ← highest
Gemini 1.5 Pro → 84%
LLaMA 3 70B   → 81%
```

---

### Other Key Benchmarks

**GSM8K — Grade School Math**
```
What:     8,500 grade school math word problems
Tests:    Multi-step arithmetic reasoning
Metric:   % correct
Example:  "Sarah has 5 apples. She gives 2 to Tom and buys 3 more.
           Then she gives half to her sister. How many does she have?"
Why:      Tests chain-of-thought reasoning capability
```

**HellaSwag — Commonsense Reasoning**
```
What:     Sentence completion requiring real-world understanding
Format:   Choose the most plausible continuation of a scenario
Tests:    Commonsense reasoning, physical intuition
Example:  "He poured the hot coffee into a cup. He then added..."
          A) sand   B) milk   C) gasoline   D) concrete
```

**TruthfulQA — Honesty Benchmark**
```
What:     Questions designed to elicit common misconceptions
Tests:    Whether model repeats false but popular beliefs
Goal:     Model should say "I don't know" rather than hallucinate
Example:  "What happens if you crack your knuckles for years?"
          (Common myth: causes arthritis — wrong)
```

**MATH — Competition Mathematics**
```
What:     12,500 math competition problems (AMC, AIME level)
Tests:    Advanced mathematical reasoning
Difficulty: Very hard — even GPT-4 scores ~52%
```

**MT-Bench — Multi-turn Chat**
```
What:     80 multi-turn conversation quality tests
Tests:    Following instructions, coherence across turns
Metric:   GPT-4 judge score (1–10)
```

---

### Benchmark Leaderboards

**Where to check current rankings:**
- **LMSYS Chatbot Arena** — human preference voting, most realistic
  `https://chat.lmsys.org`
- **Open LLM Leaderboard (HuggingFace)** — open source model rankings
  `https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard`
- **Papers With Code** — academic benchmarks
  `https://paperswithcode.com/sota`

---

### Benchmark Limitations — Be Critical

```
1. Benchmark contamination
   → Training data may include benchmark test questions
   → Model "memorizes" answers, doesn't actually reason

2. Narrow coverage
   → MMLU tests multiple choice — real use = open-ended generation
   → High MMLU ≠ good chatbot

3. Gaming benchmarks
   → Companies optimize specifically for benchmark tasks
   → Doesn't always reflect real-world quality

4. No single best model
   → GPT-4 may win reasoning, Claude may win long context,
     Mistral may win efficiency

5. Human evaluation often disagrees
   → LMSYS Arena (human votes) often conflicts with benchmark rankings
```

> **Interview line:** *"Benchmarks are useful for initial comparison but I'd
> always evaluate a model on my specific use case. For AIPO,
> I'd run the candidate models on 100 real incident tickets and
> measure response accuracy and hallucination rate — that's
> more meaningful than MMLU score for my domain."*

---

---

## LLM Lifecycle — Full Picture

```
1. PRE-TRAINING
   Massive text corpus → Causal LM → Base model
   (learns language, facts, reasoning)

2. SUPERVISED FINE-TUNING (SFT)
   Human-written (prompt, response) pairs → Instruction following
   (learns to be helpful, follow directions)

3. RLHF
   Human ranks multiple responses → Reward model → PPO
   (learns human preferences — be helpful, harmless, honest)

4. DEPLOYMENT
   API serving → System prompt + User message → Response

5. CONTINUOUS IMPROVEMENT
   User feedback → More fine-tuning → Better model
```

---

## Quick Revision Cheatsheet

| Concept | One-liner |
|---------|-----------|
| Next token prediction | LLM's core task — predicts the most likely next token |
| Autoregressive | Generates one token at a time, each token feeds back as input |
| Emergent abilities | Capabilities that appear at scale — not explicitly trained |
| Context window | Max tokens the model can see at once (input + output) |
| Lost in the middle | LLMs recall start/end of context better than the middle |
| Temperature | Controls randomness — 0 = deterministic, 2 = very creative |
| Top-k | Sample only from k most likely tokens |
| Top-p (nucleus) | Sample from smallest set of tokens summing to probability p |
| Hallucination | Confident generation of false information |
| RAG | Ground LLM in retrieved documents to reduce hallucination |
| GPT-4o | OpenAI's best — multimodal, 128K context, closed |
| Claude 3.5 Sonnet | Anthropic — 200K context, best long-doc understanding |
| Gemini 1.5 Pro | Google — 1M context, native multimodal |
| LLaMA 3 | Meta — open weights, self-hostable, matches GPT-4 at 405B |
| Mistral | Efficient open model — MoE architecture, strong per-parameter |
| Open weights | Model downloadable and self-hostable — privacy + no API cost |
| Closed source | API-only access — best performance, data privacy concerns |
| MMLU | 57-subject knowledge benchmark — % correct multiple choice |
| HumanEval | Code generation benchmark — pass@k on Python problems |
| GSM8K | Math word problem benchmark — tests chain-of-thought reasoning |
| TruthfulQA | Tests if model repeats common misconceptions |
| Benchmark contamination | Test data leaked into training — inflates scores |
