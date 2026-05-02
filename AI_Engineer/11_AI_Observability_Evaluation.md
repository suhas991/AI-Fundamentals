# 11 · AI Observability & Evaluation

---

## 1. LLMOps vs MLOps

| Aspect | MLOps (Traditional ML) | LLMOps |
|--------|----------------------|--------|
| **Artifact** | Trained model weights | Prompts + model version + chain |
| **Versioning** | Model checkpoints | Prompt versions + model configs |
| **Evaluation** | Accuracy, F1, RMSE | Faithfulness, relevance, helpfulness |
| **Monitoring** | Data drift, feature drift | Prompt drift, output quality drift |
| **Feedback loop** | Retrain on new labels | RLHF / prompt iteration |
| **Latency** | Inference latency | Token latency (TTFT, TBT) |
| **Cost** | Compute cost | Token cost per call |

---

## 2. Tracing AI Calls

**Tracing** captures the full execution of an LLM call or agent run — inputs, outputs, tool calls, latency, and cost.

### Key Tracing Tools

| Tool | Type | Strengths |
|------|------|-----------|
| **LangSmith** | Commercial | Deep LangChain integration; evals; prompt hub |
| **Langfuse** | Open-source | Self-hostable; sessions; scoring |
| **Arize Phoenix** | Open-source | Local tracing; evaluation UI; OTEL support |
| **Helicone** | Commercial | Lightweight logging; caching; cost tracking |
| **Weights & Biases Weave** | Commercial | ML experiment + LLM tracing |
| **Traceloop** | Open-source | OpenTelemetry native |

### What to Trace
```
Every LLM call:
  - model name + version
  - system prompt + user prompt
  - full response
  - input tokens / output tokens
  - latency (ms)
  - cost ($)

Every tool call:
  - tool name
  - input arguments
  - output / result
  - duration (ms)

Per session:
  - user ID
  - session ID
  - total turns
  - total cost
  - final outcome (success / failure)
```

### LangSmith Quick Setup
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls-..."
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# All LangChain calls are now automatically traced
```

---

## 3. Evaluation Metrics

### Text Quality — BLEU & ROUGE

#### BLEU (Bilingual Evaluation Understudy)
Measures **n-gram precision** between generated and reference text. Primarily used for machine translation.

```
BLEU = BP × exp(Σ wₙ × log pₙ)
  BP = brevity penalty (penalises short outputs)
  pₙ = n-gram precision (n=1,2,3,4)
```

**Limitations:** Doesn't capture fluency or semantics; high BLEU ≠ good output.

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
Measures **recall** of n-grams from reference in generated text. Used for summarisation.

| Variant | Measures |
|---------|---------|
| **ROUGE-N** | n-gram overlap (ROUGE-1, ROUGE-2) |
| **ROUGE-L** | Longest common subsequence |
| **ROUGE-S** | Skip-bigram overlap |

```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, prediction)
```

---

### RAG Evaluation — Faithfulness, Relevance, Groundedness

| Metric | Question | How Measured |
|--------|----------|-------------|
| **Faithfulness** | Is the answer supported by the retrieved context? | LLM-as-judge; check each claim |
| **Answer Relevance** | Does the answer address the question? | Cosine sim of answer ↔ question embeddings |
| **Context Precision** | Are the retrieved chunks actually relevant? | Fraction of relevant chunks retrieved |
| **Context Recall** | Did retrieval find all necessary facts? | Coverage of ground-truth facts |

**RAGAS** — popular RAG evaluation library:
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

---

### Code Evaluation — Pass@k

Measures if any of k generated code samples pass all unit tests.

```
Pass@k = 1 - C(n-c, k) / C(n, k)
  n = total samples generated
  c = samples that pass all tests
  k = samples shown to user
```

**HumanEval** and **MBPP** are standard benchmarks for code generation evaluation.

---

## 4. A/B Testing Prompts

Compare two or more prompt variants on real traffic to determine which performs better.

### Process
```
1. Define success metric (user rating, task completion, cost)
2. Split traffic: 50% → Prompt A, 50% → Prompt B
3. Log all calls with variant label
4. Run until statistical significance achieved
5. Ship winning variant
```

### Statistical Significance
- Use a t-test or chi-squared test depending on the metric type
- Aim for p < 0.05 and sufficient sample size (≥ 200 per variant)

### Tools
- **LangSmith Experiments** — built-in A/B eval
- **Langfuse Experiments** — compare prompt versions
- Custom: log variant + score → analyse in notebook

---

## 5. Cost & Latency Monitoring

### Latency Metrics
| Metric | Description |
|--------|-------------|
| **TTFT** (Time to First Token) | Latency until first token arrives |
| **TBT** (Time Between Tokens) | Inter-token latency for streaming |
| **Total Latency** | Full round-trip time |
| **p50 / p95 / p99** | Percentile latencies |

### Cost Tracking
```python
# Calculate cost per call
input_cost  = (input_tokens  / 1_000_000) * price_per_1M_input
output_cost = (output_tokens / 1_000_000) * price_per_1M_output
total_cost  = input_cost + output_cost
```

**Optimisation levers:**
- Downgrade model for simpler tasks
- Cache repeated prompts
- Reduce system prompt length
- Use batch API (50% cheaper for async)

---

## 6. Guardrails — Input/Output Validation

### Input Guardrails
- Detect and block prompt injection attempts
- Filter PII before sending to LLM
- Classify topic and enforce allowed topics
- Rate limiting per user

### Output Guardrails
- Validate JSON schema compliance
- Check for PII in response (redact if found)
- Content moderation (toxicity, NSFW)
- Hallucination detection (groundedness check)
- Length / format validation

### Guardrail Libraries
| Library | Notes |
|---------|-------|
| **Guardrails AI** | Schema validation, custom validators |
| **NeMo Guardrails** | Nvidia; programmable rails |
| **LlamaGuard** | Meta; safety classification model |
| **Azure Content Safety** | Managed API; multimodal |

---

## Quick Reference

```
Trace everything    → LangSmith / Langfuse (inputs, outputs, cost, latency)
Text eval           → BLEU (MT) / ROUGE (summarisation) / BERTScore (semantic)
RAG eval            → RAGAS (faithfulness, relevance, precision, recall)
Code eval           → Pass@k on unit tests
Prompt testing      → A/B with statistical significance
Cost control        → Monitor token usage + model tier selection
Safety              → Guardrails on input AND output
```
