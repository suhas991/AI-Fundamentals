# Latency vs Quality Trade-offs

---

## Why This Trade-off Exists

Higher quality often requires:
- Larger models
- More context
- More generation tokens
- Additional steps (retrieval, reranking, tool calls)

Each can increase latency and cost.

---

## Latency Budget Breakdown

Typical end-to-end latency:
1. Network and API overhead
2. Retrieval and reranking
3. Prompt assembly
4. First-token latency (model think time)
5. Token generation stream time
6. Post-processing/validation

Interview tip: optimizing only generation speed can miss major delays in retrieval or orchestration.

---

## High-Impact Optimization Levers

### 1. Reduce input size
- Better chunk selection
- Prompt compression
- Remove redundant context

### 2. Reduce output size
- Clear output constraints
- Lower max tokens where possible

### 3. Model routing
- Small model first, escalate only when needed

### 4. Parallelize pipeline steps
- Retrieval + safety checks in parallel where architecture allows

### 5. Cache aggressively
- Semantic cache for repeated intents
- Prompt/result cache for deterministic tasks

---

## Quality Preservation Tactics

- Keep critical instructions at top.
- Use structured outputs to reduce retries.
- Use rerankers to improve context quality at small latency cost.
- Add confidence/uncertainty handling instead of forced answers.

---

## Decision Matrix

| Scenario | Prefer Lower Latency | Prefer Higher Quality |
|---|---|---|
| Live chat support | Yes | Moderate |
| Legal/medical drafting | No | Yes |
| Search preview snippets | Yes | Moderate |
| Executive reports | Moderate | Yes |
| Agentic multi-step workflows | Moderate | Yes |

---

## Metrics to Track

- p50 and p95 latency
- Time-to-first-token (TTFT)
- Task success rate
- Hallucination/groundedness score
- Cost per successful outcome

---

## Key Takeaways

- Optimize for business objective, not raw model benchmark score.
- Tail latency (p95) matters more than average in user experience.
- Routing, caching, and prompt discipline deliver major wins.
- The best system balances speed, quality, and cost per use case.
