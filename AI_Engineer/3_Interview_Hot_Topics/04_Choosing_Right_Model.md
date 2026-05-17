# Choosing the Right Model for a Task

---

## The Interview Framework

Choose models by constraints, not hype.

Ask in this order:
1. What is the task type?
2. What quality bar is required?
3. What latency and cost limits exist?
4. What deployment/compliance constraints exist?

---

## Task-to-Model Mapping (Practical)

| Task | Typical Best Starting Point |
|---|---|
| High-stakes reasoning | Larger frontier model |
| Summarization at scale | Mid-sized instruction model |
| Classification/extraction | Small model or fine-tuned classifier |
| Code generation | Code-specialized model |
| Multilingual customer support | Strong multilingual model |
| On-device/offline use | Small quantized model |

---

## Selection Criteria

### 1. Capability
- Reasoning depth
- Domain performance
- Tool/function calling quality

### 2. Efficiency
- Cost per 1K tokens
- Tokens per second
- Context-window efficiency

### 3. Reliability
- Hallucination rate
- Format adherence (JSON/schema)
- Determinism for repeated tasks

### 4. Operational fit
- Hosting model: API vs self-hosted
- Data residency/compliance
- Monitoring and fallback support

---

## Model Choice Patterns

1. Route simple tasks to cheaper/smaller models.
2. Escalate difficult prompts to stronger models.
3. Cache common queries and deterministic outputs.
4. Use RAG before larger-model escalation when missing knowledge is the issue.

---

## Evaluation Before Final Choice

- Build a representative benchmark set.
- Score quality with task-specific rubrics.
- Measure latency p50/p95 and cost/query.
- Test failure modes and adversarial prompts.
- Run A/B test in staging or limited rollout.

---

## Common Interview Mistakes

- Picking the largest model by default.
- Ignoring p95 latency and tail behavior.
- Optimizing average quality while failing critical edge cases.
- No fallback strategy for outages/rate limits.

---

## Key Takeaways

- Model selection is a multi-objective optimization problem.
- Start with required outcome, then fit cost/latency constraints.
- Routing and fallback strategies often beat single-model designs.
- Evaluate with real prompts and production-like constraints.
