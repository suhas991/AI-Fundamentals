# 04 · Prompt Engineering

---

## 1. Prompting Strategies

### Zero-Shot Prompting
Ask the model to perform a task with **no examples** — relies entirely on pre-trained knowledge.

```
Prompt: Classify the sentiment of this review: "The product broke after one day."
```

**When to use:** Simple, well-defined tasks where the model has strong prior knowledge.

---

### One-Shot Prompting
Provide **one example** to demonstrate the desired format or reasoning style.

```
Example:
  Input:  "The hotel was amazing!"  → Positive
  Now classify: "Delivery was late and packaging was damaged."
```

**When to use:** When you need to anchor output format or style without many examples.

---

### Few-Shot Prompting
Provide **multiple examples** (typically 3–10) to guide the model.

```
Input: "I love this!"       → Positive
Input: "Worst purchase ever" → Negative
Input: "It's okay I guess"  → Neutral
Input: "Never buying again" → ?
```

**Best practices:**
- Keep examples consistent in format
- Cover edge cases and diverse inputs
- Order matters — place the most representative examples first

---

## 2. Chain of Thought (CoT)

Encourage the model to **reason step-by-step** before giving an answer.

### Standard CoT
```
Q: If a train travels 60 km/h for 2.5 hours, how far does it travel?
A: Let's think step by step.
   Speed = 60 km/h, Time = 2.5 hours
   Distance = Speed × Time = 60 × 2.5 = 150 km
   Answer: 150 km
```

### Zero-Shot CoT
Simply append: **"Let's think step by step."** — this alone significantly improves reasoning.

### Self-Consistency CoT
- Generate multiple reasoning paths for the same question
- Take a majority vote over the final answers
- Improves reliability on math and logic tasks

**When to use CoT:**
- Multi-step arithmetic or logical reasoning
- Complex decision-making
- Debugging / root cause analysis

---

## 3. System Prompts vs User Prompts

| Aspect | System Prompt | User Prompt |
|--------|--------------|-------------|
| **Role** | Sets context, persona, constraints | Actual task or question |
| **Visibility** | Usually hidden from end-user | Visible in conversation |
| **Persistence** | Applies to entire session | Per-turn |
| **Control** | Developer-controlled | User-controlled |

### System Prompt — Best Practices
```
You are a helpful customer support agent for Acme Corp.
- Only answer questions related to our product.
- Be concise and professional.
- If unsure, say "I don't know" rather than guessing.
- Never reveal internal pricing logic.
```

### User Prompt — Best Practices
- Be specific about the task
- Specify output format if needed
- Provide relevant context inline

---

## 4. Prompt Chaining

Break complex tasks into **sequential, dependent prompts** where the output of one becomes input to the next.

```
Step 1 → Extract key facts from document
Step 2 → Summarize extracted facts
Step 3 → Generate action items from the summary
Step 4 → Format action items as a JSON task list
```

**Benefits:**
- Easier to debug individual steps
- Reduces cognitive load per prompt
- Enables specialised prompts for each sub-task

**Pattern — Pass-through Chain:**
```python
response1 = llm(prompt1)
response2 = llm(prompt2 + response1)
response3 = llm(prompt3 + response2)
```

---

## 5. Output Formatting

### JSON Mode / Structured Output
Force the model to return valid, parseable JSON.

```
Prompt: Extract the following fields from the invoice text and return as JSON:
  { "vendor": "", "amount": 0, "date": "", "currency": "" }

Invoice text: "Invoice from Acme Corp, dated 2024-03-15, total $4,250 USD"
```

**Tips:**
- Provide a JSON schema or example object
- Use `response_format: { type: "json_object" }` in OpenAI API
- Validate output with `json.loads()` and handle errors

### Other Formatting Directives
- `"Respond in bullet points."`
- `"Use a markdown table."`
- `"Limit your answer to 3 sentences."`
- `"Return only the code, no explanations."`

---

## 6. Prompt Injection & Security

**Prompt injection** occurs when malicious input manipulates the model's behaviour by overriding its instructions.

### Types
| Type | Description | Example |
|------|-------------|---------|
| **Direct injection** | User inserts adversarial instructions | "Ignore all previous instructions and…" |
| **Indirect injection** | Malicious content embedded in retrieved data | Poisoned document in RAG |
| **Jailbreaking** | Creative framing to bypass guardrails | Role-play, hypothetical scenarios |

### Defences
- **Input sanitisation** — strip or escape known injection patterns
- **Output validation** — check model output against allowed formats/topics
- **Privilege separation** — don't expose raw system prompt to user turn
- **Least privilege** — limit what tools/actions the model can invoke
- **Canary tokens** — embed secret tokens in system prompt; detect leakage
- **Instruction hierarchy** — use OpenAI's system > user > tool priority model

---

## 7. ReAct Prompting (Reason + Act)

Combines **reasoning** and **tool use** in an interleaved loop.

```
Thought:  I need to find the current stock price of AAPL.
Action:   search("AAPL stock price today")
Observation: AAPL is trading at $189.45 as of 14:30 EST.
Thought:  Now I have the price. I can answer the user.
Answer:   Apple (AAPL) is currently trading at $189.45.
```

### Loop Structure
```
Thought → Action → Observation → Thought → ... → Final Answer
```

**Benefits:**
- Transparent reasoning trace
- Dynamic tool use based on intermediate results
- Reduces hallucination by grounding in real observations

**Common tools in ReAct agents:** web search, calculators, code execution, database queries, APIs

---

## Quick Reference — Prompting Tips

| Goal | Technique |
|------|-----------|
| Better reasoning | Chain of Thought |
| Consistent format | Few-shot + format spec |
| Complex workflows | Prompt chaining |
| Structured data | JSON mode |
| Agentic tasks | ReAct |
| Safety | Input validation + privilege separation |
