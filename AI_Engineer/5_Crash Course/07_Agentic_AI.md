# 07 · Agentic AI

---

## 1. What is an AI Agent?

An AI agent is a system that uses an LLM as its **reasoning engine** to autonomously plan and execute multi-step tasks by interacting with tools and its environment.

**Key distinction from a chatbot:**
- Chatbot: one-turn or simple conversational
- Agent: multi-step, goal-directed, uses external tools, has memory

---

## 2. Agent Loop — Perceive → Think → Act → Observe

```
┌──────────────────────────────────────────┐
│               AGENT LOOP                 │
│                                          │
│  Perceive  →  Think   →  Act             │
│  (Input)     (LLM)      (Tool call)      │
│     ↑                        │           │
│     └────── Observe ─────────┘           │
│             (Tool result)                │
│                                          │
│  Repeat until goal achieved              │
└──────────────────────────────────────────┘
```

| Phase | Description |
|-------|-------------|
| **Perceive** | Receive user goal + environment state |
| **Think** | LLM reasons about what to do next |
| **Act** | Execute a tool call or response |
| **Observe** | Process the result and update context |

---

## 3. Tools & Function Calling

Tools extend the LLM beyond text generation — enabling it to **interact with the real world**.

### How Function Calling Works
1. Developer defines tool schemas (name, description, parameters as JSON Schema)
2. LLM decides when and how to call a tool
3. Application executes the call and returns results
4. LLM incorporates results into its next response

```json
// Tool definition example
{
  "name": "search_web",
  "description": "Search the web for current information",
  "parameters": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "Search query" }
    },
    "required": ["query"]
  }
}
```

### Common Tool Categories
| Category | Examples |
|----------|---------|
| **Search** | Web search, document search, database query |
| **Code execution** | Python REPL, bash shell |
| **APIs** | REST calls, GraphQL, webhooks |
| **File I/O** | Read/write files, parse PDFs |
| **Browser** | Web scraping, form filling |
| **Communication** | Send email, Slack, calendar |

---

## 4. Memory Types

| Type | Description | Storage | Lifespan |
|------|-------------|---------|---------|
| **Short-term (In-context)** | Current conversation history in the context window | RAM / prompt | Single session |
| **Long-term (External)** | Past facts and summaries stored externally | DB / Vector store | Persistent |
| **Episodic** | Records of past interactions / events | DB | Persistent |
| **Semantic** | General world knowledge baked into weights | Model weights | Fixed at training |

### Memory Management Techniques
- **Summarisation** — compress old messages into summaries
- **Sliding window** — keep only the last N turns
- **Entity tracking** — extract and store named entities separately
- **Vector memory** — embed and retrieve relevant memories by similarity

---

## 5. Planning Strategies

### ReAct (Reason + Act)
Interleave thought and action traces:
```
Thought: I need to check today's weather in Mumbai.
Action: get_weather(city="Mumbai")
Observation: 32°C, partly cloudy.
Thought: Now I can answer the user.
Answer: It's currently 32°C and partly cloudy in Mumbai.
```

### Chain of Thought (CoT) Planning
Produce a full reasoning chain before taking any action — better for single-step complex problems.

### Plan-and-Execute
1. **Plan phase** — Generate a complete step-by-step plan upfront
2. **Execute phase** — Execute each step sequentially, updating plan if needed

```
Plan:
  1. Search for the company's revenue in 2023
  2. Search for their revenue in 2022
  3. Calculate YoY growth rate
  4. Format as a table
```

### Tree of Thoughts (ToT)
Explore multiple reasoning branches in parallel, evaluate each, backtrack if needed.

---

## 6. Multi-Agent Systems

Multiple specialised agents collaborate to solve complex tasks.

### Topologies

| Pattern | Description |
|---------|-------------|
| **Sequential** | Agent A → Agent B → Agent C (pipeline) |
| **Parallel** | Multiple agents run simultaneously, results merged |
| **Hierarchical** | Orchestrator agent delegates to sub-agents |
| **Debate** | Agents critique each other's outputs |

### Benefits
- Specialisation (one agent per domain)
- Parallelism for speed
- Redundancy / cross-checking
- Better at tasks exceeding single context window

---

## 7. Agent Frameworks

| Framework | Creator | Strengths |
|-----------|---------|-----------|
| **LangChain** | LangChain Inc. | Large ecosystem; chains; integrations |
| **LlamaIndex** | LlamaIndex | RAG-focused; data connectors |
| **CrewAI** | João Moura | Role-based multi-agent; easy setup |
| **AutoGen** | Microsoft | Multi-agent conversations; code execution |
| **LangGraph** | LangChain Inc. | Graph-based agent flows; cycles |
| **Smolagents** | Hugging Face | Lightweight; minimal abstraction |
| **Pydantic AI** | Pydantic | Type-safe agents |

---

## 8. Human-in-the-Loop (HITL)

Inject human oversight at critical decision points to maintain safety and accuracy.

### HITL Patterns
- **Approval gate** — human approves before high-stakes action (send email, delete file)
- **Clarification** — agent asks user for more info when uncertain
- **Review loop** — human reviews and edits draft before finalisation
- **Interrupt** — human can pause/redirect the agent mid-task

```python
# Example: approval gate
if action.type == "send_email":
    approval = ask_human(f"Approve sending to {action.recipient}?")
    if not approval:
        return "Action cancelled"
```

---

## 9. Agent Evaluation & Observability

### Evaluation Dimensions
| Dimension | Description |
|-----------|-------------|
| **Task success rate** | Did the agent complete the goal? |
| **Step efficiency** | Minimum steps vs actual steps taken |
| **Tool call accuracy** | Correct tool + correct arguments |
| **Faithfulness** | No hallucinated facts in output |
| **Safety** | No harmful or unauthorised actions |

### Observability Tools
| Tool | Features |
|------|---------|
| **LangSmith** | Traces, evals, prompt management |
| **Arize Phoenix** | OSS tracing and evaluation |
| **Helicone** | Cost/latency logging |
| **Langfuse** | OSS LLMOps platform |

### What to Trace
- Every LLM call (prompt, response, tokens, latency)
- Every tool call (name, args, result, duration)
- Agent decision points
- Total cost and turn count per task

---

## Quick Reference

```
Single agent   → ReAct loop + tools
Multi-agent    → CrewAI / AutoGen / LangGraph
Memory         → In-context (short) + vector DB (long)
Planning       → ReAct / Plan-and-Execute / ToT
Observability  → LangSmith / Langfuse traces
Safety         → HITL approval gates + guardrails
```
