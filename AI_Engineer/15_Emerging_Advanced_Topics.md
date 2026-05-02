# 15 · Emerging & Advanced Topics

---

## 1. GraphRAG — Knowledge Graphs + RAG

Traditional RAG retrieves isolated text chunks. **GraphRAG** builds a **knowledge graph** from documents, enabling multi-hop reasoning and relationship queries.

### Architecture
```
Documents → Entity extraction → Knowledge Graph
                                      ↓
Query → Graph traversal + vector search → Context → LLM → Answer
```

### When GraphRAG Wins Over Naive RAG
- "What is the relationship between X and Y?"
- "What are all products connected to this supplier?"
- Multi-hop: "Find colleagues of the person who manages the Paris office"

### Tools
- **Microsoft GraphRAG** — open-source; community + global graph summaries
- **LlamaIndex Knowledge Graph Index**
- **Neo4j + LLM** — property graph + vector search
- **NebulaGraph**

---

## 2. Long Context Models vs RAG

| Aspect | Long Context Models | RAG |
|--------|---------------------|-----|
| **Max context** | 128K–2M tokens | Virtually unlimited (external) |
| **Latency** | Scales with context length | Retrieval + generation |
| **Cost** | High (all tokens billed) | Lower (only relevant chunks) |
| **Staleness** | Fixed at inference | Real-time via DB update |
| **Precision** | May lose focus in long context | Focused on retrieved chunks |
| **Setup** | Simple — just send document | Pipeline required |

**Rule of thumb:**
- ≤ ~100K tokens → long context may suffice
- Millions of tokens / dynamic data → RAG is better
- Best of both → hybrid (long context + retrieval)

---

## 3. Mixture of Experts (MoE)

Instead of activating all model parameters for every token, MoE selectively activates **only a subset of "expert" sub-networks**.

```
Input token → Router → Expert 1 ─┐
                    ↘ Expert 7 ──┤→ Weighted sum → Output
                       (2 of 64 experts activated per token)
```

### Benefits
- **Sparse activation** — only 2–8 experts active per forward pass
- **More parameters, same compute** — e.g., Mixtral 8×7B has 47B total params but uses ~13B per token
- **Specialisation** — different experts learn different knowledge domains

### Notable MoE Models
| Model | Architecture | Notes |
|-------|-------------|-------|
| **Mixtral 8×7B** | 8 experts, 2 active | Strong open-source |
| **GPT-4** | Believed to be MoE | Unconfirmed |
| **Gemini 1.5** | MoE-based | Long context |
| **Qwen 2 MoE** | Dense + sparse layers | Efficient |

---

## 4. Constitutional AI (CAI)

Anthropic's technique for training safe AI systems using a **written set of principles** (a "constitution") to guide self-critique.

### Process
```
1. Generate initial response
2. Ask model to critique its own response against constitutional principles
3. Ask model to revise based on critique
4. Use revised response as training data for RLHF
```

### Example Principles (from Anthropic's constitution)
- "Please rewrite the response to avoid content that could harm the user."
- "Choose the response that is least likely to contain harmful or unethical content."

### Benefits
- Reduces need for human labelling of harmful examples
- Scalable and auditable (written principles)
- Enables AI to improve its own safety properties

---

## 5. Multimodal Agents

Agents that can **perceive and act across multiple modalities**: text, images, audio, video, code execution, browser control.

### Capabilities
```
Perceive:  Text, screenshots, audio transcripts, PDFs
Reason:    LLM with multimodal understanding
Act:       Type, click, scroll, API call, code execution
```

### Examples
- **Computer use** (Anthropic Claude) — controls desktop UI
- **GPT-4V + code interpreter** — analyses charts, generates Python
- **Gemini + YouTube** — watches and summarises video

### Challenges
- Spatial reasoning about UI elements
- Action grounding (OCR + coordinate mapping)
- Long task coherence
- Error recovery

---

## 6. AI Memory Architectures

### Mem0
Intelligent memory layer that automatically extracts, stores, and retrieves relevant memories for personalised AI experiences.

```python
from mem0 import Memory

memory = Memory()
memory.add("User prefers Python over JavaScript", user_id="alice")

# Later, in a new session:
relevant = memory.search("What language does this user prefer?", user_id="alice")
# Returns: Python preference
```

### MemGPT
OS-inspired memory management for LLMs — treats the context window as "main memory" and uses external storage as a "disk".

```
Main context (in-context) ← page in/out → Archival storage (external)
     ↓                                           ↑
  Working memory                         Long-term memory DB
```

**Key idea:** Agent can self-direct which memories to load/unload from context.

### Memory Architecture Summary
| Type | Where Stored | Duration | Access |
|------|-------------|---------|--------|
| Working | Context window | Per session | Automatic |
| Episodic | DB / vector store | Persistent | Semantic search |
| Semantic | Model weights | Permanent | Implicit |
| Procedural | System prompt / tools | Configurable | Explicit |

---

## 7. MCP — Model Context Protocol

An open protocol by Anthropic that standardises how AI models connect to external data sources and tools.

```
AI App (Host)
    │
    └─── MCP Client ─── MCP Server ─── [Data Source / Tool]
                                        (Files, DBs, APIs, etc.)
```

### Core Primitives
| Primitive | Description |
|-----------|-------------|
| **Tools** | Functions the model can call |
| **Resources** | Data the model can read (files, DB records) |
| **Prompts** | Reusable prompt templates |
| **Sampling** | Model can request completions from the host |

### Benefits
- Standardised interface — build once, connect anywhere
- Works across Claude, OpenAI-compatible hosts
- Replaces bespoke tool integrations

```python
# MCP server example (Python SDK)
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("my-tool-server")

@app.tool()
async def get_weather(city: str) -> str:
    """Get current weather for a city"""
    return f"Weather in {city}: 28°C, sunny"
```

---

## 8. AI Gateways & Routing — LiteLLM

**LiteLLM** provides a unified API interface over 100+ LLM providers.

```python
import litellm

# Same interface regardless of provider
response = litellm.completion(
    model="anthropic/claude-sonnet-4-5",  # or "openai/gpt-4o" or "gemini/gemini-1.5-pro"
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Gateway Features
| Feature | Description |
|---------|-------------|
| **Load balancing** | Distribute across multiple endpoints |
| **Fallback routing** | If GPT-4 fails, try Claude |
| **Cost tracking** | Unified spend across all providers |
| **Rate limit handling** | Automatic retry + backoff |
| **Caching** | Semantic and exact caching |
| **Logging** | Send to Langfuse, Helicone, etc. |

### LiteLLM as Proxy
```bash
litellm --model gpt-4o --model claude-sonnet-4-5
# Exposes OpenAI-compatible endpoint at localhost:4000
# Any OpenAI SDK can now use multiple models via proxy
```

---

## Quick Reference

```
GraphRAG          → Knowledge graph + vector search; multi-hop queries
Long context      → Simple but expensive; use for ≤100K tokens
MoE               → Sparse expert activation; more params, same compute
Constitutional AI → Self-critique via principles; Anthropic's alignment method
Multimodal agents → See, type, click, code across UI and docs
Mem0 / MemGPT     → Persistent personalised memory across sessions
MCP               → Standard protocol for tools + data sources
LiteLLM           → Single API for 100+ LLMs; routing + fallback + cost
```
