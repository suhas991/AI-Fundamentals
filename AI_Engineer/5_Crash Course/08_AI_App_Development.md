# 08 · AI Application Development

---

## 1. LLM APIs

### Major Providers

| Provider | Models | Base URL |
|----------|--------|---------|
| **OpenAI** | GPT-4o, GPT-4.1, o3 | `api.openai.com/v1` |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus | `api.anthropic.com/v1` |
| **Google** | Gemini 1.5 Pro, Gemini 2.0 | `generativelanguage.googleapis.com` |
| **Mistral** | Mistral Large, Mixtral | `api.mistral.ai/v1` |
| **Cohere** | Command R+ | `api.cohere.ai/v1` |

### OpenAI SDK — Basic Call
```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum entanglement."}
    ],
    max_tokens=500,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Anthropic SDK
```python
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-...")

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(message.content[0].text)
```

---

## 2. LangChain / LlamaIndex Fundamentals

### LangChain Core Concepts
| Concept | Description |
|---------|-------------|
| **Chain** | Sequence of calls (LLM + tools + logic) |
| **Prompt Template** | Parameterised prompt with variables |
| **Memory** | Persist conversation state between calls |
| **Agent** | LLM that decides which tools to call |
| **Retriever** | Fetch relevant documents |

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"role": "historian", "question": "When did WW2 end?"})
```

### LlamaIndex Core Concepts
| Concept | Description |
|---------|-------------|
| **Document** | Raw text unit |
| **Node** | Chunk with metadata |
| **Index** | Indexed collection (VectorStore, Tree, Keyword) |
| **QueryEngine** | Retrieves + synthesises answer |
| **Retriever** | Finds relevant nodes |

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

docs = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(docs)
engine = index.as_query_engine()
response = engine.query("What is the refund policy?")
```

---

## 3. Streaming Responses

Stream tokens as they are generated for better UX (no long wait).

```python
# OpenAI streaming
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem."}],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

```python
# Anthropic streaming
with client.messages.stream(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a poem."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

**Frontend (SSE):**
```javascript
const response = await fetch("/api/chat", { method: "POST", body: payload });
const reader = response.body.getReader();
// Read chunks and update UI
```

---

## 4. Conversation History Management

LLMs are stateless — you must send full history every request.

```python
history = []

def chat(user_message):
    history.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are helpful."}] + history
    )
    
    assistant_message = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_message})
    return assistant_message
```

### Handling Long Histories
- **Sliding window** — keep last N messages
- **Summarisation** — compress old messages: "Earlier we discussed X, Y, Z."
- **Selective retrieval** — vector-search relevant past messages

---

## 5. Function / Tool Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Mumbai?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if model wants to call a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result = get_weather(**args)  # your actual function
```

---

## 6. Structured Outputs

Force JSON schema-compliant output (OpenAI feature):

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract: John Doe, 30, from New York."}],
    response_format=Person
)

person = response.choices[0].message.parsed
print(person.name)  # "John Doe"
```

---

## 7. Token Optimisation & Cost Management

### Token Count Rules of Thumb
- 1 token ≈ 4 characters (English)
- 1 token ≈ 0.75 words
- 1 page of text ≈ 500–750 tokens

### Cost Reduction Strategies
| Strategy | Description |
|----------|-------------|
| **Smaller model** | Use gpt-4o-mini for simple tasks |
| **Reduce system prompt** | Remove unnecessary verbosity |
| **Truncate history** | Sliding window on conversation |
| **Prompt compression** | LLMLingua-style compression |
| **Batch API** | 50% cheaper for async tasks (OpenAI Batch) |
| **Output limits** | Set appropriate `max_tokens` |
| **Cache common prompts** | Avoid redundant calls |

### Model Tiers (OpenAI example)
```
gpt-4o          → Full capability, higher cost
gpt-4o-mini     → Fast, cheap, surprisingly capable
o3-mini         → Reasoning tasks
gpt-3.5-turbo   → Legacy, very cheap
```

---

## 8. Caching Strategies

| Strategy | Tools | Notes |
|----------|-------|-------|
| **Exact cache** | Redis, dict | Cache identical prompts |
| **Semantic cache** | GPTCache, LangChain cache | Cache similar prompts |
| **Prompt caching** | Anthropic/OpenAI native | Cache repeated system prompts server-side |
| **CDN / edge cache** | CloudFront, Cloudflare | Cache static API responses |

```python
# Simple in-memory cache
import hashlib, json

cache = {}

def cached_llm_call(prompt):
    key = hashlib.md5(prompt.encode()).hexdigest()
    if key in cache:
        return cache[key]
    result = llm.generate(prompt)
    cache[key] = result
    return result
```

---

## Quick Reference

```
API setup       → SDK init + API key + model selection
Streaming       → stream=True + iterate chunks
History         → Send full messages[] array each call
Tool calling    → Define schema → parse tool_calls → execute → append result
Structured out  → response_format with Pydantic model
Cost control    → Right-size model + cache + batch + compress prompts
```
