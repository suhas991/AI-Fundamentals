# 13 · MLOps & Deployment

---

## 1. Model Serving

### REST API
The most common serving pattern — expose model inference behind an HTTP endpoint.

```python
# FastAPI example
from fastapi import FastAPI
from pydantic import BaseModel
import openai

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": request.message}]
    )
    return {"response": response.choices[0].message.content}
```

### gRPC
- Binary protocol (Protocol Buffers) — lower latency than REST
- Streaming support (server/client/bidirectional)
- Used by: TensorFlow Serving, Triton Inference Server
- Best for: high-throughput internal microservice communication

### Serving Frameworks

| Framework | Best For |
|-----------|---------|
| **vLLM** | High-throughput open-source LLM serving (PagedAttention) |
| **Triton** | NVIDIA; multi-model; GPU optimised |
| **TGI** (HuggingFace) | Optimised transformer serving |
| **Ollama** | Local LLM serving; developer-friendly |
| **LiteLLM** | Unified API gateway across 100+ models |

---

## 2. Containerising AI Apps

### Docker Basics for AI Apps
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt (typical AI app)
```
fastapi
uvicorn
openai
langchain
langchain-openai
chromadb
pydantic
python-dotenv
```

### Docker Compose for AI Stack
```yaml
version: '3.8'
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [chromadb]

  chromadb:
    image: chromadb/chroma
    ports: ["8001:8000"]
    volumes: ["./chroma_data:/chroma/chroma"]
```

---

## 3. Deploying on Cloud

### Azure OpenAI
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://<resource>.openai.azure.com/",
    api_key="...",
    api_version="2024-02-01"
)
response = client.chat.completions.create(
    model="gpt-4o",  # deployment name
    messages=[{"role": "user", "content": "Hello"}]
)
```

### AWS Bedrock
```python
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}]
    })
)
result = json.loads(response["body"].read())
```

### GCP Vertex AI
```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="my-project", location="us-central1")
model = GenerativeModel("gemini-1.5-pro")
response = model.generate_content("Hello")
```

### Cloud Comparison

| Feature | Azure OpenAI | AWS Bedrock | GCP Vertex AI |
|---------|-------------|------------|--------------|
| **Models** | OpenAI + Meta | 30+ models (Claude, Llama, Titan) | Google + partners |
| **Private endpoint** | ✅ VNet integration | ✅ VPC | ✅ VPC |
| **Fine-tuning** | ✅ GPT-4o | Limited | ✅ Gemini |
| **Compliance** | SOC2, HIPAA | SOC2, HIPAA | SOC2, HIPAA |

---

## 4. Model Versioning

```
model_registry/
├── experiment_id: exp_2024_03
├── model_name: customer_support_bot
├── version: v2.1.0
├── base_model: gpt-4o
├── prompt_version: prompts/v3.md
├── fine_tune_id: ft-abc123
├── eval_score: 0.87
└── deployed_to: production
```

**Tools:**
- **MLflow** — model registry + experiment tracking
- **Weights & Biases** — experiment tracking
- **DVC** — data version control (datasets)
- **Langsmith Prompt Hub** — prompt versioning

---

## 5. Batch vs Real-Time Inference

| Aspect | Real-Time | Batch |
|--------|-----------|-------|
| **Trigger** | API request | Scheduled job |
| **Latency** | <1s required | Minutes to hours OK |
| **Cost** | Standard pricing | Often 50% cheaper |
| **Use cases** | Chatbots, copilots | Document processing, embeddings |
| **Tools** | REST API, streaming | OpenAI Batch, AWS Batch |

```python
# OpenAI Batch API
batch = client.batches.create(
    input_file_id="file-abc123",
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
```

---

## 6. Latency Optimisation

| Technique | Impact | Notes |
|-----------|--------|-------|
| **Smaller model** | High | gpt-4o-mini vs gpt-4o |
| **Streaming** | UX | First token fast; perceived speed |
| **Prompt caching** | Medium | Avoid re-processing static context |
| **Reduce output tokens** | High | Shorter max_tokens |
| **Edge deployment** | High | Deploy close to user |
| **Speculative decoding** | Medium | Draft model + verifier |
| **Quantisation** | Medium | INT4/INT8 for self-hosted |
| **KV cache** | High | Built-in; don't clear between turns |
| **Parallel tool calls** | High | Request multiple tools simultaneously |

---

## 7. GPU vs CPU Inference

| Aspect | GPU | CPU |
|--------|-----|-----|
| **Speed** | 10–100× faster | Baseline |
| **Cost** | High (H100 ~$3/hr) | Low |
| **Model size** | Large (70B+) | Small–medium (≤13B) |
| **Best for** | Production; large models | Dev/test; quantised small models |
| **Libraries** | CUDA, vLLM, TRT-LLM | llama.cpp, ONNX Runtime |
| **Quantisation** | GPTQ, AWQ | GGUF (llama.cpp) |

### GPU Memory Requirements (Approximate)
```
7B  model (FP16)  ≈ 14 GB   → RTX 3090 / 4090
13B model (FP16)  ≈ 26 GB   → A100 40GB
70B model (FP16)  ≈ 140 GB  → 2× A100 80GB
70B model (INT4)  ≈ 35 GB   → A100 40GB
```

---

## Quick Reference

```
Serve open-source  → vLLM (GPU) or Ollama (local)
Containerise       → Docker + FastAPI + uvicorn
Cloud LLM          → Azure OpenAI / AWS Bedrock / GCP Vertex
Batch jobs         → OpenAI Batch API (50% cheaper)
Latency            → Smaller model + streaming + caching
GPU sizing         → 2× model size in GB (FP16); 0.5× (INT4)
Versioning         → MLflow / W&B + Langsmith prompt hub
```
