# 14 · Cloud AI Services

---

## 1. Azure AI Services

### Azure OpenAI Service
Managed deployment of OpenAI models inside Azure infrastructure.

**Key features:**
- Private endpoints (VNet integration — data never leaves your tenant)
- Role-based access control (RBAC)
- Content filtering (built-in Azure Content Safety)
- Model fine-tuning support (GPT-4o, GPT-3.5)
- Provisioned throughput (PTUs) for guaranteed capacity

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://<resource-name>.openai.azure.com/",
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version="2024-02-01"
)

response = client.chat.completions.create(
    model="gpt-4o",           # deployment name in Azure
    messages=[{"role": "user", "content": "Summarise this report"}],
    temperature=0.3
)
```

---

### Azure AI Search (formerly Cognitive Search)
Enterprise search with built-in hybrid (vector + keyword) capabilities.

**Key features:**
- Integrated vectorisation (embed during indexing)
- Hybrid search (BM25 + vector)
- Semantic re-ranking
- Connects directly to Azure Blob, SQL, Cosmos DB
- Powers the "Add your data" feature in Azure OpenAI

**Typical RAG stack on Azure:**
```
Blob Storage → Azure AI Search (index) → Azure OpenAI (generate)
```

---

### Azure Document Intelligence (formerly Form Recognizer)
Extract structured data from documents using pre-built or custom models.

**Models:**
| Model | Use Case |
|-------|---------|
| **Read** | OCR for scanned documents |
| **Layout** | Tables, paragraphs, sections |
| **Invoice** | Vendor, amount, line items |
| **Receipt** | Merchant, total, items |
| **ID Document** | Passport, driver's licence |
| **Custom** | Train on your own forms |

---

### Azure AI Foundry (formerly Azure Machine Learning)
End-to-end platform for building, deploying, and managing AI applications.

**Features:**
- Model catalogue (OpenAI, Meta, Mistral, Cohere, etc.)
- Prompt flow — visual pipeline builder for LLM apps
- Evaluation tools (built-in metrics)
- Online + batch endpoints for deployment
- Responsible AI dashboard

---

## 2. AWS AI Services

### AWS Bedrock
Managed API access to foundation models from multiple providers.

**Available model families:**
| Provider | Models |
|----------|--------|
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Haiku, Opus |
| **Meta** | Llama 3.1 8B, 70B, 405B |
| **Amazon** | Titan Text, Titan Embeddings |
| **Mistral** | Mistral Large, Mixtral |
| **Cohere** | Command R, Command R+ |
| **Stability AI** | Stable Diffusion |

**Key features:** Serverless; private; AWS IAM; Guardrails for Bedrock; Agents for Bedrock

```python
import boto3, json

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock_runtime.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello!"}]
    }),
    contentType="application/json"
)
output = json.loads(response["body"].read())
```

---

### AWS SageMaker
Fully managed platform for building, training, and deploying ML models.

**Key components:**
| Component | Purpose |
|-----------|---------|
| **Studio** | IDE for ML development |
| **Training Jobs** | Managed distributed training |
| **Inference Endpoints** | Deploy models as REST APIs |
| **Pipelines** | MLOps workflow automation |
| **JumpStart** | Pre-built models + solutions |
| **Feature Store** | Centralised ML features |

---

### AWS Rekognition
Computer vision service for image and video analysis.

**Capabilities:**
- Object and scene detection
- Facial analysis and comparison
- Text in image (OCR)
- Celebrity recognition
- Unsafe content detection
- Video activity detection

---

## 3. GCP AI Services

### Vertex AI
Google Cloud's unified ML platform.

**Key features:**
- Model Garden — access Gemini, Llama, Claude, and open-source models
- Auto ML — no-code model training
- Custom training with managed infrastructure
- Online and batch prediction endpoints
- Feature Store
- Model monitoring and explainability

```python
import vertexai
from vertexai.generative_models import GenerativeModel, Part

vertexai.init(project="my-gcp-project", location="us-central1")
model = GenerativeModel("gemini-1.5-pro-002")

response = model.generate_content([
    "Describe what you see in this image.",
    Part.from_uri("gs://my-bucket/image.jpg", mime_type="image/jpeg")
])
print(response.text)
```

---

### Gemini API (Google AI Studio / Vertex AI)
Direct access to Gemini models.

| Model | Context Window | Strengths |
|-------|---------------|-----------|
| **Gemini 2.0 Flash** | 1M tokens | Fast; multimodal; tool use |
| **Gemini 1.5 Pro** | 2M tokens | Long context; complex reasoning |
| **Gemini 1.5 Flash** | 1M tokens | Cost-efficient; fast |

**Multimodal inputs:** Text, images, video, audio, PDF, code

---

### Google Vision AI
Pre-trained computer vision APIs.

**Capabilities:**
- Label detection
- OCR (text extraction)
- Face detection
- Object localisation
- Safe search (content moderation)
- Logo and landmark detection

---

## Quick Comparison — Cloud AI Services

| Use Case | Azure | AWS | GCP |
|----------|-------|-----|-----|
| **OpenAI models** | Azure OpenAI ✅ | Bedrock (no OpenAI) | ❌ |
| **Multi-provider LLMs** | AI Foundry | Bedrock ✅ | Vertex Model Garden |
| **Managed vector search** | Azure AI Search ✅ | OpenSearch | Vertex AI Vector Search |
| **Document extraction** | Document Intelligence ✅ | Textract | Document AI |
| **Computer vision** | Azure Vision | Rekognition | Vision AI |
| **ML platform** | Azure ML / AI Foundry | SageMaker | Vertex AI |
| **Serverless LLM** | ✅ | ✅ Bedrock | ✅ Vertex |
| **Enterprise compliance** | ✅ HIPAA, ISO | ✅ | ✅ |

---

## Quick Reference

```
Azure stack   → Azure OpenAI + AI Search + Document Intelligence + AI Foundry
AWS stack     → Bedrock + SageMaker + Rekognition
GCP stack     → Vertex AI + Gemini API + Vision AI

Best for OpenAI models  → Azure OpenAI (private, compliant)
Best multi-model choice → AWS Bedrock (30+ models)
Best long-context LLM   → GCP Gemini 1.5 Pro (2M tokens)
Best doc extraction     → Azure Document Intelligence
Best computer vision    → AWS Rekognition or GCP Vision AI
```
