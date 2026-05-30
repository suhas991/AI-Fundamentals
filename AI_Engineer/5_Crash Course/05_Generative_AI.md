# 05 · Generative AI

---

## 1. Text Generation

LLMs generate text by predicting the next token given a context window.

### Key Concepts
| Parameter | Effect |
|-----------|--------|
| **Temperature** (0–2) | Higher = more creative/random; lower = more deterministic |
| **Top-p (nucleus)** | Sample from the top-p probability mass |
| **Top-k** | Sample from the top-k most likely tokens |
| **Max tokens** | Hard cap on output length |
| **Frequency penalty** | Penalise repeated tokens |

### Use Cases
- Summarisation, translation, Q&A
- Drafting emails, reports, code comments
- Chatbots and conversational agents

---

## 2. Image Generation — Diffusion Models

### How Diffusion Works
1. **Forward process** — Gradually add Gaussian noise to a training image until it is pure noise
2. **Reverse process** — Train a neural network (U-Net) to iteratively denoise; learn to reverse noise step-by-step
3. **Inference** — Start from random noise, run denoising steps conditioned on a text prompt

```
Noise → Denoise (T steps) → Image
         ↑ guided by text embedding
```

### Key Models

| Model | Creator | Notes |
|-------|---------|-------|
| **DALL·E 3** | OpenAI | Text-image via API; strong prompt following |
| **Stable Diffusion** | Stability AI | Open-source; runs locally; highly customisable |
| **Midjourney** | Midjourney Inc. | High aesthetic quality; Discord/API |
| **Imagen** | Google DeepMind | Strong text rendering |
| **Flux** | Black Forest Labs | Open weights; state-of-the-art realism |

### Important Concepts
- **CFG (Classifier-Free Guidance)** — scale controlling how closely image follows the prompt
- **LoRA / DreamBooth** — fine-tune on custom styles or subjects
- **ControlNet** — guide generation with depth maps, poses, edges
- **Latent Diffusion** — diffusion in compressed latent space (faster, smaller)

---

## 3. Code Generation

### Tools & Models
| Tool | Underlying Model | Integration |
|------|-----------------|-------------|
| **GitHub Copilot** | GPT-4o / Codex | IDE plugin |
| **CodeLlama** | Meta | Open-source; 7B–70B |
| **Cursor** | GPT-4 / Claude | AI-first IDE |
| **Amazon Q** | Amazon | AWS-integrated |
| **Gemini Code Assist** | Gemini | Google Cloud / IDE |

### Capabilities
- Autocomplete (line / block level)
- Function generation from docstrings
- Unit test generation
- Code explanation and review
- Bug fixing and refactoring
- Multi-file context understanding

### Evaluation — Pass@k
> The probability that at least one of k generated solutions passes all unit tests.

```
Pass@k = 1 - C(n-c, k) / C(n, k)
  where n = total samples, c = correct samples
```

---

## 4. Audio Generation

| Capability | Models / Tools |
|------------|---------------|
| **Text-to-Speech (TTS)** | ElevenLabs, OpenAI TTS, Bark, XTTS |
| **Music generation** | Suno, Udio, MusicGen (Meta) |
| **Speech-to-Speech** | Real-time voice cloning |
| **Audio effects** | AudioCraft (Meta) |

**Key techniques:** WaveNet, neural vocoders, diffusion-based audio models

---

## 5. Video Generation

| Model | Creator | Notes |
|-------|---------|-------|
| **Sora** | OpenAI | High-quality, long clips |
| **Gen-3 Alpha** | Runway | Commercial, fast |
| **Kling** | Kuaishou | Strong motion quality |
| **Veo 2** | Google DeepMind | High resolution |
| **CogVideoX** | Zhipu AI | Open-source |

**Challenges:** Temporal consistency, physics simulation, long-video coherence

---

## 6. Multimodal Models

Models that accept and/or generate **multiple modalities** (text, image, audio, video).

### Input–Output Matrix

| Model | Text In | Image In | Audio In | Text Out | Image Out |
|-------|---------|---------|---------|---------|----------|
| GPT-4o | ✅ | ✅ | ✅ | ✅ | ✅ |
| Claude 3.5 | ✅ | ✅ | ❌ | ✅ | ❌ |
| Gemini 1.5 Pro | ✅ | ✅ | ✅ | ✅ | ❌ |
| LLaVA | ✅ | ✅ | ❌ | ✅ | ❌ |

### Architecture Patterns
- **Early fusion** — combine modalities before the model
- **Late fusion** — separate encoders; fuse at decision layer
- **Cross-attention** — image patch tokens attend to text tokens (e.g., Flamingo)

---

## 7. Evaluation of GenAI Outputs

### Text Quality
| Metric | Description |
|--------|-------------|
| **BLEU** | n-gram overlap with reference (mainly MT) |
| **ROUGE** | Recall-oriented overlap (summarisation) |
| **BERTScore** | Semantic similarity using BERT embeddings |
| **Perplexity** | How "surprised" the model is by real text |

### Human Evaluation Dimensions
- **Fluency** — grammatically and stylistically natural
- **Coherence** — logically consistent throughout
- **Faithfulness** — no hallucinated facts
- **Relevance** — answers the actual question
- **Helpfulness** — useful to the end user

### Image Quality
| Metric | Description |
|--------|-------------|
| **FID** (Fréchet Inception Distance) | Distributional similarity between real/generated images |
| **CLIP Score** | Text-image alignment |
| **IS** (Inception Score) | Diversity and quality |

### LLM-as-Judge
Use a powerful LLM (e.g., GPT-4) to score outputs on rubric criteria — scalable alternative to human eval.

---

## Quick Reference

```
Text Gen   → LLMs + sampling parameters
Image Gen  → Diffusion models (noise → denoise)
Code Gen   → Fine-tuned LLMs + IDE integration
Audio Gen  → Neural vocoders / diffusion audio
Video Gen  → Temporal diffusion / transformer models
Multimodal → Cross-modal encoders + LLM backbone
Evaluation → Automated metrics + human / LLM-as-judge
```
