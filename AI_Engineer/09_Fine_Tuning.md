# 09 · Fine-Tuning & Model Customisation

---

## 1. When to Fine-Tune vs RAG

| Criterion | Fine-Tune | RAG |
|-----------|-----------|-----|
| **Static knowledge** | ✅ Good fit | ❌ Overkill |
| **Dynamic / changing data** | ❌ Needs retraining | ✅ Update the store |
| **Style / tone / format** | ✅ Bakes in behaviour | ❌ Prompting is better |
| **Proprietary docs** | Possible but expensive | ✅ Natural fit |
| **Latency sensitivity** | ✅ No retrieval overhead | ❌ Adds retrieval step |
| **Cost** | High upfront | Lower ongoing |
| **Citability / traceability** | ❌ Black box | ✅ Sources traceable |

**Rule of thumb:** Try RAG + prompt engineering first. Fine-tune only when they fall short.

---

## 2. Supervised Fine-Tuning (SFT)

Train on labelled input-output pairs to teach the model a specific task or style.

```
Dataset:  [{"prompt": "Classify sentiment: ...", "completion": "Positive"}]
Training: Minimise cross-entropy loss on the completion tokens
```

### SFT Pipeline
```
1. Curate & clean dataset (100–10,000 examples)
2. Format as instruction-following pairs
3. Fine-tune on base or chat model
4. Evaluate on held-out test set
5. Deploy
```

### Dataset Format (JSONL)
```jsonl
{"messages": [
  {"role": "system", "content": "You are a legal contract analyser."},
  {"role": "user", "content": "What are the termination clauses?"},
  {"role": "assistant", "content": "The contract specifies..."}
]}
```

---

## 3. RLHF — Reinforcement Learning from Human Feedback

Used to align models with human preferences (safety, helpfulness, honesty).

### Three Stages

```
Stage 1 — SFT
  Supervised fine-tune on high-quality demonstrations

Stage 2 — Reward Model Training
  Human annotators rank model outputs (A > B > C)
  Train a reward model to predict human preferences

Stage 3 — RL via PPO
  Fine-tune the policy (LLM) using PPO
  Maximise reward model score while minimising KL divergence from SFT model
```

### RLAIF (RL from AI Feedback)
Replace human annotators with a powerful AI judge (e.g., Claude as a constitutional AI critic) — scalable alternative.

---

## 4. LoRA & QLoRA — Parameter-Efficient Fine-Tuning (PEFT)

### The Problem with Full Fine-Tuning
A 7B parameter model requires ~28 GB GPU RAM just to store weights — updating all of them is prohibitively expensive.

### LoRA (Low-Rank Adaptation)
Freeze all original weights. Add small trainable rank-decomposition matrices **A** and **B** alongside original weight matrix **W**.

```
W_updated = W + ΔW
ΔW = A × B    (where A ∈ R^{d×r}, B ∈ R^{r×k}, r << d)

Only A and B are trained — typically <1% of total parameters
```

**Key hyperparameter:** `rank r` (common: 4, 8, 16, 32) — higher r = more capacity, more memory.

### QLoRA (Quantised LoRA)
LoRA + 4-bit quantisation of the base model weights.

```
Base model (frozen, 4-bit) + LoRA adapters (full precision, trainable)
→ Fine-tune 65B model on a single 48GB GPU
```

### LoRA vs Full Fine-Tuning

| Aspect | Full Fine-Tuning | LoRA | QLoRA |
|--------|-----------------|------|-------|
| GPU RAM | Very high | Moderate | Low |
| Training speed | Slow | Fast | Fast |
| Storage | Full model copy | Adapter only (~10 MB) | Adapter only |
| Quality | ✅ Best | ✅ Near-best | Slight degradation |

---

## 5. Dataset Preparation

### Quality > Quantity
- 500 high-quality examples often outperform 10,000 noisy ones
- Diverse coverage of edge cases is critical

### Checklist
- [ ] Define clear input-output pairs
- [ ] Remove duplicates and low-quality examples
- [ ] Balance classes (for classification tasks)
- [ ] Include edge cases and failure modes
- [ ] Split: train / validation / test (e.g., 80/10/10)
- [ ] Format consistently (JSONL for most frameworks)
- [ ] Review for PII and sensitive data

### Tools
- **Label Studio** — annotation UI
- **Argilla** — annotation for NLP
- **Doccano** — open-source annotation

---

## 6. Hugging Face Ecosystem

| Component | Description |
|-----------|-------------|
| **Hub** | Model and dataset repository |
| **Transformers** | Model loading, training, inference |
| **Datasets** | Efficient data loading and processing |
| **PEFT** | LoRA, Prefix Tuning, Prompt Tuning |
| **TRL** | RLHF, SFT, reward modelling |
| **Accelerate** | Distributed training abstraction |
| **BitsAndBytes** | Quantisation library (4/8-bit) |

### Quick SFT with TRL
```python
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"])

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=2048,
)
trainer.train()
```

---

## 7. Model Quantisation

Reduce model size and memory footprint by using lower-precision weights.

| Format | Bits | Size vs FP32 | Quality Loss |
|--------|------|-------------|-------------|
| FP32 | 32 | 1× baseline | None |
| FP16 / BF16 | 16 | 0.5× | Minimal |
| INT8 | 8 | 0.25× | Small |
| INT4 | 4 | 0.125× | Moderate |
| GGUF (2–8 bit) | Variable | Very small | Varies |

### Quantisation Libraries
- **BitsAndBytes** — INT4/INT8 for HuggingFace models
- **GPTQ** — accurate post-training quantisation
- **AWQ** — activation-aware weight quantisation (better quality than GPTQ)
- **llama.cpp** — GGUF format for CPU inference
- **ExLlamaV2** — fast GPTQ inference on GPU

```python
# Load 4-bit quantised model
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
```

---

## Quick Reference

```
Need style/format change → SFT (small dataset)
Need alignment/safety    → RLHF / RLAIF
Limited GPU              → LoRA or QLoRA
Want to run locally      → Quantise (INT4/GGUF)
Dynamic knowledge        → RAG instead of fine-tuning
Dataset prep             → Quality > quantity; JSONL format
HuggingFace stack        → Transformers + PEFT + TRL + Accelerate
```
