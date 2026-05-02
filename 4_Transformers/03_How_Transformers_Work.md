# How Transformers Work

## Complete Forward Pass

### Input Processing

**1. Tokenization**
```
Text: "Hello world"
Tokens: [15496, 995]  (GPT-2 tokenizer IDs)
```

**2. Embedding Lookup**
```
Token IDs → Embedding Matrix → Token Embeddings
[15496, 995] → [768, 768] → [2, 768]
```

**3. Positional Encoding**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Input = Token Embeddings + Positional Encodings
```

### Encoder Layer (Repeated N times)

**Each encoder layer contains:**

1. **Multi-Head Self-Attention**
   ```
   Q = X × W_Q
   K = X × W_K
   V = X × W_V

   Attention = softmax(QK^T / √d_k) × V
   Output = Attention × W_O
   ```

2. **Add & Norm (Residual + LayerNorm)**
   ```
   X = LayerNorm(X + Attention_Output)
   ```

3. **Feed-Forward Network**
   ```
   FFN(x) = GELU(x × W_1 + b_1) × W_2 + b_2

   Expands: d_model → 4 × d_model
   Projects: 4 × d_model → d_model
   ```

4. **Add & Norm**
   ```
   X = LayerNorm(X + FFN_Output)
   ```

### Decoder Layer (Repeated N times)

**Each decoder layer contains:**

1. **Masked Multi-Head Self-Attention**
   - Same as encoder but with causal mask
   - Can only attend to previous positions

2. **Add & Norm**

3. **Cross-Attention (Encoder-Decoder Attention)**
   ```
   Q = Decoder_Output × W_Q
   K = Encoder_Output × W_K
   V = Encoder_Output × W_V

   Attention = softmax(QK^T / √d_k) × V
   ```

4. **Add & Norm**

5. **Feed-Forward Network**

6. **Add & Norm**

### Output Processing

**1. Final Linear Layer**
```
Decoder_Output → Linear → Logits
[batch, seq_len, d_model] → [batch, seq_len, vocab_size]
```

**2. Softmax**
```
Logits → Probabilities
```

**3. Sampling**
```
Probabilities → Next Token
```

## Training Process

### 1. Data Preparation

**Input-Output Pairs:**
```
Translation: "Hello world" → "Bonjour le monde"
Summarization: Long text → Short summary
Next token: "The cat" → "sat"
```

**Batching:**
- Group multiple examples
- Pad to same length
- Create attention masks

### 2. Forward Pass

```
Input → Encoder → Context → Decoder → Logits → Loss
```

**Loss Computation:**
```
Loss = CrossEntropy(Predicted, Target)
```

### 3. Backward Pass

```
Loss → Gradients → Parameter Updates
```

**Optimization:**
- Adam optimizer (typically)
- Learning rate scheduling
- Gradient clipping

### 4. Iteration

Repeat for millions/billions of steps across massive datasets.

## Key Components Deep Dive

### Positional Encoding

**Why needed?**
- Attention has no inherent position information
- "cat sat" vs "sat cat" would be identical without position

**Sinusoidal (original):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Learned (common):**
- Trainable embedding vectors
- Simpler, often works better

**Relative (advanced):**
- Learn relative position biases
- Better generalization to longer sequences

### Layer Normalization

**Purpose:**
- Stabilize hidden state dynamics
- Reduce internal covariate shift
- Enable higher learning rates

**Formula:**
```
LN(x) = γ × (x - μ) / √(σ² + ε) + β

μ = mean(x)
σ² = variance(x)
γ, β = learnable parameters
```

**Placement:**
- **Pre-LN**: Before attention/FFN (modern standard)
- **Post-LN**: After attention/FFN (original paper)

### Residual Connections

**Purpose:**
- Enable training of very deep networks
- Preserve gradient information
- Allow identity mapping

**Formula:**
```
Output = x + Sublayer(x)
```

**Benefits:**
- Gradients flow directly through skip connections
- Network can learn to ignore layers if needed
- Mitigates vanishing gradient problem

## Inference (Generation)

### Autoregressive Generation

**1. Encode Input**
```
Input text → Encoder → Context vectors
```

**2. Generate Token by Token**
```
Start with <SOS> token
Repeat:
  - Get decoder input (previous tokens)
  - Forward pass through decoder
  - Sample next token
  - Append to output
  - Stop at <EOS> or max length
```

**Sampling Strategies:**

**Greedy:**
```
token = argmax(probabilities)
```

**Beam Search:**
```
Keep top-k candidates at each step
Expand all candidates
Keep top-k overall
```

**Temperature Sampling:**
```
probabilities = softmax(logits / temperature)
temperature < 1: more focused
temperature > 1: more diverse
```

**Top-k / Top-p (Nucleus):**
```
Top-k: Keep only k most likely tokens
Top-p: Keep tokens summing to probability p
```

### KV Cache (Optimization)

**Problem:**
- Recomputing attention for all previous tokens is wasteful

**Solution:**
- Cache Key and Value matrices
- Only compute attention for new token

**Speedup:**
- O(n) instead of O(n²) per generated token
- Critical for fast generation

## Model Variants

### BERT (Encoder-only)

**Architecture:**
- 12 transformer encoder layers
- Bidirectional self-attention
- No decoder

**Training:**
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)

**Use cases:**
- Classification
- Named Entity Recognition
- Question Answering

### GPT (Decoder-only)

**Architecture:**
- Transformer decoder layers only
- Causal (masked) self-attention
- No encoder

**Training:**
- Next token prediction
- Autoregressive language modeling

**Use cases:**
- Text generation
- Code generation
- Chatbots

### T5 (Encoder-Decoder)

**Architecture:**
- Full encoder-decoder
- Text-to-text framework

**Training:**
- Span corruption (mask random spans)
- Various tasks framed as text-to-text

**Use cases:**
- Translation
- Summarization
- Question answering

## Scaling Laws

**Performance improves predictably with:**

1. **Model size** (parameters)
2. **Training data** (tokens)
3. **Compute** (FLOPs)

**Key insight:**
- Don't optimize architecture
- Just scale up compute, data, and parameters
- Performance follows power law

**Chinchilla scaling:**
- Optimal compute-efficient training
- Balance model size and data
- Don't over-train small models

## Common Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| d_model | 512-12288 | Model capacity |
| d_ff | 2048-49152 | FFN capacity (4× d_model) |
| n_layers | 6-96 | Depth |
| n_heads | 8-96 | Parallel attention |
| dropout | 0.1-0.3 | Regularization |
| learning rate | 1e-4 - 5e-4 | Training speed |
| batch size | 32- millions | Training stability |

## Training Tips

1. **Warmup** - Gradually increase learning rate
2. **Learning rate decay** - Decrease over time
3. **Gradient clipping** - Prevent exploding gradients
4. **Mixed precision** - Use FP16 for speed
5. **Distributed training** - Scale across multiple GPUs
6. **Checkpointing** - Save intermediate states

## Why Transformers Scale

1. **Parallelizable** - Train on entire sequence at once
2. **Data efficient** - Transfer learning works well
3. **Architecture simple** - Easy to optimize
4. **Hardware friendly** - Matrix operations on GPUs
5. **Emergent abilities** - New capabilities at scale

This combination of architectural simplicity and scaling efficiency enabled the LLM revolution.