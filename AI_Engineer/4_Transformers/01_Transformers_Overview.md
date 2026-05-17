# Transformers Overview

## What is a Transformer?

A Transformer is a deep learning architecture introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. It revolutionized NLP by replacing recurrent neural networks (RNNs) with attention mechanisms.

## Key Innovation

**Attention Is All You Need** - Transformers rely entirely on attention mechanisms, eliminating the need for recurrence or convolution.

## Architecture

### Encoder-Decoder Structure

```
Input → Encoder → Context Vector → Decoder → Output
```

**Encoder:**
- Processes input sequence
- Creates contextual representations
- Uses self-attention to understand relationships between all input tokens

**Decoder:**
- Generates output sequence
- Uses encoder output and previous predictions
- Employs masked self-attention for autoregressive generation

### Modern Variants

**Encoder-only (BERT):**
- Great for understanding tasks (classification, NER)
- Bidirectional context
- Pre-train: Masked Language Modeling

**Decoder-only (GPT):**
- Excellent for generation tasks
- Unidirectional (left-to-right) context
- Pre-train: Next token prediction

**Encoder-Decoder (T5, BART):**
- Best for sequence-to-sequence tasks
- Translation, summarization
- Pre-train: Span corruption

## Core Components

### 1. Self-Attention
Allows each token to attend to all other tokens in the sequence:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### 2. Multi-Head Attention
Runs multiple attention operations in parallel:
- 8-12 heads typical
- Each head learns different relationships
- Outputs concatenated and projected

### 3. Positional Encoding
Since attention has no inherent notion of position:
- Sinusoidal functions (original paper)
- Learned positional embeddings (common practice)
- Added to token embeddings

### 4. Feed-Forward Networks
Two-layer MLP after each attention layer:
- Expands dimensionality (typically 4x)
- Applies non-linearity (ReLU/GELU)
- Projects back to original dimension

### 5. Layer Normalization
Stabilizes training:
- Applied before attention (Pre-LN) or after (Post-LN)
- Normalizes across features
- Helps with gradient flow

### 6. Residual Connections
Skip connections around each sub-layer:
- `LayerNorm(x + Sublayer(x))`
- Enables training of very deep networks
- Preserves gradient information

## Why Transformers Work

### Parallelization
- Unlike RNNs, processes all tokens simultaneously
- Enables massive parallel training on GPUs
- Scales to billions of parameters

### Long-Range Dependencies
- Attention connects any two tokens directly
- No vanishing gradient problem over long sequences
- O(1) path length between any positions

### Scalability
- Performance improves with more data and compute
- Emergent abilities at scale (reasoning, coding)
- Foundation for modern LLMs

## Model Sizes

| Model | Parameters | Layers | Hidden Size | Heads |
|--------|-----------|--------|-------------|-------|
| BERT-Base | 110M | 12 | 768 | 12 |
| GPT-2 | 1.5B | 48 | 1600 | 25 |
| GPT-3 | 175B | 96 | 12288 | 96 |
| GPT-4 | ~1.7T | ~120 | ~12288 | ~96 |

## Training Stages

1. **Pre-training** - Learn from massive text corpora
2. **Fine-tuning** - Adapt to specific tasks
3. **RLHF** - Align with human preferences (for chat models)

## Key Advantages Over RNNs

- **Parallel processing** - Train on entire sequence at once
- **Long-term memory** - No sequential bottleneck
- **Better performance** - State-of-the-art on most NLP tasks
- **Transfer learning** - Pre-trained models work well across domains

## Limitations

- **Quadratic complexity** - O(n²) attention computation
- **Memory intensive** - Stores all attention weights
- **Position bias** - May struggle with very long sequences
- **Data hungry** - Requires massive pre-training data

## Applications

- Machine translation
- Text generation
- Question answering
- Text summarization
- Code generation
- Image processing (Vision Transformers)
- Multi-modal tasks (text + image)