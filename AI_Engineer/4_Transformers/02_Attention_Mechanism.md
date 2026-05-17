# Attention Mechanism

## What is Attention?

Attention is a mechanism that allows the model to focus on relevant parts of the input when producing each part of the output. It's inspired by human visual attention.

## Core Intuition

When reading "The animal didn't cross the street because **it** was too tired", attention helps the model understand that "**it**" refers to "**animal**", not "**street**".

## Scaled Dot-Product Attention

### The Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Components

**Query (Q):** What I'm looking for
- Represents the current token's "question"
- "What information do I need?"

**Key (K):** What's available
- Represents each token's "label"
- "What information can I provide?"

**Value (V):** The actual information
- Represents the content to be retrieved
- "Here's the information"

**d_k:** Dimension of keys (scaling factor)
- Prevents softmax saturation
- Keeps gradients stable

### Step-by-Step

1. **Compute similarity scores**
   ```
   scores = QK^T
   ```
   - Matrix multiplication of queries and keys
   - Each position (i,j) = similarity between query i and key j

2. **Scale scores**
   ```
   scaled_scores = scores / √d_k
   ```
   - Divides by square root of key dimension
   - Prevents extremely large values that saturate softmax

3. **Apply softmax**
   ```
   attention_weights = softmax(scaled_scores)
   ```
   - Converts scores to probabilities (sum to 1)
   - Higher scores get more weight

4. **Weight values**
   ```
   output = attention_weights × V
   ```
   - Each value is weighted by its attention score
   - Produces context-aware representation

### Example

**Sentence:** "The cat sat on the mat"

**Query for "sat":**
- High attention to "cat" (subject)
- High attention to "mat" (object)
- Low attention to "the" (determiner)

## Multi-Head Attention

### Why Multiple Heads?

Different heads can learn different types of relationships:
- Head 1: Subject-verb relationships
- Head 2: Adjective-noun relationships
- Head 3: Long-range dependencies
- Head 4: Syntactic patterns

### Architecture

```
Input → Linear → [Head 1, Head 2, ..., Head h] → Concat → Linear → Output
```

### Process

1. **Project inputs**
   - Each head has its own Q, K, V projections
   - Smaller dimension per head (d_k = d_model / h)

2. **Parallel attention**
   - Each head computes attention independently
   - Different heads learn different patterns

3. **Concatenate**
   - Combine outputs from all heads
   - `Concat(head_1, head_2, ..., head_h)`

4. **Final projection**
   - Linear layer to combine head outputs
   - Projects back to d_model dimension

### Typical Configuration

| Model | Heads | d_model | d_k (per head) |
|-------|-------|---------|----------------|
| BERT-Base | 12 | 768 | 64 |
| GPT-2 | 25 | 1600 | 64 |
| GPT-3 | 96 | 12288 | 128 |

## Self-Attention vs Cross-Attention

### Self-Attention
- Q, K, V all come from the same sequence
- Token attends to other tokens in the same sequence
- Used in encoder and decoder

### Cross-Attention
- Q from decoder, K and V from encoder
- Decoder attends to encoder outputs
- Used in decoder of encoder-decoder models

## Masked Attention

### Purpose

Prevent decoder from seeing future tokens during training (autoregressive property).

### Implementation

```
masked_scores = scores + mask
attention_weights = softmax(masked_scores)
```

**Mask values:**
- `-∞` for positions to mask
- `0` for allowed positions

**Result:**
- Future tokens get zero attention weight
- Model can only attend to past tokens

## Attention Patterns

### What Heads Learn

**Positional heads:**
- Attend to specific relative positions
- "i-1", "i+2" patterns

**Syntactic heads:**
- Subject-verb agreement
- Noun-adjective modification

**Semantic heads:**
- Coreference resolution
- Entity relationships

**Rare word heads:**
- Attend to rare words
- Help with vocabulary coverage

## Visualizing Attention

### Heatmap Example

For "The quick brown fox":

```
        The   quick  brown  fox
The     [0.9   0.05   0.03   0.02]
quick   [0.1   0.8    0.07   0.03]
brown   [0.05  0.1    0.75   0.1]
fox     [0.02  0.03   0.1    0.85]
```

Diagonal shows self-attention, off-diagonal shows cross-token attention.

## Computational Complexity

### Standard Attention
- Time: O(n² × d)
- Space: O(n²)
- n = sequence length, d = model dimension

### Bottleneck
- Quadratic in sequence length
- Limits maximum sequence length
- Memory intensive for long sequences

### Efficient Variants
- **Sparse attention** - Only attend to subset of positions
- **Linear attention** - Approximate with linear complexity
- **Flash attention** - Optimized memory access patterns

## Key Insights

1. **Attention is learnable** - The model learns what to attend to
2. **Parallel computation** - All attention scores computed simultaneously
3. **Interpretability** - Attention weights show model's focus
4. **Flexibility** - Can attend to any position regardless of distance
5. **Compositionality** - Multiple heads compose complex patterns

## Why "Attention Is All You Need"

Before Transformers:
- RNNs: Sequential processing, limited long-range memory
- CNNs: Local patterns, limited receptive field

With Attention:
- Direct connections between any positions
- No sequential bottleneck
- Learns complex relationships automatically
- Scales to massive models and datasets

This simple mechanism replaced entire architectures and enabled the LLM revolution.