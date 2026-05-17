# How Transformers and Attention Work

---

## Why Transformers Matter

Transformers replaced RNN-style sequence processing with parallel processing and attention, enabling large-scale training and better long-range context handling.

> Interview one-liner: A transformer learns which tokens matter to each other by computing attention scores, then combines information accordingly.

---

## Core Building Blocks

1. Token embeddings: convert tokens into vectors.
2. Positional encoding: inject order information.
3. Multi-head self-attention: each token attends to all tokens.
4. Feed-forward network (FFN): non-linear transformation per token.
5. Residual connections + layer normalization: improve stability and training depth.

---

## Self-Attention Intuition

Each token creates three vectors:
- Query (Q): what this token is looking for
- Key (K): what this token offers
- Value (V): information this token carries

Attention score is based on query-key similarity.

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Interpretation:
- $QK^T$ gives relevance scores.
- Division by $\sqrt{d_k}$ prevents extreme values.
- softmax turns scores into probabilities.
- Weighted sum of $V$ gives context-aware representation.

---

## Multi-Head Attention

Instead of one attention map, transformer uses several heads in parallel.

Why it helps:
- Different heads learn different relationships (syntax, coreference, topic).
- Improves representational power.

```
Token states
  -> Head 1 attention
  -> Head 2 attention
  -> ...
  -> Head h attention
  -> Concatenate + projection
```

---

## Encoder vs Decoder (Interview Context)

| Component | Typical Use | Masking |
|---|---|---|
| Encoder | Understanding/input representation (BERT-like) | No causal mask |
| Decoder | Text generation (GPT-like) | Causal mask (can only see past tokens) |

Most modern chat LLMs are decoder-only transformers.

---

## Common Interview Questions

### Why is attention better than RNN recurrence for long context?
- Attention allows direct token-to-token paths.
- RNNs compress history through sequential hidden state, which weakens long-range dependencies.

### Why scale by $\sqrt{d_k}$?
- Dot products grow with dimension; scaling keeps gradients stable and softmax less saturated.

### What does causal masking do?
- Prevents the model from seeing future tokens during generation training/inference.

### What are transformer weaknesses?
- Attention cost grows roughly as $O(n^2)$ with sequence length.
- High memory usage at long context windows.

---

## Key Takeaways

- Transformer = attention + FFN blocks stacked deeply.
- Self-attention computes context-aware token representations.
- Multi-head attention captures diverse relationships.
- Decoder-only transformers power modern LLMs.
- Long context is powerful but expensive.
