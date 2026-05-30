# AI Foundations — Study Notes
> For SDE-1 / AI Engineer Interview Prep | Suhas N H

---

## 1. Types of AI

### Narrow AI (Weak AI) — What exists today
AI designed to do **one specific task** extremely well.

- Cannot generalize outside its trained domain
- Examples:
  - ChatGPT → text generation
  - AlphaGo → playing Go
  - Face ID → facial recognition
  - Spam filters → email classification
  - YouTube recommendation → content ranking

> **Interview line:** *"All AI we use today is Narrow AI — it's highly optimized for one domain but completely helpless outside it."*

---

### General AI (AGI — Artificial General Intelligence) — Does not exist yet
AI that can **reason, learn, and perform any intellectual task** a human can.

- Can transfer knowledge across domains
- Self-directed learning
- No agreed-upon timeline — estimated 10–50 years away
- Debated heavily in research (OpenAI, DeepMind actively pursuing)

> **Interview line:** *"AGI is the north star of AI research — a system that can write code, diagnose disease, and play chess using the same underlying intelligence, like a human does."*

---

### Super AI (ASI — Artificial Super Intelligence) — Hypothetical
AI that **surpasses human intelligence** in every domain — creativity, problem-solving, emotional intelligence.

- Beyond AGI by definition
- Philosophical and existential debate territory
- Figures like Elon Musk, Geoffrey Hinton raise safety concerns around this

> **Interview line:** *"ASI is more philosophy than engineering today — it's the 'what happens after AGI' question."*

---

### Quick Comparison

| Type | Exists? | Example | Scope |
|------|---------|---------|-------|
| Narrow AI | ✅ Yes | ChatGPT, Alexa | Single task |
| General AI | ❌ No | Hypothetical | Any task |
| Super AI | ❌ No | Sci-fi | Beyond human |

---

---

## 2. AI vs Machine Learning vs Deep Learning

Think of it as **nested circles** — each is a subset of the one before it.

```
┌─────────────────────────────────────┐
│              AI                     │
│  (any technique making machines     │
│   simulate human intelligence)      │
│                                     │
│   ┌─────────────────────────────┐   │
│   │     Machine Learning        │   │
│   │  (learns from data,         │   │
│   │   no explicit programming)  │   │
│   │                             │   │
│   │   ┌─────────────────────┐   │   │
│   │   │    Deep Learning    │   │   │
│   │   │  (neural networks   │   │   │
│   │   │  with many layers)  │   │   │
│   │   └─────────────────────┘   │   │
│   └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### AI (Artificial Intelligence)
- Umbrella term for any system that mimics human intelligence
- Includes rule-based systems (if-else logic), expert systems, ML, DL
- Example: A chess engine using pre-programmed rules = AI, but not ML

### Machine Learning
- Subset of AI where the system **learns patterns from data** instead of being explicitly programmed
- You give it data → it figures out the rules
- Example: Show it 10,000 spam emails → it learns what spam looks like

### Deep Learning
- Subset of ML using **multi-layered neural networks**
- Excels at unstructured data — images, audio, text
- Requires large datasets and compute (GPUs)
- Example: GPT models, image recognition, speech-to-text

### Key Differences

| | AI | Machine Learning | Deep Learning |
|---|---|---|---|
| Data needed | Low (can use rules) | Medium | High (millions) |
| Compute | Low | Medium | High (GPU) |
| Interpretability | High | Medium | Low (black box) |
| Best for | Logic, rules | Structured data | Images, text, audio |

---

---

## 3. Types of Machine Learning

### Supervised Learning
The model learns from **labeled data** — input + correct answer pairs.

- You tell it: "This email is spam", "This image is a cat"
- It learns to map inputs → outputs
- Goal: Predict output for new, unseen inputs

**Algorithms:** Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM, Neural Networks

**Use cases:**
- Email spam detection
- House price prediction
- Image classification
- Sentiment analysis

```
Training data:          Model learns:
[Image: Cat] → "Cat"
[Image: Dog] → "Dog"    →   Pattern: Cat vs Dog
[Image: Cat] → "Cat"

New image → Model predicts: "Cat" ✅
```

---

### Unsupervised Learning
The model finds **hidden patterns in unlabeled data** — no correct answers given.

- You give it raw data with no labels
- It discovers structure, clusters, or relationships on its own
- Goal: Understand the structure of data

**Algorithms:** K-Means Clustering, DBSCAN, PCA (dimensionality reduction), Autoencoders

**Use cases:**
- Customer segmentation
- Anomaly detection
- Topic modeling
- Recommendation systems

```
Unlabeled customer data →  Model discovers:
                           Group A: High spenders
                           Group B: Occasional buyers
                           Group C: Window shoppers
```

---

### Reinforcement Learning
The model learns by **trial and error through interaction with an environment**, receiving rewards or penalties.

- Agent takes actions → environment gives feedback (reward/penalty)
- Goal: Maximize cumulative reward over time
- Learns the best strategy (policy) through experience

**Key concepts:**
- **Agent** — the model making decisions
- **Environment** — what the agent interacts with
- **Action** — what the agent does
- **Reward** — feedback signal (+ve or -ve)
- **Policy** — strategy the agent learns

**Use cases:**
- Game playing (AlphaGo, OpenAI Five)
- Robot locomotion
- RLHF — fine-tuning LLMs with human feedback (very relevant for AI engineers!)
- Self-driving cars
- Trading strategies

```
Agent (AI) plays a game:
  Move left → loses life → penalty (-1)
  Move right → gets coin → reward (+1)
  Over thousands of games → learns optimal strategy
```

---

### Quick Comparison

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| Labels needed | Yes | No | No (uses rewards) |
| Goal | Predict | Discover | Optimize |
| Feedback | Direct | None | Delayed reward |
| Example | Spam filter | Customer clusters | Game AI |

---

---

## 4. Neural Networks — Basics

### What is a Neural Network?
A system loosely inspired by the human brain — made of interconnected **nodes (neurons)** organized in **layers** that transform data to make predictions.

### Structure

```
INPUT LAYER      HIDDEN LAYERS       OUTPUT LAYER
(raw data)       (feature learning)  (prediction)

  x1  ──┐
        ├──→ [neuron] ──→ [neuron] ──→  "Cat" (0.92)
  x2  ──┤                              "Dog" (0.06)
        ├──→ [neuron] ──→ [neuron] ──→  "Bird"(0.02)
  x3  ──┘
```

### Key Components

**Neuron (Node)**
- Receives inputs, applies a weight, sums them up, passes through an activation function
- Formula: `output = activation(Σ weight × input + bias)`

**Weights**
- Numbers that determine how much each input matters
- Learned during training
- Initially random, refined via backpropagation

**Bias**
- Extra parameter that shifts the output
- Helps the model fit even when all inputs are zero

**Activation Function**
- Adds non-linearity — without it, the network is just linear math
- Common ones:
  - **ReLU** — max(0, x) — most common in hidden layers
  - **Sigmoid** — squashes to 0–1, used in binary classification
  - **Softmax** — squashes to probabilities that sum to 1, used in output layer for multi-class

**Layers**
- **Input layer** — raw data (pixels, tokens, features)
- **Hidden layers** — feature extraction and transformation
- **Output layer** — final prediction

### How it Learns — Backpropagation

```
1. Forward pass  → data flows input → output, prediction made
2. Loss calculated → how wrong was the prediction? (e.g., Cross-Entropy, MSE)
3. Backward pass → error propagated back through network
4. Weights updated → using gradient descent to reduce error
5. Repeat → thousands of times (epochs) until loss is minimized
```

**Gradient Descent:** The optimizer that adjusts weights in the direction that reduces loss.
- **Learning rate** — how big each step is. Too big = overshoots. Too small = slow.
- **Adam optimizer** — most commonly used today (adaptive learning rate)

### Deep Neural Network vs Shallow
- **Shallow** — 1 hidden layer → simple patterns
- **Deep** — many hidden layers → complex hierarchical features
  - Early layers: edges, shapes
  - Later layers: faces, objects, concepts

---

---

## 5. Model Training, Inference, and Fine-tuning

### Model Training
The process of **teaching a model by exposing it to data** and adjusting its weights to minimize error.

```
Dataset → Split → Train/Val/Test
              ↓
         Forward Pass
              ↓
         Calculate Loss
              ↓
         Backpropagation
              ↓
         Update Weights     ← repeat for N epochs
              ↓
         Evaluate on Val Set
              ↓
         Final Test on Test Set
```

**Key concepts:**
- **Epoch** — one full pass through the training dataset
- **Batch size** — how many samples processed before weights update
- **Loss function** — measures how wrong the model is (MSE for regression, Cross-Entropy for classification)
- **Overfitting** — model memorizes training data, fails on new data → fix with dropout, regularization, more data
- **Underfitting** — model too simple, can't capture patterns → fix with more complexity
- **Train/Val/Test split** — e.g., 70/15/15 — train on train, tune on val, final eval on test

---

### Inference
Using a **trained model to make predictions on new data**. No learning happens — weights are frozen.

```
New input → Trained Model (frozen weights) → Prediction
```

**Key concerns at inference time:**
- **Latency** — how fast is the response? (critical for real-time APIs)
- **Throughput** — how many requests per second?
- **Cost** — GPU inference is expensive, optimize with batching, quantization
- **Context window** — for LLMs, how many tokens fit in one call

**Inference optimization techniques:**
- **Quantization** — reduce weight precision (FP32 → INT8) → faster, smaller
- **Caching** — cache repeated computations (KV cache in Transformers)
- **Batching** — process multiple requests together

> **Interview line:** *"Training is a one-time expensive process. Inference is what runs in production — and that's where latency, cost, and scalability matter."*

---

### Fine-tuning
**Starting from a pre-trained model** and continuing training on a smaller, task-specific dataset to adapt it to your use case.

```
Pre-trained Model           Fine-tuned Model
(trained on internet)  →   (specialized for your task)
e.g., GPT-4 base           e.g., Customer support bot
                           trained on your FAQ data
```

**Why fine-tune instead of training from scratch?**
- Training from scratch needs massive data + compute (millions of $)
- Fine-tuning needs far less data and compute
- Pre-trained models already understand language/images — you just redirect them

**Types of fine-tuning:**

| Type | What's updated | When to use |
|------|---------------|-------------|
| Full fine-tuning | All weights | Small model, lots of data |
| LoRA / QLoRA | Small adapter layers only | Large model, limited data/compute |
| Prompt tuning | Just the prompt embeddings | Minimal compute |

**Fine-tune vs RAG — when to use which:**

| Scenario | Use |
|----------|-----|
| Model needs to know new facts | RAG |
| Model needs a different writing style/tone | Fine-tuning |
| Confidential data you can't send to API | Fine-tuning |
| Dynamic, frequently updated knowledge | RAG |
| Cost-sensitive, quick to set up | RAG |

---

### The Full Lifecycle — Summary

```
1. DATA COLLECTION
   Gather labeled/unlabeled data relevant to your task

2. PRE-PROCESSING
   Clean, tokenize, normalize, split into train/val/test

3. MODEL SELECTION
   Choose architecture (CNN, Transformer, etc.)

4. TRAINING
   Forward pass → loss → backprop → weight update → repeat

5. EVALUATION
   Measure on validation set, tune hyperparameters

6. FINE-TUNING (optional)
   Adapt pre-trained model to your specific task

7. INFERENCE / DEPLOYMENT
   Serve the model via API, optimize for latency

8. MONITORING
   Track accuracy, drift, latency, cost in production
```

---

## Quick Revision Cheatsheet

| Concept | One-liner |
|---------|-----------|
| Narrow AI | One task, very well |
| AGI | Human-level, all tasks — doesn't exist yet |
| AI ⊃ ML ⊃ DL | Nested subsets |
| Supervised | Learns from labeled data |
| Unsupervised | Finds hidden patterns, no labels |
| Reinforcement | Learns via rewards and penalties |
| Neuron | Takes inputs, applies weights, outputs signal |
| Backpropagation | How the network learns — propagates error backward |
| Training | Teach the model by minimizing loss |
| Inference | Use the trained model to predict — weights frozen |
| Fine-tuning | Adapt pre-trained model to your task with less data |
| LoRA | Efficient fine-tuning — only trains small adapter layers |
| Overfitting | Memorizes training data, fails on new data |
| Gradient Descent | Optimization algorithm that minimizes loss |
