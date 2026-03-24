# Deep Learning (DL): Types and Categories

## What Is Deep Learning?

Deep Learning (DL) is a subset of machine learning that uses multi-layer neural networks to learn hierarchical representations directly from data.

## Major Architecture Families

### 1. Feedforward Networks (MLP / Dense Networks)

- Data flows from input to output without recurrent loops
- Best suited for tabular problems and baseline modeling
- Common uses: classification and regression on structured data

### 2. Convolutional Neural Networks (CNN)

- Designed for grid-like signals (images, spectrograms, video frames)
- Core operations: convolution, nonlinearity, pooling
- Common uses: image classification, detection, segmentation, medical imaging

### 3. Recurrent Sequence Models (RNN, LSTM, GRU)

- Process ordered sequences with state over time
- Useful for time series and sequence tasks
- Many NLP tasks have moved from RNNs to Transformers, but RNNs remain useful in some low-latency settings

### 4. Transformers

- Built on self-attention for parallel sequence processing
- Handles long-range dependencies effectively
- Variants:
  - Encoder-only (BERT-like)
  - Decoder-only (GPT-like)
  - Encoder-decoder (T5/BART-like)
- Used in language, vision, speech, and multimodal systems

### 5. Generative Deep Models

- **Autoencoders / VAEs**: latent representation learning and generation
- **GANs**: adversarial training for realistic synthetic outputs
- **Diffusion models**: iterative denoising for high-fidelity generation

### 6. Graph Neural Networks (GNN)

- Operate on graph-structured data
- Variants: GCN, GraphSAGE, GAT
- Use cases: recommendation, molecular property prediction, fraud and network analysis

### 7. Attention Mechanisms

- Components that weight relevant information dynamically
- Types: self-attention, cross-attention, multi-head attention
- Fundamental to modern LLM and multimodal architectures

## DL by Application Domain

| Domain | Common Architectures | Example Use Cases |
|--------|----------------------|-------------------|
| Computer Vision | CNN, Vision Transformer (ViT) | Classification, detection, segmentation |
| NLP | Transformer | Translation, summarization, QA |
| Speech | CNN + Transformer, conformer-like models | ASR, TTS |
| Time Series | RNN, Transformer | Forecasting, anomaly detection |
| Multimodal | CLIP-like, vision-language models | Captioning, VQA, retrieval |

## Training Paradigms in DL

- **Supervised learning**: learns from labeled data
- **Self-supervised learning**: learns representations from unlabeled data
- **Unsupervised learning**: structure discovery and generation
- **Transfer learning**: starts from pretrained weights and adapts to a target task

## Key Takeaways

1. DL is defined by representation learning through deep neural architectures.
2. Transformers are the dominant architecture in many modern domains.
3. Architecture choice should follow data type and task constraints.