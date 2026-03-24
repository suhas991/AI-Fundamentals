# Machine Learning (ML): Types and Categories

## What Is ML?

Machine Learning (ML) is a subset of AI where models learn patterns from data and then make predictions or decisions, rather than relying only on manually written rules.

## Core Learning Paradigms

### 1. Supervised Learning

Learns from labeled examples, where each input has a known target output.

**Main task types**:

- **Classification**: Predict a class label
  - Binary (spam vs not spam)
  - Multi-class (cat/dog/bird)
  - Multi-label (multiple tags per item)
- **Regression**: Predict a continuous value
  - House price, demand forecast, risk score

**Common algorithms**:

- Linear Regression, Logistic Regression
- Decision Trees, Random Forest
- Support Vector Machines (SVM)
- k-Nearest Neighbors (k-NN)
- Gradient Boosting (XGBoost, LightGBM, CatBoost)

### 2. Unsupervised Learning

Learns structure from unlabeled data.

**Main task types**:

- **Clustering**: Group similar examples
  - Customer segmentation, behavioral cohorts
- **Dimensionality Reduction**: Compress features while preserving key structure
  - PCA, UMAP, t-SNE (often for visualization)
- **Association Discovery**: Find co-occurrence patterns
  - Market basket analysis

**Common algorithms**:

- k-Means, DBSCAN, Hierarchical Clustering
- Gaussian Mixture Models
- Apriori / FP-Growth
- Autoencoders (representation learning)

### 3. Reinforcement Learning (RL)

An agent learns by interacting with an environment to maximize cumulative reward.

**Method families**:

- **Value-based**: Q-Learning, DQN
- **Policy-based**: REINFORCE, policy gradients
- **Actor-critic**: A2C/A3C, PPO, SAC

**Typical use cases**:

- Game playing
- Robotics and control
- Dynamic resource allocation

## Additional ML Settings

### Semi-Supervised Learning

- Uses a small labeled set with a larger unlabeled set
- Useful when annotation cost is high

### Self-Supervised Learning

- Creates proxy labels from raw data
- Widely used for pretraining foundation models

### Transfer Learning

- Reuses knowledge from a pretrained model on a new task
- Commonly applied through fine-tuning

### Ensemble Learning

- Combines multiple models for better robustness and accuracy
- Examples: bagging, boosting, stacking

## ML by Problem Formulation

| Problem Type | Goal | Example |
|--------------|------|---------|
| Classification | Predict category labels | Fraud detection |
| Regression | Predict numeric values | Demand forecasting |
| Clustering | Discover natural groups | Customer segmentation |
| Ranking | Order results by relevance | Search and recommendation |

## Key Takeaways

1. Supervised, unsupervised, and reinforcement learning are the three core paradigms.
2. Problem formulation (classification/regression/clustering/ranking) drives model choice.
3. Modern production ML often combines pretraining, transfer learning, and ensembles.