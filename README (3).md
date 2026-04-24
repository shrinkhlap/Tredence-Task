# 🧠 Self-Pruning Neural Network

A neural network that **learns to prune itself** during training using learnable sigmoid gates and L1 regularization — trained on CIFAR-10 and tracked with MLflow.

---

## 📌 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Mathematical Formulation](#mathematical-formulation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Visualizations](#visualizations)
- [Training Details](#training-details)
- [MLflow Integration](#mlflow-integration)
- [Running the Code](#running-the-code)

---

## Overview

Traditional pruning requires a two-stage pipeline: train → prune → fine-tune. This project collapses all three into one.

Each linear layer is replaced with a **`PrunableLinear`** module containing learnable gate scores. During forward pass, weights are masked by sigmoid-activated gates. An L1 penalty on the gate activations forces the network to **zero out irrelevant connections automatically** — pruning emerges as a natural consequence of gradient descent.

Experiments are tracked across different sparsity strengths (λ) using **MLflow**.

---

## How It Works

### Why L1 on Sigmoid Gates Encourages Sparsity

An L1 penalty on sigmoid gates encourages sparsity because it directly penalizes each gate's activation magnitude, pushing many toward zero. Since sigmoid outputs lie in `[0, 1]`:

- Keeping a gate **ON** (→ 1) incurs a **higher penalty**
- Keeping a gate **OFF** (→ 0) incurs a **lower penalty**

Unlike L2 regularization (which shrinks smoothly), **L1 often drives values exactly to zero**, effectively removing the corresponding weight from the network.

> The model learns to keep only the small subset of gates that are essential — everything else gets pruned away.

---

## Mathematical Formulation

Effective weights are computed by element-wise gating of the raw weight matrix:

$$W_{\text{eff}} = W \odot \sigma(g)$$

Where:
- $g$ — learnable gate score
- $\sigma(g) \in [0, 1]$ — sigmoid gate value
- $\odot$ — element-wise multiplication

The training objective combines classification accuracy with a sparsity penalty:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \sum_{i=1}^{N} |\sigma(g_i)|$$

Where:
- $\mathcal{L}_{\text{CE}}$ — Cross-Entropy classification loss
- $\lambda$ — sparsity regularization coefficient (controls pruning aggressiveness)
- $\sum |\sigma(g)|$ — L1 penalty on all gate activations

### Intuition

| Condition | Effect |
|-----------|--------|
| Gate → 0 | Weight effectively removed (pruned) |
| Gate → 1 | Weight retained (important feature) |
| Higher λ | More aggressive pruning |
| Lower λ | Near-baseline behavior |

---

## Model Architecture

**Input:** CIFAR-10 images (32×32×3, flattened to 3072 features)

```
Input (3072)
    │
    ▼
[PrunableLinear] FC1: 3072 → 512  +  Sigmoid Gates  →  ReLU
    │
    ▼
[PrunableLinear] FC2: 512 → 256   +  Sigmoid Gates  →  ReLU
    │
    ▼
[PrunableLinear] FC3: 256 → 10    +  Sigmoid Gates
    │
    ▼
Output (10 classes)
```

Each `PrunableLinear` layer contains:
- Standard learnable **weights**
- Learnable **gate scores** (same shape as weights)
- **Sigmoid gating** applied element-wise during forward pass

---

## Results

Experiments were run at three λ values over 20 epochs. Higher λ leads to more aggressive sparsity.

| λ (Lambda) | Test Accuracy (%) | Sparsity (%) |
|:----------:|:-----------------:|:------------:|
| `1e-6`     | 54.49             | 0.52         |
| `1e-5`     | 55.52             | 20.93        |
| `1e-4`     | **56.83**         | **81.30**    |

### Key Findings

- **Higher sparsity → better accuracy**: Pruning 80% of weights at λ=1e-4 *improved* accuracy over the near-dense baseline. This is because:
  - Pruning forces the remaining 20% of weights to become **independent and robust**
  - Neurons can no longer rely on each other to compensate for poor features (reducing **co-adaptation**)
  - The result is a more **generalized internal representation** of the data

- **Epoch scheduling matters**: In shorter (10-epoch) runs, the L1 penalty prunes weights before they've had a chance to learn useful features. With 20 epochs:
  - **Epochs 1–10**: Network identifies important features
  - **Epochs 11–20**: L1 penalty cleans up non-contributing weights — the network settles into its pruned architecture

---

## Visualizations

### Gate Value Distribution
Shows that the network cleanly separates weights into two populations — pruned (near 0) and retained (away from 0).

- **Large spike near 0** → pruned, irrelevant connections
- **Cluster away from 0** → important, retained weights

### Sparsity vs Accuracy vs Training Loss
Demonstrates that increasing λ simultaneously increases sparsity and (counterintuitively) improves test accuracy — confirming that pruning acts as implicit regularization.

### Box Plot — Gate Distribution per λ
Visualizes the spread and quartiles of gate values across the three experimental conditions.

### Scatter Plot — Weight Importance
Each point represents an individual gate. The clustering behavior confirms the bimodal gate distribution predicted by theory.

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | `1e-3` |
| Epochs | 20 |
| Classification Loss | Cross-Entropy |
| Sparsity Loss | L1 on sigmoid gates |
| Dataset | CIFAR-10 |
| λ values tested | `1e-6`, `1e-5`, `1e-4` |

---

## MLflow Integration

All experiments are tracked with **MLflow** for reproducibility and comparison.

### Logged Parameters
- `lambda` — sparsity regularization coefficient
- `epochs` — number of training epochs

### Logged Metrics (per epoch)
- `train_loss` — total composite loss
- `accuracy` — test set accuracy
- `sparsity` — fraction of gates below threshold

### Logged Artifacts
- Trained model checkpoint (`.pt`)

### Launch MLflow UI

```bash
mlflow ui --port 5000
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Running the Code

### Install Dependencies

```bash
pip install torch torchvision mlflow matplotlib
```

### Run Training

```bash
python main.py
```

Training will run across all λ values and log results to MLflow automatically.

---

## Expected Gate Behavior

After training, inspecting the gate value histogram should show:

```
Frequency
    │
    █                         ← spike near 0 (pruned weights)
    █
    █     
    █                    █    ← cluster away from 0 (retained)
    █                    █
    └────────────────────────→ Gate Value (0 to 1)
```

This bimodal distribution is the hallmark of a well-trained self-pruning network.

---

## Project Structure

```
.
├── main.py           ← Entry point: training loop + MLflow logging
├── model.py          ← PrunableLinear module + network definition
├── train.py          ← Training and evaluation logic
├── plots.py          ← Visualization utilities
└── README.md
```

---

## License

MIT
