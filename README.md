# 🧠 Self-Pruning Neural Network for CIFAR-10

A PyTorch implementation of a self-pruning Multi-Layer Perceptron (MLP) that learns to remove unnecessary weights during training using **learnable gate scores** and **L1 sparsity regularization**. Training is tracked with **MLflow**.

---

## 📌 Overview

This project explores structured neural network pruning by introducing a novel `PrunableLinear` layer. Instead of post-training pruning, the network learns *which weights to keep* as part of the training objective itself — achieving a balance between classification accuracy and model sparsity.

The model is trained and evaluated on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset across multiple sparsity-controlling lambda values.

---

## 🏗️ Architecture

### `PrunableLinear` Layer
Each linear layer is augmented with a learnable `gate_scores` tensor of the same shape as the weight matrix. During the forward pass:

1. Gate scores are passed through a **Sigmoid** function → values in `[0, 1]`
2. Gates are multiplied element-wise with the weights (**soft masking**)
3. A standard linear transformation is applied: `y = x(W ⊙ G)ᵀ + b`

Gates initialized near `0` (Sigmoid(0) = 0.5), meaning the network starts with roughly half the weights active.

### `SelfPruningNet` (MLP)
```
Input (3072) → PrunableLinear(3072→512) → ReLU
             → PrunableLinear(512→256)  → ReLU
             → PrunableLinear(256→10)
```

---

## 📉 Loss Function

```
Total Loss = CrossEntropyLoss + λ × Σ|gates|
```

- **CrossEntropyLoss** — standard classification objective
- **L1 Sparsity Loss** — sum of absolute gate values, encouraging gates to push toward 0 (pruned)
- **λ (lambda)** — controls the sparsity-accuracy trade-off

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision mlflow matplotlib numpy
```

### Run Training

```bash
python main.py
```

This trains the model for **3 different lambda values** (`1e-6`, `1e-5`, `1e-4`) over **20 epochs** each. CIFAR-10 data is downloaded automatically on first run.

---

## 📊 Experiment Tracking (MLflow)

All runs are logged with MLflow. To view the dashboard:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

**Logged per run:**
| Parameter / Metric | Description |
|---|---|
| `lambda` | Sparsity regularization strength |
| `epochs` | Number of training epochs |
| `train_loss` | Loss per epoch |
| `accuracy` | Final test accuracy (%) |
| `sparsity` | % of weights with gate < 0.01 |
| `model` | Full PyTorch model artifact |

---

## 📈 Results

After training, a summary table is printed:

```
Lambda     | Accuracy (%)    | Sparsity (%)   
---------------------------------------------
1e-06      | ~53.xx          | ~low
1e-05      | ~52.xx          | ~medium
1e-04      | ~xx.xx          | ~high
```

A gate value distribution histogram is also generated for the `λ=1e-5` run, showing how many weights were effectively pruned.

> Actual values depend on hardware and random seed.

---

## 📁 Project Structure

```
.
├── main.py          # Full training pipeline
├── data/            # CIFAR-10 dataset (auto-downloaded)
└── mlruns/          # MLflow experiment logs (auto-generated)
```

---

## ⚙️ Configuration

Key hyperparameters can be adjusted in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `lambdas` | `[1e-6, 1e-5, 1e-4]` | Sparsity regularization values to sweep |
| `epochs` | `20` | Training epochs per run |
| `batch_size` | `64` | DataLoader batch size |
| `lr` | `1e-3` | Adam optimizer learning rate |
| `threshold` | `1e-2` | Gate value below which a weight is "pruned" |

---

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- mlflow
- matplotlib
- numpy
- CUDA (optional, falls back to CPU automatically)

---

## 📚 Key Concepts

- **Soft Pruning** — gates are continuous during training; hard pruning can be applied post-training using the threshold
- **L1 Regularization** — promotes sparsity by penalizing non-zero gate values
- **Lambda Sweep** — higher lambda → more sparsity, potentially lower accuracy

---

## 📄 License

MIT License. Feel free to use and adapt.
