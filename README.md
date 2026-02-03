# AIether: Procedural Growth of Neural Networks via Adaptive Geometric Extrapolation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AIether** is a procedural growth system for deep neural networks that uses adaptive geometric extrapolation to intelligently expand model architectures during training. By analyzing the optimization trajectory's geometric properties, AIether can detect stagnation and initialize new layers in informed regions of parameter space.

---

## 🎯 Key Features

- **Geometric Trajectory Analysis**: Quantifies optimization dynamics using metrics like Temporal Stretch Factor (τ) and Effective Curvature (κ)
- **Adaptive Initialization**: Uses spectral decomposition (SVD) to identify productive directions and construct informed layer initializations
- **Stagnation Detection**: Automatically detects when training plateaus and triggers architectural expansion
- **Tensor-wise Extrapolation**: Handles 2D matrices and 1D vectors uniformly with multi-scale adaptive coefficients
- **Seamless Integration**: Compatible with Hugging Face Trainer and standard PyTorch training loops

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Examples](#examples)
- [Hyperparameters](#hyperparameters)
- [Mathematical Foundation](#mathematical-foundation)
- [Citation](#citation)

---

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- numpy
- datasets
- wandb (optional, for experiment tracking)

### Install from source

```bash
git clone https://github.com/Rochafurada/AIether.git
cd AIether/aiether
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### Basic Training Example

```python
from aiether.training import train_with_hf_trainer

# Train a GPT model with procedural growth
if __name__ == "__main__":
    train_with_hf_trainer()
```

### Command Line Usage

```bash
python scripts/train.py \
    --n_layer 6 \
    --n_head 6 \
    --n_embd 384 \
    --num_epochs 3 \
    --batch_size 8 \
    --lr 5e-4 \
    --dataset_path /path/to/dataset \
    --beta0 0.1 \
    --gamma0 0.15 \
    --eta0 0.03 \
    --k_subspace 10 \
    --ell_orth 5 \
    --pg_patience 3 \
    --pg_threshold 0.15
```

---

## 🧠 Core Concepts

### Geometric Metrics

AIether analyzes the optimization trajectory using three key metrics:

1. **Temporal Stretch Factor (τ)**: Measures trajectory efficiency
   - `τ ≈ 1`: Efficient, nearly rectilinear movement
   - `τ > 2.5`: Strong evidence of stagnation

2. **Effective Curvature (κ)**: Quantifies local tortuosity
   - `κ ≈ 0`: Smooth trajectory
   - `κ > 1`: Excessive lateral wandering

3. **Spectral Analysis**: Identifies productive directions via SVD decomposition

### Adaptive Extrapolation

New layer initialization combines:

```
W_new = W_last + β(τ)·V̂_U + γ(κ)·Â_U + η(τ,κ)·v_escape
```

Where:
- `β(τ)`: Linear term adapted to trajectory efficiency
- `γ(κ)`: Curvature correction adapted to smoothness
- `η(τ,κ)`: Exploration term activated during stagnation

---

## 🏗️ Architecture

```
AIether_Refactored/
├── aiether/
│   ├── callbacks/          # Training callbacks
│   │   ├── procedural_growth_callback.py
│   │   └── debug_callback.py
│   ├── managers/           # State management
│   │   ├── growth_manager.py
│   │   └── layer_state_manager.py
│   ├── models/             # Model implementations
│   │   └── gpt.py
│   ├── training/           # Training orchestration
│   │   └── trainer.py
│   └── utils/              # Core utilities
│       ├── geometry.py     # Geometric extrapolation
│       ├── config.py       # Configuration parsing
│       └── logging.py      # Custom logging
├── scripts/
│   └── train.py           # Training entry point
└── README.md
```

---

## ⚙️ Configuration

### Model Configuration

```python
config = {
    'model': {
        'n_layer': 6,        # Initial number of layers
        'n_head': 6,         # Attention heads
        'n_embd': 384,       # Embedding dimension
        'dropout': 0.1,
        'bias': True
    }
}
```

### Procedural Growth Configuration

```python
config = {
    'Procedural_growth': {
        'patience': 3,                    # Stagnation patience
        'threshold': 0.15,                # Improvement threshold
        'warmup_steps': 400,              # Steps before first check
        'history_window_size': 15,        # Trajectory history size
        'collect_interval_steps': 100     # Snapshot collection interval
    }
}
```

### Geometric Extrapolation Parameters

```python
config = {
    'experiment': {
        'beta0': 0.1,        # Base linear gain
        'gamma0': 0.15,      # Base curvature gain
        'eta0': 0.03,        # Base escape gain
        'k_subspace': 10,    # Dynamic subspace dimension
        'ell_orth': 5,       # Secondary modes for escape
        'tau_crit': 2.0,     # TSF threshold for stagnation
        'kappa_crit': 0.5    # Curvature threshold for stagnation
    }
}
```

---

## 📊 Examples

### Example 1: Train GPT-2 Style Model

```bash
python scripts/train.py \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --dataset_path ./data/tokenized_dataset \
    --num_epochs 5 \
    --beta0 0.12 \
    --gamma0 0.18 \
    --eta0 0.04
```

### Example 2: Custom Stagnation Detection

```bash
python scripts/train.py \
    --pg_patience 5 \
    --pg_threshold 0.10 \
    --tau_crit 2.5 \
    --kappa_crit 0.6
```

### Example 3: Ablation Study

```bash
# Linear extrapolation only (no curvature)
python scripts/train.py --beta0 0.15 --gamma0 0.0 --eta0 0.0

# With curvature correction
python scripts/train.py --beta0 0.12 --gamma0 0.18 --eta0 0.0

# Full extrapolation with escape
python scripts/train.py --beta0 0.1 --gamma0 0.15 --eta0 0.03
```

---

## 🎛️ Hyperparameters

### Recommended Ranges

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `beta0` | [0.05, 0.5] | 0.1 | Linear velocity gain |
| `gamma0` | [0.05, 0.3] | 0.15 | Acceleration gain |
| `eta0` | [0.01, 0.1] | 0.03 | Escape gain |
| `k_subspace` | [5, 30] | 10 | Dynamic subspace dimension |
| `ell_orth` | [3, 10] | 5 | Secondary escape modes |
| `tau_crit` | [1.5, 3.0] | 2.0 | TSF stagnation threshold |
| `kappa_crit` | [0.3, 1.0] | 0.5 | Curvature stagnation threshold |
| `pg_patience` | [3, 10] | 5 | Stagnation patience |
| `history_window` | [5, 15] | 5 | Trajectory window size |

### Tuning Guidelines

- **Small models (<100M parameters)**: Use default values
- **Large models (>1B parameters)**: Reduce `beta0`, `gamma0` by 30-50% for stability
- **High non-convexity tasks**: Increase `tau_crit`, `kappa_crit` to avoid premature expansion
- **Smooth landscapes**: Increase `beta0`, `gamma0` for faster convergence

---

## 🔬 Mathematical Foundation

The complete mathematical formulation is available in our technical paper. Key concepts:

### Trajectory Metrics

**Temporal Stretch Factor (TSF)**:
```
τ = path_length / direct_distance
```

**Effective Curvature**:
```
κ = Σ||V_perp|| / Σ||V_parallel||
```

### Extrapolation Formula

```
W_new = W_last + β(τ)·V̂_U + γ(κ)·Â_U + η(τ,κ)·v_escape
```

Where:
- `β(τ) = β₀/(τ + ε)`: Inversely proportional to inefficiency
- `γ(κ) = γ₀/(1 + κ)`: Inversely proportional to curvature
- `η(τ,κ) = η₀·(τ - τ_crit)·(κ - κ_crit)`: Activated during stagnation

### Core Components

#### GeometricExtrapolator

Implements the geometric extrapolation algorithm:

```python
from aiether.utils import GeometricExtrapolator

extrapolator = GeometricExtrapolator(
    beta0=0.1,
    gamma0=0.15,
    eta0=0.03,
    k_subspace=10,
    ell_orth=5
)

# Extrapolate from trajectory history
W_new, metrics = extrapolator.extrapolate(W_history)
```

#### ProceduralGrowthCallback

Monitors training and triggers layer growth:

```python
from aiether.callbacks import ProceduralGrowthCallback

callback = ProceduralGrowthCallback(
    patience=3,
    threshold=0.15,
    strategy="GeometricExtrapolator",
    warmup_steps=400,
    growth_params={'beta0': 0.1, 'gamma0': 0.15, 'eta0': 0.03}
)
```

#### LayerStateManager

Manages layer states and metadata:

```python
from aiether.managers import LayerStateManager

manager = LayerStateManager(output_dir="./checkpoints")
manager.save_state(layer_id=0, state_name="L0", state_dict=state, step=100)
```

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Hugging Face team for the transformers library
- PyTorch team for the deep learning framework
- The deep learning research community for inspiration and tools

---

## 🗺️ Roadmap

- [x] Core geometric extrapolation implementation
- [x] Integration with Hugging Face Trainer
- [x] Stagnation detection and automatic growth
- [ ] Multi-layer simultaneous growth
- [ ] Online hyperparameter adaptation
- [ ] Integration with Neural Architecture Search
- [ ] Comprehensive benchmark suite
- [ ] Pre-trained model zoo

---
### 📝 Transparency Note

The core algorithms, mathematical derivations, and code implementation in this repository are original works by the author. The English translation of code comments, documentation sections, and the technical report was refined with the assistance of **Anthropic's Claude** to ensure linguistic precision and adherence to academic standards.

**Made with ❤️ for the deep learning research community**
