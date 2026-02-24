# AIether: Procedural Growth of Neural Networks via Adaptive Geometric Extrapolation

<div align="center">
    
<img src="assets/AIetherLogo.jpeg" width="100%" alt="AIether Banner">
    
<br><br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 
[![📄 Read Technical Report](https://img.shields.io/badge/📄_Read_Technical_Report-blue?style=for-the-badge)](./AIether_Technical_Report.pdf)

**AIether** is a procedural growth system for deep neural networks that uses adaptive geometric extrapolation to intelligently expand model architectures during training. By analyzing the optimization trajectory's geometric properties, AIether detects stagnation and initializes new layers in informed regions of parameter space.

[Features](#-key-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Technical Report](./AIether_Technical_Report.pdf)

</div>

---

## 🎯 Key Features

- **🔬 Geometric Trajectory Analysis**: Quantifies optimization dynamics using Temporal Stretch Factor (τ) and Effective Curvature (κ)
- **🧮 Spectral Decomposition**: Uses SVD to identify productive directions in parameter space and construct informed initializations
- **🎯 Automatic Stagnation Detection**: Monitors training quality and triggers architectural expansion when plateaus are detected
- **⚡ Tensor-Agnostic**: Handles 2D weight matrices and 1D bias vectors uniformly with multi-scale adaptive coefficients

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Core Concepts](#-core-concepts)
- [Usage Examples](#-usage-examples)
- [Hyperparameters](#️-hyperparameters)
- [Mathematical Foundation](#-mathematical-foundation)
- [Project Structure](#️-project-structure)
- [Experiments & Benchmarks](./BENCHMARKS.md)
- [Roadmap](#️-roadmap)
- [Citation](#-citation)
- [License](#-license)

---

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- numpy
- datasets
- wandb (optional, for experiment tracking)

### Install from Source
```bash
git clone https://github.com/Rochafurada/AIether.git
cd AIether
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### Minimal Training Example

The simplest way to train with AIether requires only the **5 mandatory geometric parameters**:
```bash
python scripts/train.py \
    --beta0 0.1 \
    --gamma0 0.15 \
    --eta0 0.03 \
    --k_subspace 10 \
    --ell_orth 5
```

All other parameters (model architecture, dataset, training hyperparameters) will use sensible defaults.

### Complete Training Example
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
    --ell_orth 5
```

### Programmatic Usage
```python
from aiether.training import train_with_hf_trainer

if __name__ == "__main__":
    train_with_hf_trainer()
```

---

## 🧠 Core Concepts

### 1. Geometric Trajectory Metrics

AIether analyzes the optimization trajectory using quantitative geometric metrics:

#### **Temporal Stretch Factor (τ)**
Measures trajectory efficiency as the ratio between total path length and direct displacement:
```
τ = (Σ ||W_t - W_{t-1}||) / ||W_final - W_initial||
```

- **τ ≈ 1.0–1.5**: Efficient, nearly rectilinear movement (healthy convergence)
- **τ ≈ 1.5–2.5**: Moderate exploration (stable training)
- **τ > 2.5**: Excessive oscillation (stagnation indicator)

#### **Effective Curvature (κ)**
Quantifies local tortuosity by measuring perpendicular deviation relative to net progress:
```
κ = (Σ ||V_perp||) / (Σ ||V_parallel||)
```

- **κ ≈ 0.0–0.3**: Smooth, directed trajectory
- **κ ≈ 0.3–0.8**: Healthy stochastic exploration
- **κ > 1.0**: Lateral wandering (stagnation indicator)

### 2. Adaptive Geometric Extrapolation

New layers are initialized using a combination of three components:
```
W_new = W_last + β(τ)·V̂_U + γ(κ)·Â_U + η(τ,κ)·v_escape
```

Where:

| Term | Description | Adaptive Coefficient |
|------|-------------|---------------------|
| **β(τ)·V̂_U** | Linear velocity trend projected onto dynamic subspace | β(τ) = β₀/(τ + ε) |
| **γ(κ)·Â_U** | Curvature correction from acceleration | γ(κ) = γ₀/(1 + κ) |
| **η(τ,κ)·v_escape** | Orthogonal exploration for stagnation escape | η(τ,κ) = η₀·(τ - τ_crit)·(κ - κ_crit) |

### 3. Stagnation Detection & Growth Protocol

The system monitors trajectory quality and triggers growth when:

1. **Geometric stagnation**: τ ≥ τ_crit **AND** κ ≥ κ_crit
2. **Persistent**: Condition holds for `patience` consecutive evaluations
3. **No recent improvement**: Validation loss improvement < threshold

Upon detection, the growth protocol:
1. Freezes existing layers to preserve learned representations
2. Initializes new layer using geometric extrapolation
3. Performs isolated warmup (typically 400-500 steps)
4. Unfreezes all layers and resumes joint training

---

## 📊 Usage Examples

### Example 1: Train GPT-2 Style Model
```bash
python scripts/train.py \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --dataset_path ./data/fineweb_edu \
    --num_epochs 5 \
    --batch_size 16 \
    --lr 3e-4 \
    --beta0 0.12 \
    --gamma0 0.18 \
    --eta0 0.04 \
    --k_subspace 15 \
    --ell_orth 7
```

### Example 2: Conservative Growth for Noisy Landscapes

For tasks with high gradient variance (e.g., RL, sparse rewards):
```bash
python scripts/train.py \
    --beta0 0.08 \
    --gamma0 0.12 \
    --eta0 0.02 \
    --k_subspace 8 \
    --ell_orth 4 \
    --tau_crit 2.5 \
    --kappa_crit 0.7 \
    --pg_patience 5
```

### Example 3: Aggressive Growth for Smooth Landscapes

For well-behaved optimization (e.g., vision tasks, large batch sizes):
```bash
python scripts/train.py \
    --beta0 0.15 \
    --gamma0 0.20 \
    --eta0 0.05 \
    --k_subspace 12 \
    --ell_orth 6 \
    --tau_crit 1.8 \
    --kappa_crit 0.4 \
    --pg_patience 3
```

### Example 4: Ablation Studies
```bash
# Baseline: Linear extrapolation only
python scripts/train.py --beta0 0.15 --gamma0 0.0 --eta0 0.0 --k_subspace 10 --ell_orth 5

# Add curvature correction
python scripts/train.py --beta0 0.12 --gamma0 0.18 --eta0 0.0 --k_subspace 10 --ell_orth 5

# Full system with escape mechanism
python scripts/train.py --beta0 0.1 --gamma0 0.15 --eta0 0.03 --k_subspace 10 --ell_orth 5
```

---

## 🎛️ Hyperparameters

### Mandatory Geometric Parameters

These **5 parameters are required** for all training runs:

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `--beta0` | float | [0.05, 0.5] | 0.1 | Base gain for linear velocity term |
| `--gamma0` | float | [0.05, 0.3] | 0.15 | Base gain for curvature correction term |
| `--eta0` | float | [0.01, 0.1] | 0.03 | Base gain for orthogonal escape term |
| `--k_subspace` | int | [5, 30] | 10 | Dimension of dynamic subspace (SVD truncation) |
| `--ell_orth` | int | [3, 10] | 5 | Number of secondary modes for escape direction |

### Optional Model & Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n_layer` | int | 6 | Initial number of transformer layers |
| `--n_head` | int | 6 | Number of attention heads |
| `--n_embd` | int | 384 | Embedding dimension |
| `--dropout` | float | 0.1 | Dropout rate |
| `--num_epochs` | int | 3 | Number of training epochs |
| `--batch_size` | int | 8 | Training batch size |
| `--lr` | float | 5e-4 | Learning rate |
| `--dataset_path` | str | None | Path to tokenized dataset |

### Optional Stagnation Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pg_patience` | int | 5 | Number of consecutive stagnation checks before growth |
| `--pg_threshold` | float | 0.15 | Minimum relative improvement to avoid stagnation |
| `--tau_crit` | float | 2.0 | TSF threshold for stagnation detection |
| `--kappa_crit` | float | 0.5 | Curvature threshold for stagnation detection |
| `--warmup_steps` | int | 400 | Steps before first stagnation check |
| `--history_window_size` | int | 15 | Number of checkpoints to maintain in trajectory history |
| `--collect_interval_steps` | int | 100 | Steps between trajectory snapshot collection |

### Tuning Guidelines

**For small models (<100M parameters)**:
- Use default values
- Consider slightly higher `beta0` (0.12-0.15) for faster convergence

**For large models (>1B parameters)**:
- Reduce `beta0`, `gamma0` by 30-50% for stability
- Increase `k_subspace` to 15-20 to capture richer trajectory structure
- Consider more conservative `tau_crit` (2.5-3.0)

**For high non-convexity tasks**:
- Increase `tau_crit` (2.5-3.0) and `kappa_crit` (0.6-0.8)
- Increase `pg_patience` (7-10) to avoid premature expansion
- Reduce `eta0` (0.01-0.02) for more conservative escape

**For smooth landscapes**:
- Increase `beta0` (0.15-0.20) and `gamma0` (0.20-0.25)
- Reduce `tau_crit` (1.5-1.8) for earlier stagnation detection
- Reduce `pg_patience` (3) for faster growth response

---

## 🔬 Mathematical Foundation

The complete mathematical formulation is available in our [technical report](./AIether_Technical_Report.pdf). Key concepts:

### Trajectory Metrics

**Temporal Stretch Factor (TSF)**:
```
τ = Σ||V_t|| / ||Δ||  where Δ = W_final - W_initial
```
Quantifies global trajectory inefficiency (path length vs. direct distance).

**Effective Curvature**:
```
κ = Σ||V_t^⊥|| / Σ||V_t^∥||
```
Quantifies local tortuosity through parallel-perpendicular decomposition.

### Dynamic Subspace Decomposition

The centered velocity matrix undergoes SVD:
```
Ṽ = U Σ Q^T
```
Where `Q_k = [q_1, ..., q_k]` spans the k-dimensional dynamic subspace capturing dominant trajectory modes.

### Adaptive Extrapolation Formula
```
W_new = W_last + β(τ)·V̂_U + γ(κ)·Â_U + η(τ,κ)·v_escape
```

**Adaptive coefficients**:
- `β(τ) = β₀/(τ + ε)`: Inversely proportional to trajectory inefficiency
- `γ(κ) = γ₀/(1 + κ)`: Inversely proportional to curvature
- `η(τ,κ) = η₀·max(0, τ - τ_crit)·max(0, κ - κ_crit)`: Activated during stagnation

**Components**:
- `V̂_U`: Normalized mean velocity projected onto dynamic subspace
- `Â_U`: Normalized mean acceleration projected onto dynamic subspace
- `v_escape`: Spectral escape direction from secondary modes `q_{k+1}, ..., q_{k+ℓ}`

---

## 🏗️ Project Structure
```
AIether/
├── callbacks/                      # Training callbacks
│   ├── procedural_growth_callback.py   # Main growth orchestration
│   └── debug_callback.py               # Debugging utilities
├── managers/                       # State management
│   ├── growth_manager.py               # Growth decision logic
│   └── layer_state_manager.py          # Layer state persistence
├── models/                         # Model implementations
│   └── gpt.py                          # GPT architecture with growth support
├── training/                       # Training orchestration
│   └── trainer.py                      # HuggingFace Trainer integration
├── utils/                          # Core utilities
│   ├── geometry.py                     # GeometricExtrapolator implementation
│   ├── config.py                       # Configuration parsing
│   └── logging.py                      # Custom logging
├── scripts/
│   └── train.py                        # Training entry point
├── assets/                         # Media resources
├── AIether_Technical_Report.pdf    # Full mathematical documentation
└── README.md
```

### Core Components

#### GeometricExtrapolator

Implements the geometric extrapolation algorithm:
```python
from aiether.utils.geometry import GeometricExtrapolator

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

**Key methods**:
- `compute_tau()`: Computes Temporal Stretch Factor
- `compute_kappa_eff()`: Computes Effective Curvature
- `extrapolate()`: Performs adaptive geometric extrapolation

#### ProceduralGrowthCallback

Monitors training and triggers layer growth:
```python
from aiether.callbacks import ProceduralGrowthCallback

callback = ProceduralGrowthCallback(
    patience=5,
    threshold=0.15,
    strategy="GeometricExtrapolator",
    warmup_steps=400,
    growth_params={
        'beta0': 0.1, 
        'gamma0': 0.15, 
        'eta0': 0.03,
        'k_subspace': 10,
        'ell_orth': 5
    }
)
```

#### LayerStateManager

Manages layer states and metadata:
```python
from aiether.managers import LayerStateManager

manager = LayerStateManager(output_dir="./checkpoints")
manager.save_state(
    layer_id=0, 
    state_name="L0", 
    state_dict=state, 
    step=100
)
```

---

## 📈 Experiments & Benchmarks

Detailed experimental results, ablation studies, and benchmark comparisons are available in:

**[📊 View Benchmarks & Results](./BENCHMARKS.md)**

---

## 🗺️ Roadmap

### Core Capabilities (Implemented)
- [x] **Core Spectral Extrapolation Engine:** SVD-based trajectory analysis.
- [x] **Geometric Diagnostics:** Tensor-wise calculation of Temporal Stretch ($\tau$) and Effective Curvature ($\kappa$).
- [x] **Autonomous Stagnation Detection:** Hysteresis-based triggers for architectural expansion.
- [x] **Hugging Face Integration:** Seamless callback support for `Trainer` and `Accelerate`.
- [x] **Tensor-wise Granularity:** Support for 2D weight matrices and 1D bias/norm vectors.

### Research & Engineering Targets
- [ ] **Optimizer State Transfer (Momentum Injection):** Preserving optimizer momentum ($\mu$) post-expansion to eliminate warmup cycles.
- [ ] **Non-Sequential Topologies:** Extension to Horizontal (Width) and Diagonal growth patterns.
- [ ] **Distributed Training Support:** Native compatibility with FSDP (Fully Sharded Data Parallel) and DeepSpeed for 10B+ models.
- [ ] **Riemannian Manifold Constraints:** Geodesic extrapolation for orthogonal matrices and attention heads (Stiefel manifold).
- [ ] **Spectral Auto-tuning:** Dynamic selection of subspace dimension ($k$) based on singular value gaps.
- [ ] **Gradient Conditioning:** Analysis of Hessian stability during multi-layer simultaneous growth.

---

## 📚 Citation

If you use AIether in your research, please cite:
```bibtex
@article{rocha2025aiether,
  title={AIether: Procedural Growth of Neural Networks via Adaptive Geometric Extrapolation},
  author={Rocha, Samuel},
  journal={Technical Report},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Andrej Karpathy** for [nanoGPT](https://github.com/karpathy/nanoGPT), which provided the clean GPT implementation serving as the structural backbone of this framework
- **Hugging Face** team for the transformers library and Trainer API
- **PyTorch** team for the deep learning framework
- The deep learning research community for inspiration, tools, and open science

---

## 📝 Transparency Note

The core algorithms, mathematical derivations, and code implementation in this repository are original works by the author. English translation of documentation and the technical report was refined with assistance from **Anthropic's Claude** to ensure linguistic precision and adherence to academic standards.

---

<div align="center">

**Made with ❤️ for the deep learning research community**

[⬆ Back to Top](#aiether-procedural-growth-of-neural-networks-via-adaptive-geometric-extrapolation)

</div>
