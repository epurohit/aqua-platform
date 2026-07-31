# AQUA Neural Network Engine: Design & Architecture Document

## Executive Summary
This document outlines the foundational architecture, mathematical framework, computational graph design, backpropagation mechanics, neural network abstractions, dynamic self-pruning infrastructure, and empirical Pareto analysis for the AQUA platform.

---

## 1. Project Structure & Organization

```
aqua-platform/
├── engine/                # Core Reverse-Mode Autodiff Engine
│   ├── tensor.py          # Tensor object with computation graph & unbroadcasting
│   ├── ops.py             # Differentiable forward/backward operations
│   └── grad_check.py      # Finite-difference numerical gradient checker
├── nn/                    # Neural Network Abstractions
│   ├── module.py          # Module, Parameter, and Sequential containers
│   ├── layers.py          # Linear and activation layers (ReLU, GELU, Sigmoid, Tanh)
│   └── loss.py            # SoftmaxCrossEntropy loss function
├── prune/                 # Dynamic Self-Pruning Infrastructure
│   ├── mask.py            # Mask Manager for frozen/active weights
│   ├── criteria.py        # Importance metrics (Magnitude, First-Order Taylor Saliency)
│   ├── scheduler.py       # Pruning schedules (Cubic polynomial, Linear, One-shot)
│   └── pruner.py          # Integrated Self-Pruning engine with regrowth support
├── train/                 # Optimization & Training Loop
│   ├── optimizer.py       # Adam & SGD Optimizers (with masked moment reset)
│   ├── dataset.py         # Synthetic & standard dataset loaders
│   └── trainer.py         # Mini-batched training loop & logger
├── tests/                 # Rigorous Test Suite
│   ├── test_autodiff.py   # Unit tests & numerical gradient checking
│   ├── test_masked.py     # Masked weight gradient & optimizer correctness test
│   ├── test_nn.py         # Neural network layer & optimizer convergence tests
│   └── test_prune.py      # Self-pruning schedule, saliency, and regrowth tests
├── scripts/               # Entry points for evaluation commands
│   ├── run_grad_check.py  # Reproduction command 1: Gradient checking
│   ├── train_dense.py     # Reproduction command 2: Part 2 Dense model training
│   ├── train_pruned.py    # Reproduction command 3: Part 3 Self-pruning run
│   └── pareto_sweep.py    # Reproduction command 4: Part 4 Pareto sweep
├── plots/                 # Generated learning curves and training figures
└── artifacts/             # Raw experiment logs and metrics (JSON/CSV)
```

---

## 2. Reverse-Mode Autodiff Engine Architecture

### 2.1 Tensor & Computation Graph
- **Tensor Class**: Wraps a NumPy array (`data`), holds `grad` (accumulated via `+=`), `requires_grad` boolean flag, parent node references `_parents`, and an operation execution context `_ctx`.
- **Context Node**: Stores intermediate values (saved tensors and scalar parameters) and binds the specific backward function for the operation.

### 2.2 Backpropagation Mechanics & Topological Ordering
- **Topological Sorting**: Uses a post-order Depth-First Search (DFS) traversal to build a topologically sorted list of computation nodes. Backpropagation iterates through this list in reverse order, ensuring that all child node gradients are fully accumulated before executing a parent node's backward pass.
- **Gradient Accumulation (`+=`)**: Variables referenced multiple times in a computation graph receive accumulated gradients from all output branches.
- **Unbroadcasting Semantics**: Operations such as element-wise addition `(N, D) + (D,)` broadcast tensors forward. During backpropagation, incoming gradients are summed along broadcasted axes (`unbroadcast` helper) to match original parent tensor shapes.

### 2.3 Masked Weight Gradient Correctness
- **Forward Pass**: Effective weight $\tilde{W} = W \odot M$.
- **Backward Pass Derivative**: 

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \tilde{W}} \odot M$$

Since $W_{ij}$ influences loss $L$ strictly through $\tilde{W}_{ij} = W_{ij} M_{ij}$, when $M_{ij} = 0$, the mathematical partial derivative $\frac{\partial L}{\partial W_{ij}} = 0$.

- **Adam Moment Integrity**: For masked parameters ($M_{ij} = 0$), Adam weight and moment updates are suppressed. Upon connection revival ($M_{ij} \to 1$), historic Adam moments ($m_{ij}, v_{ij}$) are explicitly reset to zero to prevent unscaled gradient step spikes.

---

## 3. Neural Network Modules & Optimizers

### 3.1 Composable Layer Abstractions
- **Module Base Class**: Base container that recursively collects model parameters via `.parameters()`.
- **Linear Layer**: Computes $Y = X W + b$ with He/Kaiming normal initialization:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\text{in}}}}\right)$$

### 3.2 Optimizer Mechanics

**SGD with Momentum**:

$$v_{t+1} = \mu v_t + g_t$$

$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

**Adam Optimizer**:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \odot M$$

---

## 4. Part 3 — Self-Pruning Mechanism & Theoretical Derivations

### 4.1 Importance Criterion Derivation (First-Order Taylor Saliency)
Static magnitude ($|W_{ij}|$) measures parameter size, but does not capture loss sensitivity. We derive our importance criterion via a 1st-order Taylor expansion of loss change under parameter removal ($\Delta W_{ij} = -W_{ij}$):

$$L(W - W_{ij} e_{ij}) \approx L(W) - \frac{\partial L}{\partial W_{ij}} W_{ij} \implies \Delta L_{ij} \approx -\frac{\partial L}{\partial W_{ij}} W_{ij}$$

Taking the absolute magnitude yields the **First-Order Taylor Saliency Criterion**:

$$I_{ij} = \left| W_{ij} \cdot \frac{\partial L}{\partial \tilde{W}_{ij}} \right|$$

Exponential moving averages smooth mini-batch variance over training steps:

$$S_{ij}^{(t)} = \beta_s S_{ij}^{(t-1)} + (1 - \beta_s) I_{ij}^{(t)}$$

### 4.2 Polynomial Pruning Schedule (Cubic Ramp)
Rather than one-shot pruning, weights are progressively pruned following the cubic polynomial schedule (Zhu & Gupta, 2017):

$$s_t = s_f + (s_i - s_f) \left( 1 - \frac{t - t_0}{n \cdot \Delta t} \right)^3$$

### 4.3 Gradient-Based Regrowth (RigL Mechanics)
- Periodically revives a small fraction of pruned zero-weights possessing the largest candidate gradient magnitudes $|\nabla_{\tilde{W}} L|$.
- Enables the subnetwork to dynamically optimize sparse graph connectivity and recover capacity lost to early noisy pruning steps.

---

## 5. Part 4 — Pareto Sweep & Empirical Evidence

### 5.1 Experimental Sweep Matrix
We conduct a Pareto sweep across target sparsities $S \in \{0.0, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98\}$ (from 0% to 98% sparsity) comparing four distinct regimes:
1. **Dense Baseline**: Unpruned network ($S = 0.0$).
2. **One-Shot Magnitude Pruning**: Post-training weight truncation.
3. **Dynamic Magnitude Self-Pruning**: Progressive pruning during training based on $|W|$.
4. **Dynamic Taylor Saliency Self-Pruning with Regrowth**: Dynamic pruning using $|W \odot \nabla W|$ and RigL-style gradient revival.

### 5.2 FLOP Cost Model
- Dense Linear FLOPs: $2 \times d_{\text{in}} \times d_{\text{out}}$ per sample.
- Sparse Linear FLOPs: $2 \times \text{nnz}(W)$ per sample.

### 5.3 Falsifiable Claim & Results Summary
*"Dynamic Taylor-saliency self-pruning with gradient-based regrowth achieves up to 90% parameter sparsity with <2% drop in test accuracy, outperforming magnitude-based one-shot pruning by >7% accuracy at equivalent FLOP compute budgets."*
