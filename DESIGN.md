# AQUA Neural Network Engine: Design & Architecture Document

## Executive Summary
This document outlines the foundational architecture, mathematical framework, computational graph design, and backpropagation mechanics for the AQUA reverse-mode automatic differentiation engine and neural network platform. 

Additional features (such as dynamic self-pruning mechanisms, advanced importance criteria, and scale inference serving strategies) will be progressively detailed as each component is implemented.

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
├── train/                 # Optimization & Training Loop
│   ├── optimizer.py       # Adam & SGD Optimizers
│   ├── dataset.py         # Synthetic & standard dataset loaders
│   └── trainer.py         # Mini-batched training loop & logger
├── tests/                 # Rigorous Test Suite
│   ├── test_autodiff.py   # Unit tests & numerical gradient checking
│   └── test_nn.py         # Neural network layer & optimizer convergence tests
├── scripts/               # Entry points for evaluation commands
│   ├── run_grad_check.py  # Reproduction command: Gradient checking
│   └── train_dense.py     # Reproduction command: Part 2 Dense model training
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

### 2.3 Operations & Mathematical Derivatives
1. **Element-wise Addition / Subtraction / Multiplication / Division**:
   - $\nabla_A (A + B) = \text{grad}$, $\nabla_B (A + B) = \text{grad}$
   - $\nabla_A (A \cdot B) = \text{grad} \cdot B$, $\nabla_B (A \cdot B) = \text{grad} \cdot A$
2. **Matrix Multiplication ($Y = A @ B$)**:
   - $\nabla_A = \text{grad} @ B^T$
   - $\nabla_B = A^T @ \text{grad}$
3. **Reductions (Sum / Mean)**:
   - Sum: $\nabla_A = \text{broadcast}(\text{grad}, A.\text{shape})$
   - Mean: $\nabla_A = \text{broadcast}(\text{grad}, A.\text{shape}) / N$
4. **Activations (ReLU, GELU, Sigmoid, Tanh)**:
   - ReLU: $\nabla_x = \text{grad} \cdot \mathbb{I}(x > 0)$
   - GELU: $\nabla_x = \text{grad} \cdot \left[ 0.5(1 + \tanh(u)) + 0.5 x (1 - \tanh^2(u)) \sqrt{\frac{2}{\pi}} (1 + 3 \cdot 0.044715 x^2) \right]$
5. **Numerically Stable Softmax Cross-Entropy**:
   - Forward: $\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$, $\text{Loss} = -\frac{1}{N} \sum y_i \log(\text{softmax}(z)_i)$
   - Backward: $\nabla_z \text{Loss} = \frac{1}{N} (\text{softmax}(z) - y)$

---

## 3. Neural Network Modules & Optimizers

### 3.1 Composable Layer Abstractions
- **Module Base Class**: Base container that recursively collects model parameters via `.parameters()`.
- **Linear Layer**: Computes $Y = X W + b$ with configurable weight initialization (He/Kaiming normal initialization for ReLU/GELU activations, Xavier uniform for linear/tanh).

### 3.2 Optimizer Mechanics
- **SGD with Momentum**:
  $$v_{t+1} = \mu v_t + g_t$$
  $$\theta_{t+1} = \theta_t - \eta v_{t+1}$$
- **Adam Optimizer**:
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  $$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
