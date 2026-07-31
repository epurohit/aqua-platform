"""
AQUA Optimizers (SGD with Momentum, Adam with Masked Weight Support).
"""
import numpy as np
from typing import List, Dict
from nn.module import Parameter


class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, params: List[Parameter], lr: float):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        """Zero out gradients for all tracked parameters."""
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with Momentum.

    v_{t+1} = momentum * v_t + grad + weight_decay * theta
    theta_{t+1} = theta_t - lr * v_{t+1}
    """
    def __init__(self, params: List[Parameter], lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities: Dict[int, np.ndarray] = {}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
                
            grad = p.grad.copy()
            if self.weight_decay > 0:
                grad += self.weight_decay * p.data
                
            p_id = id(p)
            if p_id not in self.velocities:
                self.velocities[p_id] = np.zeros_like(p.data)
                
            v = self.velocities[p_id]
            v = self.momentum * v + grad
            self.velocities[p_id] = v
            
            # Mask check if parameter has weight mask
            mask = getattr(p, "mask", None)
            if mask is not None:
                p.data -= self.lr * (v * mask)
            else:
                p.data -= self.lr * v


class Adam(Optimizer):
    """
    Adam Optimizer from scratch with explicit masked weight support and moment reset.

    Formula:
      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
      m_hat = m_t / (1 - beta1^t)
      v_hat = v_t / (1 - beta2^t)
      theta_{t+1} = theta_t - lr * m_hat / (sqrt(v_hat) + eps)
    """
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}

    def reset_moments(self, parameter: Parameter, mask: np.ndarray = None):
        """
        Resets Adam moment estimates for specified parameter or unmasked indices upon revival.
        """
        p_id = id(parameter)
        if p_id in self.m:
            if mask is None:
                self.m[p_id].fill(0.0)
                self.v[p_id].fill(0.0)
            else:
                self.m[p_id] *= mask
                self.v[p_id] *= mask

    def step(self):
        self.t += 1
        
        for p in self.params:
            if p.grad is None:
                continue
                
            grad = p.grad.copy()
            if self.weight_decay > 0:
                grad += self.weight_decay * p.data
                
            p_id = id(p)
            if p_id not in self.m:
                self.m[p_id] = np.zeros_like(p.data)
                self.v[p_id] = np.zeros_like(p.data)
                
            m = self.m[p_id]
            v = self.v[p_id]
            
            mask = getattr(p, "mask", None)
            
            # Mask gradient if parameter is pruned
            if mask is not None:
                grad = grad * mask
                
            # Update moments
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
            
            if mask is not None:
                m = m * mask
                v = v * mask
                
            self.m[p_id] = m
            self.v[p_id] = v
            
            # Bias correction
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)
            
            step_update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            if mask is not None:
                p.data -= step_update * mask
            else:
                p.data -= step_update
