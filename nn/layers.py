"""
Neural Network Layers (Linear, Activation layers).
"""
import numpy as np
from engine.tensor import Tensor
from nn.module import Module, Parameter


class Linear(Module):
    """
    Fully connected linear transformation layer: Y = X @ W + b

    Weight initialization uses Kaiming (He) normal initialization:
    std = sqrt(2.0 / in_features), preventing vanishing/exploding gradients.

    Args:
        in_features (int): Size of input feature dimension.
        out_features (int): Size of output feature dimension.
        bias (bool): Whether to include trainable bias vector. Defaults to True.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Kaiming (He) normal weight initialization
        std = np.sqrt(2.0 / in_features)
        w_data = np.random.randn(in_features, out_features) * std
        self.weight = Parameter(w_data)
        
        if bias:
            b_data = np.zeros((out_features,))
            self.bias = Parameter(b_data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    """Rectified Linear Unit activation layer."""
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    """Gaussian Error Linear Unit activation layer."""
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class Sigmoid(Module):
    """Sigmoid activation layer."""
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    """Hyperbolic Tangent activation layer."""
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()
