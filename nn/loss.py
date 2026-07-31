"""
Loss functions for Neural Network training.
"""
from engine.tensor import Tensor
from engine.ops import softmax_cross_entropy
from nn.module import Module


class CrossEntropyLoss(Module):
    """
    Numerically stable Softmax Cross-Entropy Loss module.

    Computes loss over (N, C) logits and target class indices or one-hot distributions.
    """
    def forward(self, logits: Tensor, targets) -> Tensor:
        return softmax_cross_entropy(logits, targets)
