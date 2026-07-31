"""
Importance criteria algorithms (Magnitude vs First-Order Taylor Saliency).
"""
import numpy as np
from typing import Optional
from nn.module import Parameter


def compute_magnitude_importance(param: Parameter) -> np.ndarray:
    """
    Computes magnitude-based importance: Score = |W|
    """
    return np.abs(param.data)


def compute_saliency_importance(
    param: Parameter,
    accum_saliency: Optional[np.ndarray] = None,
    beta: float = 0.9
) -> np.ndarray:
    """
    Computes First-Order Taylor Saliency importance: Score = |W * grad|

    Approximates the scalar increase in loss if connection W_ij is set to zero.
    Optionally applies exponential moving average smoothing over mini-batches.
    """
    if param.grad is None:
        raw_score = np.abs(param.data)
    else:
        raw_score = np.abs(param.data * param.grad)
        
    if accum_saliency is None:
        return raw_score
        
    # Exponential moving average smoothing
    smoothed_score = beta * accum_saliency + (1.0 - beta) * raw_score
    return smoothed_score
