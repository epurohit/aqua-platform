"""
Weight Mask Manager for tracking active and pruned parameter connections.
"""
import numpy as np
from typing import List, Dict
from nn.module import Module, Parameter


class MaskManager:
    """
    Manages binary masks M in {0, 1}^shape for trainable parameters.

    Attaches `param.mask = M` directly to each parameter tensor, ensuring
    optimizers and forward passes respect parameter zeroing.
    """
    def __init__(self, model: Module):
        self.model = model
        self.masks: Dict[int, np.ndarray] = {}
        
        # Initialize 1s mask for all parameters
        for p in self.model.parameters():
            p_id = id(p)
            self.masks[p_id] = np.ones_like(p.data)
            p.mask = self.masks[p_id]

    def get_mask(self, param: Parameter) -> np.ndarray:
        """Returns the binary mask array for a given parameter."""
        return self.masks[id(param)]

    def set_mask(self, param: Parameter, mask: np.ndarray):
        """Updates the binary mask for a parameter."""
        p_id = id(param)
        self.masks[p_id] = mask.astype(np.float64)
        param.mask = self.masks[p_id]
        # Apply mask immediately to parameter data
        param.data *= param.mask

    def get_total_sparsity(self) -> float:
        """
        Computes the global model parameter sparsity ratio (zero_params / total_params).
        """
        total_params = 0
        zero_params = 0
        for p in self.model.parameters():
            mask = self.masks[id(p)]
            total_params += mask.size
            zero_params += int(np.sum(mask == 0.0))
        return zero_params / max(total_params, 1)
