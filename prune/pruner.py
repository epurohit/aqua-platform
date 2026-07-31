"""
Self-Pruning Orchestrator with Gradient-Based Regrowth and Masked Moment Integration.
"""
import numpy as np
from typing import Dict, List, Optional
from nn.module import Module, Parameter
from train.optimizer import Adam, Optimizer
from prune.mask import MaskManager
from prune.scheduler import PruningScheduler
from prune.criteria import compute_magnitude_importance, compute_saliency_importance


class SelfPruner:
    """
    Self-Pruning Engine that progressively removes least-useful connections
    during training under a target compute/sparsity schedule.

    Args:
        model (Module): The neural network model to prune.
        optimizer (Optimizer): Optimizer instance (Adam/SGD) for resetting moments on regrowth.
        scheduler (PruningScheduler): Pruning schedule instance.
        criterion (str): Importance metric ('saliency' or 'magnitude'). Defaults to 'saliency'.
        allow_regrowth (bool): Whether to enable gradient-based connection regrowth (RigL style).
        regrowth_fraction (float): Fraction of pruned connections to revive during regrowth steps.
    """
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: PruningScheduler,
        criterion: str = "saliency",
        allow_regrowth: bool = True,
        regrowth_fraction: float = 0.1
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion.lower()
        self.allow_regrowth = allow_regrowth
        self.regrowth_fraction = regrowth_fraction
        
        self.mask_manager = MaskManager(model)
        self.step_count = 0
        self.accum_saliency: Dict[int, np.ndarray] = {}

    def step(self):
        """
        Increments step counter, updates saliency estimates, and executes pruning/regrowth
        when scheduled.
        """
        self.step_count += 1
        
        # 1. Update saliency estimates for weight parameters
        for p in self.model.parameters():
            if p.data.ndim < 2:  # Skip 1D bias vectors, prune only weights
                continue
            p_id = id(p)
            if self.criterion == "saliency":
                current_accum = self.accum_saliency.get(p_id, None)
                updated_saliency = compute_saliency_importance(p, accum_saliency=current_accum, beta=0.9)
                self.accum_saliency[p_id] = updated_saliency

        # 2. Check if pruning is scheduled for current step
        if not self.scheduler.should_prune(self.step_count):
            return

        target_sparsity = self.scheduler.get_sparsity(self.step_count)
        if target_sparsity <= 0.0:
            return

        # 3. Collect all weight parameter importance scores
        weight_params: List[Parameter] = [p for p in self.model.parameters() if p.data.ndim >= 2]
        all_scores = []
        
        for p in weight_params:
            p_id = id(p)
            if self.criterion == "saliency":
                score = self.accum_saliency[p_id]
            else:
                score = compute_magnitude_importance(p)
            all_scores.append(score.flatten())

        concatenated_scores = np.concatenate(all_scores)
        k_prune = int(np.floor(target_sparsity * len(concatenated_scores)))
        if k_prune <= 0:
            return

        threshold = np.partition(concatenated_scores, k_prune - 1)[k_prune - 1]

        # 4. Apply pruning masks based on threshold
        for p in weight_params:
            p_id = id(p)
            score = self.accum_saliency[p_id] if self.criterion == "saliency" else compute_magnitude_importance(p)
            new_mask = (score >= threshold).astype(np.float64)
            
            # Check for newly revived connections if regrowth enabled
            old_mask = self.mask_manager.get_mask(p)
            
            if self.allow_regrowth and p.grad is not None:
                pruned_indices = (new_mask == 0.0)
                num_pruned = int(np.sum(pruned_indices))
                num_regrow = int(np.floor(self.regrowth_fraction * num_pruned))
                
                if num_regrow > 0:
                    grad_mag = np.abs(p.grad)
                    grad_mag_pruned = grad_mag * pruned_indices
                    top_k_grad_thresh = np.partition(grad_mag_pruned.flatten(), -num_regrow)[-num_regrow]
                    regrow_mask = (grad_mag_pruned >= top_k_grad_thresh) & pruned_indices
                    new_mask[regrow_mask] = 1.0

            # Find connections that were zero and become 1 (revived)
            revived = (old_mask == 0.0) & (new_mask == 1.0)
            if np.any(revived) and isinstance(self.optimizer, Adam):
                # Reset Adam moments for revived connections
                self.optimizer.reset_moments(p, mask=(~revived).astype(np.float64))

            self.mask_manager.set_mask(p, new_mask)

    def get_sparsity(self) -> float:
        """Returns current global model parameter sparsity ratio."""
        return self.mask_manager.get_total_sparsity()
