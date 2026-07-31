"""
Pruning schedules (Cubic Polynomial Ramp vs One-Shot).
"""
import numpy as np


class PruningScheduler:
    """Base pruning schedule interface."""
    def get_sparsity(self, step: int) -> float:
        raise NotImplementedError

    def should_prune(self, step: int) -> bool:
        raise NotImplementedError


class CubicPruningScheduler(PruningScheduler):
    """
    Polynomial cubic pruning schedule (Zhu & Gupta, 2017).

    s_t = s_f + (s_i - s_f) * (1 - (t - t_0) / (n * dt))^3

    Args:
        target_sparsity (float): Target final model sparsity s_f in [0, 1].
        start_step (int): Step t_0 when progressive pruning begins.
        end_step (int): Step t_end when pruning finishes.
        prune_freq (int): Frequency dt (in steps) at which pruning is executed.
        initial_sparsity (float): Initial sparsity s_i. Defaults to 0.0.
    """
    def __init__(
        self,
        target_sparsity: float,
        start_step: int = 100,
        end_step: int = 1000,
        prune_freq: int = 20,
        initial_sparsity: float = 0.0
    ):
        self.target_sparsity = target_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.prune_freq = prune_freq
        self.initial_sparsity = initial_sparsity

    def should_prune(self, step: int) -> bool:
        if step < self.start_step or step > self.end_step:
            return False
        return (step - self.start_step) % self.prune_freq == 0 or step == self.end_step

    def get_sparsity(self, step: int) -> float:
        if step < self.start_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.target_sparsity
            
        progress = (step - self.start_step) / float(self.end_step - self.start_step)
        s_t = self.target_sparsity + (self.initial_sparsity - self.target_sparsity) * ((1.0 - progress) ** 3)
        return float(s_t)


class OneShotScheduler(PruningScheduler):
    """One-shot pruning schedule at a designated step."""
    def __init__(self, target_sparsity: float, prune_step: int = 500):
        self.target_sparsity = target_sparsity
        self.prune_step = prune_step

    def should_prune(self, step: int) -> bool:
        return step == self.prune_step

    def get_sparsity(self, step: int) -> float:
        if step >= self.prune_step:
            return self.target_sparsity
        return 0.0
