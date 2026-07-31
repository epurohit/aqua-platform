"""
AQUA Dynamic Self-Pruning Package
"""
from prune.mask import MaskManager
from prune.criteria import compute_magnitude_importance, compute_saliency_importance
from prune.scheduler import PruningScheduler, CubicPruningScheduler, OneShotScheduler
from prune.pruner import SelfPruner

__all__ = [
    "MaskManager",
    "compute_magnitude_importance", "compute_saliency_importance",
    "PruningScheduler", "CubicPruningScheduler", "OneShotScheduler",
    "SelfPruner"
]
