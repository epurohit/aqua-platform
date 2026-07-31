"""
AQUA Training Package
"""
from train.optimizer import Optimizer, SGD, Adam
from train.dataset import make_spirals, load_digits_dataset
from train.trainer import train_epoch, evaluate, compute_accuracy

__all__ = [
    "Optimizer", "SGD", "Adam",
    "make_spirals", "load_digits_dataset",
    "train_epoch", "evaluate", "compute_accuracy"
]
