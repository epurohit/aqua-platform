"""
Training loop, mini-batching, and model evaluation utilities.
"""
import numpy as np
from typing import Tuple, Dict, List
from engine.tensor import Tensor
from nn.module import Module
from train.optimizer import Optimizer


def compute_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Computes top-1 classification accuracy."""
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds == targets))


def evaluate(model: Module, loss_fn: Module, X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> Tuple[float, float]:
    """
    Evaluates model on dataset (X, y) without computing parameter gradients.

    Returns:
        Tuple[float, float]: (average_loss, accuracy)
    """
    N = len(X)
    total_loss = 0.0
    all_logits = []
    
    for i in range(0, N, batch_size):
        X_batch = Tensor(X[i:i + batch_size], requires_grad=False)
        y_batch = y[i:i + batch_size]
        
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        
        total_loss += loss.item() * len(X_batch.data)
        all_logits.append(logits.data)
        
    avg_loss = total_loss / N
    all_logits_arr = np.concatenate(all_logits, axis=0)
    accuracy = compute_accuracy(all_logits_arr, y)
    
    return avg_loss, accuracy


def train_epoch(
    model: Module,
    optimizer: Optimizer,
    loss_fn: Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 32
) -> Tuple[float, float]:
    """
    Executes one epoch of mini-batched training.

    Returns:
        Tuple[float, float]: (average_train_loss, train_accuracy)
    """
    N = len(X_train)
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    total_loss = 0.0
    all_logits = []
    
    for i in range(0, N, batch_size):
        batch_idx = indices[i:i + batch_size]
        X_batch = Tensor(X_train[batch_idx], requires_grad=False)
        y_batch = y_train[batch_idx]
        
        model.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_idx)
        all_logits.append((logits.data, y_batch))
        
    avg_loss = total_loss / N
    
    # Calculate epoch accuracy
    preds_list = [np.argmax(l[0], axis=-1) for l in all_logits]
    targets_list = [l[1] for l in all_logits]
    all_preds = np.concatenate(preds_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)
    accuracy = float(np.mean(all_preds == all_targets))
    
    return avg_loss, accuracy
