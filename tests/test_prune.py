"""
Unit tests for Self-Pruning infrastructure (Cubic scheduler, Taylor Saliency, Regrowth).
"""
import pytest
import numpy as np
from engine.tensor import Tensor
from nn.module import Sequential
from nn.layers import Linear, ReLU
from nn.loss import CrossEntropyLoss
from train.optimizer import Adam
from train.dataset import load_digits_dataset
from train.trainer import train_epoch, evaluate
from prune.scheduler import CubicPruningScheduler, OneShotScheduler
from prune.pruner import SelfPruner


def test_cubic_scheduler_sparsity_ramp():
    scheduler = CubicPruningScheduler(target_sparsity=0.90, start_step=100, end_step=1000, prune_freq=10)
    
    assert scheduler.get_sparsity(50) == 0.0
    assert scheduler.get_sparsity(1000) == 0.90
    
    # Verify monotonic increase
    s1 = scheduler.get_sparsity(200)
    s2 = scheduler.get_sparsity(500)
    s3 = scheduler.get_sparsity(800)
    
    assert 0.0 < s1 < s2 < s3 < 0.90


def test_self_pruning_reaches_target_sparsity():
    """
    Verifies that SelfPruner hits target sparsity on a trained MLP model.
    """
    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_digits_dataset(seed=42)
    
    model = Sequential(
        Linear(64, 64),
        ReLU(),
        Linear(64, 32),
        ReLU(),
        Linear(32, 10)
    )
    optimizer = Adam(model.parameters(), lr=0.005)
    loss_fn = CrossEntropyLoss()
    
    epochs = 15
    batch_size = 32
    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    total_steps = epochs * steps_per_epoch
    
    scheduler = CubicPruningScheduler(target_sparsity=0.90, start_step=10, end_step=int(0.8 * total_steps), prune_freq=10)
    pruner = SelfPruner(model, optimizer, scheduler, criterion="saliency", allow_regrowth=True, regrowth_fraction=0.02)
    
    step = 0
    for epoch in range(epochs):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i + batch_size]
            X_b = Tensor(X_train[batch_idx], requires_grad=False)
            y_b = y_train[batch_idx]
            
            model.zero_grad()
            logits = model(X_b)
            loss = loss_fn(logits, y_b)
            loss.backward()
            
            optimizer.step()
            pruner.step()
            step += 1

    final_sparsity = pruner.get_sparsity()
    assert final_sparsity >= 0.85, f"Expected final sparsity >=85%, got {final_sparsity:.2%}"
    
    # Verify test accuracy remains high (>85%) under 85-90% sparsity
    _, test_acc = evaluate(model, loss_fn, X_test, y_test)
    assert test_acc > 0.85, f"Expected test accuracy >85% at high sparsity, got {test_acc:.2%}"
