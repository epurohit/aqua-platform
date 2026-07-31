"""
Reproduction script for Part 3: Self-Pruning neural network with Taylor Saliency & Regrowth.
Run via: python scripts/train_pruned.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from nn.module import Sequential
from nn.layers import Linear, ReLU
from nn.loss import CrossEntropyLoss
from train.optimizer import Adam
from train.dataset import load_digits_dataset
from train.trainer import evaluate, compute_accuracy
from engine.tensor import Tensor
from prune.scheduler import CubicPruningScheduler
from prune.pruner import SelfPruner


def main():
    print("=" * 70)
    print(" AQUA Part 3: Self-Pruning Neural Network Run (90% Target Sparsity)")
    print("=" * 70)
    
    # 1. Load Dataset
    X_train, y_train, X_test, y_test = load_digits_dataset(seed=42)
    print(f"Dataset Loaded: Train shape={X_train.shape}, Test shape={X_test.shape}")
    
    # 2. Build Model
    model = Sequential(
        Linear(64, 64),
        ReLU(),
        Linear(64, 32),
        ReLU(),
        Linear(32, 10)
    )
    
    optimizer = Adam(model.parameters(), lr=0.005)
    loss_fn = CrossEntropyLoss()
    
    epochs = 30
    batch_size = 32
    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    total_steps = epochs * steps_per_epoch
    
    # Configure Cubic Pruning Scheduler: start step 10%, end step 80%
    start_step = int(0.10 * total_steps)
    end_step = int(0.80 * total_steps)
    scheduler = CubicPruningScheduler(
        target_sparsity=0.90,
        start_step=start_step,
        end_step=end_step,
        prune_freq=10
    )
    
    pruner = SelfPruner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion="saliency",
        allow_regrowth=True,
        regrowth_fraction=0.05
    )
    
    history = {
        "epoch": [], "train_loss": [], "test_acc": [], "sparsity": []
    }
    
    print("\nStarting Self-Pruning Training Loop...")
    print("-" * 70)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Test Acc':<12} | {'Sparsity (%)':<14} | {'Target Sparsity':<15}")
    print("-" * 70)
    
    step_count = 0
    for epoch in range(1, epochs + 1):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        
        total_loss = 0.0
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
            step_count += 1
            total_loss += loss.item() * len(batch_idx)
            
        tr_loss = total_loss / len(X_train)
        _, te_acc = evaluate(model, loss_fn, X_test, y_test, batch_size=batch_size)
        curr_sparsity = pruner.get_sparsity()
        target_sp = scheduler.get_sparsity(step_count)
        
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["test_acc"].append(te_acc)
        history["sparsity"].append(curr_sparsity)
        
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{epoch:<8} | {tr_loss:<12.4f} | {te_acc * 100:<11.2f}% | {curr_sparsity * 100:<13.2f}% | {target_sp * 100:<14.2f}%")
            
    print("-" * 70)
    print(f"Final Model Sparsity: {history['sparsity'][-1] * 100:.2f}%")
    print(f"Final Test Accuracy:  {history['test_acc'][-1] * 100:.2f}%")
    
    # 3. Plot Self-Pruning Trajectory
    os.makedirs("plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history["epoch"], [s * 100 for s in history["sparsity"]], color="purple", linewidth=2, label="Actual Sparsity")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Sparsity (%)")
    ax1.set_title("Part 3: Sparsity Trajectory over Epochs")
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(history["epoch"], [a * 100 for a in history["test_acc"]], color="green", linewidth=2, label="Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Part 3: Accuracy under Progressive Pruning")
    ax2.grid(True)
    ax2.legend()
    
    plot_path = os.path.join("plots", "part3_self_pruning.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Self-pruning trajectory plot saved to {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
