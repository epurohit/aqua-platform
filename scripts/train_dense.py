"""
Reproduction script for Part 2: Dense MLP model training with Adam optimizer.
Run via: python scripts/train_dense.py
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
from train.trainer import train_epoch, evaluate


def main():
    print("=" * 65)
    print(" AQUA Part 2: Dense Neural Network Training with Adam Optimizer")
    print("=" * 65)
    
    # Set global random seed for exact reproducibility
    np.random.seed(42)
    
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
    
    epochs = 25
    batch_size = 32
    
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": []
    }
    
    print("\nStarting Training Loop...")
    print("-" * 65)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Test Loss':<12} | {'Test Acc':<12}")
    print("-" * 65)
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, optimizer, loss_fn, X_train, y_train, batch_size=batch_size)
        te_loss, te_acc = evaluate(model, loss_fn, X_test, y_test, batch_size=batch_size)
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{epoch:<8} | {tr_loss:<12.4f} | {tr_acc * 100:<11.2f}% | {te_loss:<12.4f} | {te_acc * 100:<11.2f}%")
            
    print("-" * 65)
    print(f"Final Test Accuracy: {history['test_acc'][-1] * 100:.2f}%")
    
    # 3. Save Plot
    os.makedirs("plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="blue")
    ax1.plot(range(1, epochs + 1), history["test_loss"], label="Test Loss", color="red", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Part 2: Loss Curve")
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], label="Train Acc", color="blue")
    ax2.plot(range(1, epochs + 1), [a * 100 for a in history["test_acc"]], label="Test Acc", color="red", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Part 2: Accuracy Curve")
    ax2.grid(True)
    ax2.legend()
    
    plot_path = os.path.join("plots", "part2_dense_learning_curve.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Learning curve plot saved to {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
