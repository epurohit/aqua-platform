"""
Reproduction script for Part 4: Pareto Sweep (Accuracy vs Sparsity vs FLOP Cost).
Run via: python scripts/pareto_sweep.py
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

from nn.module import Sequential
from nn.layers import Linear, ReLU
from nn.loss import CrossEntropyLoss
from train.optimizer import Adam
from train.dataset import load_digits_dataset
from train.trainer import train_epoch, evaluate
from engine.tensor import Tensor
from prune.scheduler import CubicPruningScheduler, OneShotScheduler
from prune.pruner import SelfPruner


def compute_model_flops(model: Sequential) -> Tuple[int, int]:
    """
    Computes total dense FLOPs and active sparse FLOPs per sample.
    Linear layer FLOPs = 2 * non_zero_weights.
    """
    dense_flops = 0
    sparse_flops = 0
    
    for layer in model.layers:
        if isinstance(layer, Linear):
            w = layer.weight.data
            mask = getattr(layer.weight, "mask", np.ones_like(w))
            d_flops = 2 * w.size
            s_flops = 2 * int(np.sum(mask > 0))
            dense_flops += d_flops
            sparse_flops += s_flops
            
    return dense_flops, sparse_flops


def run_experiment(method: str, target_sparsity: float, X_tr, y_tr, X_te, y_te, epochs=25, seed=42):
    """
    Runs a single training experiment under a given pruning strategy and target sparsity.
    """
    np.random.seed(seed)
    model = Sequential(
        Linear(64, 64),
        ReLU(),
        Linear(64, 32),
        ReLU(),
        Linear(32, 10)
    )
    optimizer = Adam(model.parameters(), lr=0.005)
    loss_fn = CrossEntropyLoss()
    
    batch_size = 32
    steps_per_epoch = int(np.ceil(len(X_tr) / batch_size))
    total_steps = epochs * steps_per_epoch
    
    pruner = None
    if target_sparsity > 0.0:
        if method == "one_shot_magnitude":
            scheduler = OneShotScheduler(target_sparsity=target_sparsity, prune_step=int(0.8 * total_steps))
            pruner = SelfPruner(model, optimizer, scheduler, criterion="magnitude", allow_regrowth=False)
        elif method == "dynamic_magnitude":
            scheduler = CubicPruningScheduler(target_sparsity=target_sparsity, start_step=int(0.1*total_steps), end_step=int(0.8*total_steps), prune_freq=10)
            pruner = SelfPruner(model, optimizer, scheduler, criterion="magnitude", allow_regrowth=False)
        elif method == "dynamic_saliency_regrowth":
            scheduler = CubicPruningScheduler(target_sparsity=target_sparsity, start_step=int(0.1*total_steps), end_step=int(0.8*total_steps), prune_freq=10)
            pruner = SelfPruner(model, optimizer, scheduler, criterion="saliency", allow_regrowth=True, regrowth_fraction=0.02)
            
    start_time = time.time()
    for epoch in range(epochs):
        indices = np.arange(len(X_tr))
        np.random.shuffle(indices)
        for i in range(0, len(X_tr), batch_size):
            batch_idx = indices[i:i + batch_size]
            X_b = Tensor(X_tr[batch_idx], requires_grad=False)
            y_b = y_tr[batch_idx]
            
            model.zero_grad()
            logits = model(X_b)
            loss = loss_fn(logits, y_b)
            loss.backward()
            
            optimizer.step()
            if pruner is not None:
                pruner.step()
                
    elapsed_sec = time.time() - start_time
    _, test_acc = evaluate(model, loss_fn, X_te, y_te, batch_size=batch_size)
    
    actual_sparsity = pruner.get_sparsity() if pruner is not None else 0.0
    dense_flops, sparse_flops = compute_model_flops(model)
    
    return {
        "method": method,
        "target_sparsity": float(target_sparsity),
        "actual_sparsity": float(actual_sparsity),
        "test_accuracy": float(test_acc),
        "dense_flops": dense_flops,
        "sparse_flops": sparse_flops,
        "train_time_sec": float(elapsed_sec)
    }


def main():
    print("=" * 75)
    print(" AQUA Part 4: Pareto Sweep & Baseline Comparison")
    print("=" * 75)
    
    # Set global random seed for exact reproducibility across runs
    np.random.seed(42)
    
    X_tr, y_tr, X_te, y_te = load_digits_dataset(seed=42)
    
    sparsities = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
    methods = ["dynamic_saliency_regrowth", "dynamic_magnitude", "one_shot_magnitude"]
    
    results = []
    
    # 1. Dense baseline (S=0.0)
    print("\nRunning Dense Baseline (0% Sparsity)...")
    dense_res = run_experiment("dense_baseline", 0.0, X_tr, y_tr, X_te, y_te, seed=42)
    results.append(dense_res)
    print(f"Dense Baseline | Acc: {dense_res['test_accuracy']*100:.2f}% | FLOPs: {dense_res['dense_flops']}")
    
    # 2. Sweep across methods and sparsities
    for target_s in sparsities[1:]:
        for method in methods:
            print(f"Running {method:<26} | Target Sparsity: {target_s * 100:.0f}%...")
            res = run_experiment(method, target_s, X_tr, y_tr, X_te, y_te, seed=42)
            results.append(res)
            print(f"  --> Acc: {res['test_accuracy']*100:.2f}% | Actual Sparsity: {res['actual_sparsity']*100:.2f}% | FLOPs: {res['sparse_flops']}")
            
    # 3. Save Raw Artifacts
    os.makedirs("artifacts", exist_ok=True)
    json_path = os.path.join("artifacts", "pareto_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw metrics saved to {json_path}")
    
    # 4. Generate Pareto Plots
    os.makedirs("plots", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        "dense_baseline": "black",
        "one_shot_magnitude": "red",
        "dynamic_magnitude": "blue",
        "dynamic_saliency_regrowth": "green"
    }
    labels = {
        "dense_baseline": "Dense Baseline",
        "one_shot_magnitude": "One-Shot Magnitude",
        "dynamic_magnitude": "Dynamic Magnitude",
        "dynamic_saliency_regrowth": "Dynamic Saliency + Regrowth (Ours)"
    }
    
    # Ax1: Accuracy vs Sparsity
    for method in ["dense_baseline"] + methods:
        m_res = [r for r in results if r["method"] == method]
        x_val = [r["actual_sparsity"] * 100 for r in m_res]
        y_val = [r["test_accuracy"] * 100 for r in m_res]
        
        ax1.plot(x_val, y_val, marker='o', label=labels[method], color=colors[method], linewidth=2)
        
    ax1.set_xlabel("Model Sparsity (%)")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Part 4: Pareto Curve (Accuracy vs Sparsity)")
    ax1.grid(True)
    ax1.legend()
    
    # Ax2: Accuracy vs FLOPs
    for method in ["dense_baseline"] + methods:
        m_res = [r for r in results if r["method"] == method]
        x_val = [r["sparse_flops"] for r in m_res]
        y_val = [r["test_accuracy"] * 100 for r in m_res]
        
        ax2.plot(x_val, y_val, marker='s', label=labels[method], color=colors[method], linewidth=2)
        
    ax2.set_xlabel("FLOP Cost per Sample")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Part 4: FLOP Efficiency (Accuracy vs Compute)")
    ax2.grid(True)
    ax2.legend()
    
    plot_path = os.path.join("plots", "part4_pareto_curve.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Pareto curve plot saved to {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
