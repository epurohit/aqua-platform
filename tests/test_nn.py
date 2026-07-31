"""
Unit tests for Neural Network layers, Adam optimizer, and training convergence.
"""
import pytest
import numpy as np
from engine.tensor import Tensor
from nn.layers import Linear, ReLU, GELU
from nn.module import Sequential
from nn.loss import CrossEntropyLoss
from train.optimizer import Adam, SGD
from train.dataset import load_digits_dataset
from train.trainer import train_epoch, evaluate


def test_linear_layer_forward_backward():
    np.random.seed(42)
    layer = Linear(10, 5)
    x = Tensor(np.random.randn(8, 10), requires_grad=True)
    
    out = layer(x)
    assert out.shape == (8, 5)
    
    loss = out.sum()
    loss.backward()
    
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (10, 5)
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (5,)


def test_adam_optimizer_step():
    np.random.seed(42)
    layer = Linear(4, 2)
    opt = Adam(layer.parameters(), lr=0.01)
    
    initial_w = layer.weight.data.copy()
    x = Tensor(np.random.randn(2, 4), requires_grad=True)
    out = layer(x).sum()
    out.backward()
    
    opt.step()
    updated_w = layer.weight.data
    
    assert not np.allclose(initial_w, updated_w), "Adam step failed to update weights"


def test_part2_dense_model_convergence():
    """
    Verifies that a 2-layer MLP trained with Adam converges on Digits dataset
    achieving >90% test accuracy without NaNs.
    """
    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_digits_dataset(seed=42)
    
    model = Sequential(
        Linear(64, 32),
        ReLU(),
        Linear(32, 10)
    )
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = CrossEntropyLoss()
    
    initial_loss, _ = evaluate(model, loss_fn, X_test, y_test)
    
    # Train for 20 epochs
    for epoch in range(20):
        train_loss, train_acc = train_epoch(model, optimizer, loss_fn, X_train, y_train, batch_size=32)
        assert not np.isnan(train_loss), f"NaN loss encountered at epoch {epoch}"
        
    final_loss, final_acc = evaluate(model, loss_fn, X_test, y_test)
    
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
    assert final_acc > 0.90, f"Expected >90% test accuracy, got {final_acc:.2%}"
