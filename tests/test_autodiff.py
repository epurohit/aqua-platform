"""
Unit tests for Reverse-Mode Autodiff Engine & Gradient Checker
"""
import pytest
import numpy as np
from engine.tensor import Tensor
from engine.grad_check import check_gradient


def test_add_grad():
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = Tensor(np.random.randn(3, 4), requires_grad=True)
    
    def fn(inputs):
        return (inputs[0] + inputs[1]).sum()
        
    passed, max_err, _, _ = check_gradient(fn, [x, y])
    assert passed, f"Add gradient check failed with max rel err: {max_err}"


def test_broadcasting_add_grad():
    x = Tensor(np.random.randn(5, 4), requires_grad=True)
    b = Tensor(np.random.randn(4,), requires_grad=True)
    
    def fn(inputs):
        return (inputs[0] + inputs[1]).sum()
        
    passed, max_err, _, _ = check_gradient(fn, [x, b])
    assert passed, f"Broadcasting add gradient check failed with max rel err: {max_err}"


def test_matmul_grad():
    X = Tensor(np.random.randn(4, 8), requires_grad=True)
    W = Tensor(np.random.randn(8, 6), requires_grad=True)
    
    def fn(inputs):
        return (inputs[0] @ inputs[1]).sum()
        
    passed, max_err, _, _ = check_gradient(fn, [X, W])
    assert passed, f"Matmul gradient check failed with max rel err: {max_err}"


def test_activations_grad():
    x = Tensor(np.random.randn(4, 4), requires_grad=True)
    
    # ReLU
    passed, err, _, _ = check_gradient(lambda inputs: inputs[0].relu().sum(), [x])
    assert passed, f"ReLU grad check failed: {err}"
    
    # GELU
    passed, err, _, _ = check_gradient(lambda inputs: inputs[0].gelu().sum(), [x])
    assert passed, f"GELU grad check failed: {err}"
    
    # Sigmoid
    passed, err, _, _ = check_gradient(lambda inputs: inputs[0].sigmoid().sum(), [x])
    assert passed, f"Sigmoid grad check failed: {err}"
    
    # Tanh
    passed, err, _, _ = check_gradient(lambda inputs: inputs[0].tanh().sum(), [x])
    assert passed, f"Tanh grad check failed: {err}"


def test_softmax_cross_entropy_grad():
    from engine.ops import softmax_cross_entropy
    logits = Tensor(np.random.randn(8, 5), requires_grad=True)
    targets = np.array([0, 2, 1, 4, 3, 2, 0, 1])
    
    def fn(inputs):
        return softmax_cross_entropy(inputs[0], targets)
        
    passed, max_err, _, _ = check_gradient(fn, [logits])
    assert passed, f"Softmax cross-entropy gradient check failed with max rel err: {max_err}"
