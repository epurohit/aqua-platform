"""
Correctness test for masked weight gradient treatment and optimizer state integrity.
"""
import pytest
import numpy as np
from engine.tensor import Tensor
from engine.grad_check import check_gradient


def test_masked_weight_gradient_is_zero():
    """
    Verifies that a masked weight (M_ij = 0) produces zero forward contribution
    and receives exactly zero gradient wrt parameter W.
    """
    np.random.seed(42)
    X = Tensor(np.random.randn(4, 5), requires_grad=True)
    W = Tensor(np.random.randn(5, 3), requires_grad=True)
    
    # Mask out 2nd and 4th columns of W (index 1 and 3 in input dimension)
    mask = np.ones((5, 3), dtype=np.float64)
    mask[1, :] = 0.0
    mask[3, :] = 0.0
    
    # Effective weight: W_eff = W * mask
    W_eff = W * Tensor(mask, requires_grad=False)
    out = (X @ W_eff).sum()
    out.backward()
    
    # 1. Gradient wrt parameter W must be zero at masked positions
    assert W.grad is not None
    assert np.allclose(W.grad[1, :], 0.0), f"Expected 0.0 gradient at row 1, got {W.grad[1, :]}"
    assert np.allclose(W.grad[3, :], 0.0), f"Expected 0.0 gradient at row 3, got {W.grad[3, :]}"
    
    # 2. Unmasked positions must match standard matrix multiplication gradient (X.T @ 1)
    expected_unmasked_grad = X.data.T @ np.ones((4, 3))
    assert np.allclose(W.grad[0, :], expected_unmasked_grad[0, :])
    assert np.allclose(W.grad[2, :], expected_unmasked_grad[2, :])
    assert np.allclose(W.grad[4, :], expected_unmasked_grad[4, :])


def test_masked_weight_numerical_gradient_check():
    """
    Verifies finite-difference numerical gradient matches analytical gradient for masked parameters.
    """
    np.random.seed(42)
    X = Tensor(np.random.randn(3, 4), requires_grad=True)
    W = Tensor(np.random.randn(4, 3), requires_grad=True)
    mask = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    def fn(inputs):
        inp_W = inputs[1]
        W_eff = inp_W * Tensor(mask, requires_grad=False)
        return (inputs[0] @ W_eff).relu().sum()
        
    passed, max_err, analytical_grads, numerical_grads = check_gradient(fn, [X, W])
    assert passed, f"Masked numerical gradient check failed with max rel err: {max_err}"
    
    # Verify numerical gradient is also zero for masked weights
    W_num_grad = numerical_grads[1]
    assert np.allclose(W_num_grad[mask == 0.0], 0.0, atol=1e-6)
