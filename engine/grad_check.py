"""
Finite-difference numerical gradient checking utility for autograd verification.
"""
import numpy as np
from typing import Callable, Tuple, List
from engine.tensor import Tensor


def compute_numerical_gradient(
    fn: Callable[[List[Tensor]], Tensor],
    inputs: List[Tensor],
    eps: float = 1e-6
) -> List[np.ndarray]:
    """
    Computes numerical gradient for a multi-tensor function using central finite differences:
    g_num_i = (f(x_i + eps) - f(x_i - eps)) / (2 * eps)

    Args:
        fn: Function taking a list of input Tensors and returning a scalar output Tensor.
        inputs: List of input Tensors with requires_grad=True.
        eps: Small step size for finite differences.

    Returns:
        List of np.ndarray numerical gradients corresponding to each input Tensor.
    """
    num_grads = []
    
    for inp in inputs:
        grad_arr = np.zeros_like(inp.data)
        it = np.nditer(inp.data, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            original_val = inp.data[idx]
            
            # Plus eps
            inp.data[idx] = original_val + eps
            out_plus = fn(inputs).item()
            
            # Minus eps
            inp.data[idx] = original_val - eps
            out_minus = fn(inputs).item()
            
            # Reset original value
            inp.data[idx] = original_val
            
            # Central difference
            grad_arr[idx] = (out_plus - out_minus) / (2.0 * eps)
            it.iternext()
            
        num_grads.append(grad_arr)
        
    return num_grads


def check_gradient(
    fn: Callable[[List[Tensor]], Tensor],
    inputs: List[Tensor],
    eps: float = 1e-6,
    tol: float = 1e-5
) -> Tuple[bool, float, List[np.ndarray], List[np.ndarray]]:
    """
    Compares analytical gradients computed by backward() against finite-difference gradients.

    Args:
        fn: Function returning scalar output Tensor.
        inputs: List of input Tensors.
        eps: Finite difference step size.
        tol: Maximum allowable relative error tolerance.

    Returns:
        Tuple of (passed: bool, max_relative_error: float, analytical_grads, numerical_grads)
    """
    # 1. Forward and analytical backward
    for inp in inputs:
        inp.zero_grad()
        inp.requires_grad = True
        
    out = fn(inputs)
    out.backward()
    
    analytical_grads = [inp.grad.copy() for inp in inputs]
    
    # 2. Numerical gradient computation
    numerical_grads = compute_numerical_gradient(fn, inputs, eps=eps)
    
    # 3. Compute maximum relative error across all tensors
    max_rel_error = 0.0
    for a_grad, n_grad in zip(analytical_grads, numerical_grads):
        denom = np.maximum(np.abs(a_grad), np.abs(n_grad))
        denom = np.maximum(denom, 1e-8)
        rel_error = np.abs(a_grad - n_grad) / denom
        max_rel_error = max(max_rel_error, float(np.max(rel_error)))
        
    passed = max_rel_error < tol
    return passed, max_rel_error, analytical_grads, numerical_grads
