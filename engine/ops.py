"""
Differentiable mathematical and activation operations for reverse-mode autograd engine.
"""
import numpy as np
from typing import Union, Tuple, Optional
from engine.tensor import Tensor, Context, unbroadcast


def _ensure_tensor(x: Union[Tensor, np.ndarray, float, int]) -> Tensor:
    """Helper to wrap raw scalars or arrays into non-differentiable Tensors."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False)


def add(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """
    Element-wise addition: out = a + b
    Backward: da = grad, db = grad
    """
    a_t = _ensure_tensor(a)
    b_t = _ensure_tensor(b)
    
    out_data = a_t.data + b_t.data
    requires_grad = a_t.requires_grad or b_t.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad)
    
    if requires_grad:
        out._parents = {a_t, b_t}
        def backward_fn(ctx: Context, grad: np.ndarray):
            return grad, grad
        out._ctx = Context("add", backward_fn, saved_tensors=[a_t, b_t])
    return out


def sub(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """
    Element-wise subtraction: out = a - b
    Backward: da = grad, db = -grad
    """
    a_t = _ensure_tensor(a)
    b_t = _ensure_tensor(b)
    
    out_data = a_t.data - b_t.data
    requires_grad = a_t.requires_grad or b_t.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad)
    
    if requires_grad:
        out._parents = {a_t, b_t}
        def backward_fn(ctx: Context, grad: np.ndarray):
            return grad, -grad
        out._ctx = Context("sub", backward_fn, saved_tensors=[a_t, b_t])
    return out


def mul(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """
    Element-wise multiplication: out = a * b
    Backward: da = grad * b, db = grad * a
    """
    a_t = _ensure_tensor(a)
    b_t = _ensure_tensor(b)
    
    out_data = a_t.data * b_t.data
    requires_grad = a_t.requires_grad or b_t.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad)
    
    if requires_grad:
        out._parents = {a_t, b_t}
        def backward_fn(ctx: Context, grad: np.ndarray):
            return grad * b_t.data, grad * a_t.data
        out._ctx = Context("mul", backward_fn, saved_tensors=[a_t, b_t])
    return out


def div(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """
    Element-wise division: out = a / b
    Backward: da = grad / b, db = -grad * a / (b^2)
    """
    a_t = _ensure_tensor(a)
    b_t = _ensure_tensor(b)
    
    out_data = a_t.data / b_t.data
    requires_grad = a_t.requires_grad or b_t.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad)
    
    if requires_grad:
        out._parents = {a_t, b_t}
        def backward_fn(ctx: Context, grad: np.ndarray):
            ga = grad / b_t.data
            gb = -grad * a_t.data / (b_t.data ** 2)
            return ga, gb
        out._ctx = Context("div", backward_fn, saved_tensors=[a_t, b_t])
    return out


def neg(a: Tensor) -> Tensor:
    """Unary negation: out = -a, Backward: da = -grad"""
    out_data = -a.data
    out = Tensor(out_data, requires_grad=a.requires_grad)
    if a.requires_grad:
        out._parents = {a}
        def backward_fn(ctx: Context, grad: np.ndarray):
            return -grad,
        out._ctx = Context("neg", backward_fn, saved_tensors=[a])
    return out


def pow_op(a: Tensor, power: Union[float, int]) -> Tensor:
    """Scalar power: out = a^p, Backward: da = grad * p * a^(p-1)"""
    out_data = a.data ** power
    out = Tensor(out_data, requires_grad=a.requires_grad)
    if a.requires_grad:
        out._parents = {a}
        def backward_fn(ctx: Context, grad: np.ndarray):
            return grad * power * (a.data ** (power - 1)),
        out._ctx = Context("pow", backward_fn, saved_tensors=[a], saved_values=[power])
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication: out = a @ b
    Backward: da = grad @ b^T, db = a^T @ grad
    """
    a_t = _ensure_tensor(a)
    b_t = _ensure_tensor(b)
    
    out_data = a_t.data @ b_t.data
    requires_grad = a_t.requires_grad or b_t.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad)
    
    if requires_grad:
        out._parents = {a_t, b_t}
        def backward_fn(ctx: Context, grad: np.ndarray):
            b_transp = np.swapaxes(b_t.data, -1, -2)
            a_transp = np.swapaxes(a_t.data, -1, -2)
            return grad @ b_transp, a_transp @ grad
        out._ctx = Context("matmul", backward_fn, saved_tensors=[a_t, b_t])
    return out


def sum_op(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Sum reduction over specified axes."""
    out_data = np.sum(a.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a.requires_grad)
    
    if a.requires_grad:
        out._parents = {a}
        def backward_fn(ctx: Context, grad: np.ndarray):
            target_grad = grad
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else axis
                shape = list(a.shape)
                for ax in sorted(axes):
                    if ax < 0:
                        ax = len(shape) + ax
                    shape[ax] = 1
                target_grad = grad.reshape(shape)
            return np.broadcast_to(target_grad, a.shape),
        out._ctx = Context("sum", backward_fn, saved_tensors=[a])
    return out


def mean_op(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """Mean reduction over specified axes."""
    out_data = np.mean(a.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a.requires_grad)
    
    if a.requires_grad:
        out._parents = {a}
        def backward_fn(ctx: Context, grad: np.ndarray):
            target_grad = grad
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else axis
                shape = list(a.shape)
                for ax in sorted(axes):
                    if ax < 0:
                        ax = len(shape) + ax
                    shape[ax] = 1
                target_grad = grad.reshape(shape)
            numel = a.data.size / out_data.size
            return np.broadcast_to(target_grad, a.shape) / numel,
        out._ctx = Context("mean", backward_fn, saved_tensors=[a])
    return out


def relu(a: Tensor) -> Tensor:
    """Rectified Linear Unit: out = max(a, 0), Backward: da = grad * (a > 0)"""
    out_data = np.maximum(a.data, 0.0)
    out = Tensor(out_data, requires_grad=a.requires_grad)
    if a.requires_grad:
        out._parents = {a}
        def backward_fn(ctx: Context, grad: np.ndarray):
            return grad * (a.data > 0.0),
        out._ctx = Context("relu", backward_fn, saved_tensors=[a])
    return out


def gelu(a: Tensor) -> Tensor:
    """
    Gaussian Error Linear Unit (GELU) with exact derivative formula.
    Forward: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    k = np.sqrt(2.0 / np.pi)
    x = a.data
    inner = k * (x + 0.044715 * (x ** 3))
    tanh_inner = np.tanh(inner)
    out_data = 0.5 * x * (1.0 + tanh_inner)
    
    out = Tensor(out_data, requires_grad=a.requires_grad)
    if a.requires_grad:
        out._parents = {a}
        def backward_fn(ctx: Context, grad: np.ndarray):
            dtanh = 1.0 - tanh_inner ** 2
            dinner_dx = k * (1.0 + 3.0 * 0.044715 * (x ** 2))
            dx = 0.5 * (1.0 + tanh_inner) + 0.5 * x * dtanh * dinner_dx
            return grad * dx,
        out._ctx = Context("gelu", backward_fn, saved_tensors=[a])
    return out


def sigmoid(a: Tensor) -> Tensor:
    """Sigmoid activation: out = 1 / (1 + exp(-x)), Backward: da = grad * s * (1 - s)"""
    s = 1.0 / (1.0 + np.exp(-np.clip(a.data, -500.0, 500.0)))
    out = Tensor(s, requires_grad=a.requires_grad)
    if a.requires_grad:
        out._parents = {a}
        def backward_fn(ctx: Context, grad: np.ndarray):
            return grad * s * (1.0 - s),
        out._ctx = Context("sigmoid", backward_fn, saved_tensors=[a])
    return out


def tanh(a: Tensor) -> Tensor:
    """Hyperbolic tangent: out = tanh(x), Backward: da = grad * (1 - tanh(x)^2)"""
    t = np.tanh(a.data)
    out = Tensor(t, requires_grad=a.requires_grad)
    if a.requires_grad:
        out._parents = {a}
        def backward_fn(ctx: Context, grad: np.ndarray):
            return grad * (1.0 - t ** 2),
        out._ctx = Context("tanh", backward_fn, saved_tensors=[a])
    return out


def softmax_cross_entropy(logits: Tensor, targets: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Numerically stable Softmax Cross-Entropy Loss.
    `logits`: (N, C) or (C,) logits tensor
    `targets`: (N, C) one-hot array/tensor or (N,) / () integer class indices
    """
    logits_data = logits.data
    targets_data = targets.data if isinstance(targets, Tensor) else np.array(targets)
    
    is_1d = (logits_data.ndim == 1)
    if is_1d:
        logits_data = np.expand_dims(logits_data, axis=0)
    if targets_data.ndim == 0:
        targets_data = np.expand_dims(targets_data, axis=0)
        
    N = logits_data.shape[0]
    shift_logits = logits_data - np.max(logits_data, axis=-1, keepdims=True)
    exps = np.exp(shift_logits)
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    
    if targets_data.ndim == 1:
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(N), targets_data.astype(int)] = 1.0
    else:
        one_hot = targets_data
        
    loss_val = -np.mean(np.sum(one_hot * np.log(np.maximum(probs, 1e-15)), axis=-1))
    out = Tensor(loss_val, requires_grad=logits.requires_grad)
    
    if logits.requires_grad:
        out._parents = {logits}
        def backward_fn(ctx: Context, grad: np.ndarray):
            dlogits = (probs - one_hot) / N * grad
            if is_1d:
                dlogits = np.squeeze(dlogits, axis=0)
            return dlogits,
        out._ctx = Context("softmax_cross_entropy", backward_fn, saved_tensors=[logits])
    return out
