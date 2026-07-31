import numpy as np
from typing import Optional, Set, Tuple, Union, List


def unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reduces gradient dimensions to match parent target shape after forward broadcasting.

    When NumPy broadcasts arrays during forward operations (e.g., (N, D) + (D,)),
    the output gradient has shape (N, D). This function sums out broadcasted axes
    to reduce the gradient back to the parent's target shape (D,).

    Args:
        grad (np.ndarray): Incoming gradient array of shape (G_1, G_2, ..., G_k).
        target_shape (Tuple[int, ...]): Target shape (T_1, T_2, ..., T_m) of parent tensor.

    Returns:
        np.ndarray: Reduced gradient array matching target_shape exactly.
    """
    if grad.shape == target_shape:
        return grad
    
    if target_shape == () or target_shape == (1,):
        if target_shape == (1,) and grad.shape == (1,):
            return grad
        return np.sum(grad).reshape(target_shape)
    
    # 1. Sum out extra leading dimensions created by broadcasting
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = np.sum(grad, axis=tuple(range(ndim_diff)))
    
    # 2. Sum out dimensions where target_shape has size 1
    axes_to_sum = tuple(i for i, dim in enumerate(target_shape) if dim == 1)
    if axes_to_sum:
        grad = np.sum(grad, axis=axes_to_sum, keepdims=True)
        
    return grad.reshape(target_shape)


class Context:
    """Stores intermediate variables and backward function for an autograd node."""
    def __init__(self, op_name: str, backward_fn, saved_tensors: List['Tensor'] = None, saved_values: List = None):
        self.op_name = op_name
        self.backward_fn = backward_fn
        self.saved_tensors = saved_tensors if saved_tensors is not None else []
        self.saved_values = saved_values if saved_values is not None else []


class Tensor:
    """
    NumPy-backed differentiable Tensor supporting reverse-mode automatic differentiation.

    Attributes:
        data (np.ndarray): Underlying numerical data array (float64 by default).
        requires_grad (bool): Flag indicating if gradients should be computed.
        grad (Optional[np.ndarray]): Accumulated gradient array, initialized to None.
        _ctx (Optional[Context]): Autograd context holding backward function and parents.
        _parents (Set['Tensor']): Set of direct ancestor Tensors in computation graph.
    """
    def __init__(self, data: Union[np.ndarray, list, float, int], requires_grad: bool = False, dtype=np.float64):
        if isinstance(data, np.ndarray):
            self.data = data.astype(dtype, copy=False)
        else:
            self.data = np.array(data, dtype=dtype)
            
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._ctx: Optional[Context] = None
        self._parents: Set['Tensor'] = set()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def zero_grad(self):
        """Reset gradient buffer to None."""
        self.grad = None

    def backward(self, grad: Optional[np.ndarray] = None):
        """
        Executes reverse-mode automatic differentiation starting from this tensor.

        Builds a post-order Depth-First Search (DFS) topological ordering of ancestor
        tensors in the computation graph. Traverses nodes in reverse order, executing
        each backward function and accumulating gradients (+-) into parent tensors.

        Args:
            grad (Optional[np.ndarray]): Incoming gradient array from root loss.
                                         Defaults to ones array for scalar outputs.
        """
        if not self.requires_grad:
            raise RuntimeError("Called backward() on a Tensor with requires_grad=False.")

        if grad is None:
            if self.shape == () or self.shape == (1,):
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError("Grad argument must be specified for non-scalar Tensors.")
        else:
            if isinstance(grad, Tensor):
                grad = grad.data
            grad = np.array(grad, dtype=self.dtype)

        # Initialize root gradient
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad += grad

        # Topological sort via DFS
        topo: List['Tensor'] = []
        visited: Set['Tensor'] = set()

        def build_topo(node: 'Tensor'):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    build_topo(parent)
                topo.append(node)

        build_topo(self)

        # Backpropagation loop
        for node in reversed(topo):
            if node._ctx is None or node.grad is None:
                continue

            parent_grads = node._ctx.backward_fn(node._ctx, node.grad)
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)

            for parent, p_grad in zip(node._ctx.saved_tensors, parent_grads):
                if parent is not None and parent.requires_grad and p_grad is not None:
                    p_grad_unbroadcast = unbroadcast(p_grad, parent.shape)
                    if parent.grad is None:
                        parent.grad = p_grad_unbroadcast.copy()
                    else:
                        parent.grad += p_grad_unbroadcast

    def item(self) -> float:
        return float(self.data.item())

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    # Operator Overloading methods
    def __add__(self, other):
        from engine.ops import add
        return add(self, other)

    def __radd__(self, other):
        from engine.ops import add
        return add(other, self)

    def __sub__(self, other):
        from engine.ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        from engine.ops import sub
        return sub(other, self)

    def __mul__(self, other):
        from engine.ops import mul
        return mul(self, other)

    def __rmul__(self, other):
        from engine.ops import mul
        return mul(other, self)

    def __truediv__(self, other):
        from engine.ops import div
        return div(self, other)

    def __rtruediv__(self, other):
        from engine.ops import div
        return div(other, self)

    def __matmul__(self, other):
        from engine.ops import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        from engine.ops import matmul
        return matmul(other, self)

    def __neg__(self):
        from engine.ops import neg
        return neg(self)

    def __pow__(self, power):
        from engine.ops import pow_op
        return pow_op(self, power)

    def sum(self, axis=None, keepdims=False):
        from engine.ops import sum_op
        return sum_op(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        from engine.ops import mean_op
        return mean_op(self, axis=axis, keepdims=keepdims)

    def relu(self):
        from engine.ops import relu
        return relu(self)

    def gelu(self):
        from engine.ops import gelu
        return gelu(self)

    def sigmoid(self):
        from engine.ops import sigmoid
        return sigmoid(self)

    def tanh(self):
        from engine.ops import tanh
        return tanh(self)
