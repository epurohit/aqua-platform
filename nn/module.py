"""
Base Module, Parameter, and Sequential container abstractions.
"""
from typing import List, Union
from engine.tensor import Tensor


class Parameter(Tensor):
    """A trainable parameter Tensor (requires_grad=True by default)."""
    def __init__(self, data, dtype=None):
        super().__init__(data, requires_grad=True, dtype=dtype if dtype is not None else float)


class Module:
    """
    Base Neural Network Module.
    Tracks parameters recursively.
    """
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def parameters(self) -> List[Parameter]:
        """
        Recursively collects and returns all trainable Parameters in the module.

        Returns:
            List[Parameter]: List of all Parameter instances.
        """
        params = []
        for name, param in self.__dict__.items():
            if isinstance(param, Parameter):
                params.append(param)
            elif isinstance(param, Module):
                params.extend(param.parameters())
            elif isinstance(param, list):
                for item in param:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Sequential(Module):
    """Executes a list of sub-modules sequentially."""
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)

    def parameters(self) -> List[Parameter]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
