from typing import Any
from torchlet.nn import Module


class Sequential(Module):
    """Sequential container."""

    def __init__(self, *args: Module) -> None:
        super().__init__()
        self._modules = {str(i): module for i, module in enumerate(args)}

    def forward(self, x: Any) -> Any:
        for module in self._modules.values():
            x = module(x)
        return x

    def __repr__(self) -> str:
        return f"Sequential({', '.join([str(module) for module in self._modules.values()])})"
