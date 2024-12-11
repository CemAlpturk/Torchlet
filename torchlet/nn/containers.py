from typing import Any, Iterator
from torchlet.nn import Module


class Sequential(Module):
    """Sequential container."""

    def __init__(self, *args: Module) -> None:
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, x: Any) -> Any:
        for module in self:
            x = module(x)
        return x

    def __repr__(self) -> str:
        return f"Sequential({', '.join([str(module) for module in self._modules.values()])})"
