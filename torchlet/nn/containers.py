from typing import Any, Iterator, Iterable
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


class ModuleList(Module):
    """ModuleList container."""

    def __init__(self, modules: Iterable[Module]) -> None:
        super().__init__()

        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)


class ModuleDict(Module):
    """ModuleDict container."""

    def __init__(self, modules: dict[str, Module]) -> None:
        super().__init__()

        for key, module in modules.items():
            self.add_module(key, module)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)
