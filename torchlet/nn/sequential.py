from collections import defaultdict

from torchlet import Tensor
from torchlet.nn import Module


class Sequential(Module):
    """
    Model that chains multiple modules together.
    """

    modules: tuple[Module, ...]

    def __init__(self, *args: Module) -> None:
        self.modules = args
        self._adjust_module_names()
        self.train()

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

    def parameters(self) -> dict[str, Tensor]:
        params = {}
        for module in self.modules:
            # params.extend(module.parameters())
            params.update(module.parameters())
        return params

    def train(self) -> None:
        for module in self.modules:
            module.train()
        self.training = True

    def eval(self) -> None:
        for module in self.modules:
            module.eval()
        self.training = False

    def __getitem__(self, index: int) -> Module:
        return self.modules[index]

    def __repr__(self) -> str:
        return f"Sequential({', '.join([str(module) for module in self.modules])})"

    def _adjust_module_names(self) -> None:
        names = defaultdict(int)
        for module in self.modules:
            names[module.__class__.__name__] += 1
            module.name = (
                f"{module.__class__.__name__}_{names[module.__class__.__name__]}"
            )
