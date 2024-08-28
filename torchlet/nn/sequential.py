from torchlet import Tensor
from torchlet.nn import Module


class Sequential(Module):
    """
    Model that chains multiple modules together.
    """

    modules: tuple[Module, ...]

    def __init__(self, *args: Module) -> None:
        self.modules = args
        self.train()

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

    def parameters(self) -> list[Tensor]:
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def train(self) -> None:
        for module in self.modules:
            module.train()
        self.training = True

    def eval(self) -> None:
        for module in self.modules:
            module.eval()
        self.training = False

    def __repr__(self) -> str:
        return f"Sequential({', '.join([str(module) for module in self.modules])})"
