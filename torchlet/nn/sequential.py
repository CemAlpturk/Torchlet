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

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

    def state_dict(self) -> dict[str, Tensor]:
        state = {}
        counts = defaultdict(int)
        for module in self.modules:
            name = module.__class__.__name__
            counts[name] += 1
            count = counts[name]
            prefix = f"{name}{count}"
            state.update(module.state_dict(prefix=prefix))
        return state

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
        s = "Sequential(\n"
        for i, module in enumerate(self.modules):
            s += f"    ({i}) : {module}\n"
        s += ")"
        return s
