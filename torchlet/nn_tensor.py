from abc import ABC, abstractmethod
import numpy as np
from torchlet.tensor import Tensor


class Module(ABC):
    """Base class for all neural network modules."""

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = Tensor(np.zeros_like(p.array))

    def parameters(self) -> list[Tensor]:
        return []

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Linear(Module):
    """
    Linear layer.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.W = Tensor(
            np.random.uniform(-1, 1, (in_features, out_features)),
            name=f"Linear/W: {in_features}x{out_features}",
        )
        self.b = Tensor(np.zeros((1, out_features)), name=f"Linear/b: {out_features}")

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def parameters(self) -> list[Tensor]:
        return [self.W, self.b]

    def __repr__(self) -> str:
        return f"Linear({self.W.array.shape[0]}, {self.W.array.shape[1]})"


class ReLU(Module):
    """
    ReLU activation function.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sequential(Module):
    """
    Sequential container.
    """

    def __init__(self, *args: Module) -> None:
        self.modules = args

    def parameters(self) -> list[Tensor]:
        return [p for module in self.modules for p in module.parameters()]

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

    def __repr__(self) -> str:
        return f"Sequential({', '.join([str(module) for module in self.modules])})"
