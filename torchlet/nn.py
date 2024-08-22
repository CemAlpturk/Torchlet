from abc import ABC, abstractmethod
import numpy as np

from torchlet.tensor import Tensor


class Module(ABC):
    """Base class for all neural network modules."""

    def parameters(self) -> list[Tensor]:
        """Return a list of all learnable parameters."""
        return []

    def zero_grad(self) -> None:
        """
        Sets the gradients of all parameters to zero.

        Returns:
            None
        """
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    @abstractmethod
    def forward(self, *args: tuple, **kwargs: dict) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)


class Linear(Module):
    """
    Linear layer.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.W = Tensor(data=np.random.uniform(-1, 1, (in_features, out_features)))
        self.b = Tensor(data=np.zeros((out_features,)))

    def forward(self, x: Tensor) -> Tensor:
        # z = x @ self.W

        # # Manually add the backward function
        # # This is due to the broadcast of the bias term
        # # Until I find a better way to do this

        # result = z.data + self.b.data

        # def _backward(dy: np.ndarray) -> None:
        #     z.grad += dy
        #     if z.shape != self.b.shape:
        #         dy = dy.sum(axis=0)
        #     self.b.grad += dy

        # out = Tensor(
        #     data=result,
        #     dtype=z.dtype,
        #     name="bias_broadcast",
        #     backward_fn=_backward,
        # )

        # return out

        return x @ self.W + self.b

    def parameters(self) -> list[Tensor]:
        return [self.W, self.b]

    def __repr__(self) -> str:
        return f"Linear({self.W.data.shape[0]}, {self.W.data.shape[1]})"


class ReLU(Module):

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        self.modules = args

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

    def parameters(self) -> list[Tensor]:
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def __repr__(self) -> str:
        return f"Sequential({', '.join([str(module) for module in self.modules])})"
