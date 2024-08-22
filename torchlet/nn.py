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
        z = x @ self.W

        # Manually add the backward function
        # This is due to the broadcast of the bias term
        # Until I find a better way to do this

        result = z.data + self.b.data

        def _backward(dy: np.ndarray) -> None:
            z.grad += dy
            if z.shape != self.b.shape:
                dy = dy.sum(axis=0)
            self.b.grad += dy

        out = Tensor(
            data=result,
            dtype=z.dtype,
            name="bias_broadcast",
            backward_fn=_backward,
        )

        return out

    def parameters(self) -> list[Tensor]:
        return [self.W, self.b]
