from abc import ABC, abstractmethod
import numpy as np
from torchlet import Tensor


class Module(ABC):
    """
    Base class for all neural network modules.
    """

    training: bool
    name: str

    def __init__(self, name: str) -> None:
        self.training = True
        self.name = name

    def parameters(self) -> dict[str, Tensor]:
        """
        Return a dict of all learnable parameters.
        """
        return {}

    def zero_grad(self) -> None:
        """
        Sets the gradients of all parameters to zero.

        Returns:
            None
        """
        for p in self.parameters().values():
            p.grad = np.zeros_like(p.data)

    @abstractmethod
    def forward(self, *args: tuple, **kwargs: dict) -> Tensor:
        """
        Forward pass of the module.
        """
        raise NotImplementedError

    def train(self) -> None:
        """
        Set the module to training mode.
        """
        self.training = True

    def eval(self) -> None:
        """
        Set the module to evaluation mode.
        """
        self.training = False

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)
