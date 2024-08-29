from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np
from torchlet import Tensor


class Module(ABC):
    """
    Base class for all neural network modules.
    """

    training: bool

    def __init__(self) -> None:
        self.training = True

    def state_dict(self, prefix: str = "") -> dict[str, Tensor]:
        """
        Returns the state of the module as a dictionary.
        """
        return {}

    def parameters(self) -> Iterator[Tensor]:
        """
        Return an iterator over the parameters of the module.
        """
        for param in self.state_dict().values():
            yield param

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
