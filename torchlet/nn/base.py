from abc import ABC, abstractmethod
import numpy as np
from torchlet import Tensor


class Module(ABC):
    """
    Base class for all neural network modules.
    """

    def parameters(self) -> list[Tensor]:
        """
        Return a list of all learnable parameters.
        """
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
        """
        Forward pass of the module.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)
