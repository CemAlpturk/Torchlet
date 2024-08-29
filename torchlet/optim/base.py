from abc import ABC, abstractmethod
from typing import Iterator
from torchlet import Tensor


class Optimizer(ABC):
    """
    Base class for all optimizers.
    """

    params: list[Tensor]

    def __init__(self, params: Iterator[Tensor]) -> None:
        self.params = list(params)

    def zero_grad(self) -> None:
        """
        Sets the gradients of all parameters to zero.
        """

        for param in self.params:
            param.zero_grad()

    @abstractmethod
    def step(self) -> None:
        """
        Updates the parameters.
        """
        pass
