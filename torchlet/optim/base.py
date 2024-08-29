from abc import ABC, abstractmethod
from torchlet import Tensor


class Optimizer(ABC):
    """
    Base class for all optimizers.
    """

    params: list[Tensor]

    def __init__(self, params: list[Tensor]) -> None:
        self.params = params

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
