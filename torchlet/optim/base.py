from abc import ABC, abstractmethod
from torchlet import Tensor


class Optimizer(ABC):
    """
    Base class for all optimizers.
    """

    params: dict[str, Tensor]

    def __init__(self, params: dict[str, Tensor]) -> None:
        self.params = params

    def zero_grad(self) -> None:
        """
        Sets the gradients of all parameters to zero.
        """

        for param in self.params.values():
            param.zero_grad()

    @abstractmethod
    def step(self) -> None:
        """
        Updates the parameters.
        """
        pass
