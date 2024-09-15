from typing import Sequence
from abc import ABC, abstractmethod

from torchlet.nn import Parameter


class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    """

    params: Sequence[Parameter]

    def __init__(self, params: Sequence[Parameter]) -> None:
        self.params = params

    def zero_grad(self) -> None:
        """
        Sets all gradients to zero for parameters that are being optimized.
        """
        for p in self.params:
            if p.value is None:
                # Don't know why this is here
                continue

            if hasattr(p.value, "grad"):

                p.value.grad = None

    @abstractmethod
    def step(self) -> None:
        """
        Updates the parameters based on the gradients.
        """
        ...
