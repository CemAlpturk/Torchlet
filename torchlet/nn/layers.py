import numpy as np

from torchlet import Tensor
from torchlet.nn import Module


class Linear(Module):
    """
    Linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        # TODO: Initialization methods

        if in_features <= 0:
            raise ValueError("in_features must be greater than 0")
        if out_features <= 0:
            raise ValueError("out_features must be greater than 0")

        self.W = Tensor(
            data=np.random.uniform(-1, 1, (in_features, out_features)),
            requires_grad=True,
        )
        if bias:
            self.b = Tensor(
                data=np.zeros((out_features,)),
                requires_grad=True,
            )
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        z = x @ self.W
        if self.b is not None:
            z += self.b

        return z

    def parameters(self) -> list[Tensor]:
        return [self.W] + ([self.b] if self.b is not None else [])

    def __repr__(self) -> str:
        return f"Linear({self.W.data.shape[0]}, {self.W.data.shape[1]}, bias={self.b is not None})"
