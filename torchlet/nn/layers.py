import numpy as np

from torchlet import Tensor
from torchlet.nn import Module


class Linear(Module):
    """
    Linear layer.
    """

    W: Tensor
    b: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str = "Linear",
    ) -> None:
        # TODO: Initialization methods
        super().__init__(name=name)

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

    def parameters(self) -> dict[str, Tensor]:
        # return [self.W] + ([self.b] if self.b is not None else [])
        params = {f"{self.name}/W": self.W}
        if self.b is not None:
            params[f"{self.name}/b"] = self.b
        return params

    def __repr__(self) -> str:
        return f"{self.name}({self.W.data.shape[0]}, {self.W.data.shape[1]}, bias={self.b is not None})"


class Dropout(Module):
    """
    Dropout layer.
    """

    prob: float
    _coeff: float

    def __init__(
        self,
        prob: float,
        name: str = "Dropout",
    ) -> None:
        super().__init__(name=name)

        if prob < 0 or prob >= 1:
            raise ValueError("Dropout probability must be in the range [0, 1)")
        self.prob = prob
        self._coeff = 1 / (1 - prob)
        self.name = name or "Dropout"

    def forward(self, x: Tensor) -> Tensor:

        if self.training:
            mask = np.random.uniform(0, 1, x.data.shape) > self.prob
            mask = mask * self._coeff
            mask = Tensor(mask, dtype=x.dtype)
            return x * mask

        return x

    def __repr__(self) -> str:
        return f"{self.name}(p={self.prob})"
