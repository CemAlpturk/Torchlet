import numpy as np

from torchlet import Tensor
from torchlet.nn import Module


class Linear(Module):
    """
    Linear layer.
    """

    weight: Tensor
    b: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        # TODO: Initialization methods
        super().__init__()

        if in_features <= 0:
            raise ValueError("in_features must be greater than 0")
        if out_features <= 0:
            raise ValueError("out_features must be greater than 0")

        self.weight = Tensor(
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
        z = x @ self.weight
        if self.b is not None:
            z += self.b

        return z

    def state_dict(self, prefix: str = "") -> dict[str, Tensor]:
        prefix = prefix + "." if prefix else ""
        state = {f"{prefix}weight": self.weight}
        if self.b is not None:
            state[f"{prefix}b"] = self.b
        return state

    def __repr__(self) -> str:
        return f"Linear(in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, bias={self.b is not None})"


class Dropout(Module):
    """
    Dropout layer.
    """

    prob: float
    _coeff: float

    def __init__(self, prob: float) -> None:
        super().__init__()

        if prob < 0 or prob >= 1:
            raise ValueError("Dropout probability must be in the range [0, 1)")
        self.prob = prob
        self._coeff = 1 / (1 - prob)

    def forward(self, x: Tensor) -> Tensor:

        if self.training:
            mask = np.random.uniform(0, 1, x.data.shape) > self.prob
            mask = mask * self._coeff
            mask = Tensor(mask, dtype=x.dtype)
            return x * mask

        return x

    def __repr__(self) -> str:
        return f"Dropout(prob={self.prob})"
