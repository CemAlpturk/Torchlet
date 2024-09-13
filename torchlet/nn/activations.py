from torchlet import Tensor
from torchlet.nn import Module


class ReLU(Module):
    """Rectified Linear Unit activation function."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation function."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"
