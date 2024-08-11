import numpy as np
from torchlet.engine import Value


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad *= np.zeros_like(p.data)

    def parameters(self) -> list[Value]:
        return []

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    """
    Linear layer.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.W = Value(
            np.random.uniform(-1, 1, (in_features, out_features)),
            _op=f"Linear/W: {in_features}x{out_features}",
        )
        self.b = Value(np.zeros((1, out_features)), _op=f"Linear/b: {out_features}")

    def forward(self, x: Value) -> Value:
        return x @ self.W + self.b

    def parameters(self) -> list[Value]:
        return [self.W, self.b]

    def __repr__(self) -> str:
        return f"Linear({self.W.data.shape[0]}, {self.W.data.shape[1]})"


class ReLU(Module):
    """
    ReLU activation function.
    """

    def forward(self, x: Value) -> Value:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sequential(Module):
    """
    Sequential container.
    """

    def __init__(self, *layers: Module) -> None:
        self.layers = layers

    def forward(self, x: Value) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def __repr__(self) -> str:
        return f"Sequential({', '.join([str(layer) for layer in self.layers])})"
