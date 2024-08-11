from typing import Callable, Set, Union, TypeAlias
import numpy as np

# Define type alias
Arrayable: TypeAlias = Union["Value", float, np.ndarray]


class Value:
    """Stores a value and its gradient."""

    def __init__(
        self,
        data: float | list[float] | np.ndarray,
        _children: tuple["Value", ...] = (),
        _op: str = "",
    ) -> None:

        self.data = np.array(data)
        if len(self.data.shape) < 2:
            self.data = self.data.reshape(1, -1)

        self.grad = np.zeros_like(self.data, dtype=np.float64)

        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["Value"] = set(_children)
        self._op: str = _op

    def __add__(self, other: Arrayable) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Arrayable) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other: int | float) -> "Value":
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __matmul__(self, other: Arrayable) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(np.maximum(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def mean(self) -> "Value":
        out = Value(float(np.mean(self.data)), (self,), "mean")

        def _backward():
            grad_divisor = self.data.size
            self.grad += out.grad / grad_divisor

        out._backward = _backward
        return out

    def sum(self) -> "Value":
        out = Value(float(np.sum(self.data)), (self,), "sum")

        def _backward():
            self.grad += out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        # Topological order all of the children in the graph
        topo: list["Value"] = []
        visited = set()

        def build_topo(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        # self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def transpose(self) -> "Value":
        if isinstance(self.data, np.ndarray):
            return Value(self.data.T, (self,), "transpose")
        else:
            return self

    @property
    def T(self) -> "Value":
        return self.transpose()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape if isinstance(self.data, np.ndarray) else ()

    def __neg__(self) -> "Value":
        return self * -1

    def __radd__(self, other: Arrayable) -> "Value":
        return self + other

    def __sub__(self, other: Arrayable) -> "Value":
        return self + (-other)

    def __rsub__(self, other: Arrayable) -> "Value":
        return other + (-self)

    def __rmul__(self, other: Arrayable) -> "Value":
        return self * other

    def __truediv__(self, other: Arrayable) -> "Value":
        return self * other**-1

    def __rtruediv__(self, other: Arrayable) -> "Value":
        return other * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
