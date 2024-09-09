from typing import Callable, Iterable
import math


# Scalar operations
def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Check if x is less than y."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if x is equal to y."""
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    """Check if x is within tol of y."""
    return abs(x - y) < tol


def sigmoid(x: float) -> float:
    """Compute the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU function."""
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    """Compute the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the reciprocal."""
    return 1.0 / x


# Simple backward implementations
def log_back(x: float, grad: float) -> float:
    """Backward pass for the natural logarithm."""
    return grad / x


def inv_back(x: float, grad: float) -> float:
    """Backward pass for the reciprocal."""
    return -grad / (x * x)


def relu_back(x: float, grad: float) -> float:
    """Backward pass for the ReLU function."""
    return grad if x > 0.0 else 0.0


def sigmoid_back(x: float, grad: float) -> float:
    """Backward pass for the sigmoid function."""
    sig = sigmoid(x)
    return grad * sig * (1 - sig)


def exp_back(x: float, grad: float) -> float:
    """Backward pass for the exponential function."""
    return grad * math.exp(x)


# Functional ops
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Map a function over an iterable."""

    def mapped(xs: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in xs]

    return mapped


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Zip two iterables with a function."""

    def zipped(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(xs, ys)]

    return zipped


def reduce(
    fn: Callable[[float, float], float], init: float
) -> Callable[[Iterable[float]], float]:
    """Reduce an iterable with a function and initial value."""

    def reduced(xs: Iterable[float]) -> float:
        total = init
        for x in xs:
            total = fn(total, x)

        return total

    return reduced


# Simple functional utilities
def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list."""
    return map(neg)(xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists elementwise."""
    return zipWith(add)(xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum all elements in a list."""
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    """Multiply all elements in a list."""
    return reduce(mul, 1.0)(xs)
