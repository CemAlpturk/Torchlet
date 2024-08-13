import numbers
import uuid
import logging

import numpy as np
import numpy.typing
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

DEFAULT_TYPE = np.float32


class Tensor:
    """
    N-dimensional grid of numbers.
    """

    def __init__(
        self,
        array: ArrayLike,
        requires_grad: bool = True,
        args: tuple["Tensor", ...] | None = None,
        dtype: np.typing.DTypeLike | None = None,
        name: str | None = None,
        _id: int | None = None,
    ) -> None:
        """
        Initialize a new Tensor object.
        TODO: docstring
        """

        if isinstance(array, Tensor):
            self = array
            return

        self._id = _id or uuid.uuid4().int
        self._parents = None

        self.array = np.array(array)

        if dtype is None:
            dtype = DEFAULT_TYPE

        self.array = self.array.astype(dtype)
        self.grad = None

        self.requires_grad = requires_grad
        self.args = args

        self._backward = lambda: None
        self.name = name

    def backward(self) -> None:
        """
        Compute the gradient of the tensor.
        """
        pass

    def __hash__(self) -> int:
        return self._id

    def __add__(self, other: float | "Tensor") -> "Tensor":
        """
        Add two tensors.
        """
        if isinstance(other, numbers.Number):
            # The gradient of the number can be ignored.
            out = Tensor(
                array=self.array + other,
                args=(self,),
                name=f"+ {other}",
            )

            def _backward() -> None:
                self.grad += out.grad

            out._backward = _backward

        elif isinstance(other, Tensor):
            # The gradient of the tensor must be computed.
            out = Tensor(
                array=self.array + other.array,
                args=(self, other),
                name="+",
            )

            def _backward() -> None:
                self.grad += out.grad
                other.grad += out.grad

            out._backward = _backward

        else:
            raise NotImplementedError(f"Cannot add {type(self)} and {type(other)}")

        return out

    def __radd__(self, other: float | "Tensor") -> "Tensor":
        return self + other

    def __iadd__(self, other: float | "Tensor") -> "Tensor":
        return self + other

    def __sub__(self, other: float | "Tensor") -> "Tensor":
        """
        Subtract two tensors.
        """
        if isinstance(other, numbers.Number):
            # The gradient of the number can be ignored.
            out = Tensor(
                array=self.array - other,
                args=(self,),
                name=f"- {other}",
            )

            def _backward() -> None:
                self.grad += out.grad

            out._backward = _backward

        elif isinstance(other, "Tensor"):
            # The gradient of the tensor must be computed.
            out = Tensor(
                array=self.array - other.array,
                args=(self, other),
                name="-",
            )

            def _backward() -> None:
                self.grad += out.grad
                other.grad -= out.grad

            out._backward = _backward

        else:
            raise NotImplementedError(f"Cannot subtract {type(self)} and {type(other)}")

        return out

    def __rsub__(self, other: float | "Tensor") -> "Tensor":
        return -self.__sub__(other)

    def __isub__(self, other: float | "Tensor") -> "Tensor":
        return self.__sub__(other)

    def __mul__(self, other: float | "Tensor") -> "Tensor":
        """
        Multiply two tensors.
        """

        if isinstance(other, numbers.Number):
            # The gradient of the number can be ignored
            out = Tensor(
                array=self.array * other,
                args=(self,),
                name=f"* {other}",
            )

            def _backward() -> None:
                self.grad += other * out.grad

            out._backward = _backward

        elif isinstance(other, "Tensor"):
            # The gradient of the tensor must be computed
            out = Tensor(
                array=self.array * other.array,
                args=(self, other),
                name="*",
            )

            def _backward() -> None:
                self.grad += other.array * out.grad
                other.grad += self.array * out.grad

            out._backward = _backward

        else:
            raise NotImplementedError(f"Cannot multiply {type(self)} and {type(other)}")

        return out

    def __rmul__(self, other: float | "Tensor") -> "Tensor":
        return self.__mul__(other)

    def __imul__(self, other: float | "Tensor") -> "Tensor":
        return self.__mul__(other)

    def __neg__(self) -> "Tensor":
        return self * -1

    def __pow__(self, other: int | float | "Tensor") -> "Tensor":
        """
        Exponention.
        """

        if isinstance(other, numbers.Number):
            out = Tensor(
                array=self.array**other,
                args=(self,),
                name=f"** {other}",
            )

            def _backward() -> None:
                self.grad += (other * self.array ** (other - 1)) * out.grad

            out._backward = _backward

        elif isinstance(other, Tensor):
            raise NotImplementedError("Exponentiation of two tensors is not supported.")

        else:
            raise NotImplementedError(
                f"Cannot exponentiate {type(self)} and {type(other)}"
            )

        return out

    def __rpow__(self, other: int | float | "Tensor") -> "Tensor":
        return self.__pow__(other)

    def __ipow__(self, other: float | "Tensor") -> "Tensor":
        return self.__pow__(other)

    def __truediv__(self, other: float | "Tensor") -> "Tensor":
        """
        Division.
        """
        pass
