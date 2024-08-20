from typing import Union, Callable, Sequence, Any
import uuid
import logging

import numpy as np
from numpy.typing import DTypeLike, ArrayLike

logger = logging.getLogger(__name__)

DEFAULT_TYPE = np.float32

# INP_TYPE: TypeAlias = Union[float, int, "Tensor"]


class Tensor:
    """
    N-dimensional grid of numbers.
    """

    def __init__(
        self,
        array: ArrayLike,
        requires_grad: bool = True,
        dtype: DTypeLike | None = None,
        name: str | None = None,
        args: tuple["Tensor", ...] | None = None,
        backward_fn: Callable[["Tensor"], None] | None = None,
        _id: int | None = None,
    ) -> None:
        """
        Initialize a new Tensor object.
        TODO: docstring
        """
        # Needed?
        if isinstance(array, Tensor):
            self = array
            return

        # if np.isscalar(array):
        #     self.array = np.array([array])
        # else:
        #     self.array = np.array(array)
        self.array = np.array(array)

        if dtype is None:
            dtype = DEFAULT_TYPE

        self.array = self.array.astype(dtype)
        self.requires_grad = requires_grad

        self.name = name
        self._id = _id or uuid.uuid1().int
        self.args = args or tuple()

        def _backward(dy: "Tensor") -> None:
            return None

        if self.requires_grad:
            self._backward = backward_fn if backward_fn is not None else _backward
            self.grad = Tensor(np.zeros_like(self.array), requires_grad=False)

    def backward(self) -> None:
        """
        Compute the gradient of the tensor.
        TODO: implement
        """
        topo: list["Tensor"] = []
        visited = set()

        def build_topo(v: "Tensor") -> None:
            if v not in visited:
                visited.add(v)
                for child in v.args:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = Tensor(np.ones_like(self.array), requires_grad=False)
        for v in reversed(topo):
            v._backward(v.grad)

    @property
    def shape(self) -> Sequence[int]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        # TODO: Implement
        raise NotImplementedError

    def sum(self) -> "Tensor":
        def _backward(dy: "Tensor") -> None:
            self.grad += Tensor(np.ones_like(self.array)) * dy

        out = Tensor(
            array=np.sum(self.array),
            args=(self,),
            name="sum",
            backward_fn=_backward,
            dtype=self.dtype,
        )

        return out

    def mean(self) -> "Tensor":
        def _backward(dy: "Tensor") -> None:
            grad_divisor = self.array.size
            self.grad += dy / grad_divisor  # type: ignore

        out = Tensor(
            array=np.mean(self.array),
            args=(self,),
            name="mean",
            backward_fn=_backward,
            dtype=self.dtype,
        )

        return out

    def transpose(self) -> "Tensor":
        def _backward(dy: "Tensor") -> None:
            self.grad += dy.transpose()

        out = Tensor(
            array=self.array.T,
            args=(self,),
            name="transpose",
            backward_fn=_backward,
            dtype=self.dtype,
        )

        return out

    def dot(self, other: "Tensor") -> "Tensor":
        def _backward(dy: "Tensor") -> None:
            self.grad += dy.dot(other.T)
            other.grad += self.T.dot(dy)

        out = Tensor(
            array=self.array.dot(other.array),
            args=(self, other),
            name="dot",
            backward_fn=_backward,
            dtype=np.result_type(self.array, other.array),
        )
        return out

    def relu(self) -> "Tensor":
        def _backward(dy: "Tensor") -> None:
            self.grad += dy * (self > 0)  # type: ignore

        out = Tensor(
            array=np.maximum(0, self.array),
            args=(self,),
            name="ReLU",
            backward_fn=_backward,
            dtype=self.dtype,
        )

        return out

    def __hash__(self) -> int:
        """
        Return the hash value of the object.

        Returns:
            int: The hash value of the object.
        """
        return self._id

    def __add__(self, other: Union[int, float, "Tensor"]) -> "Tensor":

        if isinstance(other, (int, float)):
            # The gradient of the number can be ignored.
            def _backward(dy: "Tensor") -> None:
                self.grad += dy

            out = Tensor(
                array=self.array + other,
                args=(self,),
                name=f"+ {other}",
                backward_fn=_backward,
                dtype=np.result_type(self.array, other),
            )

        elif isinstance(other, Tensor):
            # The gradient of the tensor must be computed.
            def _backward(dy: "Tensor") -> None:
                self.grad += dy
                other.grad += dy

            out = Tensor(
                array=self.array + other.array,
                args=(self, other),
                name="+",
                backward_fn=_backward,
                dtype=np.result_type(self.array, other.array),
            )

        else:
            raise NotImplementedError(f"Cannot add {type(self)} and {type(other)}")

        return out

    def __radd__(self, other: Union[int, float]) -> "Tensor":
        return self + other

    def __iadd__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        return self + other

    def __sub__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Subtract two tensors.
        """
        if isinstance(other, (int, float)):
            # The gradient of the number can be ignored.
            def _backward(dy: "Tensor") -> None:
                self.grad += dy

            out = Tensor(
                array=self.array - other,
                args=(self,),
                name=f"- {other}",
                backward_fn=_backward,
                dtype=np.result_type(self.array, other),
            )

        elif isinstance(other, Tensor):
            # The gradient of the tensor must be computed.
            def _backward(dy: "Tensor") -> None:
                self.grad += dy
                other.grad -= dy

            out = Tensor(
                array=self.array - other.array,
                args=(self, other),
                name="-",
                backward_fn=_backward,
                dtype=np.result_type(self.array, other.array),
            )

        else:
            raise NotImplementedError(f"Cannot subtract {type(self)} and {type(other)}")

        return out

    def __rsub__(self, other: Union[int, float]) -> "Tensor":
        return other + (-self)

    def __isub__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        return self - other

    def __mul__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Multiply two tensors.
        """

        if isinstance(other, (int, float)):
            # The gradient of the number can be ignored
            def _backward(dy: "Tensor") -> None:
                self.grad += other * dy

            out = Tensor(
                array=self.array * other,
                args=(self,),
                name=f"* {other}",
                backward_fn=_backward,
                dtype=np.result_type(self.array, other),
            )

        elif isinstance(other, Tensor):
            # The gradient of the tensor must be computed
            def _backward(dy: "Tensor") -> None:
                self.grad += other * dy
                other.grad += self * dy

            out = Tensor(
                array=self.array * other.array,
                args=(self, other),
                name="*",
                backward_fn=_backward,
                dtype=np.result_type(self.array, other.array),
            )

        else:
            raise NotImplementedError(f"Cannot multiply {type(self)} and {type(other)}")

        return out

    def __rmul__(self, other: Union[int, float]) -> "Tensor":
        # Multiplication is commutative
        return self * other

    def __imul__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        return self * other

    def __neg__(self) -> "Tensor":
        return self * (-1.0)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication.
        """

        # def _backward(dy: "Tensor") -> None:
        #     self.grad.array += np.dot(dy.array, other.array.T)
        #     other.grad.array += np.dot(self.array.T, dy.array)

        # out = Tensor(
        #     array=np.dot(self.array, other.array),
        #     args=(self, other),
        #     name="@",
        #     backward_fn=_backward,
        #     dtype=np.result_type(self.array, other.array),
        # )

        def _backward(dy: "Tensor") -> None:
            self.grad.array += np.tensordot(dy.array, other.array, axes=(-1, -1))
            indices = list(range(dy.ndim - 1))
            other.grad.array += np.tensordot(
                self.array, dy.array, axes=(indices, indices)
            )

        out = Tensor(
            array=np.tensordot(self.array, other.array, axes=(-1, 0)),
            args=(self, other),
            name="@",
            backward_fn=_backward,
            dtype=np.result_type(self.array, other.array),
        )

        return out

    def __pow__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Exponention.
        """

        if isinstance(other, (int, float)):

            def _backward(dy: "Tensor") -> None:
                self.grad += (other * self ** (other - 1)) * dy

            out = Tensor(
                array=self.array**other,
                args=(self,),
                name=f"** {other}",
                backward_fn=_backward,
                dtype=np.result_type(self.array, other),
            )

        elif isinstance(other, Tensor):
            raise NotImplementedError("Exponentiation of two tensors is not supported.")

        else:
            raise NotImplementedError(
                f"Cannot exponentiate {type(self)} and {type(other)}"
            )

        return out

    def __rpow__(self, other: Union[int, float]) -> "Tensor":
        raise NotImplementedError("Exponentiation is not commutative.")

    def __ipow__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        return self**other

    def __truediv__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Division.
        """
        if isinstance(other, (int, float)):
            return self * (1 / other)

        elif isinstance(other, Tensor):
            return self * (other**-1)
        else:
            raise NotImplementedError(f"Cannot divide {type(self)} by {type(other)}")

    def __rtruediv__(self, other: Union[int, float]) -> "Tensor":
        if isinstance(other, (int, float)):
            return other * (self**-1)
        else:
            raise NotImplementedError(f"Cannot divide {type(other)} by {type(self)}")

    def __itruediv__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        return self / other

    def __lt__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Less than comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.array < other.array)

        return Tensor(self.array < other)

    def __le__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Less than or equal to comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.array <= other.array)

        return Tensor(self.array <= other)

    def __eq__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Equal to comparison.
        """
        if isinstance(other, Tensor):
            if other._id == self._id:
                return Tensor(True)
            return Tensor(self.array == other.array)

        return Tensor(self.array == other)

    def __ne__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Not equal to comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.array != other.array)

        return Tensor(self.array != other)

    def __gt__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Greater than comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.array > other.array)

        return Tensor(self.array > other)

    def __ge__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Greater than or equal to comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.array >= other.array)

        return Tensor(self.array >= other)

    def __repr__(self) -> str:
        name = f", name={self.name}" if self.name is not None else ""
        return f"Tensor({self.array.__str__()}{name})"

    def __getitem__(self, idx: Any) -> "Tensor":
        def _backward(dy: "Tensor") -> None:
            self.grad[idx] += dy

        return Tensor(
            self.array[idx],
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            name="getitem",
            args=(self,),
            backward_fn=_backward,
        )

    def __setitem__(self, idx: Any, value: Union[int, float, "Tensor"]) -> None:
        # TODO: Implement gradient computation
        if isinstance(value, Tensor):
            self.array[idx] = value.array
        else:
            self.array[idx] = value
