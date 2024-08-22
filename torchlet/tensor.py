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
        data: ArrayLike,
        # requires_grad: bool = True,
        dtype: DTypeLike | None = None,
        name: str | None = None,
        args: tuple["Tensor", ...] | None = None,
        backward_fn: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """
        Initialize a new Tensor object.
        TODO: docstring
        """
        # Needed?
        if isinstance(data, Tensor):
            self = data
            return

        self.data = np.array(data)

        if dtype is None:
            dtype = DEFAULT_TYPE

        self.data = self.data.astype(dtype)
        # self.requires_grad = requires_grad

        self.name = name
        self._id = uuid.uuid1().int
        self.args = args or tuple()

        # TODO: Init as None to save memory?
        self.grad = np.zeros_like(self.data)
        self.backward_fn = backward_fn

    def backward(self) -> None:
        """
        Compute the gradient of the tensor.
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

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            if v.backward_fn is not None:
                v.backward_fn(v.grad)

    @property
    def shape(self) -> Sequence[int]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        # TODO: Implement
        raise NotImplementedError

    def sum(self) -> "Tensor":
        def _backward(dy: np.ndarray) -> None:
            self.grad += np.ones_like(self.data) * dy

        out = Tensor(
            data=np.sum(self.data),
            dtype=self.dtype,
            name="sum",
            args=(self,),
            backward_fn=_backward,
        )

        return out

    def mean(self) -> "Tensor":
        def _backward(dy: np.ndarray) -> None:
            grad_divisor = self.data.size
            self.grad += dy / grad_divisor

        out = Tensor(
            data=np.mean(self.data),
            dtype=self.dtype,
            name="mean",
            args=(self,),
            backward_fn=_backward,
        )

        return out

    def transpose(self) -> "Tensor":

        # TODO: Make sure this is correct
        def _backward(dy: np.ndarray) -> None:
            self.grad += dy.transpose()

        out = Tensor(
            data=self.data.T,
            dtype=self.dtype,
            name="transpose",
            args=(self,),
            backward_fn=_backward,
        )

        return out

    def relu(self) -> "Tensor":
        def _backward(dy: np.ndarray) -> None:
            self.grad += dy * (self.data > 0)

        out = Tensor(
            data=np.maximum(0, self.data),
            dtype=self.dtype,
            name="ReLU",
            args=(self,),
            backward_fn=_backward,
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

        if isinstance(other, Tensor):

            def _backward(dy: np.ndarray) -> None:
                self.grad += dy
                other.grad += dy

            out = Tensor(
                data=self.data + other.data,
                dtype=np.result_type(self.data, other.data),
                name="+",
                args=(self, other),
                backward_fn=_backward,
            )

        else:

            def _backward(dy: np.ndarray) -> None:
                self.grad += dy

            out = Tensor(
                data=self.data + other,
                dtype=np.result_type(self.data, other),
                name=f"+ {other}",
                args=(self,),
                backward_fn=_backward,
            )
        return out

    def __radd__(self, other: Union[int, float]) -> "Tensor":
        return self + other

    def __iadd__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        self = self + other
        return self

    def __sub__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Subtract two tensors.
        """
        if isinstance(other, Tensor):
            # The gradient of the tensor must be computed.
            def _backward(dy: np.ndarray) -> None:
                self.grad += dy
                other.grad -= dy

            out = Tensor(
                data=self.data - other.data,
                dtype=np.result_type(self.data, other.data),
                name="-",
                args=(self, other),
                backward_fn=_backward,
            )

        else:

            def _backward(dy: np.ndarray) -> None:
                self.grad += dy

            out = Tensor(
                data=self.data - other,
                dtype=np.result_type(self.data, other),
                name=f"- {other}",
                args=(self,),
                backward_fn=_backward,
            )

        return out

    def __rsub__(self, other: Union[int, float]) -> "Tensor":
        return other + (-self)

    def __isub__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        self = self - other
        return self

    def __mul__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Multiply two tensors.
        """
        if isinstance(other, Tensor):
            # The gradient of the tensor must be computed
            def _backward(dy: np.ndarray) -> None:
                self.grad += other.data * dy
                other.grad += self.data * dy

            out = Tensor(
                data=self.data * other.data,
                dtype=np.result_type(self.data, other.data),
                name="*",
                args=(self, other),
                backward_fn=_backward,
            )
        else:
            # The gradient of the number can be ignored
            def _backward(dy: np.ndarray) -> None:
                self.grad += other * dy

            out = Tensor(
                data=self.data * other,
                dtype=np.result_type(self.data, other),
                name=f"* {other}",
                args=(self,),
                backward_fn=_backward,
            )

        return out

    def __rmul__(self, other: Union[int, float]) -> "Tensor":
        return self * other

    def __imul__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        self = self * other
        return self

    def __neg__(self) -> "Tensor":
        return self * (-1.0)

    def __matmul__(self, other: "Tensor") -> "Tensor":

        assert isinstance(other, Tensor), "Only supporting matmul with another tensor"

        # Interesting implementation
        def _backward(dy: np.ndarray) -> None:
            self.grad += np.tensordot(dy, other.data, axes=(-1, -1))
            indices = list(range(dy.ndim - 1))
            other.grad += np.tensordot(self.data, dy, axes=(indices, indices))

        return Tensor(
            data=np.tensordot(self.data, other.data, axes=(-1, 0)),
            dtype=np.result_type(self.data, other.data),
            name="@",
            args=(self, other),
            backward_fn=_backward,
        )

    def __pow__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Exponention.
        """
        if isinstance(other, Tensor):
            raise NotImplementedError("Exponentiation of two tensors is not supported.")

        def _backward(dy: np.ndarray) -> None:
            self.grad += (other * self.data ** (other - 1)) * dy

        out = Tensor(
            data=self.data**other,
            dtype=np.result_type(self.data, other),
            name=f"** {other}",
            args=(self,),
            backward_fn=_backward,
        )

        return out

    def __rpow__(self, other: Union[int, float]) -> "Tensor":
        raise NotImplementedError("Exponentiation is not commutative.")

    def __ipow__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        self = self**other
        return self

    def __truediv__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Division.
        """
        if isinstance(other, Tensor):
            return self * (other**-1)

        else:
            return self * (1 / other)

    def __rtruediv__(self, other: Union[int, float]) -> "Tensor":
        return other * self**-1

    def __itruediv__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        self = self / other
        return self

    def __lt__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Less than comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data < other.data)

        return Tensor(self.data < other)

    def __le__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Less than or equal to comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data <= other.data)

        return Tensor(self.data <= other)

    def __eq__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Equal to comparison.
        """
        if isinstance(other, Tensor):
            if other._id == self._id:
                return Tensor(True)
            return Tensor(self.data == other.data)

        return Tensor(self.data == other)

    def __ne__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Not equal to comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data != other.data)

        return Tensor(self.data != other)

    def __gt__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Greater than comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data > other.data)

        return Tensor(self.data > other)

    def __ge__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        """
        Greater than or equal to comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data >= other.data)

        return Tensor(self.data >= other)

    def __repr__(self) -> str:
        name = f", name={self.name}" if self.name is not None else ""
        return f"Tensor({self.data.__str__()}{name})"

    def __getitem__(self, idx: Any) -> "Tensor":
        def _backward(dy: np.ndarray) -> None:
            self.grad[idx] += dy

        return Tensor(
            data=self.data[idx],
            dtype=self.dtype,
            name="getitem",
            args=(self,),
            backward_fn=_backward,
        )

    def __setitem__(self, idx: Any, value: Union[int, float, "Tensor"]) -> None:
        # TODO: Implement gradient computation
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value
