from __future__ import annotations
from typing import Union, Callable, Sequence, Any
import uuid
import logging

import numpy as np
from numpy.typing import DTypeLike, ArrayLike

logger = logging.getLogger(__name__)

DEFAULT_TYPE = np.float32


class Tensor:
    """
    A class representing a tensor object.

    Args:
        data (ArrayLike): The data of the tensor.
        dtype (DTypeLike | None, optional): The data type of the tensor. Defaults to None.
        name (str | None, optional): The name of the tensor. Defaults to None.
        args (tuple[Tensor, ...] | None, optional): The arguments of the tensor. Defaults to None.
        backward_fn (Callable[[np.ndarray], None] | None, optional): The backward function of the tensor. Defaults to None.

    """

    data: np.ndarray
    name: str | None
    _id: int
    args: tuple[Tensor, ...]
    grad: np.ndarray
    backward_fn: Callable[[np.ndarray], None] | None
    requires_grad: bool

    def __init__(
        self,
        data: ArrayLike,
        dtype: DTypeLike | None = None,
        name: str | None = None,
        args: tuple[Tensor, ...] | None = None,
        backward_fn: Callable[[np.ndarray], None] | None = None,
        requires_grad: bool = False,
    ) -> None:
        """
        Initializes a Tensor object.
        Args:
            data (ArrayLike): The input data for the tensor.
            dtype (DTypeLike | None, optional): The data type of the tensor. Defaults to None.
            name (str | None, optional): The name of the tensor. Defaults to None.
            args (tuple[Tensor, ...] | None, optional): The arguments passed to the tensor. Defaults to None.
            backward_fn (Callable[[np.ndarray], None] | None, optional): The backward function for gradient computation. Defaults to None.
            requires_grad (bool, optional): Whether the tensor requires gradient computation. Defaults to False.
        """

        # Needed?
        if isinstance(data, Tensor):
            self = data
            return

        self.data = np.array(data)

        if dtype is None:
            dtype = DEFAULT_TYPE

        self.data = self.data.astype(dtype)

        self.name = name
        self._id = uuid.uuid1().int
        self.args = args or tuple()

        # TODO: Init as None to save memory?
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self.backward_fn = backward_fn if requires_grad else None

    def backward(self) -> None:
        """
        Compute the gradient of the tensor.
        """
        if not self.requires_grad:
            raise ValueError("Tensor does not require gradient computation")

        topo: list[Tensor] = []
        visited = set()

        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v.args:
                    if child.requires_grad:
                        build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            if v.backward_fn is not None and v.requires_grad:
                v.backward_fn(v.grad)

    def zero_grad(self) -> None:
        """
        Zero the gradient of the tensor.
        """
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

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
    def T(self) -> Tensor:
        return self.transpose()

    def item(self) -> Union[int, float]:
        if self.data.size == 1:
            return self.data.item()
        else:
            raise ValueError("Only scalar tensors can be converted to Python scalars")

    def to_numpy(self) -> np.ndarray:
        return self.data

    def reshape(self, shape: Sequence[int]) -> Tensor:
        # TODO: Implement
        raise NotImplementedError

    def sum(self, axis: int | None = None) -> Tensor:
        from torchlet.functions import sum

        return sum(self, axis=axis)

    def mean(self) -> Tensor:
        from torchlet.functions import mean

        return mean(self)

    def transpose(self) -> Tensor:
        from torchlet.functions import transpose

        return transpose(self)

    def relu(self) -> Tensor:
        from torchlet.functions import relu

        return relu(self)

    def sigmoid(self) -> Tensor:
        from torchlet.functions import sigmoid

        return sigmoid(self)

    def tanh(self) -> Tensor:
        from torchlet.functions import tanh

        return tanh(self)

    def size(self) -> int:
        return self.data.size

    def __hash__(self) -> int:
        """
        Return the hash value of the object.

        Returns:
            int: The hash value of the object.
        """
        return self._id

    def __add__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        if isinstance(other, Tensor):
            from torchlet.functions import tensor_add

            return tensor_add(self, other)
        else:
            from torchlet.functions import constant_add

            return constant_add(self, other)

    def __radd__(self, other: np.ndarray | float | int) -> Tensor:
        return self + other

    def __iadd__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        self = self + other
        return self

    def __sub__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        if isinstance(other, Tensor):
            from torchlet.functions import tensor_subtract

            return tensor_subtract(self, other)
        else:
            from torchlet.functions import constant_subtract

            return constant_subtract(self, other)

    def __rsub__(self, other: np.ndarray | float | int) -> Tensor:
        return other + (-self)

    def __isub__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        self = self - other
        return self

    def __mul__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        if isinstance(other, Tensor):
            from torchlet.functions import tensor_multiply

            return tensor_multiply(self, other)
        else:
            from torchlet.functions import constant_multiply

            return constant_multiply(self, other)

    def __rmul__(self, other: np.ndarray | float | int) -> Tensor:
        return self * other

    def __imul__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        self = self * other
        return self

    def __neg__(self) -> Tensor:
        return self * (-1.0)

    def __matmul__(self, other: Tensor) -> Tensor:

        assert isinstance(other, Tensor), "Only supporting matmul with another tensor"

        from torchlet.functions import tensor_matmul

        return tensor_matmul(self, other)

    def __pow__(self, other: float | int) -> Tensor:
        from torchlet.functions import scalar_power

        return scalar_power(self, other)

    def __rpow__(self, other: float | int) -> Tensor:
        raise NotImplementedError("Exponentiation is not commutative.")

    def __ipow__(self, other: float | int) -> Tensor:
        self = self**other
        return self

    def __truediv__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        if isinstance(other, Tensor):
            return self * (other**-1)

        else:
            return self * (1 / other)

    def __rtruediv__(self, other: np.ndarray | float | int) -> Tensor:
        return other * self**-1

    def __itruediv__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        self = self / other
        return self

    def __lt__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        """
        Less than comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data < other.data)

        return Tensor(self.data < other)

    def __le__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        """
        Less than or equal to comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data <= other.data)

        return Tensor(self.data <= other)

    def __eq__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        """
        Equal to comparison.
        """
        if isinstance(other, Tensor):
            if other._id == self._id:
                return Tensor(True)
            return Tensor(self.data == other.data)

        return Tensor(self.data == other)

    def __ne__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        """
        Not equal to comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data != other.data)

        return Tensor(self.data != other)

    def __gt__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        """
        Greater than comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data > other.data)

        return Tensor(self.data > other)

    def __ge__(self, other: Tensor | np.ndarray | float | int) -> Tensor:
        """
        Greater than or equal to comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data >= other.data)

        return Tensor(self.data >= other)

    def __repr__(self) -> str:
        name = f", name={self.name}" if self.name is not None else ""
        return f"Tensor({self.data.__str__()}{name})"

    def __getitem__(self, idx: Any) -> Tensor:
        from torchlet.functions import tensor_getitem

        return tensor_getitem(self, idx)

    def __setitem__(self, idx: Any, value: Union[int, float, Tensor]) -> None:
        from torchlet.functions import tensor_setitem

        tensor_setitem(self, idx, value)
