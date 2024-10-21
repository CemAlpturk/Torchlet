from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import torchlet
from torchlet import scalar_ops
from torchlet.autodiff import Context, Variable, backpropagate
from torchlet.tensor_data import TensorData, IndexingError
from torchlet import tensor_functions as F

if TYPE_CHECKING:
    from typing import Any, Iterable, Sequence, TypeAlias, Union

    import numpy.typing as npt

    from torchlet.tensor_data import (
        Shape,
        Storage,
        Strides,
        UserIndex,
        UserShape,
        UserStrides,
    )
    from torchlet.tensor_functions import Function
    from torchlet.tensor_ops import TensorBackend

    TensorLike: TypeAlias = Union[float, int, "Tensor"]


@dataclass
class History:
    """
    `History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: type[Function] | None = None
    ctx: Context | None = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """
    Tensor class that represents a multidimensional array.
    """

    f: TensorBackend
    history: History | None
    grad: Tensor | None
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: History | None = None,
        name: str | None = None,
        backend: TensorBackend | None = None,
        requires_grad: bool = False,
    ) -> None:
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count

        assert isinstance(v, TensorData), f"Expected TensorData, got {type(v)}"
        assert backend is not None, "Backend must be provided"  # TODO: Remove this

        self._tensor = v
        self.f = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        if requires_grad and torchlet.is_grad_enabled():
            self.history = back if back is not None else History()
        else:
            self.history = None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """
        Returns:
            numpy.ndarray: The tensor as a numpy array.
        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    # Properties
    @property
    def shape(self) -> UserShape:
        """
        Returns:
            UserShape: The shape of the tensor.
        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """
        Returns:
            int: The size of the tensor.
        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """
        Returns:
            int: The number of dimensions of the tensor.
        """
        return self._tensor.dims

    @property
    def requires_grad(self) -> bool:
        return self.history is not None

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """
        Turns a python number into a tensor with the same backend.
        """
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.f)
        else:
            b._type_(self.f)
            c = b
        return c

    def require_grad(self, requires_grad: bool = True) -> None:
        if requires_grad:
            if self.history is None:
                self.history = History()
        else:
            self.history = None

    def reset_history(self) -> None:
        if self.requires_grad:
            self.history = History()
        else:
            self.history = None

    # Functions
    def __add__(self, b: TensorLike) -> Tensor:
        return F.Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return F.Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        return F.Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b: TensorLike) -> Tensor:
        return F.Mul.apply(self, F.Inv.apply(self._ensure_tensor(b)))

    def __rtrudiv__(self, b: TensorLike) -> Tensor:
        return F.Mul.apply(F.Inv.apply(self), self._ensure_tensor(b))

    def __matmul__(self, b: Tensor) -> Tensor:
        return F.MatMul.apply(self, b)

    def __lt__(self, b: TensorLike) -> Tensor:
        return F.LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:
        return F.EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return F.LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        return F.Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        return self + b

    def __rmul__(self, b: TensorLike) -> Tensor:
        return self * b

    def __rsub__(self, b: TensorLike) -> Tensor:
        return -(self - b)

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return F.Inv.apply(self) * b

    def all(self, dim: int | None = None) -> Tensor:
        if dim is None:
            return F.All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return F.All.apply(self, self._ensure_tensor(dim))

    def is_close(self, y: Tensor) -> Tensor:
        return F.IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        return F.Sigmoid.apply(self)

    def relu(self) -> Tensor:
        return F.ReLU.apply(self)

    def log(self) -> Tensor:
        return F.Log.apply(self)

    def exp(self) -> Tensor:
        return F.Exp.apply(self)

    def item(self) -> float:
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def sum(self, dim: int | None = None) -> Tensor:
        if dim is None:
            return F.Sum.apply(
                self.contiguous().view(self.size), self._ensure_tensor(0)
            )
        else:
            return F.Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: int | None = None) -> Tensor:
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum(dim) / self.size

    def permute(self, *order: int) -> Tensor:
        return F.Permute.apply(self, torchlet.tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """Change the shape of the tensor to a new shape with the same size."""
        return F.View.apply(self, torchlet.tensor(list(shape)))

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data."""
        return F.Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(
        self, key: int | UserIndex | slice | tuple[int | slice, ...]
    ) -> Tensor:
        # TODO: Implement ellipsis?

        # Convert input to tuple of ints or slices
        if isinstance(key, (int, slice)):
            key = (key,)

        if isinstance(key, list):
            key = tuple(key)

        # if length of key is less than the number of dimensions, pad with slices
        if len(key) < self.dims:
            key = tuple(list(key) + [slice(None)] * (self.dims - len(key)))

        # Calculate new shape
        new_shape = []
        new_strides = []
        start_offset = 0
        end_offset = 0
        for i, k in enumerate(key):
            if isinstance(k, int):
                ki = k + self.shape[i] if k < 0 else k

                # Handle out of bounds
                if ki < 0 or ki >= self.shape[i]:
                    raise IndexingError(
                        f"Index {ki} is out of bounds for dimension {i} with size {self.shape[i]}"
                    )
                start = ki
                stop = ki + 1
                step = 1
            else:
                start, stop, step = k.indices(self.shape[i])
                new_shape.append((stop - start) // step)
                new_strides.append(self._tensor._strides[i] * step)
            start_offset += start * self._tensor._strides[i]
            end_offset += (stop - 1) * self._tensor._strides[i]

        new_shape = tuple(new_shape)
        new_strides = tuple(new_strides)

        new_storage = self._tensor._storage[start_offset : end_offset + 1]
        return Tensor.make(new_storage, new_shape, strides=new_strides, backend=self.f)

    def __setitem__(self, key: int | UserIndex, val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods for autodiff
    def _type_(self, backend: TensorBackend) -> None:
        self.f = backend

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.f)

    @staticmethod
    def make(
        storage: Storage | list[float],
        shape: UserShape,
        strides: UserStrides | None = None,
        backend: TensorBackend | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """
        Create a new tensor from data.
        """
        return Tensor(
            TensorData(storage, shape, strides),
            backend=backend,
            requires_grad=requires_grad,
        )

    def expand(self, other: Tensor) -> Tensor:
        """
        Method used to allow for backprop over broadcasting.
        This methid is called when the output of `backward`
        is a different size than the input of `forward`.

        Args:
            other (Tensor): backward tensor (must broadcast with self).

        Returns:
            Tensor: Expanded version of `other` with the right derivatives.
        """

        # Case 1: Both the same shape
        if self.shape == other.shape:
            return other

        # Case 2: Backward is smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = torchlet.zeros(true_shape, backend=self.f)
        self.f.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.f.add_reduce(out, dim)

        assert out.size == self.size, f"{out.shape} {self.shape}"

        return Tensor.make(out._tensor._storage, self.shape, backend=self.f)

    def tuple(self) -> tuple[Storage, Shape, Strides]:
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        return Tensor(self._tensor, backend=self.f)

    def unsqueeze(self, dim: int) -> Tensor:
        """Adds a singleton dimension at the specified position."""
        shape = list(self.shape)
        shape.insert(dim, 1)
        return self.view(*shape)

    def squeeze(self, dim: int | None = None) -> Tensor:
        """Removes singleton dimensions."""
        if dim is not None:
            if self.shape[dim] != 1:
                raise ValueError(
                    f"Cannot squeeze dimension {dim} with size {self.shape[dim]}"
                )
            shape = list(self.shape)
            new_shape = shape[:dim] + shape[dim + 1 :]
            return self.view(*new_shape)
        else:
            # Squeeze all singleton dimensions
            new_shape = tuple(dim for dim in self.shape if dim != 1)
            return self.view(*new_shape)

    def numpy(self) -> np.ndarray:

        data = self._tensor._storage
        shape = self._tensor._shape
        return np.array(data).reshape(shape)

    # Variable elements

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the derivetive accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x (Any): The value to add to the derivative.
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(scalar_ops.prod(self.shape)),
                self.shape,
                backend=self.f,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable is created by the user (no `last_fn`)."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is a constant."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Tensor | None = None) -> None:
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.f)
            self.grad = grad_output
        backpropagate(self, grad_output)

    def zero_grad(self) -> None:
        """Reset the derivative on this variable."""
        self.grad = None
