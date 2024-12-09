from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

import torchlet
from torchlet import scalar_ops
from torchlet.autodiff import Context
from torchlet.tensor_ops import TensorBackend, SimpleBackend
from torchlet.scalar_ops import prod

if TYPE_CHECKING:
    from typing import Any, Sequence
    from ._tensor import Tensor, Shape, Strides
    from .tensor_data import UserShape, UserIndex


def wrap_tuple(x: Any) -> tuple:
    """
    Wrap a single value into a tuple if it is not already a tuple.

    Args:
        x (Any): The value to wrap.

    Returns:
        tuple: The wrapped value.
    """
    if isinstance(x, tuple):
        return x
    return (x,)


class Function(ABC):

    @staticmethod
    @abstractmethod
    def forward(ctx: Context, *inps: Tensor) -> Tensor:
        pass

    @staticmethod
    @abstractmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tensor | tuple[Tensor, ...]:
        pass

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        raw_vals: list[Tensor] = []
        need_grad = False
        for v in vals:
            if v.requires_grad:
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context
        ctx = Context(not need_grad)

        # Call forward with variables
        c = cls._forward(ctx, *raw_vals)

        assert isinstance(
            c, torchlet.Tensor
        ), f"Forward must return a tensor. Got {type(c)}"

        back = None
        if need_grad:
            back = torchlet.History(cls, ctx, vals)
        return torchlet.Tensor(c._tensor, back, backend=c.f, requires_grad=need_grad)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.neg_map(a)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tensor:
        return grad_out.f.neg_map(grad_out)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return a.f.inv_map(a)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tensor:
        (a,) = ctx.saved_values
        return grad_out.f.inv_back_zip(a, grad_out)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.add_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> tuple[Tensor, Tensor]:
        return grad_out, grad_out


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> tuple[Tensor, Tensor]:
        a, b = ctx.saved_values
        d_a = grad_out.f.mul_zip(b, grad_out)
        d_b = grad_out.f.mul_zip(a, grad_out)

        return d_a, d_b


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return a.f.sigmoid_map(a)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tensor:
        (a,) = ctx.saved_values
        return grad_out.f.sigmoid_back_zip(a, grad_out)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return a.f.relu_map(a)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tensor:
        (a,) = ctx.saved_values
        return grad_out.f.relu_back_zip(a, grad_out)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return a.f.log_map(a)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tensor:
        (a,) = ctx.saved_values
        return grad_out.f.log_back_zip(a, grad_out)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return a.f.exp_map(a)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tensor:
        (a,) = ctx.saved_values
        return grad_out.f.exp_back_zip(a, grad_out)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> tuple[Tensor, float]:
        a_shape, dim = ctx.saved_values
        return grad_out, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(prod(a.shape))), 0.0)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tensor:
        raise NotImplementedError("All does not support backward")


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> tuple[Tensor, Tensor]:
        a, b = ctx.saved_values
        return zeros(a.shape, backend=a.f), zeros(b.shape, backend=b.f)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> tuple[Tensor, Tensor]:
        a, b = ctx.saved_values
        return zeros(a.shape, backend=a.f), zeros(b.shape, backend=b.f)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.is_close_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("IsClose does not support backward")


class Permute(Function):
    @staticmethod
    def to_tuple(t: Tensor | Shape | Strides) -> tuple[int, ...]:
        """
        Converts a tensor or numpy array to an int tuple
        """
        if isinstance(t, torchlet.Tensor):
            return tuple([int(t[i].item()) for i in range(t.size)])
        return tuple([int(t[i]) for i in range(len(t))])

    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        ctx.save_for_backward(order)
        new_tensor_data = a._tensor.permute(*Permute.to_tuple(order))
        new_t_storage, new_t_shape, new_t_strides = new_tensor_data.tuple()

        return torchlet.Tensor.make(
            new_t_storage,
            Permute.to_tuple(new_t_shape),
            strides=Permute.to_tuple(new_t_strides),
            backend=a.f,
        )

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> tuple[Tensor, float]:
        (order,) = ctx.saved_values
        restore_order = zeros((order.size,), backend=grad_out.f)

        # Generate inverse permutation
        for i in range(order.size):
            restore_order[int(order[i].item())] = i

        new_tensor_data = grad_out._tensor.permute(*Permute.to_tuple(restore_order))
        new_t_storage, new_t_shape, new_t_strides = new_tensor_data.tuple()

        return (
            torchlet.Tensor.make(
                new_t_storage,
                Permute.to_tuple(new_t_shape),
                strides=Permute.to_tuple(new_t_strides),
                backend=grad_out.f,
            ),
            0.0,
        )


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Input tensor must be contiguous"
        shape2 = [int(shape[i].item()) for i in range(shape.size)]
        return torchlet.Tensor.make(a._tensor._storage, tuple(shape2), backend=a.f)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> tuple[Tensor, float]:
        (original,) = ctx.saved_values
        return (
            torchlet.Tensor.make(
                grad_out._tensor._storage, original, backend=grad_out.f
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tensor:
        return grad_out


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.f.matrix_multiply(a, b)

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> tuple[Tensor, Tensor]:
        a, b = ctx.saved_values

        # Keep track of whether the inputs are vectors
        a_is_vector = a.dims == 1
        b_is_vector = b.dims == 1

        # Promote vectors to 2D tensors
        if a_is_vector:
            a = a.unsqueeze(0)  # (1, n)
        if b_is_vector:
            b = b.unsqueeze(1)  # (n, 1)

        # Promote grad_out if necessary
        if grad_out.dims == 0:
            # grad_out is a scalar
            # unlikely to happen in practice
            grad_out = grad_out.unsqueeze(0).unsqueeze(0)  # (1, 1)
        elif grad_out.dims == 1:
            if a_is_vector and not b_is_vector:
                grad_out = grad_out.unsqueeze(0)  # (1, p)
            elif not a_is_vector and b_is_vector:
                grad_out = grad_out.unsqueeze(1)  # (m, 1)
            elif a_is_vector and b_is_vector:
                grad_out = grad_out.unsqueeze(0).unsqueeze(0)  # (1, 1)

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        # Compute gradients
        grad_a = grad_out @ transpose(b)
        grad_b = transpose(a) @ grad_out

        # Remove extra dimensions
        if a_is_vector:
            grad_a = grad_a.squeeze(0)
        if b_is_vector:
            grad_b = grad_b.squeeze(1)

        return grad_a, grad_b


# Helper functions
def zeros(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape (UserShape): The shape of the tensor.
        backend (TensorBackend): The backend to use.
        requires_grad (bool): Whether to compute gradients.

    Returns:
        Tensor: The zero tensor.
    """
    return torchlet.Tensor.make(
        [0.0] * int(scalar_ops.prod(shape)),
        shape,
        backend=backend,
        requires_grad=requires_grad,
    )


def ones(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a ones tensor of size `shape`.

    Args:
        shape (UserShape): The shape of the tensor.
        backend (TensorBackend): The backend to use.
        requires_grad (bool): Whether to compute gradients.

    Returns:
        Tensor: The ones tensor.
    """
    return torchlet.Tensor.make(
        [1.0] * int(scalar_ops.prod(shape)),
        shape,
        backend=backend,
        requires_grad=requires_grad,
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape (UserShape): The shape of the tensor.
        backend (TensorBackend): The backend to use.
        requires_grad (bool): Whether to compute gradients.

    Returns:
        Tensor: The random tensor.
    """
    vals = [random.random() for _ in range(int(scalar_ops.prod(shape)))]
    tensor = torchlet.Tensor.make(
        vals,
        shape,
        backend=backend,
        requires_grad=requires_grad,
    )
    return tensor


def arange(
    end: int,
    start: int = 0,
    step: int = 1,
    *,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a range tensor from `start` to `end` with step `step`.

    Args:
        end (int): The end of the range.
        start (int): The start of the range.
        step (int): The step of the range.
        backend (TensorBackend): The backend to use.
        requires_grad (bool): Whether to compute gradients.

    Returns:
        Tensor: The range tensor.
    """
    vals = [float(x) for x in range(start, end, step)]
    tensor = torchlet.Tensor.make(
        vals,
        (len(vals),),
        backend=backend,
        requires_grad=requires_grad,
    )
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data `ls` and shape `shape`.

    Args:
        ls (Any): The data of the tensor.
        shape (UserShape): The shape of the tensor.
        backend (TensorBackend): The backend to use.
        requires_grad (bool): Whether to compute gradients.

    Returns:
        Tensor: The tensor.
    """

    tensor = torchlet.Tensor.make(
        ls,
        shape,
        backend=backend,
        requires_grad=requires_grad,
    )

    return tensor


def tensor(
    ls: Any,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data and shape from `ls`.

    Args:
        ls (Any): The data of the tensor.
        backend (TensorBackend): The backend to use.
        requires_grad (bool): Whether to compute gradients.

    Returns:
        :class: `Tensor` : The tensor.
    """

    def shape(ls: Any) -> list[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> list[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    curr = flatten(ls)
    shape2 = shape(ls)
    return _tensor(
        curr,
        tuple(shape2),
        backend=backend,
        requires_grad=requires_grad,
    )


def concat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """
    Concatenate a sequence of tensors along a given dimension.
    Does not support backpropagation.
    This implementation creates a new tensor and copies data from the input tensors, which can be inefficient for large tensors.
    A more efficient approach would be to directly manipulate the underlying storage of the tensors to avoid unnecessary data copying.

    Args:
        tensors (Sequence[Tensor]): The sequence of tensors to concatenate.
        dim (int): The dimension along which to concatenate.

    Returns:
        Tensor: The concatenated tensor.
    """
    # Ensure all tensors have the same shape except for the concatenation dimension
    shapes = [tensor.shape for tensor in tensors]
    for i in range(len(shapes[0])):
        if i != dim:
            for shape in shapes:
                if shape[i] != shapes[0][i]:
                    raise ValueError(
                        "All tensors must have the same shape except for the concatenation dimension"
                    )

    # Calculate the new shape
    new_shape = list(shapes[0])
    new_shape[dim] = sum(shape[dim] for shape in shapes)

    # Bad way of doing this, fix this later
    concatenated_tensor = zeros(tuple(new_shape), backend=tensors[0].f)

    pos = 0
    for t, shape in zip(tensors, shapes):
        idx = [slice(None)] * len(new_shape)
        idx[dim] = slice(pos, pos + shape[dim])
        concatenated_tensor[tuple(idx)] = t
        pos += shape[dim]

    return concatenated_tensor


# Gradient check for tensors


def grad_central_difference(
    f: Any,
    *vals: Tensor,
    arg: int = 0,
    epsilon: float = 1e-6,
    ind: UserIndex,
) -> float:
    x = vals[arg]
    up = zeros(x.shape, backend=x.f)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0].item() / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    for x in vals:
        x.require_grad(True)
        x.zero_grad()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Inpit %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""
    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind].item(),
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind].item(), i, ind, check),
        )
