from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Protocol, Any

import numpy as np

import torchlet
from torchlet import scalar_ops
from torchlet.tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from torchlet.tensor_data import Index, Shape, Strides, Storage

    from torchlet._tensor import Tensor


# Is this needed?
class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Tensor | None = ..., /) -> Tensor: ...


class ZipProto(Protocol):
    def __call__(self, a: Tensor, b: Tensor, /) -> Tensor: ...


class ReduceProto(Protocol):
    def __call__(self, a: Tensor, dim: int, /) -> Tensor: ...


class TensorOps(ABC):
    @staticmethod
    @abstractmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    @abstractmethod
    def zip(fn: Callable[[float, float], float]) -> ZipProto:
        pass

    @staticmethod
    @abstractmethod
    def reduce(fn: Callable[[float, float], float], init: float) -> ReduceProto:
        pass

    @staticmethod
    @abstractmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        pass


class TensorBackend:
    def __init__(self, ops: type[TensorOps]) -> None:
        """
        Dynamically construct a tensor backend on a `tensor_ops` object
        that implements map, zip and reduce functions.

        Args:
            ops (type[TensorOps]): The tensor operations object.

        Returns:
            A collection of tensor functions.
        """

        # Maps
        self.neg_map = ops.map(scalar_ops.neg)
        self.sigmoid_map = ops.map(scalar_ops.sigmoid)
        self.relu_map = ops.map(scalar_ops.relu)
        self.log_map = ops.map(scalar_ops.log)
        self.exp_map = ops.map(scalar_ops.exp)
        self.id_map = ops.map(scalar_ops.id)
        self.inv_map = ops.map(scalar_ops.inv)

        # Zips
        self.add_zip = ops.zip(scalar_ops.add)
        self.mul_zip = ops.zip(scalar_ops.mul)
        self.lt_zip = ops.zip(scalar_ops.lt)
        self.eq_zip = ops.zip(scalar_ops.eq)
        self.is_close_zip = ops.zip(scalar_ops.is_close)
        self.relu_back_zip = ops.zip(scalar_ops.relu_back)
        self.sigmoid_back_zip = ops.zip(scalar_ops.sigmoid_back)
        self.log_back_zip = ops.zip(scalar_ops.log_back)
        self.exp_back_zip = ops.zip(scalar_ops.exp_back)
        self.inv_back_zip = ops.zip(scalar_ops.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(scalar_ops.add, 0.0)
        self.mul_reduce = ops.reduce(scalar_ops.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:

        f = tensor_map(fn)

        def ret(a: Tensor, out: Tensor | None = None) -> Tensor:
            if out is None:
                out = torchlet.zeros(a.shape, backend=a.f)

            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> ZipProto:

        f = tensor_zip(fn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape

            out = torchlet.zeros(c_shape, backend=a.f)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(fn: Callable[[float, float], float], start: float) -> ReduceProto:

        f = tensor_reduce(fn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = torchlet.zeros(tuple(out_shape), backend=a.f)
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Matrix multiplication for 2-D tensors.
        Broadcast vectors to 2-D tensors.
        """

        assert isinstance(a, torchlet.Tensor) and isinstance(b, torchlet.Tensor)

        a_is_vector = False
        b_is_vector = False

        if a.dims == 1:
            a_is_vector = True
            a = a.unsqueeze(0)

        if b.dims == 1:
            b_is_vector = True
            b = b.unsqueeze(1)

        assert a.dims == 2
        assert b.dims == 2

        assert a.shape[1] == b.shape[0]

        out_shape = (a.shape[-2], b.shape[-1])
        out = torchlet.zeros(out_shape, backend=a.f)

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        if a_is_vector and b_is_vector:
            out = out.squeeze()

        elif a_is_vector:
            out = out.squeeze(0)

        elif b_is_vector:
            out = out.squeeze(1)

        return out


# Implementations


def tensor_map(fn: Callable[[float], float]) -> Any:
    """
    Low level implementation of the map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:

        out_index: Index = np.zeros(len(out_shape), dtype=np.int32)
        in_index: Index = np.zeros(len(in_shape), dtype=np.int32)

        for i in range(out.size):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            val = in_storage[index_to_position(in_index, in_strides)]
            out[index_to_position(out_index, out_strides)] = fn(val)

    return _map


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    """
    Low level implementation of the zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        out_index: Index = np.zeros(len(out_shape), dtype=np.int32)
        a_index: Index = np.zeros(len(a_shape), dtype=np.int32)
        b_index: Index = np.zeros(len(b_shape), dtype=np.int32)

        for i in range(out.size):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            a_val = a_storage[index_to_position(a_index, a_strides)]
            b_val = b_storage[index_to_position(b_index, b_strides)]
            out[index_to_position(out_index, out_strides)] = fn(a_val, b_val)

    return _zip


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
        dim: int,
    ) -> None:

        index: Index = np.zeros(len(out_shape), dtype=np.int32)

        reduce_dim_a_stride: int = in_strides[dim]
        reduce_dim_total_len: int = reduce_dim_a_stride * in_shape[dim]

        for i in range(out.size):
            to_index(i, out_shape, index)

            a_start_ind = index_to_position(index, in_strides)
            out_ind = index_to_position(index, out_strides)

            for a_ind in range(
                a_start_ind, a_start_ind + reduce_dim_total_len, reduce_dim_a_stride
            ):
                out[out_ind] = fn(in_storage[a_ind], out[out_ind])

    return _reduce


def tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    Low level implementation of the matrix multiplication.
    Batched 2-D tensors only for now.
    """

    # Containers for indices
    out_index: Index = np.zeros(len(out_shape), dtype=np.int32)
    a_index: Index = np.zeros(len(a_shape), dtype=np.int32)
    b_index: Index = np.zeros(len(b_shape), dtype=np.int32)

    common_dim = a_shape[-1]

    for i in range(out_shape[0]):
        out_index[0] = i
        a_index[0] = i
        for j in range(out_shape[1]):
            out_index[1] = j
            b_index[1] = j
            val = 0.0
            for k in range(common_dim):
                a_index[1] = k
                b_index[0] = k

                a_val = a_storage[index_to_position(a_index, a_strides)]
                b_val = b_storage[index_to_position(b_index, b_strides)]

                val += a_val * b_val

            out[index_to_position(out_index, out_strides)] = val


SimpleBackend = TensorBackend(SimpleOps)
