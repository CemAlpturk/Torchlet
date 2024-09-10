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
                out = torchlet.zeros(
                    a.shape,
                    backend=a.f,
                    requires_grad=a.requires_grad,
                )

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

            out = torchlet.zeros(
                c_shape,
                backend=a.f,
                requires_grad=a.requires_grad,
            )
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
            out = torchlet.zeros(
                tuple(out_shape), backend=a.f, requires_grad=a.requires_grad
            )
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement matrix multiplication
        raise NotImplementedError


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


SimpleBackend = TensorBackend(SimpleOps)
