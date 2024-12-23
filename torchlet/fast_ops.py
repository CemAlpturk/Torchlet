from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

import torchlet
from torchlet.tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

from torchlet.tensor_ops import (
    MapProto,
    ZipProto,
    ReduceProto,
    TensorOps,
    TensorBackend,
)

if TYPE_CHECKING:
    from typing import Callable
    from torchlet.tensor_data import Index, Shape, Strides, Storage
    from torchlet._tensor import Tensor


# Compile indexing functions
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Tensor | None = None) -> Tensor:
            if out is None:
                out = torchlet.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> ZipProto:
        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = torchlet.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(fn: Callable[[float, float], float], start: float = 0.0) -> ReduceProto:
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            out = torchlet.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = torchlet.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())  # type: ignore

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low level tensor_map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:

        if out_strides.size == in_strides.size and np.all(out_strides == in_strides):
            for i in prange(out.size):
                out[i] = fn(in_storage[i % in_storage.size])
        else:
            for i in prange(out.size):
                out_index: Index = np.zeros(len(out_shape), dtype=np.int32)
                in_index: Index = np.zeros(len(in_shape), dtype=np.int32)

                to_index(i, out_shape, out_index)  # type: ignore
                broadcast_index(out_index, out_shape, in_shape, in_index)  # type: ignore
                val: float = in_storage[index_to_position(in_index, in_strides)]  # type: ignore
                pos: int = index_to_position(out_index, out_strides)  # type: ignore
                out[pos] = fn(val)

    return njit(parallel=True)(_map)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA hgh order tensor zip function.
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
        if (
            out_strides.size == a_strides.size
            and out_strides.size == b_strides.size
            and np.all(out_strides == a_strides)
            and np.all(out_strides == b_strides)
        ):
            for i in prange(out.size):
                out[i] = fn(
                    a_storage[i % a_storage.size], b_storage[i % b_storage.size]
                )

        else:
            for i in prange(out.size):
                out_index: Index = np.zeros(len(out_shape), dtype=np.int32)
                a_index: Index = np.zeros(len(a_shape), dtype=np.int32)
                b_index: Index = np.zeros(len(b_shape), dtype=np.int32)

                to_index(i, out_shape, out_index)  # type: ignore
                broadcast_index(out_index, out_shape, a_shape, a_index)  # type: ignore
                broadcast_index(out_index, out_shape, b_shape, b_index)  # type: ignore

                val_a: float = a_storage[index_to_position(a_index, a_strides)]  # type: ignore
                val_b: float = b_storage[index_to_position(b_index, b_strides)]  # type: ignore
                out[i] = fn(val_a, val_b)

    return njit(parallel=True)(_zip)


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA high order tensor reduce function.
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:

        reduce_dim_a_stride: int = a_strides[reduce_dim]
        reduce_dim_total_len: int = reduce_dim_a_stride * a_shape[reduce_dim]

        for i in prange(out.size):
            index: Index = np.zeros(len(out_shape), dtype=np.int32)
            to_index(i, out_shape, index)  # type: ignore

            a_start_ind: int = index_to_position(index, a_strides)  # type: ignore
            out_ind: int = index_to_position(index, out_strides)  # type: ignore

            for a_ind in range(
                a_start_ind, a_start_ind + reduce_dim_total_len, reduce_dim_a_stride
            ):
                out[out_ind] = fn(a_storage[a_ind], out[out_ind])

    return njit(parallel=True)(_reduce)


def _tensor_matrix_multiply(
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
    NUMBA tensor matrix multiply function.
    """

    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    rows = out_shape[1]
    cols = out_shape[2]
    common_dim = a_shape[2]

    # Iterate over batches
    for n in prange(out_shape[0]):
        out_start_ind: int = out_strides[0] * n
        a_start_ind: int = a_batch_stride * n
        b_start_ind: int = b_batch_stride * n
        for i in prange(rows):
            a_ind = a_start_ind + (i * a_strides[1])
            for j in prange(cols):
                b_ind = b_start_ind + (j * b_strides[2])
                val = 0
                # TODO: See if this is better parallel or not, or with a buffer that is summed after
                for k in prange(common_dim):
                    a_val = a_storage[a_ind + (k * a_strides[2])]
                    b_val = b_storage[b_ind + (k * b_strides[1])]
                    val += a_val * b_val
                out_ind = out_start_ind + i * out_strides[1] + j * out_strides[2]
                out[out_ind] = val


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)


FastTensorBackend = TensorBackend(FastOps)
