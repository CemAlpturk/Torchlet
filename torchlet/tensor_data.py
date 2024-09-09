from __future__ import annotations
from typing import Iterable, Sequence, TypeAlias


import numpy as np
import numpy.typing as npt


MAX_DIMS = 32

Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position
    in storage based on `strides`.

    Args:
        index (Index): The index of the tensor.
        strides (Strides): The strides of the tensor.

    Returns:
        int: The position in storage.
    """

    return int((index * strides).sum())


def to_index(pos: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert a position `pos` in storage to a multidimensional tensor index with the given shape.
    Operation is performed in-place and the result is stored in `out_index`.

    Args:
        pos (int): The position in storage.
        shape (Shape): The shape of the tensor.
        out_index (OutIndex): The output index.
    """
    val = pos
    for i in range(shape.size - 1, -1, -1):
        out_index[i] = val % shape[i]
        val = val // shape[i]


def broadcast_index(
    big_index: Index,
    big_shape: Shape,
    shape: Shape,
    out_index: OutIndex,
) -> None:
    """
    Convert a `big_index` into `gib_shape` to a smaller `out_index` into shape `shape`.
    Operation is performed in-place and the result is stored in `out_index`.

    Args:
        big_index (Index): The index of the big tensor.
        big_shape (Shape): The shape of the big tensor.
        shape (Shape): The shape of the smaller tensor.
        out_index (OutIndex): The output index.
    """

    size_diff = big_shape.size - shape.size
    for i in range(shape.size - 1, -1, -1):
        if shape[i] == 1:
            out_index[i] = 0
        else:
            out_index[i] = big_index[i + size_diff]


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 (UserShape): The first shape.
        shape2 (UserShape): The second shape.

    Returns:
        UserShape: The broadcasted shape.
    """

    # Bring the shapes to the same size
    if len(shape1) < len(shape2):
        shape1 = (1,) * (len(shape2) - len(shape1)) + tuple(shape1)
    else:
        shape2 = (1,) * (len(shape1) - len(shape2)) + tuple(shape2)

    # Compute new shape
    new_shape: list[int] = []
    for i in range(len(shape1)):
        d1, d2 = shape1[i], shape2[i]

        if d1 == d2 or d1 == 1 or d2 == 1:
            # Broadcast
            new_shape.append(d1 if d1 > d2 else d2)

        else:
            raise IndexingError(
                f"The size of tensor a ({d1}) must match the size of tensor b ({d2}) at a non-singelton dimension {i}"
            )

    return tuple(new_shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    """
    Compute the strides from a shape.

    Args:
        shape (UserShape): The shape of the tensor.

    Returns:
        UserStrides: The strides of the tensor.
    """

    strides = [1]
    offset = 1
    for i in range(len(shape) - 1, -1, -1):
        strides.append(shape[i] * offset)
        offset *= shape[i]

    return tuple(reversed(strides[:-1]))


class TensorData:

    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Sequence[float] | Storage,
        shape: UserShape,
        strides: UserStrides | None = None,
    ) -> None:

        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = np.array(storage, dtype=np.float64)

        if strides is None:
            strides = strides_from_shape(shape)

        # Input sanity checks
        assert isinstance(strides, tuple), "Strides must be a tuple"
        assert isinstance(shape, tuple), "Shape must be a tuple"

        if len(strides) != len(shape):
            raise IndexError(f"Len of strides {strides} must match {shape}.")

        self._strides = np.array(strides)
        self._shape = np.array(shape)
        self.strides = strides
        self.dims = len(strides)

        # TODO: May want to implement this with native python
        self.size = int(np.prod(self._shape))
        self.shape = shape

        assert len(self._storage) == self.size

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous. Outer dimensions have bigger strides thatn inner dimensions.

        Returns:
            bool: True if the layout is contiguous.
        """

        for i, j in zip(self._strides[:-1], self._strides[1:]):
            if j > i:
                return False

        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """
        Broadcast two shapes to create a new union shape.

        Args:
            shape_a (UserShape): The first shape.
            shape_b (UserShape): The second shape.

        Returns:
            UserShape: The broadcasted shape.
        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: int | UserIndex) -> int:
        """
        Return position in storage for a given index.
        """

        if isinstance(index, int):
            aindex: Index = np.array([index], np.int32)

        # Bit weird to handle it like this not all paths lead to `aindex`
        if isinstance(index, tuple):
            aindex = np.array(index)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")

        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing
        return index_to_position(aindex, self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """
        Return all indices in the tensor.
        """

        lshape: Shape = np.array(self.shape)
        out_index: Index = np.array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def get(self, key: UserIndex) -> float:
        """
        Get the value at the given index.

        Args:
            key (UserIndex): The index to get.

        Returns:
            float: The value at the index.
        """

        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, value: float) -> None:
        """
        Set the value at the given index.

        Args:
            key (UserIndex): The index to set.
            value (float): The value to set.
        """
        self._storage[self.index(key)] = value

    def tuple(self) -> tuple[Storage, Shape, Strides]:
        """
        Return the tensor data as a tuple.
        """

        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the tensor data.

        Args:
            order (int): The order to permute the tensor.

        Returns:
            TensorData: The permuted tensor.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        idxs = np.array(order)
        new_shape: UserShape = tuple(int(x) for x in self._shape[idxs])
        new_strides = tuple(int(x) for x in self._strides[idxs])

        return TensorData(
            storage=self._storage,
            shape=new_shape,
            strides=new_strides,
        )

    def to_string(self) -> str:
        """
        Return the tensor as a string.
        """

        s = ""
        for index in self.indices():
            ls = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    ls = "\n%s[" % ("\t" * i) + ls
                else:
                    break
            s += ls
            v = self.get(index)
            s += f"{v:3.2f}"
            ls = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    ls += "]"
                else:
                    break
            if ls:
                s += ls
            else:
                s += " "
        return s
