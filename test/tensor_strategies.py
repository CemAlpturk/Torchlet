import numpy as np
from hypothesis.strategies import (
    DrawFn,
    SearchStrategy,
    composite,
    floats,
    lists,
    permutations,
    integers,
)

import torchlet
from torchlet.tensor_data import (
    TensorData,
    UserShape,
    UserIndex,
)
from torchlet.tensor_functions import TensorBackend
from torchlet.scalar_ops import prod

from .strategies import small_ints


@composite
def vals(draw: DrawFn, size: int, number: SearchStrategy[float]) -> torchlet.Tensor:
    pts = draw(
        lists(
            number,
            min_size=size,
            max_size=size,
        )
    )
    return torchlet.tensor_functions.tensor(pts)


@composite
def shapes(draw: DrawFn) -> UserShape:
    lsize = draw(lists(small_ints, min_size=1, max_size=4))
    return tuple(lsize)


@composite
def tensor_data(
    draw: DrawFn,
    numbers: SearchStrategy[float] = floats(),
    shape: UserShape | None = None,
) -> TensorData:
    if shape is None:
        shape = draw(shapes())
    size = int(np.prod(shape))
    data = draw(lists(numbers, min_size=size, max_size=size))
    permute: list[int] = draw(permutations(range(len(shape))))
    permute_shape = tuple([shape[i] for i in permute])
    z = sorted(enumerate(permute), key=lambda a: a[1])
    reverse_permute = [a[0] for a in z]
    td = TensorData(data, permute_shape)
    ret = td.permute(*reverse_permute)
    assert ret.shape[0] == shape[0]
    return ret


@composite
def indices(draw: DrawFn, layout: torchlet.Tensor) -> UserIndex:
    return tuple((draw(integers(min_value=0, max_value=s - 1)) for s in layout.shape))


@composite
def tensors(
    draw: DrawFn,
    numbers: SearchStrategy[float] = floats(
        allow_nan=False, min_value=-100, max_value=100
    ),
    backend: TensorBackend | None = None,
    shape: UserShape | None = None,
    requires_grad: bool = False,
) -> torchlet.Tensor:
    backend = torchlet.tensor_functions.SimpleBackend if backend is None else backend
    td = draw(tensor_data(numbers, shape))
    return torchlet.Tensor(td, backend=backend, requires_grad=requires_grad)


@composite
def shaped_tensors(
    draw: DrawFn,
    n: int,
    numbers: SearchStrategy[float] = floats(
        allow_nan=False, min_value=-100, max_value=100
    ),
    backend: TensorBackend | None = None,
    requires_grad: bool = False,
) -> list[torchlet.Tensor]:
    backend = torchlet.tensor_functions.SimpleBackend if backend is None else backend
    td = draw(tensor_data(numbers))
    values = []
    for i in range(n):
        data = draw(lists(numbers, min_size=td.size, max_size=td.size))
        values.append(
            torchlet.Tensor(
                TensorData(data, td.shape, td.strides),
                backend=backend,
                requires_grad=requires_grad,
            )
        )
    return values


@composite
def matmul_tensors(
    draw: DrawFn,
    numbers: SearchStrategy[float] = floats(
        allow_nan=False, min_value=-100, max_value=100
    ),
    requires_grad: bool = False,
) -> list[torchlet.Tensor]:

    i, j, k = [draw(integers(min_value=1, max_value=10)) for _ in range(3)]

    l1 = (i, j)
    l2 = (j, k)
    values = []
    for shape in [l1, l2]:
        size = int(prod(shape))
        data = draw(lists(numbers, min_size=size, max_size=size))
        values.append(
            torchlet.Tensor(
                TensorData(data, shape),
                requires_grad=requires_grad,
            ),
        )

    return values


def assert_close_tensor(a: torchlet.Tensor, b: torchlet.Tensor) -> None:
    if a.is_close(b).all().item() != 1.0:
        assert (
            False
        ), f"Tensors are not close \n x.shape={a.shape} \n x={a} \n y.shape={b.shape} \n y={b} \n Diff={a - b} {a.is_close(b)}"
