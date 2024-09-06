import numpy as np
from hypothesis.strategies import (
    DrawFn,
    SearchStrategy,
    composite,
    floats,
    lists,
    permutations,
)

from torchlet.tensor_data import (
    TensorData,
    UserShape,
)

from .strategies import small_ints


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
