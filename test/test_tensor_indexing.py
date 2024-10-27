import pytest

import torchlet
from torchlet import tensor, Tensor
from torchlet.tensor_data import IndexingError


def test_one_dim() -> None:
    """Test indexing a 1D tensor."""
    t1 = tensor([1, 2, 3, 4, 5])
    assert isinstance(t1, Tensor)
    assert t1[0].item() == 1
    assert t1[1].item() == 2
    assert t1[2].item() == 3
    assert t1[3].item() == 4
    assert t1[4].item() == 5


def test_two_dim() -> None:
    """Test indexing a 2D tensor."""
    t1 = tensor([[1, 2, 3], [4, 5, 6]])
    assert t1[0, 0].item() == 1
    assert t1[0, 1].item() == 2
    assert t1[0, 2].item() == 3
    assert t1[1, 0].item() == 4
    assert t1[1, 1].item() == 5
    assert t1[1, 2].item() == 6


def test_three_dim() -> None:
    """Test indexing a 3D tensor."""
    t1 = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert t1[0, 0, 0].item() == 1
    assert t1[0, 0, 1].item() == 2
    assert t1[0, 1, 0].item() == 3
    assert t1[0, 1, 1].item() == 4
    assert t1[1, 0, 0].item() == 5
    assert t1[1, 0, 1].item() == 6
    assert t1[1, 1, 0].item() == 7
    assert t1[1, 1, 1].item() == 8


def test_negative_index() -> None:
    """Test negative indexing."""
    t1 = tensor([1, 2, 3, 4, 5])
    assert t1[-1].item() == 5
    assert t1[-2].item() == 4
    assert t1[-3].item() == 3
    assert t1[-4].item() == 2
    assert t1[-5].item() == 1


def test_negative_index_2d() -> None:
    """Test negative indexing."""
    t1 = tensor([[1, 2, 3], [4, 5, 6]])
    assert t1[-1, -1].item() == 6
    assert t1[-1, -2].item() == 5
    assert t1[-1, -3].item() == 4
    assert t1[-2, -1].item() == 3
    assert t1[-2, -2].item() == 2
    assert t1[-2, -3].item() == 1


def test_indexing_error() -> None:
    t1 = tensor([1, 2, 3, 4, 5])
    with pytest.raises(IndexingError):
        print(t1[5])


def test_slice() -> None:
    """Test slicing."""
    t1 = tensor([1, 2, 3, 4, 5])
    b = t1[1:4]
    assert isinstance(b, Tensor)
    assert b.shape == (3,)
    assert b[0].item() == 2
    assert b[1].item() == 3
    assert b[2].item() == 4


def test_slice_2d() -> None:
    """Test slicing in 2d array."""

    t1 = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = t1[1:3, 1:3]
    assert isinstance(b, Tensor)
    assert b.shape == (2, 2)
    assert b[0, 0].item() == 5
    assert b[0, 1].item() == 6
    assert b[1, 0].item() == 8
    assert b[1, 1].item() == 9


def test_mixed_indexing() -> None:
    """Test mixed indexing."""
    t1 = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = t1[1, 1:3]
    assert isinstance(b, Tensor)
    assert b.shape == (2,)
    assert b[0].item() == 5
    assert b[1].item() == 6

    b = t1[1:3, 1]
    assert isinstance(b, Tensor)
    assert b.shape == (2,)
    assert b[0].item() == 5
    assert b[1].item() == 8

    b = t1[1:3, 1:3]
    assert isinstance(b, Tensor)
    assert b.shape == (2, 2)
    assert b[0, 0].item() == 5
    assert b[0, 1].item() == 6
    assert b[1, 0].item() == 8
    assert b[1, 1].item() == 9

    b = t1[0, :]
    assert isinstance(b, Tensor)
    assert b.shape == (3,)
    assert b[0].item() == 1
    assert b[1].item() == 2
    assert b[2].item() == 3

    b = t1[:, 0]
    assert isinstance(b, Tensor)
    assert b.shape == (3,)
    assert b[0].item() == 1
    assert b[1].item() == 4
    assert b[2].item() == 7

    b = t1[1]
    assert isinstance(b, Tensor)
    assert b.shape == (3,)
    assert b[0].item() == 4
    assert b[1].item() == 5
    assert b[2].item() == 6

    b = t1[-1]
    assert isinstance(b, Tensor)
    assert b.shape == (3,)
    assert b[0].item() == 7
    assert b[1].item() == 8
    assert b[2].item() == 9

    b = t1[:, -1]
    assert isinstance(b, Tensor)
    assert b.shape == (3,)
    assert b[0].item() == 3
    assert b[1].item() == 6
    assert b[2].item() == 9


def test_setitem_index_1d() -> None:
    """Test setting an item."""
    t1 = tensor([1, 2, 3, 4, 5])
    t1[1] = 10
    assert t1[1].item() == 10


def test_setitem_index_2d() -> None:
    """Test setting an item."""
    t1 = tensor([[1, 2, 3], [4, 5, 6]])
    t1[1, 1] = 10
    assert t1[1, 1].item() == 10


def test_setitem_index_3d() -> None:
    """Test setting an item."""
    t1 = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    t1[1, 1, 1] = 10
    assert t1[1, 1, 1].item() == 10


def test_setitem_negative_index() -> None:
    """Test setting with negative indexing."""
    t1 = tensor([1, 2, 3, 4, 5])
    t1[-1] = 50
    t1[-2] = 40
    assert t1[-1].item() == 50
    assert t1[-2].item() == 40


def test_setitem_slice() -> None:
    """Test setting a slice."""
    t1 = tensor([1, 2, 3, 4, 5])
    t1[1:4] = tensor([10, 20, 30])
    assert t1[1].item() == 10
    assert t1[2].item() == 20
    assert t1[3].item() == 30


def test_setitem_slice_2d() -> None:
    """Test setting a 2D slice."""
    t1 = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t1[1:, 1:] = tensor([[10, 20], [30, 40]])
    print(t1)
    assert t1[1, 1].item() == 10
    assert t1[1, 2].item() == 20
    assert t1[2, 1].item() == 30
    assert t1[2, 2].item() == 40

    t1 = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t1[0] = tensor([10, 20, 30])
    assert t1[0, 0].item() == 10
    assert t1[0, 1].item() == 20
    assert t1[0, 2].item() == 30


def test_setitem_slice_1d() -> None:
    tensor_1d = tensor([0, 1, 2, 3, 4])
    tensor_1d[1:4] = 99
    assert tensor_1d[1].item() == 99
    assert tensor_1d[2].item() == 99
    assert tensor_1d[3].item() == 99


def test_setitem_slice_2d_() -> None:
    tensor_2d = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tensor_2d[:, 1] = 0
    assert tensor_2d[0, 1].item() == 0
    assert tensor_2d[1, 1].item() == 0
    assert tensor_2d[2, 1].item() == 0


def test_setitem_slice_3d() -> None:
    tensor_3d = torchlet.arange(27).view(3, 3, 3)
    tensor_3d[1, :, :] = torchlet.ones((3, 3))
    assert tensor_3d[1, 0, 0].item() == 1
    assert tensor_3d[1, 0, 1].item() == 1
    assert tensor_3d[1, 0, 2].item() == 1
    assert tensor_3d[1, 1, 0].item() == 1
    assert tensor_3d[1, 1, 1].item() == 1
    assert tensor_3d[1, 1, 2].item() == 1
    assert tensor_3d[1, 2, 0].item() == 1
    assert tensor_3d[1, 2, 1].item() == 1
    assert tensor_3d[1, 2, 2].item() == 1


def test_setitem_advanced_indexing() -> None:
    tensor_3d = torchlet.arange(27).view(3, 3, 3)
    tensor_3d[1, :, 2] = 42
    assert tensor_3d[1, 0, 2].item() == 42
    assert tensor_3d[1, 1, 2].item() == 42
    assert tensor_3d[1, 2, 2].item() == 42


def test_setitem_broadcasting_scalar_to_slice() -> None:
    tensor_2d = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tensor_2d[:, :] = 7
    for idx in tensor_2d._tensor.indices():
        assert tensor_2d[idx] == 7


def test_setitem_broadcasting_tensor_to_slice() -> None:
    tensor_2d = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    replacement = tensor([[-1, -1, -1], [-2, -2, -2], [-3, -3, -3]])
    tensor_2d[:, :] = replacement
    for idx in tensor_2d._tensor.indices():
        assert tensor_2d[idx] == replacement[idx]


def test_setitem_out_of_bounds() -> None:
    tensor_1d = tensor([0, 1, 2, 3, 4])
    tensor_2d = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    with pytest.raises(IndexingError):
        tensor_1d[10] = 5  # Out of bounds for 1D tensor

    with pytest.raises(IndexingError):
        tensor_2d[3, 3] = 5  # Out of bounds for 2D tensor


def test_setitem_incorrect_shape_assignment() -> None:
    tensor_2d = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Test setting with incorrect shape (should raise error)
    with pytest.raises(ValueError):
        tensor_2d[:, :] = tensor([1, 2, 3, 4])  # Shape mismatch


def test_shared_memory() -> None:
    t1 = tensor([1, 2, 3, 4, 5])
    t2 = t1
    t2[1] = 10
    assert t1[1].item() == 10
    assert t2[1].item() == 10


def test_shared_memory_slice() -> None:
    t1 = torchlet.arange(5)
    t2 = t1[0:1]
    t2[0] = 10
    assert t1[0].item() == 10

    t1 = torchlet.arange(5)
    t2 = t1[1:3]
    t2[0] = 10
    assert t1[1].item() == 10
    assert t2[0].item() == 10
