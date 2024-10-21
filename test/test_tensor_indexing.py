import pytest

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
