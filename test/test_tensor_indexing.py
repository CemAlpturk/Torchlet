import pytest

from torchlet import tensor
from torchlet.tensor_data import IndexingError


def test_one_dim() -> None:
    """Test indexing a 1D tensor."""
    t1 = tensor([1, 2, 3, 4, 5])
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
        t1[5]
