from typing import Callable, Iterable

import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data, lists, permutations

from torchlet import Tensor, tensor
from torchlet.tensor_functions import grad_check

from .strategies import assert_close, small_floats
from .tensor_strategies import shaped_tensors, tensors
from .math_tests import MathTestVariable

one_arg, two_arg, red_arg = MathTestVariable._comp_testing()


@given(lists(small_floats, min_size=1))
def test_create(t1: list[float]) -> None:
    """Test the ability to create an index a 1D Tensor"""
    t2 = tensor(t1)
    for i in range(len(t1)):
        assert t1[i] == t2[i]


@given(tensors())
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(
    fn: tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]],
    t1: Tensor,
) -> None:
    """Test one-arg functions compared to floats"""
    name, base_fn, tensor_fn = fn
    t2 = tensor_fn(t1)
    for ind in t2._tensor.indices():
        assert_close(t2[ind], base_fn(t1[ind]))


@given(shaped_tensors(2))
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    ts: tuple[Tensor, Tensor],
) -> None:
    name, base_fn, tensor_fn = fn
    t1, t2 = ts
    t3 = tensor_fn(t1, t2)
    for ind in t3._tensor.indices():
        assert_close(t3[ind], base_fn(t1[ind], t2[ind]))


@given(tensors(requires_grad=True))
@pytest.mark.parametrize("fn", one_arg)
def test_one_derivative(
    fn: tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]],
    t1: Tensor,
) -> None:
    """Test the gradient of a none-arg tensor function"""
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(data(), tensors())
def test_permute(data: DataObject, t1: Tensor) -> None:
    """Test permute function"""
    permutation = data.draw(permutations(range(len(t1.shape))))

    def permute(a: Tensor) -> Tensor:
        return a.permute(*permutation)

    grad_check(permute, t1)


def test_grad_size() -> None:
    """Test the size of the gradient"""
    a = tensor([1], requires_grad=True)
    b = tensor([[1, 1]], requires_grad=True)

    c = (a * b).sum()
    c.backward()

    assert c.shape == (1,)
    assert a.grad is not None
    assert b.grad is not None
    assert a.shape == a.grad.shape
    assert b.shape == b.grad.shape


@given(tensors())
@pytest.mark.parametrize("fn", red_arg)
def test_grad_reduce(
    fn: tuple[str, Callable[[Iterable[float]], float], Callable[[Tensor], Tensor]],
    t1: Tensor,
) -> None:
    """Test the grad of a tensor reduce"""
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(shaped_tensors(2))
@pytest.mark.parametrize("fn", two_arg)
def test_two_grad(
    fn: tuple[str, Callable[[float], float], Callable[[Tensor, Tensor], Tensor]],
    ts: tuple[Tensor, Tensor],
) -> None:
    name, _, tensor_fn = fn
    t1, t2 = ts
    grad_check(tensor_fn, t1, t2)


@given(shaped_tensors(2))
@pytest.mark.parametrize("fn", two_arg)
def test_two_grad_broadcast(
    fn: tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    ts: tuple[Tensor, Tensor],
) -> None:
    """Test the grad of a two argument function"""
    _, _, tensor_fn = fn
    t1, t2 = ts
    grad_check(tensor_fn, t1, t2)

    t11 = t1.sum(0).detach()
    t12 = t2.detach()

    t21 = t1.detach()
    t22 = t2.sum(0).detach()

    # broadcast check
    grad_check(tensor_fn, t11, t12)
    grad_check(tensor_fn, t21, t22)


def test_fromList() -> None:
    """Test longer from list conversion"""
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)

    t = tensor([[[2, 3, 4], [4, 5, 7]]])
    assert t.shape == (1, 2, 3)


def test_view() -> None:
    """Test view"""
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t2 = t.view(6)
    assert t2.shape == (6,)
    t2 = t2.view(1, 6)
    assert t2.shape == (1, 6)
    t2 = t2.view(6, 1)
    assert t2.shape == (6, 1)
    t2 = t2.view(2, 3)
    assert t.is_close(t2).all().item() == 1.0


@given(tensors())
def test_back_view(t1: Tensor) -> None:
    """Test the gradient of view"""

    def view(a: Tensor) -> Tensor:
        a = a.contiguous()
        return a.view(a.size)

    grad_check(view, t1)


@pytest.mark.xfail
def test_permute_view() -> None:
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t2 = t.permute(1, 0)
    t2.view(6)


@pytest.mark.xfail
def test_index() -> None:
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t[50, 2]


def test_fromnumpy() -> None:
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    n = t.to_numpy()
    t2 = tensor(n.tolist())
    for ind in t._tensor.indices():
        assert t[ind] == t2[ind]


def test_reduce_forward_one_dim() -> None:
    t = tensor([[2, 3], [4, 6], [5, 7]])

    t_summed = t.sum(0)

    t_sum_expected = tensor([[11, 16]])

    assert t_summed.is_close(t_sum_expected).all().item()


def test_reduce_forward_all_dims() -> None:
    t = tensor([[2, 3], [4, 6], [5, 7]])
    t_summed_all = t.sum()
    t_summed_all_expected = tensor([27])
    assert_close(t_summed_all[0], t_summed_all_expected[0])


def test_simple_opt() -> None:

    def f(x: Tensor) -> Tensor:
        return x * x

    x = tensor([2.0], requires_grad=True)
    ys = []
    for i in range(10):
        y = f(x)
        y.backward()

        assert y.requires_grad
        assert y.grad is not None
        assert x.grad is not None
        x = x - x.grad * 0.1
        x.reset_history()
        x.grad = None
        ys.append(y.detach().item())

        assert x.requires_grad
