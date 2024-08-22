from torchlet.tensor import Tensor
from torchlet.nn import Linear  # ReLU, Sequential


def test_linear():
    linear = Linear(3, 2)
    assert linear.W.data.shape == (3, 2)
    assert linear.b.data.shape == (2,)
    assert linear.parameters() == [linear.W, linear.b]

    inp1 = Tensor([1, 2, 3])
    out1 = linear(inp1)
    assert out1.data.shape == (2,)

    y1 = out1.sum()
    y1.backward()

    assert linear.W.grad.shape == (3, 2)
    assert linear.b.grad.shape == (2,)

    inp2 = Tensor([[1, 2, 3]])
    out2 = linear(inp2)
    assert out2.data.shape == (1, 2)

    y2 = out2.sum()
    y2.backward()
    assert linear.W.grad.shape == (3, 2)
    assert linear.b.grad.shape == (2,)

    inp3 = Tensor([[1, 2, 3], [4, 5, 6]])
    out3 = linear(inp3)
    assert out3.data.shape == (2, 2)

    y3 = out3.sum()
    y3.backward()

    assert linear.W.grad.shape == (3, 2)
    assert linear.b.grad.shape == (2,)
