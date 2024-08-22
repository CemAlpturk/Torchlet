from torchlet.tensor import Tensor
from torchlet.nn_tensor import Linear  # ReLU, Sequential


def test_linear():
    linear = Linear(3, 2)
    assert linear.W.array.shape == (3, 2)
    assert linear.b.array.shape == (2,)
    assert linear.parameters() == [linear.W, linear.b]

    inp1 = Tensor([1, 2, 3])
    out1 = linear(inp1)
    assert out1.array.shape == (2,)

    y1 = out1.sum()
    y1.backward()

    assert linear.W.grad.array.shape == (3, 2)
    assert linear.b.grad.array.shape == (2,)

    inp2 = Tensor([[1, 2, 3]])
    out2 = linear(inp2)
    assert out2.array.shape == (1, 2)
    print(out2)
    print(out2.args)
    y2 = out2.sum()
    y2.backward()
    print(y2)
    print(y2.args)
    print(linear.b.grad)
    assert linear.W.grad.array.shape == (3, 2)
    # assert linear.b.grad.array.shape == (2,)

    inp3 = Tensor([[1, 2, 3], [4, 5, 6]])
    out3 = linear(inp3)
    assert out3.array.shape == (2, 2)

    y3 = out3.sum()
    y3.backward()
    print(linear.b.grad)
    assert linear.W.grad.array.shape == (3, 2)
    assert linear.b.grad.array.shape == (1, 2)
