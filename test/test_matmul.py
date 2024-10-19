import torchlet


def test_2d_matmul() -> None:
    a = torchlet.ones((2, 3))
    b = torchlet.ones((3, 2))

    c = a @ b

    assert c.shape == (2, 2)
    assert c[0, 0].item() == 3.0
    assert c[0, 1].item() == 3.0
    assert c[1, 0].item() == 3.0
    assert c[1, 1].item() == 3.0


def test_2d_matmul_backprop() -> None:
    a = torchlet.ones((2, 3), requires_grad=True)
    b = torchlet.ones((3, 2), requires_grad=True)

    c = a @ b
    y = c.sum()
    y.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape


def test_vector_matmul() -> None:
    a = torchlet.ones((3,))
    b = torchlet.ones((3, 2))

    c = a @ b

    assert c.shape == (2,)
    assert c[0].item() == 3.0
    assert c[1].item() == 3.0

    a = torchlet.ones((3, 2))
    b = torchlet.ones((2,))

    c = a @ b

    assert c.shape == (3,)
    assert c[0].item() == 2.0
    assert c[1].item() == 2.0
    assert c[2].item() == 2.0


def test_vector_matmul_backprop() -> None:
    a = torchlet.ones((3,), requires_grad=True)
    b = torchlet.ones((3, 2), requires_grad=True)

    c = a @ b
    y = c.sum()
    y.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

    a = torchlet.ones((3, 2), requires_grad=True)
    b = torchlet.ones((2,), requires_grad=True)

    c = a @ b
    y = c.sum()
    y.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
