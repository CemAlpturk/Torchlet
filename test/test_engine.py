import numpy as np
from torchlet.engine import Value


def test_gradients_numerical():

    def f(x: Value) -> Value:
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        return h + q + q * x

    x = Value(-4.0)
    y = f(x)

    y.backward()

    x_grad = x.grad.item()

    # Numerical gradient
    epsilon = 1e-9
    x.data += epsilon
    y2 = f(x)
    numerical_grad = (y2.data - y.data) / epsilon

    assert np.isclose(x_grad, numerical_grad, atol=1e-3)
