import pytest
import numpy as np
from torchlet.optim.sgd import SGD
from torchlet import Tensor


def test_sgd_step():
    # Create some dummy parameters and gradients
    params = {
        "p1": Tensor([1.0, 2.0, 3.0], requires_grad=True),
    }
    grads = [np.array([0.1, 0.2, 0.3])]

    # Create an instance of SGD optimizer
    optimizer = SGD(params, lr=0.1)

    # Set the gradients for the parameters
    for param, grad in zip(params.values(), grads):
        param.grad = grad

    # Perform a step of optimization
    optimizer.step()

    # Check if the parameters have been updated correctly
    expected_params = [Tensor([0.99, 1.98, 2.97])]
    for param, expected_param in zip(params.values(), expected_params):
        assert param.data == pytest.approx(expected_param.data)
