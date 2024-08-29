from collections.abc import Iterator

from torchlet.nn import Linear, Sequential
from torchlet import Tensor


def test_init() -> None:
    # Create a Sequential model with two Linear modules
    model = Sequential(Linear(10, 5), Linear(5, 2))

    # Check the number of modules
    assert len(model.modules) == 2


def test_forward() -> None:
    # Create a Sequential model with two Linear modules
    model = Sequential(Linear(10, 5), Linear(5, 2))

    # Create a sample input tensor
    input_tensor = Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    # Forward pass through the model
    output_tensor = model.forward(input_tensor)

    # Check the shape of the output tensor
    assert output_tensor.shape == (1, 2)


def test_state_dict() -> None:
    # Create a Sequential model with two Linear modules
    model = Sequential(Linear(10, 5), Linear(5, 2, bias=False))

    # Get the parameters of the model
    state_dict = model.state_dict()

    # Check the number of parameters
    assert len(state_dict) == 3
    assert "Linear1.weight" in state_dict
    assert "Linear1.b" in state_dict
    assert "Linear2.weight" in state_dict
    assert "Linear2.b" not in state_dict

    for key, value in state_dict.items():
        assert isinstance(value, Tensor)


def test_parameters() -> None:
    model = Sequential(Linear(10, 5), Linear(5, 2))
    params = model.parameters()

    assert isinstance(params, Iterator)
    assert len(list(params)) == 4

    for param in model.parameters():
        assert isinstance(param, Tensor)


def test_sequential_train_eval() -> None:
    # Create a Sequential model with two Linear modules
    model = Sequential(Linear(10, 5), Linear(5, 2))

    # Set the model to train mode
    model.train()

    # Check if all modules are in train mode
    for module in model.modules:
        assert module.training

    # Set the model to eval mode
    model.eval()

    # Check if all modules are in eval mode
    for module in model.modules:
        assert not module.training
