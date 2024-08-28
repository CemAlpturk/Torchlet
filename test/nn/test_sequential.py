from torchlet.nn import Linear, Sequential
from torchlet import Tensor


def test_sequential_forward() -> None:
    # Create a Sequential model with two Linear modules
    model = Sequential(Linear(10, 5), Linear(5, 2))

    # Create a sample input tensor
    input_tensor = Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    # Forward pass through the model
    output_tensor = model.forward(input_tensor)

    # Check the shape of the output tensor
    assert output_tensor.shape == (1, 2)


def test_sequential_parameters() -> None:
    # Create a Sequential model with two Linear modules
    model = Sequential(Linear(10, 5), Linear(5, 2))

    # Get the parameters of the model
    parameters = model.parameters()

    # Check the number of parameters
    assert len(parameters) == 4


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
