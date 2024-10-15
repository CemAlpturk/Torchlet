import torchlet


def test_no_grad_context() -> None:
    # Ensure gradient computation is enabled by default
    assert torchlet.is_grad_enabled()

    with torchlet.no_grad():
        # Gradient computation should be disabled within the context
        assert not torchlet.is_grad_enabled()

    # Gradient computation should be re-enabled after the context
    assert torchlet.is_grad_enabled()


def test_tensor_no_grad():
    with torchlet.no_grad():
        # Create a tensor within the no_grad context
        tensor = torchlet.tensor([1.0, 2.0, 3.0], requires_grad=True)
        # The tensor should not have gradient history
        assert tensor.history is None

    # Create a tensor outside the no_grad context
    tensor = torchlet.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # The tensor should have gradient history
    assert tensor.history is not None
