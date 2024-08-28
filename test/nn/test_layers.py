import pytest
import numpy as np
from torchlet import Tensor
from torchlet.nn import Linear, Module


@pytest.mark.parametrize("in_features", [1, 2, 3])
@pytest.mark.parametrize("out_features", [1, 2, 3])
@pytest.mark.parametrize("bias", [True, False])
class TestLinear:

    def test_init(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
    ) -> None:
        linear = Linear(in_features, out_features, bias)

        assert isinstance(linear, Module)

        assert isinstance(linear.W, Tensor)
        assert linear.W.data.shape == (in_features, out_features)

        if bias:
            assert isinstance(linear.b, Tensor)
            assert linear.b.data.shape == (out_features,)
        else:
            assert linear.b is None

    def test_forward(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
    ) -> None:
        linear = Linear(in_features, out_features, bias)

        # Single sample
        inp1 = Tensor(np.arange(in_features))
        out1 = linear(inp1)
        assert isinstance(out1, Tensor)
        assert out1.data.shape == (out_features,)

        y1 = out1.sum()
        y1.backward()

        assert linear.W.grad.shape == (in_features, out_features)
        if bias:
            assert isinstance(linear.b, Tensor)
            assert linear.b.grad.shape == (out_features,)

        # Single batched sample
        inp2 = Tensor(np.arange(in_features).reshape(1, -1))
        out2 = linear(inp2)
        assert isinstance(out2, Tensor)
        assert out2.data.shape == (1, out_features)

        y2 = out2.sum()
        y2.backward()

        assert linear.W.grad.shape == (in_features, out_features)
        if bias:
            assert isinstance(linear.b, Tensor)
            assert linear.b.grad.shape == (out_features,)

        # Multiple batched samples
        inp3 = Tensor(np.arange(in_features * 2).reshape(2, -1))
        out3 = linear(inp3)
        assert isinstance(out3, Tensor)
        assert out3.data.shape == (2, out_features)

        y3 = out3.sum()
        y3.backward()

        assert linear.W.grad.shape == (in_features, out_features)
        if bias:
            assert isinstance(linear.b, Tensor)
            assert linear.b.grad.shape == (out_features,)

    def test_parameters(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
    ) -> None:
        linear = Linear(in_features, out_features, bias)
        parameters = linear.parameters()

        assert isinstance(parameters, list)
        assert len(parameters) == 1 + (1 if bias else 0)
        assert all(isinstance(p, Tensor) for p in parameters)
        assert parameters[0].data.shape == (in_features, out_features)
        if bias:
            assert parameters[1].data.shape == (out_features,)
