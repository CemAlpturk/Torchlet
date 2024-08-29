from collections.abc import Iterator

import pytest
import numpy as np

from torchlet import Tensor
from torchlet.nn import (
    Module,
    Linear,
    Dropout,
)


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

        assert isinstance(linear.weight, Tensor)
        assert linear.weight.data.shape == (in_features, out_features)

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

        assert linear.weight.grad.shape == (in_features, out_features)
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

        assert linear.weight.grad.shape == (in_features, out_features)
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

        assert linear.weight.grad.shape == (in_features, out_features)
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
        params = linear.parameters()

        assert isinstance(params, Iterator)
        assert len(list(params)) == 1 + (1 if bias else 0)

        for param in linear.parameters():
            assert isinstance(param, Tensor)

        params = list(linear.parameters())
        assert params[0].data.shape == (in_features, out_features)
        if bias:
            assert params[1].data.shape == (out_features,)

    def test_state_dict(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
    ) -> None:
        linear = Linear(in_features, out_features, bias)
        state_dict = linear.state_dict()

        assert isinstance(state_dict, dict)
        assert len(state_dict) == 1 + (1 if bias else 0)

        assert "weight" in state_dict
        assert isinstance(state_dict["weight"], Tensor)
        assert state_dict["weight"].data.shape == (in_features, out_features)

        if bias:
            assert "b" in state_dict
            assert isinstance(state_dict["b"], Tensor)
            assert state_dict["b"].data.shape == (out_features,)

    def test_state_dict_prefix(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
    ) -> None:
        linear = Linear(in_features, out_features, bias)
        prefix = "Linear"
        state_dict = linear.state_dict(prefix=prefix)

        assert isinstance(state_dict, dict)
        assert len(state_dict) == 1 + (1 if bias else 0)

        assert f"{prefix}.weight" in state_dict
        assert isinstance(state_dict[f"{prefix}.weight"], Tensor)
        assert state_dict[f"{prefix}.weight"].data.shape == (in_features, out_features)

        if bias:
            assert f"{prefix}.b" in state_dict
            assert isinstance(state_dict[f"{prefix}.b"], Tensor)
            assert state_dict[f"{prefix}.b"].data.shape == (out_features,)


class TestDropout:

    @pytest.mark.parametrize("prob", [0.0, 0.5, 0.9])
    def test_init(self, prob: float) -> None:
        dropout = Dropout(prob)

        assert isinstance(dropout, Module)
        assert dropout.prob == prob
        assert dropout._coeff == 1 / (1 - prob)

    @pytest.mark.parametrize("prob", [0.0, 0.5, 0.9])
    def test_forward(self, prob: float) -> None:
        dropout = Dropout(prob)

        # Test during training
        dropout.train()
        x = Tensor(np.ones((10, 10)))
        y = dropout(x)

        assert isinstance(y, Tensor)
        assert y.data.shape == (10, 10)

        # Test during evaluation
        dropout.eval()
        y = dropout(x)

        assert isinstance(y, Tensor)
        assert y.data.shape == (10, 10)
        assert np.allclose(y.data, x.data)

    def test_forward_with_zero_probability(self) -> None:
        prob = 0.0
        dropout = Dropout(prob)

        # Test during training
        dropout.train()
        x = Tensor(np.ones((10, 10)))
        y = dropout(x)

        assert isinstance(y, Tensor)
        assert y.data.shape == (10, 10)
        assert np.allclose(y.data, x.data)

        # Test during evaluation
        dropout.eval()
        y = dropout(x)

        assert isinstance(y, Tensor)
        assert y.data.shape == (10, 10)
        assert np.allclose(y.data, x.data)

    def test_state_dict(self) -> None:
        dropout = Dropout(0.5)
        state_dict = dropout.state_dict()

        assert isinstance(state_dict, dict)
        assert len(state_dict) == 0

    def test_parameters(self) -> None:
        dropout = Dropout(0.5)
        params = dropout.parameters()

        assert isinstance(params, Iterator)
        assert len(list(params)) == 0
