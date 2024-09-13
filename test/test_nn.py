import pytest

import torchlet
from torchlet import Tensor
from torchlet.nn import Module, Parameter, Linear


class TestModule:

    def test_init(self) -> None:

        class Module2(Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = 1
                self.param = Parameter(torchlet.rand((), requires_grad=True))

        class Module1(Module):
            def __init__(self) -> None:
                super().__init__()
                self.y = 2
                self.module2 = Module2()
                self.param = Parameter(torchlet.rand(()))

        module = Module1()

        assert hasattr(module, "_modules")
        assert hasattr(module, "_parameters")

        assert isinstance(module._modules, dict)
        assert isinstance(module._parameters, dict)

        assert "module2" in module._modules
        assert "param" in module._parameters

        assert hasattr(module, "y")
        assert module.y == 2

        assert hasattr(module, "param")
        assert isinstance(module.param, Parameter)
        assert isinstance(module.param.value, Tensor)
        assert not module.param.value.requires_grad

        assert hasattr(module, "module2")
        assert isinstance(module.module2, Module2)
        assert hasattr(module.module2, "_modules")
        assert hasattr(module.module2, "_parameters")

        assert "param" in module.module2._parameters
        assert hasattr(module.module2, "param")
        assert isinstance(module.module2.param, Parameter)
        assert isinstance(module.module2.param.value, Tensor)
        assert module.module2.param.value.requires_grad

        assert hasattr(module.module2, "x")
        assert module.module2.x == 1


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
        assert isinstance(linear.weight, Parameter)
        assert isinstance(linear.weight.value, Tensor)
        assert linear.weight.value.shape == (in_features, out_features)

        if bias:
            assert isinstance(linear.bias, Parameter)
            assert isinstance(linear.bias.value, Tensor)
            assert linear.bias.value.shape == (out_features,)

    def test_forward(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
    ) -> None:
        linear = Linear(in_features, out_features, bias)

        # Single sample
        inp1 = torchlet.rand((in_features,))
        out1 = linear(inp1)
        assert isinstance(out1, Tensor)
        assert out1.shape == (out_features,)

        y1 = out1.sum()
        y1.backward()

        assert isinstance(linear.weight.value, Tensor)
        assert linear.weight.value.grad is not None
        assert linear.weight.value.grad.shape == (in_features, out_features)
        if bias:
            assert isinstance(linear.bias, Parameter)
            assert isinstance(linear.bias.value, Tensor)
            assert linear.bias.value.grad is not None
            assert linear.bias.value.grad.shape == (out_features,)

        # Single batched sample
        inp2 = torchlet.rand((1, in_features))
        out2 = linear(inp2)
        assert isinstance(out2, Tensor)
        assert out2.shape == (1, out_features)

        y2 = out2.sum()
        y2.backward()

        assert isinstance(linear.weight.value, Tensor)
        assert linear.weight.value.grad is not None
        assert linear.weight.value.grad.shape == (in_features, out_features)
        if bias:
            assert isinstance(linear.bias, Parameter)
            assert isinstance(linear.bias.value, Tensor)
            assert linear.bias.value.grad is not None
            assert linear.bias.value.grad.shape == (out_features,)

        # Multiple batched samples
        inp3 = torchlet.rand((2, in_features))
        out3 = linear(inp3)
        assert isinstance(out3, Tensor)
        assert out3.shape == (2, out_features)

        y3 = out3.sum()
        y3.backward()

        assert isinstance(linear.weight.value, Tensor)
        assert linear.weight.value.grad is not None
        assert linear.weight.value.grad.shape == (in_features, out_features)
        if bias:
            assert isinstance(linear.bias, Parameter)
            assert isinstance(linear.bias.value, Tensor)
            assert linear.bias.value.grad is not None
            assert linear.bias.value.grad.shape == (out_features,)
