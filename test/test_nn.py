import pytest
from unittest.mock import MagicMock
from hypothesis import given, strategies as st

import torchlet
from torchlet import Tensor
from torchlet.nn import (
    Module,
    Parameter,
    Linear,
    Sequential,
    ModuleList,
    ModuleDict,
)


class TestModule:

    @pytest.fixture(scope="function")
    def module(self) -> Module:
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

        return Module1()

    def test_init(self, module: Module) -> None:

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
        assert isinstance(module.module2, Module)
        assert hasattr(module.module2, "_modules")
        assert hasattr(module.module2, "_parameters")

        assert "param" in module.module2._parameters
        assert hasattr(module.module2, "param")
        assert isinstance(module.module2.param, Parameter)
        assert isinstance(module.module2.param.value, Tensor)
        assert module.module2.param.value.requires_grad

        assert hasattr(module.module2, "x")
        assert module.module2.x == 1

    def test_modules(self, module: Module) -> None:

        modules = module.modules()
        assert isinstance(modules, list)
        assert len(modules) == 1
        assert isinstance(modules[0], Module)
        assert modules[0] == module.module2

    def test_train(self, module: Module) -> None:
        module.train()

        assert module.training
        assert module.module2.training

    def test_eval(self, module: Module) -> None:
        module.eval()

        assert not module.training
        assert not module.module2.training

    def test_named_parameters(self, module: Module) -> None:
        named_parameters = module.named_parameters()
        assert isinstance(named_parameters, list)
        assert len(named_parameters) == 2
        assert isinstance(named_parameters[0], tuple)
        assert isinstance(named_parameters[0][0], str)
        assert isinstance(named_parameters[0][1], Parameter)
        assert isinstance(named_parameters[1], tuple)
        assert isinstance(named_parameters[1][0], str)
        assert isinstance(named_parameters[1][1], Parameter)

    def test_add_parameter(self, module: Module) -> None:
        param = torchlet.rand((), requires_grad=True)
        module.add_parameter("param2", param)

        assert "param2" in module._parameters
        assert hasattr(module, "param2")
        assert module.param2.value == param


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


class TestSimpleModule:

    @pytest.fixture(scope="function")
    def simple_module(self) -> Module:

        class MLP(Module):
            def __init__(self) -> None:
                super().__init__()

                self.linear1 = Linear(3, 2)
                self.linear2 = Linear(2, 1)

            def forward(self, x: Tensor) -> Tensor:
                x = self.linear1(x)
                x = x.relu()
                x = self.linear2(x)
                x = x.sigmoid()
                return x

        return MLP()

    def test_init(self, simple_module: Module) -> None:

        assert hasattr(simple_module, "linear1")
        assert hasattr(simple_module, "linear2")

        assert isinstance(simple_module.linear1, Linear)
        assert isinstance(simple_module.linear2, Linear)

        assert hasattr(simple_module.linear1, "weight")
        assert hasattr(simple_module.linear1, "bias")
        assert hasattr(simple_module.linear2, "weight")
        assert hasattr(simple_module.linear2, "bias")

        assert isinstance(simple_module.linear1.weight, Parameter)
        assert isinstance(simple_module.linear1.bias, Parameter)
        assert isinstance(simple_module.linear2.weight, Parameter)
        assert isinstance(simple_module.linear2.bias, Parameter)

        assert isinstance(simple_module.linear1.weight.value, Tensor)
        assert isinstance(simple_module.linear1.bias.value, Tensor)
        assert isinstance(simple_module.linear2.weight.value, Tensor)
        assert isinstance(simple_module.linear2.bias.value, Tensor)

        assert simple_module.linear1.weight.value.shape == (3, 2)
        assert simple_module.linear1.bias.value.shape == (2,)
        assert simple_module.linear2.weight.value.shape == (2, 1)
        assert simple_module.linear2.bias.value.shape == (1,)

        assert simple_module.linear1.weight.value.requires_grad
        assert simple_module.linear1.bias.value.requires_grad
        assert simple_module.linear2.weight.value.requires_grad
        assert simple_module.linear2.bias.value.requires_grad

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_forward(self, simple_module: Module, batch_size: int) -> None:
        size = (batch_size, 3) if batch_size > 1 else (3,)
        x = torchlet.rand(size)

        out = simple_module(x)

        assert isinstance(out, Tensor)
        assert out.shape == (batch_size, 1) if batch_size > 1 else (1,)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_backward(self, simple_module: Module, batch_size: int) -> None:
        size = (batch_size, 3) if batch_size > 1 else (3,)
        x = torchlet.rand(size)

        out = simple_module(x)
        y = out.sum()
        y.backward()

        assert simple_module.linear1.weight.value.grad is not None
        assert simple_module.linear1.bias.value.grad is not None
        assert simple_module.linear2.weight.value.grad is not None
        assert simple_module.linear2.bias.value.grad is not None

        assert simple_module.linear1.weight.value.grad.shape == (3, 2)
        assert simple_module.linear1.bias.value.grad.shape == (2,)
        assert simple_module.linear2.weight.value.grad.shape == (2, 1)
        assert simple_module.linear2.bias.value.grad.shape == (1,)


class TestSequential:

    def test_sequential_initialization(self) -> None:
        module1 = MagicMock(spec=Module)
        module2 = MagicMock(spec=Module)
        seq = Sequential(module1, module2)

        assert len(seq._modules) == 2
        assert seq._modules["0"] == module1
        assert seq._modules["1"] == module2

    def test_sequential_forward(self) -> None:
        module1 = MagicMock(spec=Module)
        module2 = MagicMock(spec=Module)
        module1.return_value = "output1"
        module2.return_value = "output2"

        seq = Sequential(module1, module2)
        result = seq("input")

        # Check call count
        assert module1.call_count == 1
        assert module2.call_count == 1

        # Manually compare call arguments
        assert module1.call_args == (("input",),)
        assert module2.call_args == (("output1",),)

        assert result == "output2"

    @given(st.lists(st.integers(), min_size=1, max_size=10))
    def test_sequential_forward_with_hypothesis(self, data) -> None:
        modules = [MagicMock(spec=Module) for _ in data]
        for i, module in enumerate(modules):
            module.return_value = data[i]

        seq = Sequential(*modules)
        result = seq.forward(data[0])

        for i, module in enumerate(modules):
            if i == 0:
                assert module.call_args == ((data[0],),)
            else:
                assert module.call_args == ((data[i - 1],),)

        assert result == data[-1]

    def test_sequential_empty_initialization(self) -> None:
        seq = Sequential()
        assert len(seq._modules) == 0

    def test_sequential_single_module(self) -> None:
        module = MagicMock(spec=Module)
        module.return_value = "output"

        seq = Sequential(module)
        result = seq.forward("input")

        assert module.call_args == (("input",),)
        assert result == "output"


class TestModuleList:

    def test_module_list_initialization(self) -> None:
        module1 = MagicMock(spec=Module)
        module2 = MagicMock(spec=Module)
        module_list = ModuleList([module1, module2])

        assert len(module_list) == 2
        assert module_list._modules["0"] == module1
        assert module_list._modules["1"] == module2

    def test_module_list_empty_initialization(self) -> None:
        module_list = ModuleList([])
        assert len(module_list) == 0


class TestModuleDict:

    def test_module_list_initialization(self) -> None:
        module1 = MagicMock(spec=Module)
        module2 = MagicMock(spec=Module)
        module_dict = ModuleDict({"mod1": module1, "mod2": module2})

        assert len(module_dict) == 2
        assert module_dict._modules["mod1"] == module1
        assert module_dict._modules["mod2"] == module2

    def test_module_list_empty_initialization(self) -> None:
        module_dict = ModuleDict({})
        assert len(module_dict) == 0
