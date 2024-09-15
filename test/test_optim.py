import pytest

from torchlet import Tensor, tensor
from torchlet import nn
from torchlet.optim import Optimizer, SGD


class TestSGD:

    @pytest.fixture
    def module(self) -> nn.Module:

        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = nn.Linear(2, 3)

            def forward(self, x: Tensor) -> Tensor:
                return self.linear1(x)

        return MLP()

    def test_init(self, module: nn.Module) -> None:
        optimizer = SGD(module.parameters(), lr=0.1)

        assert isinstance(optimizer, Optimizer)
        assert optimizer.lr == 0.1
        assert optimizer.params == module.parameters()

    def test_zero_grad(self, module: nn.Module) -> None:
        optimizer = SGD(module.parameters(), lr=0.1)

        x = tensor([[1.0, 2.0]])
        y = module(x).sum()
        y.backward()

        assert module.linear1.weight.value.grad is not None
        assert module.linear1.bias.value.grad is not None

        optimizer.zero_grad()

        assert module.linear1.weight.value.grad is None
        assert module.linear1.bias.value.grad is None

    def test_step(self, module: nn.Module) -> None:
        optimizer = SGD(module.parameters(), lr=0.1)

        x = tensor([[1.0, 2.0]])
        y = module(x).sum()

        optimizer.zero_grad()
        y.backward()

        weight_before = module.linear1.weight.value.detach().contiguous().numpy()
        bias_before = module.linear1.bias.value.detach().contiguous().numpy()

        optimizer.step()

        weight_after = module.linear1.weight.value.numpy()
        bias_after = module.linear1.bias.value.numpy()

        assert (weight_after != weight_before).any()
        assert (bias_after != bias_before).any()

        assert weight_after.shape == weight_before.shape
        assert bias_after.shape == bias_before.shape

        assert module.linear1.weight.value.requires_grad
        assert module.linear1.bias.value.requires_grad

    def test_simple_optim(self, module: nn.Module) -> None:

        opt = SGD(module.parameters(), lr=0.1)

        x = tensor([[1.0, 2.0]])
        loss = []
        for i in range(10):
            y = module(x).sum()

            opt.zero_grad()
            y.backward()

            assert module.linear1.weight.value.grad is not None
            assert module.linear1.bias.value.grad is not None
            opt.step()

            loss.append(y.detach().item())

            assert module.linear1.weight.value.requires_grad
            assert module.linear1.bias.value.requires_grad

            assert isinstance(module.linear1.weight.value, Tensor)
            assert isinstance(module.linear1.bias.value, Tensor)

        assert loss[0] > loss[-1]
