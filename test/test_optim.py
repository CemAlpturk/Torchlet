import pytest

from torchlet import Tensor, tensor
from torchlet import nn
from torchlet.optim import Optimizer, SGD, Adam, AdamW


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


class TestAdam:

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
        optimizer = Adam(module.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        assert isinstance(optimizer, Optimizer)
        assert optimizer.lr == 1e-3
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-8
        assert optimizer.params == module.parameters()

    def test_invalid_params(self, module: nn.Module) -> None:
        with pytest.raises(ValueError, match="Invalid learning rate"):
            Adam(module.parameters(), lr=-0.01)
        with pytest.raises(ValueError, match="Invalid epsilon"):
            Adam(module.parameters(), eps=-1e-8)
        with pytest.raises(ValueError, match="Invalid beta parameter at index 0"):
            Adam(module.parameters(), betas=(1.5, 0.999))
        with pytest.raises(ValueError, match="Invalid beta parameter at index 1"):
            Adam(module.parameters(), betas=(0.9, 1.0))
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            Adam(module.parameters(), weight_decay=-0.1)

    def test_step_updates_weights(self, module: nn.Module) -> None:
        optimizer = Adam(module.parameters(), lr=0.01)
        x = tensor([[1.0, 2.0]])
        y = module(x).sum()

        optimizer.zero_grad()
        y.backward()

        w_before = module.linear1.weight.value.detach().contiguous().numpy().copy()
        b_before = module.linear1.bias.value.detach().contiguous().numpy().copy()

        optimizer.step()

        w_after = module.linear1.weight.value.numpy()
        b_after = module.linear1.bias.value.numpy()

        assert (w_after != w_before).any()
        assert (b_after != b_before).any()
        assert w_after.shape == w_before.shape
        assert b_after.shape == b_before.shape
        assert module.linear1.weight.value.requires_grad
        assert module.linear1.bias.value.requires_grad
        assert optimizer.step_count == 1
        assert optimizer.exp_avg[0] is not None
        assert optimizer.exp_avg_sq[0] is not None

    def test_optimization_convergence(self, module: nn.Module) -> None:
        opt = Adam(module.parameters(), lr=0.05)
        x = tensor([[1.0, 2.0]])
        target = tensor([[0.5, -0.5, 1.0]])

        losses = []
        for _ in range(25):
            opt.zero_grad()
            pred = module(x)
            diff = pred - target
            loss = (diff * diff).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[0] > losses[-1]

    def test_maximize_mode(self, module: nn.Module) -> None:
        opt = Adam(module.parameters(), lr=0.01, maximize=True)
        x = tensor([[1.0, 2.0]])
        y = module(x).sum()
        optimizer_standard = Adam(module.parameters(), lr=0.01, maximize=False)

        assert opt.maximize is True
        assert optimizer_standard.maximize is False


class TestAdamW:

    @pytest.fixture
    def module(self) -> nn.Module:
        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = nn.Linear(2, 3)

            def forward(self, x: Tensor) -> Tensor:
                return self.linear1(x)

        return MLP()

    def test_init_and_inheritance(self, module: nn.Module) -> None:
        opt = AdamW(module.parameters(), lr=1e-3, weight_decay=0.05)
        assert isinstance(opt, Adam)
        assert isinstance(opt, Optimizer)
        assert opt.weight_decay == 0.05

    def test_decoupled_weight_decay(self, module: nn.Module) -> None:
        opt_w = AdamW(module.parameters(), lr=0.01, weight_decay=0.1)
        x = tensor([[1.0, 2.0]])
        y = module(x).sum()

        opt_w.zero_grad()
        y.backward()
        w_before = module.linear1.weight.value.detach().contiguous().numpy().copy()
        opt_w.step()
        w_after = module.linear1.weight.value.numpy()

        assert (w_after != w_before).any()
        assert w_after.shape == w_before.shape
