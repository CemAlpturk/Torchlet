from __future__ import annotations

from typing import Sequence
import math

from torchlet.optim.optimizer import Optimizer
from torchlet import Tensor
from torchlet.nn import Parameter
from torchlet.tensor_functions import zeros


class Adam(Optimizer):
    """
    Implements the Adam optimizer.

    Proposed in:
    Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization."
    International Conference on Learning Representations (ICLR) 2015.
    https://arxiv.org/abs/1412.6980

    Args:
        params (Sequence[Parameter]): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (tuple[float, float], optional): Coefficients for computing running
            averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): Term added to denominator for numerical stability
            (default: 1e-8).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.0).
        maximize (bool, optional): Maximize the params based on the objective,
            instead of minimizing (default: False).
    """

    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    maximize: bool
    step_count: int
    exp_avg: list[Tensor | None]
    exp_avg_sq: list[Tensor | None]

    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximize = maximize
        self.step_count = 0
        self.exp_avg = [None for _ in self.params]
        self.exp_avg_sq = [None for _ in self.params]

    def step(self) -> None:
        """Performs a single optimization step."""
        self.step_count += 1
        beta1, beta2 = self.betas

        # Bias corrections
        bias_correction1 = 1.0 - beta1**self.step_count
        bias_correction2 = 1.0 - beta2**self.step_count

        for i, param in enumerate(self.params):
            if param.value is None or not hasattr(param.value, "grad"):
                continue

            grad: Tensor = param.value.grad
            if grad is None:
                continue

            if self.maximize:
                grad = -grad

            # L2 penalty applied to gradient (standard Adam)
            if not isinstance(self, AdamW) and self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.value

            # Initialize first and second moment buffers on first active step
            if self.exp_avg[i] is None:
                self.exp_avg[i] = zeros(param.value.shape, backend=param.value.f)
                self.exp_avg_sq[i] = zeros(param.value.shape, backend=param.value.f)

            # Decay the first and second moment running average coefficient
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            m_t = beta1 * self.exp_avg[i] + (1.0 - beta1) * grad
            m_t.reset_history()
            self.exp_avg[i] = m_t

            # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            v_t = beta2 * self.exp_avg_sq[i] + (1.0 - beta2) * (grad * grad)
            v_t.reset_history()
            self.exp_avg_sq[i] = v_t

            # Bias-corrected denominator and step direction
            denom = (v_t / bias_correction2).sqrt() + self.eps
            step_size = self.lr / bias_correction1
            step_direction = m_t / denom

            if isinstance(self, AdamW) and self.weight_decay != 0.0:
                # Decoupled weight decay: theta = theta - lr * weight_decay * theta
                param_val = param.value - (self.lr * self.weight_decay) * param.value
                update = param_val - step_size * step_direction
            else:
                update = param.value - step_size * step_direction

            update.reset_history()
            param.update(update)


class AdamW(Adam):
    """
    Implements the AdamW optimizer (Adam with decoupled weight decay).

    Proposed in:
    Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization."
    International Conference on Learning Representations (ICLR) 2019.
    https://arxiv.org/abs/1711.05101

    Args:
        params (Sequence[Parameter]): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        betas (tuple[float, float], optional): Coefficients for computing running
            averages of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): Term added to denominator for numerical stability
            (default: 1e-8).
        weight_decay (float, optional): Decoupled weight decay coefficient (default: 1e-2).
        maximize (bool, optional): Maximize the params based on the objective,
            instead of minimizing (default: False).
    """

    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
