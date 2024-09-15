from typing import Sequence

from torchlet.optim.optimizer import Optimizer
from torchlet import Tensor
from torchlet.nn import Parameter


class SGD(Optimizer):
    """
    Stochastic Gradieent Descent optimizer.
    """

    lr: float
    momentum: float
    weight_decay: float
    dampening: float
    nesterov: bool
    maximize: bool
    momentum_buffer: list[Tensor | None]

    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 0.001,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
    ) -> None:
        """
        Initializes a SGD optimizer.
        Follows pytorch's implementation.

        Args:
            params (Sequence[Parameter]): Parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 0.001.
            momentum (float, optional): Momentum factor. Defaults to 0.0.
            weight_decay (float, optional): Weight decay factor. Defaults to 0.0.
            dampening (float, optional): Dampening factor. Defaults to 0.0.
            nesterov (bool, optional): Nesterov momentum. Defaults to False.
            maximize (bool, optional): Whether to maximize the loss. Defaults to False.
        """

        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize
        self.momentum_buffer = [None for _ in self.params]

    def step(self) -> None:

        for param, momentum_buff in zip(self.params, self.momentum_buffer):

            if param.value is None:
                continue

            if not hasattr(param.value, "grad"):
                continue

            grad: Tensor = param.value.grad
            if grad is None:
                continue

            if self.weight_decay != 0:
                # L2 regularization
                grad += self.weight_decay * param.value

            if self.momentum != 0:
                if momentum_buff is not None:
                    momentum_buff = (
                        self.momentum * momentum_buff + (1 - self.dampening) * grad
                    )

                else:
                    momentum_buff = grad

                if self.nesterov:
                    grad += self.momentum * momentum_buff
                else:
                    grad = momentum_buff

            if self.maximize:
                update = param.value + self.lr * grad
            else:
                update = param.value - self.lr * grad

            # Necessary to prevent scary bugs
            # TODO: Implement a better way to handle this
            update.reset_history()
            param.update(update)
