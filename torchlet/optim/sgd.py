from torchlet import Tensor
from torchlet.optim.base import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(
        self,
        params: list[Tensor],
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

        Args:
            params (list[Tensor]): List of parameters to optimize.
            lr (float, optional): Learning rate (default: 0.001).
            momentum (float, optional): Momentum factor (default: 0.0).
            weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.0).
            dampening (float, optional): Dampening for momentum (default: 0.0).
            nesterov (bool, optional): Enables Nesterov momentum (default: False).
            maximize (bool, optional): Whether to maximize the objective function (default: False).
        """
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize

        self.momentum_buffer = list(None for _ in self.params)

    def step(self) -> None:

        for param, momentum_buff in zip(self.params, self.momentum_buffer):
            grad = param.grad

            if self.weight_decay != 0:
                # L2 regularization
                grad += self.weight_decay * param.data

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
                param.data += self.lr * grad
            else:
                param.data -= self.lr * grad
