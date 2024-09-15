import torchlet
from torchlet.nn import Module, Parameter


class Linear(Module):
    """Linear Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if in_features < 1:
            raise ValueError("in_features must be greater than 0")
        if out_features < 1:
            raise ValueError("out_features must be greater than 0")

        self.weight = Parameter(
            torchlet.rand((in_features, out_features), requires_grad=True),
            "weight",
        )
        if bias:
            self.bias = Parameter(
                torchlet.rand((out_features,), requires_grad=True), "bias"
            )
        else:
            self.bias = None

    def forward(self, x: torchlet.Tensor) -> torchlet.Tensor:
        W = self.weight.value
        z = x @ W
        if self.bias is not None:
            z += self.bias.value

        return z

    def __repr__(self) -> str:
        return f"Linear(in_features={self.weight.value.shape[0]}, out_features={self.weight.value.shape[1]}, bias={self.bias is not None})"
