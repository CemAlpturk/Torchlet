from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Protocol


# Is this needed?
class Variable(Protocol):

    def accumulate_derivative(self, x: Any) -> None: ...

    @property
    def unique_id(self) -> int: ...

    def is_leaf(self) -> bool: ...

    def is_constant(self) -> bool: ...

    @property
    def parents(self) -> Iterable[Variable]: ...

    def chain_rule(self, d_output: Any) -> Iterable[tuple[Variable, Any]]: ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable (Variable): The right-most variable.

    Returns:
        Iterable[Variable]: The topological order of the computation graph with non-constant variables.
    """
    marked: set[int] = set()
    result: list[Variable] = []

    def dfs(v: Variable) -> None:
        if v.is_constant():
            return

        marked.add(v.unique_id)

        for w in v.parents:
            if w.unique_id not in marked:
                dfs(w)

        result.append(v)

    dfs(variable)

    result = list(reversed(result))
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute the derivatives for the leave nodes.

    Args:
        variable (Variable): The right-most variable.
        deriv (Any): The derivative of the output to be backpropagated.
    """

    ordered_vars: Iterable[Variable] = topological_sort(variable)
    derivatives: dict[int, Any] = {var.unique_id: 0 for var in ordered_vars}
    derivatives[variable.unique_id] = deriv

    for var in ordered_vars:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
        else:
            d_output = derivatives[var.unique_id]
            for parent_var, deriv in var.chain_rule(d_output):
                if parent_var.is_constant():
                    continue
                if parent_var.unique_id in derivatives:
                    derivatives[parent_var.unique_id] += deriv
                else:
                    # This might be redundant
                    derivatives[parent_var.unique_id] = deriv


@dataclass
class Context:
    """
    Context for the computation graph to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """
        Store the given `values` for the backward pass.
        """
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> tuple[Any, ...]:
        return self.saved_values
