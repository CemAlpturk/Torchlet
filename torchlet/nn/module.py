from __future__ import annotations
from typing import Any, Sequence


class Parameter:
    """
    A `Parameter` is a special container stored in a `Module`.
    It is designed to hold a 'Variable` which is a tensor that requires gradient.
    """

    def __init__(self, x: Any, name: str | None = None) -> None:
        self.value = x
        self.name = name
        if self.name and hasattr(self.value, "name"):
            self.value.name = self.name

    def update(self, x: Any) -> None:
        """Updates the parameter value."""
        self.value = x
        if self.name and hasattr(self.value, "name"):
            self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)


class Module:
    """
    Modules form a tree that store parameters and other submodules.
    They make up the basis of neural networks.
    """

    _modules: dict[str, Module]
    _parameters: dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m: dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Set the mode of this module and all decendent modules to `train`."""
        self.training = True

        for module in self._modules.values():
            module.train()

    def eval(self) -> None:
        """Set the mode of this module and all decendent modules to `eval`."""
        self.training = False

        for module in self._modules.values():
            module.eval()

    def named_parameters(self) -> Sequence[tuple[str, Parameter]]:
        """
        Collect all the Parameters of this module and it's decendent modules.

        Returns:
            A sequence of tuples containing the name and the parameter.
        """
        named_parameters = list(
            (name, param) for name, param in self._parameters.items()
        )

        # Handle naming convention
        for name, module in self._modules.items():
            sub_module_params = [
                (f"{name}.{k}", v) for k, v in module.named_parameters()
            ]
            named_parameters.extend(sub_module_params)

        return named_parameters

    def parameters(self) -> Sequence[Parameter]:
        """
        Enumerate over all the parameters of this module and its decendents.
        """
        parameters = list(self._parameters.values())

        # Submodules
        for module in self._modules.values():
            parameters.extend(module.parameters())

        return parameters

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the module."""
        raise NotImplementedError("Forward pass not implemented.")

    def __setattr__(self, key: str, val: Any) -> None:
        # Store all Modules and Parameters in the respective dictionaries.
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

        raise AttributeError(f"Module has no attribute '{key}'")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through Module."""
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n " + "\n ".join(lines) + "\n"

        main_str += ")"
        return main_str
