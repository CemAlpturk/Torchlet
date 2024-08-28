from typing import Any
import numpy as np

from torchlet import Tensor


def tensor_add(a: Tensor, b: Tensor) -> Tensor:
    """
    Adds two tensors element-wise.

    Args:
        a (Tensor): The first input tensor.
        b (Tensor): The second input tensor.

    Returns:
        Tensor: The resulting tensor after element-wise addition.

    Raises:
        AssertionError: If either `a` or `b` is not a tensor.

    Notes:
        - The shape of `a` and `b` must be compatible for element-wise addition.
        - If the shapes are not compatible, broadcasting is performed.
        - The resulting tensor will have the same dtype as the input tensors.
        - If either `a` or `b` has `requires_grad` set to `True`, the resulting tensor will also have `requires_grad` set to `True`.
        - The backward function `_backward` is called during backpropagation if `requires_grad` is `True`.
    """
    assert isinstance(a, Tensor) and isinstance(
        b, Tensor
    ), "Both inputs must be tensors"

    def _backward(grad: np.ndarray) -> None:
        # TODO: Move this to seperate function
        # Broadcasting
        a.grad += grad
        if a.data.shape != b.data.shape:
            grad = grad.sum(axis=0)

        b.grad += grad

    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(
        data=a.data + b.data,
        dtype=np.result_type(a.dtype, b.dtype),
        name="tensor_add",
        args=(a, b) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def constant_add(a: Tensor, b: np.ndarray | float | int) -> Tensor:
    """
    Adds a constant value to a tensor.

    Args:
        a (Tensor): The input tensor.
        b (np.ndarray | float | int): The constant value to be added.

    Returns:
        Tensor: The tensor with the constant value added.

    Raises:
        AssertionError: If the input is not a tensor.

    """
    assert isinstance(a, Tensor), "Input must be a tensor"

    def _backward(grad: np.ndarray) -> None:
        a.grad += grad

    requires_grad = a.requires_grad
    out = Tensor(
        data=a.data + b,
        dtype=a.dtype,
        name="constant_add",
        args=(a,) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def tensor_subtract(a: Tensor, b: Tensor) -> Tensor:
    """
    Subtract two tensors element-wise.

    Args:
        a (Tensor): The first input tensor.
        b (Tensor): The second input tensor.

    Returns:
        Tensor: The resulting tensor after subtracting `b` from `a` element-wise.
    """

    assert isinstance(a, Tensor) and isinstance(
        b, Tensor
    ), "Both inputs must be tensors"

    def _backward(grad: np.ndarray) -> None:
        # Broadcasting
        a.grad += grad
        if a.data.shape != b.data.shape:
            grad = grad.sum(axis=0)

        b.grad -= grad

    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(
        data=a.data - b.data,
        dtype=np.result_type(a.dtype, b.dtype),
        name="tensor_subtract",
        args=(a, b) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def constant_subtract(a: Tensor, b: np.ndarray | float | int) -> Tensor:
    """
    Subtract a constant value from a tensor.

    Args:
        a (Tensor): The input tensor.
        b (np.ndarray | float | int): The constant value to subtract from the tensor.

    Returns:
        Tensor: The resulting tensor after subtracting the constant value.

    Raises:
        AssertionError: If the input is not a tensor.

    """
    assert isinstance(a, Tensor), "Input must be a tensor"

    def _backward(grad: np.ndarray) -> None:
        a.grad += grad

    requires_grad = a.requires_grad
    out = Tensor(
        data=a.data - b,
        dtype=a.dtype,
        name="constant_subtract",
        args=(a,) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def tensor_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiplies two tensors element-wise.

    Args:
        a (Tensor): The first input tensor.
        b (Tensor): The second input tensor.

    Returns:
        Tensor: The resulting tensor after element-wise multiplication.

    Raises:
        AssertionError: If either `a` or `b` is not a tensor.

    """
    assert isinstance(a, Tensor) and isinstance(
        b, Tensor
    ), "Both inputs must be tensors"

    def _backward(grad: np.ndarray) -> None:
        a.grad += b.data * grad
        b.grad += a.data * grad

    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(
        data=a.data * b.data,
        dtype=np.result_type(a.dtype, b.dtype),
        name="tensor_multiply",
        args=(a, b) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def constant_multiply(a: Tensor, b: np.ndarray | float | int) -> Tensor:
    """
    Multiplies a tensor `a` by a constant `b`.

    Args:
        a (Tensor): The input tensor.
        b (np.ndarray | float | int): The constant to multiply with `a`.

    Returns:
        Tensor: The resulting tensor after multiplying `a` by `b`.
    """

    assert isinstance(a, Tensor), "Input must be a tensor"

    def _backward(grad: np.ndarray) -> None:
        a.grad += b * grad

    requires_grad = a.requires_grad
    out = Tensor(
        data=a.data * b,
        dtype=a.dtype,
        name="constant_multiply",
        args=(a,) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def tensor_matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Performs matrix multiplication between two tensors.

    Args:
        a (Tensor): The first input tensor.
        b (Tensor): The second input tensor.

    Returns:
        Tensor: The resulting tensor after matrix multiplication.

    Raises:
        AssertionError: If either `a` or `b` is not a tensor.

    """
    assert isinstance(a, Tensor) and isinstance(
        b, Tensor
    ), "Both inputs must be tensors"

    def _backward(grad: np.ndarray) -> None:
        a.grad += np.tensordot(grad, b.data, axes=(-1, -1))
        indices = list(range(grad.ndim - 1))
        b.grad += np.tensordot(a.data, grad, axes=(indices, indices))

    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(
        data=np.tensordot(a.data, b.data, axes=(-1, 0)),
        dtype=np.result_type(a.dtype, b.dtype),
        name="tensor_matmul",
        args=(a, b) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def scalar_power(a: Tensor, b: float | int) -> Tensor:
    """
    Computes the element-wise power of a tensor with a scalar exponent.

    Args:
        a (Tensor): The input tensor.
        b (float | int): The scalar exponent.

    Returns:
        Tensor: The resulting tensor with each element raised to the power of the scalar exponent.
    """
    assert isinstance(a, Tensor), "Input must be a tensor"
    assert isinstance(b, (float, int)), "Exponent must be a scalar"

    def _backward(grad: np.ndarray) -> None:
        a.grad += grad * b * a.data ** (b - 1)

    requires_grad = a.requires_grad
    out = Tensor(
        data=a.data**b,
        dtype=a.dtype,
        name="scalar_power",
        args=(a,) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def sum(a: Tensor) -> Tensor:
    """
    Compute the sum of all elements in the input tensor.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The tensor containing the sum of all elements.

    Raises:
        AssertionError: If the input is not a tensor.

    Examples:
        >>> a = Tensor([1, 2, 3])
        >>> sum(a)
        Tensor(6)
    """
    # TODO: Implement sum over axis
    assert isinstance(a, Tensor), "Input must be a tensor"

    def _backward(grad: np.ndarray) -> None:
        a.grad += np.ones_like(a.data) * grad

    requires_grad = a.requires_grad
    out = Tensor(
        data=a.data.sum(),
        dtype=a.dtype,
        name="sum",
        args=(a,) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def mean(a: Tensor) -> Tensor:
    """
    Compute the mean value of a tensor over all elements.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: A new tensor containing the mean value.

    Raises:
        AssertionError: If the input is not a tensor.

    Examples:
        >>> a = Tensor([1, 2, 3, 4])
        >>> mean(a)
        Tensor(2.5)
    """
    # TODO: Implement mean over axis

    assert isinstance(a, Tensor), "Input must be a tensor"

    def _backward(grad: np.ndarray) -> None:
        grad_divisor = a.data.size
        a.grad += grad / grad_divisor

    requires_grad = a.requires_grad
    out = Tensor(
        data=a.data.mean(),
        dtype=a.dtype,
        name="mean",
        args=(a,) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def transpose(a: Tensor) -> Tensor:
    """
    Transposes the input tensor.

    Args:
        a (Tensor): The input tensor to be transposed.

    Returns:
        Tensor: The transposed tensor.

    Raises:
        AssertionError: If the input is not a tensor.

    Examples:
        >>> a = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> transpose(a)
        Tensor([[1, 4],
                [2, 5],
                [3, 6]])
    """
    assert isinstance(a, Tensor), "Input must be a tensor"

    def _backward(grad: np.ndarray) -> None:
        a.grad += grad.T

    requires_grad = a.requires_grad
    out = Tensor(
        data=a.data.T,
        dtype=a.dtype,
        name="transpose",
        args=(a,) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def relu(a: Tensor) -> Tensor:
    """
    Applies the rectified linear unit (ReLU) activation function element-wise to the input tensor.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The output tensor after applying ReLU activation function.

    Raises:
        AssertionError: If the input is not a tensor.

    Examples:
        >>> x = Tensor(data=[-1, 0, 1])
        >>> relu(x)
        Tensor(data=[0, 0, 1])
    """
    assert isinstance(a, Tensor), "Input must be a tensor"

    def _backward(grad: np.ndarray) -> None:
        # a.grad += grad * (a.data > 0)
        a.grad += np.where(a.data > 0, grad, 0)

    requires_grad = a.requires_grad
    out = Tensor(
        data=np.maximum(a.data, 0),
        dtype=a.dtype,
        name="relu",
        args=(a,) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def sigmoid(a: Tensor) -> Tensor:
    """
    Applies the sigmoid function element-wise to the input tensor.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The output tensor after applying the sigmoid function.

    Raises:
        AssertionError: If the input is not a tensor.

    Examples:
        >>> a = Tensor([1, 2, 3])
        >>> sigmoid(a)
        Tensor([0.73105858, 0.88079708, 0.95257413])
    """
    assert isinstance(a, Tensor), "Input must be a tensor"
    result = 1 / (1 + np.exp(-a.data))

    def _backward(grad: np.ndarray) -> None:
        a.grad += grad * result * (1 - result)

    requires_grad = a.requires_grad
    out = Tensor(
        data=result,
        dtype=a.dtype,
        name="sigmoid",
        args=(a,) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def tanh(a: Tensor) -> Tensor:
    """
    Applies the hyperbolic tangent function element-wise to the input tensor.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The output tensor after applying the hyperbolic tangent function.

    Raises:
        AssertionError: If the input is not a tensor.

    Examples:
        >>> x = Tensor([0, 1, -1])
        >>> tanh(x)
        Tensor([-0.        ,  0.76159416, -0.76159416])
    """
    assert isinstance(a, Tensor), "Input must be a tensor"

    result = np.tanh(a.data)

    def _backward(grad: np.ndarray) -> None:
        a.grad += grad * (1 - result**2)

    requires_grad = a.requires_grad
    out = Tensor(
        data=result,
        dtype=a.dtype,
        name="tanh",
        args=(a,) if requires_grad else None,
        backward_fn=_backward if requires_grad else None,
        requires_grad=requires_grad,
    )
    return out


def tensor_getitem(a: Tensor, idx: Any) -> Tensor:
    """
    Retrieves the specified index or indices from a tensor.

    Args:
        a (Tensor): The input tensor.
        idx (Any): The index or indices to retrieve from the tensor.

    Returns:
        Tensor: The tensor containing the specified index or indices.

    Raises:
        AssertionError: If the input is not a tensor.

    Examples:
        >>> a = Tensor([1, 2, 3, 4, 5])
        >>> tensor_getitem(a, 2)
        Tensor([3])

        >>> b = Tensor([[1, 2], [3, 4]])
        >>> tensor_getitem(b, (0, 1))
        Tensor([2])
    """

    assert isinstance(a, Tensor), "Input must be a tensor"

    def _backward(grad: np.ndarray) -> None:
        a.grad[idx] += grad

    out = Tensor(
        data=a.data[idx],
        dtype=a.dtype,
        name="getitem",
        args=(a,) if a.requires_grad else None,
        backward_fn=_backward if a.requires_grad else None,
        requires_grad=a.requires_grad,
    )
    return out


def tensor_setitem(
    a: Tensor,
    idx: Any,
    value: Tensor | np.ndarray | float | int,
) -> Tensor:
    """
    Sets the value of a tensor at the specified index.

    Args:
        a (Tensor): The input tensor.
        idx (Any): The index to set the value at.
        value (Tensor | np.ndarray | float | int): The value to set at the index.

    Returns:
        Tensor: The modified tensor.

    Raises:
        AssertionError: If the input is not a tensor.

    TODO:
        - Implement backward function.
    """
    assert isinstance(a, Tensor), "Input must be a tensor"

    if isinstance(value, Tensor):
        a.data[idx] = value.data
    else:
        a.data[idx] = value
    return a
