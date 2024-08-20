import numpy as np
from torchlet.tensor import Tensor


def test_tensor_init():
    array = np.array([1, 2, 3])
    tensor = Tensor(array)
    assert np.array_equal(tensor.array, array)
    assert tensor.requires_grad
    assert tensor.dtype == np.float32
    assert tensor.name is None
    assert tensor.args == ()
    assert tensor._id is not None


def test_tensor_backward():
    array = np.array([1, 2, 3])
    tensor = Tensor(array)
    tensor.backward()
    assert np.array_equal(tensor.grad.array, np.ones_like(array))


def test_tensor_shape():
    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array)
    assert tensor.shape == (2, 2)


def test_tensor_ndim():
    array = np.array([1, 2, 3])
    tensor = Tensor(array)
    assert tensor.ndim == 1


def test_tensor_transpose():
    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array)
    transposed = tensor.transpose()
    expected_array = np.array([[1, 3], [2, 4]])
    assert np.array_equal(transposed.array, expected_array)


def test_tensor_dot():
    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[5, 6], [7, 8]])
    tensor1 = Tensor(array1)
    tensor2 = Tensor(array2)
    dot_product = tensor1.dot(tensor2)
    expected_array = array1.dot(array2)
    assert np.array_equal(dot_product.array, expected_array)


def test_tensor_relu():
    array = np.array([-1, 2, -3])
    tensor = Tensor(array)
    relu = tensor.relu()
    expected_array = np.array([0, 2, 0])
    assert np.array_equal(relu.array, expected_array)


def test_tensor_add():
    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])
    tensor1 = Tensor(array1)
    tensor2 = Tensor(array2)
    added = tensor1 + tensor2
    expected_array = np.array([5, 7, 9])
    assert np.array_equal(added.array, expected_array)

    added.backward()

    expected = np.array([1, 1, 1])
    assert np.array_equal(tensor1.grad.array, expected)
    assert np.array_equal(tensor2.grad.array, expected)

    tensor = Tensor(array1)
    value = 5
    added = tensor + value
    expected_array = np.array([6, 7, 8])
    assert np.array_equal(added.array, expected_array)
    added.backward()
    assert np.array_equal(tensor.grad.array, np.ones_like(array1))

    tensor = Tensor(array1)
    value = 5
    added = value + tensor
    expected_array = np.array([6, 7, 8])
    assert np.array_equal(added.array, expected_array)
    added.backward()
    assert np.array_equal(tensor.grad.array, np.ones_like(array1))


def test_tensor_sub():
    array1 = np.array([4, 5, 6])
    array2 = np.array([1, 2, 3])
    tensor1 = Tensor(array1)
    tensor2 = Tensor(array2)
    subtracted = tensor1 - tensor2
    expected_array = np.array([3, 3, 3])
    assert np.array_equal(subtracted.array, expected_array)
    subtracted.backward()
    assert np.array_equal(tensor1.grad.array, np.ones_like(array1))
    assert np.array_equal(tensor2.grad.array, -np.ones_like(array2))

    tensor = Tensor(array1)
    value = 2
    subtracted = tensor - value
    expected_array = np.array([2, 3, 4])
    assert np.array_equal(subtracted.array, expected_array)
    subtracted.backward()
    assert np.array_equal(tensor.grad.array, np.ones_like(array1))

    tensor = Tensor(array1)
    value = 2
    subtracted = value - tensor
    expected_array = np.array([-2, -3, -4])
    assert np.array_equal(subtracted.array, expected_array)
    subtracted.backward()
    assert np.array_equal(tensor.grad.array, -np.ones_like(array1))


def test_tensor_mul():
    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])
    tensor1 = Tensor(array1)
    tensor2 = Tensor(array2)
    multiplied = tensor1 * tensor2
    expected_array = np.array([4, 10, 18])
    assert np.array_equal(multiplied.array, expected_array)
    multiplied.backward()
    assert np.array_equal(tensor1.grad.array, array2)
    assert np.array_equal(tensor2.grad.array, array1)

    tensor = Tensor(array1)
    value = 2
    multiplied = tensor * value
    expected_array = np.array([2, 4, 6])
    assert np.array_equal(multiplied.array, expected_array)
    multiplied.backward()
    assert np.array_equal(tensor.grad.array, value * np.ones_like(array1))

    tensor = Tensor(array1)
    value = 2
    multiplied = value * tensor
    expected_array = np.array([2, 4, 6])
    assert np.array_equal(multiplied.array, expected_array)
    multiplied.backward()
    assert np.array_equal(tensor.grad.array, value * np.ones_like(array1))


def test_tensor_pow():
    array = np.array([2, 3, 4])
    tensor = Tensor(array)
    powered = tensor**2
    expected_array = np.array([4, 9, 16])
    assert np.array_equal(powered.array, expected_array)
    powered.backward()
    assert np.array_equal(tensor.grad.array, 2 * array)


def test_tensor_div():
    array1 = np.array([4, 6, 8])
    array2 = np.array([2, 3, 4])
    tensor1 = Tensor(array1)
    tensor2 = Tensor(array2)
    divided = tensor1 / tensor2
    expected_array = np.array([2, 2, 2])
    assert np.array_equal(divided.array, expected_array)
    divided.backward()
    assert np.all(np.isclose(tensor1.grad.array, 1 / array2))
    assert np.all(np.isclose(tensor2.grad.array, -array1 / array2**2))

    tensor = Tensor(array1)
    value = 2
    divided = tensor / value
    expected_array = np.array([2, 3, 4])
    assert np.array_equal(divided.array, expected_array)
    divided.backward()
    assert np.array_equal(tensor.grad.array, 1 / value * np.ones_like(array1))

    tensor = Tensor(array1)
    value = 2
    divided = value / tensor
    expected_array = np.array([0.5, 1 / 3, 0.25])
    assert np.all(np.isclose(divided.array, expected_array))
    divided.backward()
    assert np.all(np.isclose(tensor.grad.array, -value / array1**2))


def test_tensor_comparison():
    array1 = np.array([1, 2, 3])
    array2 = np.array([2, 2, 2])
    tensor1 = Tensor(array1)
    tensor2 = Tensor(array2)
    less_than = tensor1 < tensor2
    expected_array = np.array([True, False, False])
    assert np.array_equal(less_than.array, expected_array)

    less_than_or_equal = tensor1 <= tensor2
    expected_array = np.array([True, True, False])
    assert np.array_equal(less_than_or_equal.array, expected_array)

    equal = tensor1 == tensor2
    expected_array = np.array([False, True, False])
    assert np.array_equal(equal.array, expected_array)

    not_equal = tensor1 != tensor2
    expected_array = np.array([True, False, True])
    assert np.array_equal(not_equal.array, expected_array)

    greater_than = tensor1 > tensor2
    expected_array = np.array([False, False, True])
    assert np.array_equal(greater_than.array, expected_array)

    greater_than_or_equal = tensor1 >= tensor2
    expected_array = np.array([False, True, True])
    assert np.array_equal(greater_than_or_equal.array, expected_array)


def test_tensor_getitem():
    array = np.array([1, 2, 3, 4, 5])
    tensor = Tensor(array)
    sliced = tensor[1:4]
    expected_array = np.array([2, 3, 4])
    assert np.array_equal(sliced.array, expected_array)

    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array)
    sliced = tensor[1]
    expected_array = np.array([3, 4])
    assert np.array_equal(sliced.array, expected_array)

    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array)
    sliced = tensor[:, 1]
    expected_array = np.array([2, 4])
    assert np.array_equal(sliced.array, expected_array)

    # Test for negative indices
    array = np.array([1, 2, 3, 4, 5])
    tensor = Tensor(array)
    sliced = tensor[-3:]
    expected_array = np.array([3, 4, 5])
    assert np.array_equal(sliced.array, expected_array)

    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array)
    sliced = tensor[:, -1]
    expected_array = np.array([2, 4])
    assert np.array_equal(sliced.array, expected_array)

    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array)
    sliced = tensor[-1]
    expected_array = np.array([3, 4])
    assert np.array_equal(sliced.array, expected_array)

    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array)
    sliced = tensor[-1, :]
    expected_array = np.array([3, 4])
    assert np.array_equal(sliced.array, expected_array)

    # Test gradients
    array = np.array([1, 2, 3, 4, 5])
    tensor = Tensor(array)
    sliced = tensor[1:4]
    y = sliced.sum()
    y.backward()
    expected_array = np.array([0, 1, 1, 1, 0])
    assert np.array_equal(tensor.grad.array, expected_array)


def test_tensor_setitem():
    array = np.array([1, 2, 3, 4, 5])
    tensor = Tensor(array)
    tensor[1:4] = 0
    expected_array = np.array([1, 0, 0, 0, 5])
    assert np.array_equal(tensor.array, expected_array)

    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array)
    tensor[1] = 0
    expected_array = np.array([[1, 2], [0, 0]])
    assert np.array_equal(tensor.array, expected_array)

    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array)
    tensor[:, 1] = 0
    expected_array = np.array([[1, 0], [3, 0]])
    assert np.array_equal(tensor.array, expected_array)

    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array)
    tensor[1, :] = 0
    expected_array = np.array([[1, 2], [0, 0]])
    assert np.array_equal(tensor.array, expected_array)


def test_tensor_matmul():
    # TODO: fix this test
    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[5, 6], [7, 8]])
    tensor1 = Tensor(array1)
    tensor2 = Tensor(array2)
    matmul = tensor1 @ tensor2
    expected_array = array1 @ array2
    assert np.array_equal(matmul.array, expected_array)
    matmul.backward()
    print(tensor1.grad.array)
    assert np.array_equal(tensor1.grad.array, np.ones_like(expected_array) @ array2.T)
    assert np.array_equal(tensor2.grad.array, array1.T @ np.ones_like(expected_array))


def test_numerical_grad():
    def f(x: Tensor) -> Tensor:
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        return h + q + q * x

    x1 = Tensor(10.0, dtype=np.float64)
    y1 = f(x1)

    y1.backward()

    x_grad = x1.grad.array

    # Numerical gradient
    epsilon = 1e-6
    x2 = Tensor(10.0 + epsilon, dtype=np.float64)
    y2 = f(x2)
    print(y1.array)
    print(y2.array)
    numerical_grad = (y2.array - y1.array) / epsilon

    assert np.isclose(x_grad, numerical_grad, atol=1e-3)
