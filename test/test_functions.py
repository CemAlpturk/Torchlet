import numpy as np

from torchlet import Tensor
import torchlet.functions as F


def test_tensor_add():
    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])
    tensor1 = Tensor(array1, requires_grad=True)
    tensor2 = Tensor(array2, requires_grad=True)
    added = F.tensor_add(tensor1, tensor2)
    expected_array = np.array([5, 7, 9])
    assert np.array_equal(added.data, expected_array)

    added.backward()

    expected = np.array([1, 1, 1])
    assert np.array_equal(tensor1.grad, expected)
    assert np.array_equal(tensor2.grad, expected)


def test_constant_add():
    array = np.array([1, 2, 3])
    constant = 5
    tensor = Tensor(array, requires_grad=True)
    added = F.constant_add(tensor, constant)
    expected_array = np.array([6, 7, 8])
    assert np.array_equal(added.data, expected_array)

    added.backward()

    expected = np.array([1, 1, 1])
    assert np.array_equal(tensor.grad, expected)


def test_tensor_subtract():
    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])
    tensor1 = Tensor(array1, requires_grad=True)
    tensor2 = Tensor(array2, requires_grad=True)
    subtracted = F.tensor_subtract(tensor1, tensor2)
    expected_array = np.array([-3, -3, -3])
    assert np.array_equal(subtracted.data, expected_array)

    subtracted.backward()

    expected = np.array([1, 1, 1])
    assert np.array_equal(tensor1.grad, expected)
    assert np.array_equal(tensor2.grad, -expected)


def test_constant_subtract():
    array = np.array([1, 2, 3])
    constant = 5
    tensor = Tensor(array, requires_grad=True)
    subtracted = F.constant_subtract(tensor, constant)
    expected_array = np.array([-4, -3, -2])
    assert np.array_equal(subtracted.data, expected_array)

    subtracted.backward()

    expected = np.array([1, 1, 1])
    assert np.array_equal(tensor.grad, expected)


def test_tensor_multiply():
    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])
    tensor1 = Tensor(array1, requires_grad=True)
    tensor2 = Tensor(array2, requires_grad=True)
    multiplied = F.tensor_multiply(tensor1, tensor2)
    expected_array = np.array([4, 10, 18])
    assert np.array_equal(multiplied.data, expected_array)

    multiplied.backward()

    expected_grad1 = np.array([4, 5, 6])
    expected_grad2 = np.array([1, 2, 3])
    assert np.array_equal(tensor1.grad, expected_grad1)
    assert np.array_equal(tensor2.grad, expected_grad2)


def test_constant_multiply():
    array = np.array([1, 2, 3])
    constant = 5
    tensor = Tensor(array, requires_grad=True)
    multiplied = F.constant_multiply(tensor, constant)
    expected_array = np.array([5, 10, 15])
    assert np.array_equal(multiplied.data, expected_array)

    multiplied.backward()

    expected = np.array([5, 5, 5])
    assert np.array_equal(tensor.grad, expected)


def test_tensor_matmul():
    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[5, 6], [7, 8]])
    tensor1 = Tensor(array1, requires_grad=True)
    tensor2 = Tensor(array2, requires_grad=True)
    multiplied = F.tensor_matmul(tensor1, tensor2)
    expected_array = np.array([[19, 22], [43, 50]])
    assert np.array_equal(multiplied.data, expected_array)

    multiplied.backward()

    expected_grad1 = np.array([[11, 15], [11, 15]])
    expected_grad2 = np.array([[4, 4], [6, 6]])
    assert np.array_equal(tensor1.grad, expected_grad1)
    assert np.array_equal(tensor2.grad, expected_grad2)


def test_scalar_power():
    array = np.array([1, 2, 3])
    exponent = 2
    tensor = Tensor(array, requires_grad=True)
    powered = F.scalar_power(tensor, exponent)
    expected_array = np.array([1, 4, 9])
    assert np.array_equal(powered.data, expected_array)

    powered.backward()

    expected = np.array([2, 4, 6])
    assert np.array_equal(tensor.grad, expected)


def test_sum():
    array = np.array([1, 2, 3])
    tensor = Tensor(array, requires_grad=True)
    summed = F.sum(tensor)
    expected_value = 6
    assert summed.data == expected_value

    summed.backward()

    expected_grad = np.array([1, 1, 1])
    assert np.array_equal(tensor.grad, expected_grad)


def test_mean():
    array = np.array([1, 2, 3, 4])
    tensor = Tensor(array, requires_grad=True)
    mean_value = F.mean(tensor)
    expected_value = 2.5
    assert mean_value.data == expected_value

    mean_value.backward()

    expected_grad = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.array_equal(tensor.grad, expected_grad)


def test_transpose():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor = Tensor(array, requires_grad=True)
    transposed = F.transpose(tensor)
    expected_array = np.array([[1, 4], [2, 5], [3, 6]])
    assert np.array_equal(transposed.data, expected_array)

    transposed.backward()

    expected_grad = np.array([[1, 1, 1], [1, 1, 1]])
    assert np.array_equal(tensor.grad, expected_grad)


def test_relu():
    array = np.array([-1, 0, 1])
    tensor = Tensor(array, requires_grad=True)
    result = F.relu(tensor)
    expected_array = np.array([0, 0, 1])
    assert np.array_equal(result.data, expected_array)

    result.backward()

    expected_grad = np.array([0, 0, 1])
    assert np.array_equal(tensor.grad, expected_grad)


def test_sigmoid():
    array = np.array([1, 2, 3])
    tensor = Tensor(array, requires_grad=True)
    result = F.sigmoid(tensor)
    expected_array = 1 / (1 + np.exp(-array))
    assert np.allclose(result.data, expected_array)
    y = result.sum()
    y.backward()

    expected_grad = expected_array * (1 - expected_array)
    print(tensor.grad)
    print(expected_grad)
    assert np.allclose(tensor.grad, expected_grad)


def test_tanh():
    array = np.array([0, 1, -1])
    tensor = Tensor(array, requires_grad=True)
    result = F.tanh(tensor)
    expected_array = np.tanh(array)
    assert np.allclose(result.data, expected_array)
    y = result.sum()
    y.backward()

    expected_grad = 1 - expected_array**2
    assert np.allclose(tensor.grad, expected_grad)


def test_tensor_getitem():
    array = np.array([1, 2, 3, 4, 5])
    tensor = Tensor(array, requires_grad=True)
    idx = 2
    result = F.tensor_getitem(tensor, idx)
    expected_array = np.array(3)
    assert np.array_equal(result.data, expected_array)

    result.backward()

    expected_grad = np.array([0, 0, 1, 0, 0])
    assert np.array_equal(tensor.grad, expected_grad)


def test_tensor_getitem_2d():
    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array, requires_grad=True)
    idx = (0, 1)
    result = F.tensor_getitem(tensor, idx)
    expected_array = np.array(2)
    assert np.array_equal(result.data, expected_array)

    result.backward()

    expected_grad = np.array([[0, 1], [0, 0]])
    assert np.array_equal(tensor.grad, expected_grad)


def test_tensor_setitem():
    array = np.array([1, 2, 3, 4, 5])
    tensor = Tensor(array, requires_grad=True)
    idx = 2
    value = 10
    result = F.tensor_setitem(tensor, idx, value)
    expected_array = np.array([1, 2, 10, 4, 5])
    assert np.array_equal(result.data, expected_array)


def test_tensor_setitem_2d():
    array = np.array([[1, 2], [3, 4]])
    tensor = Tensor(array, requires_grad=True)
    idx = (0, 1)
    value = 10
    result = F.tensor_setitem(tensor, idx, value)
    expected_array = np.array([[1, 10], [3, 4]])
    assert np.array_equal(result.data, expected_array)
