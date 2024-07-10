from decimal import Decimal
import numpy
import pytest

from mydeeplearning.activation_func import identity_function, sigmoid, softmax
from mydeeplearning.neural_network import NeuralNetwork, Layer


@pytest.fixture
def layers():
    weight1 = [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
    weight2 = [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]
    weight3 = [[0.1, 0.3], [0.2, 0.4]]
    bias1 = [0.1, 0.2, 0.3]
    bias2 = [0.1, 0.2]
    bias3 = [0.1, 0.2]
    return (
        Layer(weight1, bias1, sigmoid),
        Layer(weight2, bias2, sigmoid),
        Layer(weight3, bias3, identity_function),
    )


def test_generate_layer():
    weight = [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
    bias = [1.0, 2.0, 3.0]
    l = Layer(weight, bias, softmax)


def test_forward_layer():
    weight = [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
    bias = [1.0, 2.0, 3.0]
    l = Layer(weight, bias, sigmoid)
    y = l.forward([1.0, 0.5])
    assert numpy.allclose(y, [0.76852478, 0.92414182, 0.97811873])


def test_forward_bad_layer():
    weight = [[1, 1, 1], [1, 1, 1]]
    bias = [1, 1]
    l = Layer(weight, bias)
    try:
        l.forward([1, 1])
        assert False
    except RuntimeError as e:
        assert (
            str(e)
            == "The length did not match between the weights and the input or the bias."
        )


def test_connect_layers(layers):
    layers[0].connect(layers[1])
    layers[1].connect(layers[2])
    assert layers[0].len_forward == 3
    assert layers[1].len_forward == 2
    assert layers[2].len_forward == 1


def test_forward_multiple_layers(layers):
    layers[0].connect(layers[1])
    layers[1].connect(layers[2])
    y = layers[0].forward([1.0, 0.5])
    assert numpy.allclose(y, [0.31682708, 0.69627909])


def test_add_hidden_layers(layers):
    net = NeuralNetwork()
    net.add(*layers)
    assert net.ndim == 3


def test_add_hidden_layers_one_by_one(layers):
    net = NeuralNetwork()
    net.add(layers[0])
    net.add(layers[1])
    net.add(layers[2])
    assert net.ndim == 3


def test_forward_neural_network(layers):
    net = NeuralNetwork()
    net.add(*layers)
    y = net.forward([1.0, 0.5])
    assert numpy.allclose(y, [0.31682708, 0.69627909])


def test_neural_network_loss(layers):
    net = NeuralNetwork()
    net.add(*layers)
    loss = net.loss([1.0, 0.5], [0, 1])
    assert Decimal("0.5213") == Decimal(loss).quantize(Decimal("1e-4"))


def test_neural_network_accuracy(layers):
    net = NeuralNetwork()
    net.add(*layers)
    accuracy = net.accuracy([[1.0, 0.5], [2.0, 3.5]], [[0, 1], [1, 0]])
    assert Decimal("0.5") == Decimal(accuracy).quantize(Decimal("1e-1"))
