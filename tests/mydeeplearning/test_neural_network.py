import numpy
import pytest

from mydeeplearning.activation import identity_function, sigmoid, softmax
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
        Layer(weight3, bias3, identity_function)
    )


def test_generate_network():
    net = NeuralNetwork()


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


def test_forward_multiple_layers(layers):
    layers[0].forwarding_to(layers[1])
    layers[1].forwarding_to(layers[2])
    y = layers[0].forward([1.0, 0.5])
    assert numpy.allclose(y, [0.31682708, 0.69627909])


def test_get_remaining_forwarding_length(layers):
    layers[0].forwarding_to(layers[1])
    layers[1].forwarding_to(layers[2])
    assert layers[0].remaining_forwarding_len == 3


def test_add_hidden_layer(layers):
    net = NeuralNetwork()
    net.add(*layers)
    assert net.dim == 3


def test_forward_neural_network(layers):
    net = NeuralNetwork(softmax)
    net.add(*layers)
    y = net.forward([1.0, 0.5])
    assert numpy.allclose(y, [0.40625907, 0.59374093])
