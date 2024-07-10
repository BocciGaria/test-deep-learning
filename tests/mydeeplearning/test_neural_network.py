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


@pytest.fixture
def net():
    return NeuralNetwork()


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


def test_add_hidden_layers(net, layers):
    net.add(*layers)
    assert net.ndim == 3


def test_add_hidden_layers_one_by_one(net, layers):
    net.add(layers[0])
    net.add(layers[1])
    net.add(layers[2])
    assert net.ndim == 3


def test_forward_neural_network(net, layers):
    net.add(*layers)
    y = net.forward([1.0, 0.5])
    assert numpy.allclose(y, [0.31682708, 0.69627909])


def test_neural_network_loss(net, layers):
    net.add(*layers)
    loss = net.loss([1.0, 0.5], [0, 1])
    assert Decimal("0.5213") == Decimal(loss).quantize(Decimal("1e-4"))


def test_neural_network_accuracy(net, layers):
    net.add(*layers)
    accuracy = net.accuracy([[1.0, 0.5], [2.0, 3.5]], [[0, 1], [1, 0]])
    assert Decimal("0.5") == Decimal(accuracy).quantize(Decimal("1e-1"))


def test_neural_network_gradient(net, layers):
    net.add(*layers)
    gradient = net.gradient([1.0, 0.5], [0, 1])
    assert numpy.allclose(
        [
            [-0.00186764, -0.00243357, -0.00268174],
            [-0.00093382, -0.00121678, -0.00134087],
        ],
        gradient[0]["W"],
    )
    assert numpy.allclose([-0.00186764, -0.00243357, -0.00268174], gradient[0]["t"])
    assert numpy.allclose(
        [
            [-0.01092468, -0.00824053],
            [-0.01270752, -0.00958533],
            [-0.01426836, -0.01076268],
        ],
        gradient[1]["W"],
    )
    assert numpy.allclose([-0.01901789, -0.01434527], gradient[1]["t"])
    assert numpy.allclose(
        [[0.25441944, -0.25441944], [0.31323004, -0.31323004]], gradient[2]["W"]
    )
    assert numpy.allclose([0.406259, -0.406259], gradient[2]["t"])
