import numpy as np
import pytest

from mydeeplearning.neuralnet import forward, init_network, SimpleNet, TwoLayerNet


def test_forward_network():
    x = [1.0, 0.5]
    y = forward(init_network(), x)
    assert np.allclose(y, np.array([0.31682708, 0.69627909]), rtol=1e-8, atol=1e-8)
    # assert not np.allclose(y, np.array([0.31682713, 0.69627914]), rtol=1e-8, atol=1e-8)
    # assert not np.allclose(y, np.array([0.31682703, 0.69627904]), rtol=1e-8, atol=1e-8)


@pytest.fixture
def simple_net():
    return SimpleNet()


def test_simplenet_instance(simple_net):
    assert simple_net.weights.shape == (2, 3)
    assert simple_net.weights.dtype == np.floating


def test_simplenet_prediction(simple_net):
    prediction = simple_net.predict([0, 0])
    assert prediction.shape == (3,)
    assert prediction.dtype == np.floating


def test_simplenet_loss(simple_net):
    loss = simple_net.loss([0, 0], [0, 0, 1])
    assert loss.ndim == 0
    assert loss.dtype == np.floating


@pytest.fixture
def two_layer_net_mnist():
    return TwoLayerNet(784, 100, 10)


@pytest.fixture
def input_mnist():
    return np.random.rand(100, 784)


@pytest.fixture
def teacher_mnist():
    return np.random.rand(100, 10)


def test_twolayernet_instance(two_layer_net_mnist):
    assert two_layer_net_mnist.params["W1"].shape == (784, 100)
    assert two_layer_net_mnist.params["b1"].shape == (100,)
    assert two_layer_net_mnist.params["W2"].shape == (100, 10)
    assert two_layer_net_mnist.params["b2"].shape == (10,)


def test_twolayernet_prediction(two_layer_net_mnist, input_mnist):
    y = two_layer_net_mnist.predict(input_mnist)
    assert y.shape == (100, 10)
    assert y.dtype == np.floating
    assert np.all([y >= 0] and [y <= 1])


def test_twolayernet_loss(two_layer_net_mnist, input_mnist, teacher_mnist):
    loss = two_layer_net_mnist.loss(input_mnist, teacher_mnist)
    assert loss.ndim == 0
    assert loss.dtype == np.floating


def test_twolayernet_accuracy(two_layer_net_mnist, input_mnist, teacher_mnist):
    a = two_layer_net_mnist.accuracy(input_mnist, teacher_mnist)
    assert a.ndim == 0
    assert a.dtype == np.floating
    assert a >= 0 and a <= 1


def test_twolayernet_numerical_gradient(
    two_layer_net_mnist, input_mnist, teacher_mnist
):
    # grads = two_layer_net_mnist.numerical_gradient(input_mnist, teacher_mnist)
    # assert grads["W1"].shape == (784, 100)
    # assert grads["b1"].shape == (100,)
    # assert grads["W2"].shape == (100, 10)
    # assert grads["b2"].shape == (10,)
    # 処理が遅いのでデータ数を減らす
    net = TwoLayerNet(64, 15, 10)
    x = np.random.rand(5, 64)
    t = np.random.rand(5, 10)
    grads = net.numerical_gradient(x, t)
    assert grads["W1"].shape == (64, 15)
    assert grads["b1"].shape == (15,)
    assert grads["W2"].shape == (15, 10)
    assert grads["b2"].shape == (10,)
