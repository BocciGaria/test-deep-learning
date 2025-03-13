import numpy as np

from mydeeplearning.layers import (
    ReLULayer,
    SigmoidLayer,
    AffineLayer,
    SoftmaxWithLossLayer,
)


def test_relu_layer():
    l = ReLULayer()
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    y = l.forward(x)
    assert np.allclose([[1.0, 0.0], [0, 3.0]], y)
    dout = l.backward(np.array([[1.0, 2.0], [-1.0, -2.0]]))
    assert np.allclose([[1.0, 0.0], [0.0, -2.0]], dout)


def test_sigmoid_layer():
    l = SigmoidLayer()
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    y = l.forward(x)
    assert np.allclose([[0.73105858, 0.37754067], [0.11920292, 0.95257413]], y)
    dout = l.backward(np.array([[1.0, 2.0], [-1.0, -2.0]]))
    assert np.allclose([[0.19661193, 0.47000742], [-0.10499359, -0.09035332]], dout)


def test_affine_layer():
    W = np.array([[1, 3, 5], [2, 4, 6]])
    b = np.array([0.5, 1.0, 1.5])
    l = AffineLayer(W, b)
    x = np.array([[6, 3], [5, 2], [4, 1]])
    assert np.allclose(
        [[12.5, 31.0, 49.5], [9.5, 24.0, 38.5], [6.5, 17.0, 27.5]], l.forward(x)
    )
    dout = np.array([[3, 2, 1], [4, 5, 6], [-1, -2, -3]])
    assert np.allclose([[14, 20], [49, 64], [-22, -28]], l.backward(dout))


def test_softmaxwithloss_layer():
    layer = SoftmaxWithLossLayer()
    # softmaxの出力が[3, 2, 5]となるように入力を[log(3), log(2), log(5)]とする
    loss = layer.forward([np.log(3), np.log(2), np.log(5)], [0, 1, 0])
    assert loss.dtype == np.floating
    assert np.isclose(loss, 1.60943741, atol=1e-8, rtol=1e-8)
    dout = layer.backward()
    assert np.allclose(dout, [0.3, -0.8, 0.5], atol=1e-8, rtol=1e-8)
