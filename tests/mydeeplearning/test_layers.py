import numpy

from mydeeplearning.layers import ReLULayer, SigmoidLayer, AffineLayer


def test_relu_layer():
    l = ReLULayer()
    x = numpy.array([[1.0, -0.5], [-2.0, 3.0]])
    y = l.forward(x)
    assert numpy.allclose([[1.0, 0.0], [0, 3.0]], y)
    dout = l.backward(numpy.array([[1.0, 2.0], [-1.0, -2.0]]))
    assert numpy.allclose([[1.0, 0.0], [0.0, -2.0]], dout)


def test_sigmoid_layer():
    l = SigmoidLayer()
    x = numpy.array([[1.0, -0.5], [-2.0, 3.0]])
    y = l.forward(x)
    assert numpy.allclose([[0.73105858, 0.37754067], [0.11920292, 0.95257413]], y)
    dout = l.backward(numpy.array([[1.0, 2.0], [-1.0, -2.0]]))
    assert numpy.allclose([[0.19661193, 0.47000742], [-0.10499359, -0.09035332]], dout)


def test_affine_layer():
    W = numpy.array([[1, 3, 5], [2, 4, 6]])
    b = numpy.array([0.5, 1.0, 1.5])
    l = AffineLayer(W, b)
    x = numpy.array([[6, 3], [5, 2], [4, 1]])
    assert numpy.allclose(
        [[12.5, 31.0, 49.5], [9.5, 24.0, 38.5], [6.5, 17.0, 27.5]], l.forward(x)
    )
    dout = numpy.array([[3, 2, 1], [4, 5, 6], [-1, -2, -3]])
    assert numpy.allclose([[14, 20], [49, 64], [-22, -28]], l.backward(dout))
