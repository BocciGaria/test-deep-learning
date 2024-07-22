import numpy

from mydeeplearning.layers import ReLULayer, SigmoidLayer


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
