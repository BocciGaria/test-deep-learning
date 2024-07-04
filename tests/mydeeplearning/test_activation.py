import numpy

from mydeeplearning.activation import step_function, relu, sigmoid


def test_step_function():
    assert step_function(0.1) == 1
    assert step_function(0) == 0
    assert step_function(-0.1) == 0


def test_step_function_array():
    out = step_function(numpy.array([0.1, 0, -0.1]))
    result = out == numpy.array([1, 0, 0])
    assert result.all()


def test_relu():
    assert relu(0.1) == 0.1
    assert relu(1.1) == 1.1
    assert relu(0) == 0
    assert relu(-0.1) == 0


def test_relu_array():
    out = relu(numpy.array([0.1, 1.1, 0, -0.1]))
    result = out == numpy.array([0.1, 1.1, 0, 0])
    assert result.all()


def test_sigmoid():
    assert numpy.isclose(sigmoid(-1), 0.26894142)
    assert sigmoid(0) == 0.5
    assert numpy.isclose(sigmoid(1), 0.73105858)
    assert numpy.isclose(sigmoid(2), 0.88079708)


def test_sigmoid_array():
    out = sigmoid(numpy.array([-1, 0, 1, 2]))
    assert numpy.allclose(out, numpy.array([0.26894142, 0.5, 0.73105858, 0.88079708]))
