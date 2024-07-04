import numpy

from mydeeplearning.activation import step_function


def test_step_function():
    assert step_function(1) == 1
    assert step_function(0) == 0
    assert step_function(-1) == 0

def test_step_function_array():
    out = step_function(numpy.array([1, 0, -1]))
    result = out == numpy.array([1, 0, 0])
    assert result.all()
