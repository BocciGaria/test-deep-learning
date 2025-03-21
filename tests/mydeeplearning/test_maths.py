from decimal import Decimal
import numpy
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mydeeplearning.maths import (
    numerical_diff,
    sum_of_square,
    numerical_gradient,
    gradient_descent,
    numerical_gradient_dirty,
)


def test_numerical_diff():
    f = lambda x: 0.01 * x**2 + 0.1 * x
    assert Decimal("0.2000") == Decimal(numerical_diff(f, 5)).quantize(Decimal("1e-4"))
    assert Decimal("0.3000") == Decimal(numerical_diff(f, 10)).quantize(Decimal("1e-4"))


def test_sum_of_square():
    assert Decimal("25.0000") == Decimal(sum_of_square([3.0, 4.0])).quantize(
        Decimal("1e-4")
    )
    assert numpy.array_equal([25.0000, 7.8125], sum_of_square([[3.0, 4.0], [1.25, 2.5]]))


def test_numerical_gradient():
    gradient = numerical_gradient(sum_of_square, [3.0, 4.0])
    assert numpy.allclose([6.0, 8.0], gradient)
    gradient = numerical_gradient(sum_of_square, [0.0, 2.0])
    assert numpy.allclose([0.0, 4.0], gradient)
    gradient = numerical_gradient(sum_of_square, [3.0, 0.0])
    assert numpy.allclose([6.0, 0.0], gradient)


def test_multidimentional_numerical_gradient():
    f = lambda params: numpy.sum([sum_of_square(x) for x in params])
    gradient = numerical_gradient(f, [[3.0, 4.0], [0.0, 2.0], [3.0, 0.0]])
    assert numpy.allclose([[6.0, 8.0], [0.0, 4.0], [6.0, 0.0]], gradient)


def test_numerical_gradient_dirty():
    f = lambda params: numpy.sum([sum_of_square(x) for x in params])
    gradient = numerical_gradient_dirty(
        f, numpy.array([[3.0, 4.0], [0.0, 2.0], [3.0, 0.0]])
    )
    assert numpy.allclose([[6.0, 8.0], [0.0, 4.0], [6.0, 0.0]], gradient)
    try:
        numerical_gradient_dirty(f, [[3.0, 4.0], [0.0, 2.0], [3.0, 0.0]])
        assert False
    except ValueError as e:
        assert str(e).startswith("The type of x should be numpy.ndarray ")


def test_gradient_descent():
    gradient = gradient_descent(sum_of_square, [-3.0, 4.0], 0.1, 100)
    assert numpy.allclose(
        [-6.11110793e-10, 8.14814391e-10], gradient, rtol=0.0, atol=1e-18
    )
