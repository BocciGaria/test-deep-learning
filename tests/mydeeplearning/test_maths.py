from decimal import Decimal
import numpy

from mydeeplearning.maths import numerical_diff, numerical_gradient


def test_numerical_diff():
    f = lambda x: 0.01 * x**2 + 0.1 * x
    assert Decimal("0.2000") == Decimal(numerical_diff(f, 5)).quantize(Decimal("1e-4"))
    assert Decimal("0.3000") == Decimal(numerical_diff(f, 10)).quantize(Decimal("1e-4"))


def test_numerical_gradient():
    f = lambda x: numpy.sum(x[0]**2 + x[1]**2)
    gradient = numerical_gradient(f, [3.0, 4.0])
    assert numpy.allclose([6.0, 8.0], gradient)
    gradient = numerical_gradient(f, [0.0, 2.0])
    assert numpy.allclose([0.0, 4.0], gradient)
    gradient = numerical_gradient(f, [3.0, 0.0])
    assert numpy.allclose([6.0, 0.0], gradient)
