from decimal import Decimal

from mydeeplearning.maths import numerical_diff


def test_numerical_diff():
    f = lambda x: 0.01 * x**2 + 0.1 * x
    assert Decimal("0.2000") == Decimal(numerical_diff(f, 5)).quantize(Decimal("1e-4"))
    assert Decimal("0.3000") == Decimal(numerical_diff(f, 10)).quantize(Decimal("1e-4"))
