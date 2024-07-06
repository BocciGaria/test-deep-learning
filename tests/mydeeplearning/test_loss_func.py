from decimal import Decimal
import pytest

from mydeeplearning.loss_func import sum_squared_error, cross_entropy_error


@pytest.fixture
def good_data():
    return {
        "y": [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
        "t": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    }


@pytest.fixture
def bad_data():
    return {
        "y": [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0],
        "t": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    }


def test_sum_squared_error(good_data, bad_data):
    error = sum_squared_error(good_data["y"], good_data["t"])
    assert Decimal("0.0975") == Decimal(error).quantize(Decimal("1e-4"))
    error = sum_squared_error(bad_data["y"], bad_data["t"])
    assert Decimal("0.5975") == Decimal(error).quantize(Decimal("1e-4"))
