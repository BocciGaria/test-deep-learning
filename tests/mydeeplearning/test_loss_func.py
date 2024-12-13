from decimal import Decimal
import os
import pytest
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

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


def test_cross_entropy_error(good_data, bad_data):
    error = cross_entropy_error(good_data["y"], good_data["t"])
    assert Decimal("0.51082546") == Decimal(error).quantize(Decimal("1e-8"))
    error = cross_entropy_error(bad_data["y"], bad_data["t"])
    assert Decimal("2.30258409") == Decimal(error).quantize(Decimal("1e-8"))


def test_cross_entropy_error_batch(good_data, bad_data):
    y = [good_data["y"], bad_data["y"]]
    t = [good_data["t"], bad_data["t"]]
    error = cross_entropy_error(y, t)
    assert Decimal("1.40670478") == Decimal(error).quantize(Decimal("1e-8"))
