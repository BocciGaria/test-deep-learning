import math
import pytest

from mydeeplearning.computational_graph import MulNode


def test_mul_layer_forward():
    l_1 = MulNode()
    l_2 = MulNode()
    al_1 = l_1.forward(100.0, 2.0)
    al_2 = l_2.forward(al_1, 1.1)
    assert math.isclose(200, al_1)
    assert math.isclose(220.0, al_2)


def test_mul_layer_backward():
    l_1 = MulNode()
    l_2 = MulNode()
    al_1 = l_1.forward(100.0, 2.0)
    al_2 = l_2.forward(al_1, 1.1)
    dx_2, dy_2 = l_2.backward(1)
    dx_1, dy_1 = l_1.backward(dx_2)
    assert math.isclose(1.1, dx_2)
    assert math.isclose(200.0, dy_2)
    assert math.isclose(2.2, dx_1)
    assert math.isclose(110, dy_1)
