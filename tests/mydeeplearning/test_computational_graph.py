import math
import pytest

from mydeeplearning.computational_graph import MulNode, AddNode


def test_mul_layer_forward():
    n_1 = MulNode()
    n_2 = MulNode()
    an_1 = n_1.forward(100.0, 2.0)
    an_2 = n_2.forward(an_1, 1.1)
    assert math.isclose(200, an_1)
    assert math.isclose(220.0, an_2)


def test_mul_layer_backward():
    n_1 = MulNode()
    n_2 = MulNode()
    an_1 = n_1.forward(100.0, 2.0)
    an_2 = n_2.forward(an_1, 1.1)
    dx_2, dy_2 = n_2.backward(1)
    dx_1, dy_1 = n_1.backward(dx_2)
    assert math.isclose(1.1, dx_2)
    assert math.isclose(200.0, dy_2)
    assert math.isclose(2.2, dx_1)
    assert math.isclose(110, dy_1)


def test_add_node_forward():
    n = AddNode()
    assert math.isclose(650.0, n.forward(200.0, 450.0))


def test_add_node_backward():
    n = AddNode()
    n.forward(200.0, 450.0)
    dx, dy = n.backward(1.1)
    assert math.isclose(1.1, dx)
    assert math.isclose(1.1, dy)
