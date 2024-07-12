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


def test_combine_mul_node_add_node():
    apple_unit_price = 100.0
    apple_num = 2.0
    orange_unit_price = 150.0
    orange_num = 3.0
    tax = 1.1

    apple_node = MulNode()
    orange_node = MulNode()
    add_node = AddNode()
    tax_node = MulNode()

    apple_price = apple_node.forward(apple_unit_price, apple_num)
    orange_price = orange_node.forward(orange_unit_price, orange_num)
    total_price = add_node.forward(apple_price, orange_price)
    total_price_with_tax = tax_node.forward(total_price, tax)

    d_total_price, d_tax = tax_node.backward(1)
    d_apple_price, d_orange_price = add_node.backward(d_total_price)
    d_orange_unit_price, d_orange_num = orange_node.backward(d_orange_price)
    d_apple_unit_price, d_apple_num = apple_node.backward(d_apple_price)

    assert math.isclose(715.0, total_price_with_tax)
    assert math.isclose(110.0, d_apple_num)
    assert math.isclose(2.2, d_apple_unit_price)
    assert math.isclose(165.0, d_orange_num)
    assert math.isclose(3.3, d_orange_unit_price)
    assert math.isclose(650.0, d_tax)
