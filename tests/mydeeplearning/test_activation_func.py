import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mydeeplearning.activation_func import (
    identity_function,
    step_function,
    relu,
    sigmoid,
    softmax,
)


def test_identity_function():
    assert identity_function(1) == 1
    assert np.array_equal(identity_function(np.array([-1, 0, 1])), [-1, 0, 1])


def test_step_function():
    assert step_function(0.1) == 1
    assert step_function(0) == 0
    assert step_function(-0.1) == 0


def test_step_function_array():
    out = step_function(np.array([0.1, 0, -0.1]))
    assert np.array_equal(out, [1, 0, 0])


def test_relu():
    assert relu(0.1) == 0.1
    assert relu(1.1) == 1.1
    assert relu(0) == 0
    assert relu(-0.1) == 0


def test_relu_array():
    out = relu(np.array([0.1, 1.1, 0, -0.1]))
    assert np.array_equal(out, [0.1, 1.1, 0, 0])


def test_sigmoid():
    assert np.isclose(sigmoid(-1), 0.26894142, atol=1e-8)
    assert sigmoid(0) == 0.5
    assert np.isclose(sigmoid(1), 0.73105858, atol=1e-8)
    assert np.isclose(sigmoid(2), 0.88079708, atol=1e-8)


def test_sigmoid_array():
    out = sigmoid([-1, 0, 1, 2])
    assert np.allclose(out, [0.26894142, 0.5, 0.73105858, 0.88079708], atol=1e-8)


def test_softmax():
    out = softmax([0.3, 2.9, 4.0])
    assert np.allclose(out, [0.01821127, 0.24519181, 0.73659691], atol=1e-8)
    out = softmax(np.random.random(10))
    # assert np.sum(out) == 1.0 # 丁度1.0にならないことはままある。
    assert np.isclose(np.sum(out), 1.0, atol=1e-8)
