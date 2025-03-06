import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mydeeplearning.perceptron import perceptron, AND, OR, NAND, XOR


def test_perceptron_guard_block():
    try:
        perceptron([1, [1]], [1, 1], 1)
    except Exception as e:
        assert (
            "入力値または重みはnumpy.arrayで変換できる必要があります。" in e.__notes__
        )
    else:
        assert False
    try:
        perceptron([1, 1], [[1], 1], 1)
    except Exception as e:
        assert (
            "入力値または重みはnumpy.arrayで変換できる必要があります。" in e.__notes__
        )
    else:
        assert False
    try:
        perceptron(1, [1, 1], 1)
    except ValueError as e:
        assert "入力値の次元数は1としてください。" in str(e)
    else:
        assert False
    try:
        perceptron([1, 1], [[1, 1], [1, 1]], 1)
    except ValueError as e:
        assert "入力値の次元数は1としてください。" in str(e)
    else:
        assert False
    try:
        perceptron([1, 2], [1, 1], 1)
    except ValueError as e:
        assert "入力値は[0, 1]のいずれかでなくてはなりません。" in str(e)
    else:
        assert False
    try:
        perceptron([1, 1], [1, 1], "a")
    except Exception as e:
        assert "入力値・重み・バイアスの組み合わせを確認してください。" in e.__notes__
    else:
        assert False
    try:
        perceptron([1, 1], [1, 1, 1], 1)
    except Exception as e:
        assert "入力値・重み・バイアスの組み合わせを確認してください。" in e.__notes__
    else:
        assert False


def test_AND():
    assert AND(0, 0) == 0
    assert AND(1, 0) == 0
    assert AND(0, 1) == 0
    assert AND(1, 1) == 1


def test_OR():
    assert OR(0, 0) == 0
    assert OR(1, 0) == 1
    assert OR(0, 1) == 1
    assert OR(1, 1) == 1


def test_NAND():
    assert NAND(0, 0) == 1
    assert NAND(1, 0) == 1
    assert NAND(0, 1) == 1
    assert NAND(1, 1) == 0


def test_XOR():
    assert XOR(0, 0) == 0
    assert XOR(1, 0) == 1
    assert XOR(0, 1) == 1
    assert XOR(1, 1) == 0
