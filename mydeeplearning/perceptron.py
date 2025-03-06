import numpy as np


def perceptron(x, w, b) -> int:
    """論理ゲート"""
    try:
        x = np.array(x)
        w = np.array(w)
    except Exception as e:
        e.add_note("入力値または重みはnumpy.arrayで変換できる必要があります。")
        raise e
    if not (x.ndim == 1 and w.ndim == 1):
        raise ValueError("入力値の次元数は1としてください。")
    if not np.isin(x, (0, 1)).all():
        raise ValueError("入力値は[0, 1]のいずれかでなくてはなりません。")
    try:
        y = np.sum(x * w) + b
        if y <= 0:
            return 0
        else:
            return 1
    except Exception as e:
        e.add_note("入力値・重み・バイアスの組み合わせを確認してください。")
        raise e


def AND(x1, x2):
    """ANDゲート

    単層パーセプトロンによるANDゲートの実装"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    return perceptron(x, w, b)


def OR(x1, x2):
    """ORゲート

    単層パーセプトロンによるORゲートの実装"""
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    return perceptron(x, w, b)


def NAND(x1, x2):
    """NANDゲート

    単層パーセプトロンによるNANDゲートの実装"""
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    return perceptron(x, w, b)


def XOR(x1, x2):
    """XORゲート

    2層パーセプトロンによるXORゲートの実装"""
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)
