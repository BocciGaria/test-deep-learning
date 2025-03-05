import numpy


def logic_gate(x: numpy.ndarray, w, b):
    """論理ゲート"""
    if x.ndim > 1:
        raise ValueError("入力値の次元数が多すぎます。x.ndim: %s" % x.ndim)
    if not numpy.isin(x, (0, 1, True, False)).all():
        raise ValueError(
            "入力値は[1, 0, True, False]のいずれかでなくてはなりません。x: %s" % x
        )
    y = numpy.sum(x * w) + b
    if y <= 0:
        return 0
    else:
        return 1


def AND(x1, x2):
    """ANDゲート"""
    x = numpy.array([x1, x2])
    w = numpy.array([0.5, 0.5])
    b = -0.7
    return logic_gate(x, w, b)


def OR(x1, x2):
    """ORゲート"""
    x = numpy.array([x1, x2])
    w = numpy.array([0.5, 0.5])
    b = -0.2
    y = numpy.sum(x * w) + b
    return logic_gate(x, w, b)


def NAND(x1, x2):
    """NANDゲート"""
    x = numpy.array([x1, x2])
    w = numpy.array([-0.5, -0.5])
    b = 0.7
    y = numpy.sum(x * w) + b
    return logic_gate(x, w, b)


def XOR(x1, x2):
    """XORゲート"""
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)
