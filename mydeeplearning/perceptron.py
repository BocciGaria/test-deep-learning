import numpy


def AND(x1, x2):
    """ANDゲート"""
    x = numpy.array([x1, x2])
    w = numpy.array([0.5, 0.5])
    b = -0.7
    y = numpy.sum(x * w) + b
    if y <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    """ORゲート"""
    x = numpy.array([x1, x2])
    w = numpy.array([0.5, 0.5])
    b = -0.2
    y = numpy.sum(x * w) + b
    if y <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    """NANDゲート"""
    x = numpy.array([x1, x2])
    w = numpy.array([-0.5, -0.5])
    b = 0.7
    y = numpy.sum(x * w) + b
    if y <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    """XORゲート"""
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)
