import numpy


def AND(x1, x2):
    """ANDゲート"""
    x = numpy.array([x1, x2])
    w = numpy.array([0.5, 0.5])
    b = -0.7
    tmp = numpy.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    """ORゲート"""
    x = numpy.array([x1, x2])
    w = numpy.array([0.5, 0.5])
    b = -0.2
    tmp = numpy.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    """NANDゲート"""
    x = numpy.array([x1, x2])
    w = numpy.array([-0.5, -0.5])
    b = 0.7
    tmp = numpy.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
