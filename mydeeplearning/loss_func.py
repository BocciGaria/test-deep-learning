"""損失関数

作成日   : 2024/07/06
作成者   : BocciGaria
最終更新日: 2024/07/06
最終更新者: BocciGaria
"""

import numpy


def sum_squared_error(y, t):
    """二乗和誤差"""
    nda_y = numpy.array(y)
    nda_t = numpy.array(t)
    return numpy.sum((nda_y - nda_t) ** 2) * 0.5


def cross_entropy_error(y, t):
    """交差エントロピー誤差"""
    delta = 1e-7
    nda_y = numpy.array(y)
    nda_t = numpy.array(t)
    return -numpy.sum(nda_t * numpy.log(nda_y + delta))
