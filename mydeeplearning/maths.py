"""基本的な数学の公式

作成日   : 2024/07/07
作成者   : BocciGaria
最終更新日: 2024/07/07
最終更新者: BocciGaria
"""

import numpy


def numerical_diff(f, x):
    """数値微分
    ひとつのパラメータを持つ関数fのx地点における数値微分を取得する

    Parameters
    ----------
    f : function
        ひとつのパラメータを持つ関数
    x : array
        関数fへのパラメータ

    Returns
    -------
    float
        fにおけるx地点の数値微分
    """
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    """勾配
    複数のパラメータを持つ関数fのx地点における偏微分を取得する

    Parameters
    ----------
    f : function
        複数のパラメータを持つ関数
    x : array
        関数fへのパラメータ群

    Returns
    -------
    array[float]
        fにおけるx地点の勾配
    """
    result = numpy.zeros_like(x)
    h = 1e-4
    nda_x = numpy.array(x)
    for idx in range(nda_x.size):
        tmp_upper_x = nda_x.copy()
        tmp_upper_x[idx] = tmp_upper_x[idx] + h
        y_upper = f(tmp_upper_x)
        tmp_lower_x = nda_x.copy()
        tmp_lower_x[idx] = tmp_lower_x[idx] - h
        y_lower = f(tmp_lower_x)
        result[idx] = (y_upper - y_lower) / (h * 2)
    return result
