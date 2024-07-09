"""基本的な数学の公式

作成日   : 2024/07/07
作成者   : BocciGaria
最終更新日: 2024/07/07
最終更新者: BocciGaria
"""

import numpy


def sum_of_square(x):
    """二乗和

    Parameters
    ----------
    x : array
        数値配列

    Returns
    -------
    float
    """
    square = numpy.array(x) ** 2
    if square.ndim == 1:
        return numpy.sum(square)
    else:
        return numpy.sum(square, axis=1)


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
        関数fへのパラメータ

    Returns
    -------
    array[float]
        関数fのパラメータxにおける出力値の勾配
    """
    result = numpy.zeros_like(x)
    h = 1e-4
    nda_x = numpy.array(x)
    iter = numpy.nditer(nda_x, flags=["multi_index"], op_flags=["readwrite"])
    while not iter.finished:
        index = iter.multi_index
        increased = nda_x.copy()
        increased[index] += h
        decreased = nda_x.copy()
        decreased[index] -= h
        result[index] = (f(increased) - f(decreased)) / (2 * h)
        iter.iternext()
    return result


def gradient_descent(f, initial_x, lr, step):
    """勾配降下法

    Parameters
    ----------
    f : function
        複数のパラメータを持つ関数
    initial_x : array
        関数fに最初に与えるパラメータ群
    lr : float
        勾配を小さくするためのステップごとのパラメータ更新量
    step : int
        パラメータの更新回数

    Returns
    -------
    array[float]
        適切な更新量によって十分な回数更新を行った場合、関数fのパラメータの極小値が得られる
    """
    x = numpy.array(initial_x)
    for _ in range(step):
        gradient = numerical_gradient(f, x)
        x -= gradient * lr
    return x
