"""活性化関数

作成日   : 2024/07/04
作成者   : BocciGaria
最終更新日: 2024/07/04
最終更新者: BocciGaria
"""

import numpy


def identity_function(x):
    """恒等関数"""
    return x


def step_function(x):
    """ステップ関数"""
    return numpy.array(x > 0, dtype=numpy.int16)


def relu(x):
    """ReLU関数"""
    return numpy.maximum(0, x)


def sigmoid(x):
    """シグモイド関数"""
    return 1 / (1 + numpy.exp(-x))
