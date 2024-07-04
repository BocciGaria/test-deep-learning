"""活性化関数

作成日   : 2024/07/04
作成者   : BocciGaria
最終更新日: 2024/07/04
最終更新者: BocciGaria
"""

import numpy


def step_function(x):
    """ステップ関数"""
    return numpy.array(x > 0, dtype=numpy.int16)
