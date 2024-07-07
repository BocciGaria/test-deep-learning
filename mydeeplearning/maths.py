"""基本的な数学の公式

作成日   : 2024/07/07
作成者   : BocciGaria
最終更新日: 2024/07/07
最終更新者: BocciGaria
"""


def numerical_diff(f, x):
    """数値微分"""
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)
