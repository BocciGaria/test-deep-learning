"""損失関数

作成日   : 2024/07/06
作成者   : BocciGaria
最終更新日: 2024/07/06
最終更新者: BocciGaria
"""

import numpy


def sum_squared_error(y, t):
    """二乗和誤差
    単一のデータセットにのみ対応

    Parameters
    ----------
    y : array
        ニューラルネットワークの出力
    t : array
        教師データ

    Returns
    -------
    float
        二乗和誤差
    """
    nda_y = numpy.array(y)
    nda_t = numpy.array(t)
    return numpy.sum((nda_y - nda_t) ** 2) * 0.5


def cross_entropy_error(y, t):
    """交差エントロピー誤差

    Parameters
    ----------
    y : array
        ニューラルネットワークの出力（複数データセットの場合は二次元）
    t : array
        教師データ（複数データセットの場合は二次元）

    Returns
    -------
    float
        複数のデータセットの場合は交差エントロピー誤差の平均値
    """
    nda_y = numpy.array(y)
    nda_t = numpy.array(t)
    delta = 1e-7
    data_size = 1 if nda_y.ndim == 1 else nda_y.shape[0]
    return -numpy.sum(nda_t * numpy.log(nda_y + delta)) / data_size
