import numpy as np


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
    nda_y = np.array(y)
    nda_t = np.array(t)
    return np.sum((nda_y - nda_t) ** 2) * 0.5


def cross_entropy_error(y, t):
    """交差エントロピー誤差

    Parameters
    ----------
    y : array
        ニューラルネットワークの出力（複数データセットの場合は二次元）
    t : array
        one-hot表現の教師データ（複数データセットの場合は二次元）

    Returns
    -------
    float
        交差エントロピー誤差または
        複数のデータセットの場合は交差エントロピー誤差の平均値
    """
    nda_y = np.array(y)
    nda_t = np.array(t)
    delta = 1e-7  # ニューラルネットワークの出力が0であるときlogが-infとなってしまうのを防止する
    # TODO: 配列の次元数が３以上の場合どうなる？
    batch_size = 1 if nda_y.ndim == 1 else nda_y.shape[0]
    return -np.sum(nda_t * np.log(nda_y + delta)) / batch_size
