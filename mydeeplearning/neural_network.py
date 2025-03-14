"""ニューラルネットワーク

作成日   : 2024/07/05
作成者   : BocciGaria
最終更新日: 2024/07/05
最終更新者: BocciGaria
"""

import numpy

from mydeeplearning.activation_func import identity_function, softmax
from mydeeplearning.loss_func import cross_entropy_error
from mydeeplearning.maths import numerical_gradient_dirty


class Layer:
    """ネットワークの層"""

    _nl: "Layer" = None

    def __init__(self, weight, bias, activation=identity_function) -> None:
        """
        ニューラルネットワークを構成する層を生成する

        Parameters
        ----------
        weight : array
            この層への入力数に応じた重み（入力と出力がともに１つでない限り配列）
        bias : array
            この層の出力数に応じたバイアス（出力が１つでない限り配列）
        activation : function
            この層のニューロンで実行する活性化関数
        """
        self.W = numpy.array(weight)
        self.b = numpy.array(bias)
        self._h = activation

    def forward(self, x) -> numpy.ndarray:
        """入力から出力への伝達処理

        Parameters
        ----------
        x : array
            入力値（入力が１つでない限り配列）

        Returns
        -------
        numpy.ndarray
            出力値（ニューロンが１つでない限り配列）
        """
        try:
            a = numpy.dot(x, self.W) + self.b
        except ValueError:
            raise RuntimeError(
                "The length did not match between the weights and the input or the bias."
            )
        if self._nl is not None:
            return self._nl.forward(self._h(a))
        else:
            return self._h(a)

    def connect(self, layer) -> None:
        """フォワード時の伝達先層を接続する"""
        self._nl = layer

    @property
    def len_forward(self) -> int:
        """フォワード時の残りの伝達回数（出力方向にある層の数に等しい）"""
        if self._nl is None:
            return 1
        else:
            return self._nl.len_forward + 1

    @property
    def last_layer(self) -> "Layer":
        """フォワード時の最終伝達先層"""
        if self._nl is None:
            return self
        else:
            return self._nl.last_layer


class NeuralNetwork(Layer):
    """ニューラルネットワーク"""

    def __init__(self) -> None:
        """
        ニューラルネットワークを生成する
        """
        super().__init__(1, 0, identity_function)
        self._layers = []

    def add(self, *layers: Layer) -> None:
        """隠し層を追加する
        追加した順に接続され、新しい層は出力方向の最後の層の後ろに接続される

        Parameters
        ----------
        *layers : tuple[Layer]
            追加する隠し層（複数追加する場合は引数に指定した順に追加される）
        """
        self._layers.extend(layers)
        if self._nl is None:
            self.connect(layers[0])
        else:
            self.last_layer.connect(layers[0])
        previous = layers[0]
        for next in layers[1:]:
            previous.connect(next)
            previous = next

    def loss(self, x, t):
        """損失関数の値
        交差エントロピー誤差による損失を取得数

        Parameters
        ----------
        x : array
            ニューラルネットワークへの入力
        t : array
            ont-hot形式の教師データ

        Returns
        -------
        float
            損失
        """
        y = softmax(self.forward(x))
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """精度を取得する
        フォワード処理をバッチ実行し出力の精度を求める

        Parameters
        ----------
        x : numpy.array
            ニューラルネットワークへの入力群（2次元配列）
        t : array
            one-hot形式の教師データ群（2次元配列）

        Returns
        -------
        float
            精度
        """
        y = self.forward(x)
        return numpy.sum(numpy.argmax(y, axis=1) == numpy.argmax(t, axis=1)) / len(x)

    def gradient(self, x, t):
        """パラメータに対する勾配を取得する

        Parameters
        ----------
        x : array
            ニューラルネットワークへの入力群
        t : array
            one-hot形式の教師データ群

        Returns
        -------
        array
            勾配
        """
        result = []
        f = lambda p: self.loss(x, t)
        for l in self._layers:
            result.append(
                {
                    "W": numerical_gradient_dirty(f, l.W),
                    "b": numerical_gradient_dirty(f, l.b),
                }
            )
        return result

    def update(self, gradient, lr) -> None:
        """パラメータの更新
        現在のパラメータに対する勾配と学習率を使用してパラメータを更新する

        Parameters
        ----------
        gradient : array
            現在のパラメータに対する勾配
        lr : float
            学習率
        """
        for i in range(len(self._layers)):
            self._layers[i].W -= gradient[i]["W"] * lr
            self._layers[i].b -= gradient[i]["b"] * lr

    @property
    def ndim(self) -> int:
        """ネットワークの次元数
        ３層ネットワークであれば３が得られる
        """
        return self.len_forward - 1
