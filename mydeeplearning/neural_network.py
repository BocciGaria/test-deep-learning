"""ニューラルネットワーク

作成日   : 2024/07/05
作成者   : BocciGaria
最終更新日: 2024/07/05
最終更新者: BocciGaria
"""

import numpy

from mydeeplearning.activation import identity_function


class Layer:
    """ネットワークの層"""
    _nl: "Layer" = None

    def __init__(self, weight, bias, activation) -> None:
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
        self._w = numpy.array(weight)
        self._b = numpy.array(bias)
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
        a = numpy.dot(x, self._w) + self._b
        z = self._h(a)
        if self._nl is not None:
            return self._nl.forward(z)
        else:
            return z

    def forwarding_to(self, layer) -> None:
        """フォワード時の伝達先層を設定する"""
        self._nl = layer

    @property
    def remaining_forwarding_len(self) -> int:
        """フォワード時の残りの伝達回数（出力方向にある層の数に等しい）"""
        if self._nl is None:
            return 1
        else:
            return self._nl.remaining_forwarding_len + 1

    @property
    def last_layer(self) -> "Layer":
        """フォワード時の最終伝達先層"""
        if self._nl is None:
            return self
        else:
            return self._nl.last_layer


class NeuralNetwork(Layer):
    """ニューラルネットワーク"""

    def __init__(self, fianl_activation=identity_function) -> None:
        """
        ニューラルネットワークを生成する

        Parameters
        ----------
        final_activation : function
            ネットワークの出力層のニューロンで実行される活性化関数
        """
        super().__init__(1, 0, identity_function)
        self._fa = fianl_activation

    def forward(self, x) -> numpy.ndarray:
        return self._fa(super().forward(x))

    def add(self, *layers: Layer) -> None:
        """隠し層を追加する
        追加した順に接続され、新しい層は出力方向の最後の層の後ろに接続される

        Parameters
        ----------
        *layers : tuple[Layer]
            追加する隠し層（複数追加する場合は引数に指定した順に追加される）
        """
        if self._nl is None:
            self.forwarding_to(layers[0])
        else:
            self.last_layer.forwarding_to(layers[0])
        previous = layers[0]
        for next in layers[1:]:
            previous.forwarding_to(next)
            previous = next

    @property
    def dim(self) -> int:
        """ネットワークの次元数
        ３層ネットワークであれば３が得られる
        """
        return self.remaining_forwarding_len - 1
