import numpy as np

from mydeeplearning.activation_func import identity_function, sigmoid, softmax
from mydeeplearning.loss_func import cross_entropy_error
from mydeeplearning.maths import numerical_gradient


def init_network():
    """初期化したニューラルネットワークを作成"""
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    """ニューラルネットワークの順方向伝播(forward propagation)"""
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


class INeuralNet:
    def predict(self, x) -> np.ndarray:
        """予測

        ニューラルネットワークへの入力に対する予測結果を得る。

        Parameters
        ----------
        x : ArrayLike
            入力の配列

        Returns
        -------
        : numpy.ndarray
            予測結果の配列
        """
        raise NotImplementedError()

    def loss(self, x, t) -> np.floating:
        """損失

        ニューラルネットワークへの入力に対する予測結果と教師データによって呼び出し時点の損失を求める。

        Parameters
        ----------
        x : ArrayLike
            入力の配列
        t : ArrayLike
            教師データ

        Returns
        -------
        : float
            損失
        """
        raise NotImplementedError()


class SimpleNet(INeuralNet):
    """単純なニューラルネットワーク"""

    def __init__(self):
        self.weights = np.random.randn(2, 3)  # ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.weights)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)


class TwoLayerNet(INeuralNet):
    """２層ニューラルネットワーク"""

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """コンストラクタ

        Parameters
        ----------
        input_size : int
            入力層のニューロンの数
        hidden_size : int
            隠れ層のニューロンの数
        output_size : int
            出力層のニューロンの数
        weight_init_std : float
            重みの初期値の標準偏差
        """
        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2

        return softmax(a2)

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """認識精度を求める

        Parameters
        ----------
        x : ArrayLike
            入力値の配列
        t : ArrayLike
            教師データの配列

        Returns
        -------
        : np.floating
            認識精度(0.0 ~ 1.0)
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        """重みパラメータに対する勾配を数値微分によって求める

        Parameters
        ----------
        x : ArrayLike
            入力値の配列
        t : ArrayLike
            教師データの配列

        Returns
        -------
        : dict[np.ndarray]
            重みパラメータに対する勾配
        """
        # Wの参照渡しによって成立している->膨大なパラメータをコピーしなくていいことのトレードオフとして副作用あり
        loss_W = lambda W: self.loss(x, t)

        grads = {
            "W1": numerical_gradient(loss_W, self.params["W1"]),
            "b1": numerical_gradient(loss_W, self.params["b1"]),
            "W2": numerical_gradient(loss_W, self.params["W2"]),
            "b2": numerical_gradient(loss_W, self.params["b2"]),
        }

        return grads
