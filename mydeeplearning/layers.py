import numpy as np

from mydeeplearning.activation_func import softmax
from mydeeplearning.computational_graph import ComputationalGraphNode
from mydeeplearning.loss_func import cross_entropy_error


class ReLULayer(ComputationalGraphNode):
    """ReLU(Rectified Linear Unit)レイヤ"""

    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x <= 0
        out = x.copy()
        out[self._mask] = 0
        return out

    def backward(self, dout):
        # doutにmutableを渡すと副作用あり
        dout[self._mask] = 0
        return dout


class SigmoidLayer(ComputationalGraphNode):
    """Sigmoidレイヤ"""

    def __init__(self):
        self._out = None

    def forward(self, x):
        self._out = 1 / (1 + np.exp(-x))
        return self._out

    def backward(self, dout):
        return dout * (1.0 - self._out) * self._out


class AffineLayer(ComputationalGraphNode):
    """Affineレイヤ"""

    def __init__(self, weight, bias) -> None:
        self._W = weight
        self._b = bias
        self._x = None
        self._dW = None
        self._db = None

    def forward(self, x):
        self._x = x
        return np.dot(x, self._W) + self._b

    def backward(self, dout):
        self._dW = np.dot(self._x.T, dout)
        self._db = np.sum(dout, axis=0)
        return np.dot(dout, self._W.T)


class SoftmaxWithLossLayer(ComputationalGraphNode):
    """Softmax-with-Lossレイヤー"""

    def __init__(self):
        self._loss = None
        self._y = None
        self._t = None

    def forward(self, x, t):
        self._t = np.array(t)
        self._y = softmax(x)
        self._loss = cross_entropy_error(self._y, self._t)

        return self._loss

    # 使われない仮引数がある→他のレイヤークラスとシグネチャーが合わない
    def backward(self, dout=1):
        batch_size = 1 if self._t.ndim == 1 else self._t.shape[0]

        return (self._y - self._t) / batch_size
