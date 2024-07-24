import numpy

from mydeeplearning.computational_graph import ComputationalGraphNode


class Layer(ComputationalGraphNode):
    """ニューラルネットワークを構成する層の基底クラス"""

    pass


class ReLULayer(Layer):
    """ReLU(Rectified Linear Unit)レイヤ"""

    def _compute(self, x):
        self._mask = x <= 0
        out = x.copy()
        out[self._mask] = 0
        return out

    def backward(self, dout):
        dout[self._mask] = 0
        return dout


class SigmoidLayer(ComputationalGraphNode):
    """Sigmoidレイヤ"""

    def _compute(self, x):
        self._out = 1 / (1 + numpy.exp(-x))
        return self._out

    def backward(self, dout):
        return dout * (1.0 - self._out) * self._out


class AffineLayer(ComputationalGraphNode):
    """Affineレイヤ"""

    def __init__(self, weight, bias) -> None:
        self.W = weight
        self.b = bias

    def _compute(self, x):
        self.x = x
        return numpy.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dout_W = numpy.dot(self.x.T, dout)
        self.dout_b = numpy.sum(dout, axis=0)
        return numpy.dot(dout, self.W.T)
