import numpy


class Layer:
    """ネットワークの層"""
    __nl: "Layer" = None

    def __init__(self, weight, bias, activation) -> None:
        self.__w = numpy.array(weight)
        self.__b = numpy.array(bias)
        self.__h = activation

    def forward(self, x) -> numpy.ndarray:
        a = numpy.dot(x, self.__w) + self.__b
        z = self.__h(a)
        if self.__nl is not None:
            return self.__nl.forward(z)
        else:
            return z

    def set_next_layer(self, l) -> None:
        self.__nl = l


class NeuralNetwork:
    """ニューラルネットワーク"""

    def forward(self, x) -> numpy.ndarray:
        return numpy.array(x)
