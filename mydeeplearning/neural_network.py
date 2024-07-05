import numpy


class Layer:
    """ネットワークの層"""
    def __init__(self, weight, bias, activation) -> None:
        self.__w = numpy.array(weight)
        self.__b = numpy.array(bias)
        self.__h = activation

    def forward(self, x) -> numpy.ndarray:
        a = numpy.dot(x, self.__w) + self.__b
        return self.__h(a)


class NeuralNetwork:
    """ニューラルネットワーク"""

    def forward(self, x) -> numpy.ndarray:
        return numpy.array(x)
