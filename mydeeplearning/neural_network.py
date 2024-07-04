import numpy


class NeuralNetwork:
    """ニューラルネットワーク"""

    def __init__(self, initial_input) -> None:
        self.__initial_input = initial_input

    @property
    def initial_input(self) -> numpy.ndarray:
        """入力層への入力値"""
        return self.__initial_input
