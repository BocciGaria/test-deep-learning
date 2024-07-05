import numpy

from mydeeplearning.activation import identity_function


class Layer:
    """ネットワークの層"""
    _nl: "Layer" = None

    def __init__(self, weight, bias, activation) -> None:
        self._w = numpy.array(weight)
        self._b = numpy.array(bias)
        self._h = activation

    def forward(self, x) -> numpy.ndarray:
        a = numpy.dot(x, self._w) + self._b
        z = self._h(a)
        if self._nl is not None:
            return self._nl.forward(z)
        else:
            return z

    def forwarding_to(self, l) -> None:
        self._nl = l

    @property
    def remaining_forwarding_len(self) -> int:
        if self._nl is None:
            return 1
        else:
            return self._nl.remaining_forwarding_len + 1

    @property
    def last_layer(self) -> "Layer":
        if self._nl is None:
            return self
        else:
            return self._nl.last_layer


class NeuralNetwork(Layer):
    """ニューラルネットワーク"""

    def __init__(self, fianl_activation=identity_function) -> None:
        super().__init__(1, 0, identity_function)
        self._fa = fianl_activation

    def forward(self, x) -> numpy.ndarray:
        return self._fa(super().forward(x))

    def add(self, *layers: Layer) -> None:
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
        return self.remaining_forwarding_len - 1
