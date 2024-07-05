import numpy

from mydeeplearning.activation import sigmoid, softmax
from mydeeplearning.neural_network import NeuralNetwork, Layer


def test_generate_network():
    net = NeuralNetwork()


def test_generate_layer():
    weight = [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
    bias = [1.0, 2.0, 3.0]
    l = Layer(weight, bias, softmax)


def test_forward_layer():
    weight = [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
    bias = [1.0, 2.0, 3.0]
    l = Layer(weight, bias, sigmoid)
    y = l.forward([1.0, 0.5])
    assert numpy.allclose(y, [0.76852478, 0.92414182, 0.97811873])



def test_add_hidden_layer():
    # ネットワークを生成する
    # 隠れ層を生成する（重み、バイアス、活性化関数）
    # 隠れ層をネットワークに挿入する
    # 何を検証する？
    pass


def test_forward_single_layered_neural_network():
    # ネットワークを生成する 初期値:[1.0, 0.5]
    # 隠れ層を生成する
    # 隠れ層を追加する
    # 入力から出力への伝達処理を行う
    # assert numpy.allclose(y, numpy.array([0.31682708, 0.69627909]))
    pass


def test_forward_neural_network():
    net = NeuralNetwork()
    y = net.forward([1.0, 0.5])
    assert numpy.array_equal(y, [1.0, 0.5])
