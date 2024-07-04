import numpy

from mydeeplearning.neural_network import NeuralNetwork


def test_generate_network():
    n1 = NeuralNetwork([1.0, 0.5])
    assert numpy.array_equal(n1.initial_input, numpy.array([1.0, 0.5]))
    n2 = NeuralNetwork(0.3)
    assert n2.initial_input == 0.3
    n3 = NeuralNetwork([3, 2, 1])
    assert numpy.array_equal(n3.initial_input, numpy.array([3, 2, 1]))
    try:
        n4 = NeuralNetwork("hoge")
    except ValueError:
        pass


def test_add_hidden_layer():
    # ネットワークを生成する
    # 隠れ層を生成する（重み、バイアス、活性化関数）
    # 隠れ層をネットワークに挿入する
    # 何を検証する？
    assert False


def test_forward_single_layered_neural_network():
    # ネットワークを生成する 初期値:[1.0, 0.5]
    # 隠れ層を生成する
    # 隠れ層を追加する
    # 入力から出力への伝達処理を行う
    # assert numpy.allclose(y, numpy.array([0.31682708, 0.69627909]))
    pass
