import numpy as np

from mydeeplearning import neuralnet


def test_forward_network():
    x = [1.0, 0.5]
    y = neuralnet.forward(neuralnet.init_network(), x)
    assert np.allclose(y, np.array([0.31682708, 0.69627909]), rtol=1e-8, atol=1e-8)
    # assert not np.allclose(y, np.array([0.31682713, 0.69627914]), rtol=1e-8, atol=1e-8)
    # assert not np.allclose(y, np.array([0.31682703, 0.69627904]), rtol=1e-8, atol=1e-8)
