import numpy as np


def identity_function(x):
    """恒等関数"""
    return x


def step_function(x):
    """ステップ関数"""
    return np.array(x > 0, dtype=np.int16)


def relu(x):
    """ReLU関数"""
    return np.maximum(0, x)


def sigmoid(x):
    """シグモイド関数"""
    return 1 / (1 + np.exp(-np.array(x)))


def softmax(x):
    """ソフトマックス関数"""
    max_x = np.max(x)
    normalized_x = x - max_x  # オーバーフロー対策
    exp_x = np.exp(normalized_x)
    return exp_x / np.sum(exp_x)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.arange(-5.0, 5.0, 0.1)
    y_id = identity_function(x)
    plt.plot(x, y_id, label="Identity")
    y_step = step_function(x)
    plt.plot(x, y_step, label="Step")
    y_sig = sigmoid(x)
    plt.plot(x, y_sig, label="Sigmoid", linestyle="--")
    y_relu = relu(x)
    plt.plot(x, y_relu, label="ReLU", linestyle="--")
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.show()
