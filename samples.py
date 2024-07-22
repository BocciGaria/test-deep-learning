from matplotlib import pyplot
import numpy
import pickle
from PIL import Image

from lib.mnist import mnist

from mydeeplearning.activation_func import (
    step_function,
    relu,
    sigmoid,
    softmax,
    identity_function,
)
from mydeeplearning.maths import (
    numerical_diff,
    sum_of_square,
    numerical_gradient,
    gradient_descent,
)
from mydeeplearning.neural_network import Layer, NeuralNetwork


def plot_activation_functions():
    x = numpy.arange(-5, 5, 0.01)

    fig = pyplot.figure(figsize=(12, 8))
    axes = fig.add_subplot()
    axes.set_title("activation functions")
    axes.set_xlabel("input")
    axes.set_ylabel("output")
    axes.set_ylim(-0.1, 1.1)
    axes.plot(x, step_function(x), label="step")
    axes.plot(x, relu(x), label="ReLU")
    axes.plot(x, sigmoid(x), label="sigmoid")
    pyplot.show()


def show_mnist_image():
    (x_train, t_train), (x_test, t_test) = mnist.load_mnist(
        flatten=True, normalize=False
    )
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)
    pil_img = Image.fromarray(numpy.uint8(x_train[0].reshape(28, 28)))
    pil_img.show()


def run_network_with_sample_weights():
    (_, _), (images, labels) = mnist.load_mnist(flatten=True, normalize=False)
    with open("./lib/mnist/sample_weight.pkl", "rb") as f:
        net_data = pickle.load(f)
    net = NeuralNetwork()
    net.add(Layer(net_data["W1"], net_data["b1"], sigmoid))
    net.add(Layer(net_data["W2"], net_data["b2"], sigmoid))
    net.add(Layer(net_data["W3"], net_data["b3"], softmax))

    accuracy_count = 0
    for i in range(len(images)):
        y = net.forward(images[i])
        p = numpy.argmax(y)
        if p == labels[i]:
            accuracy_count += 1
    print("Accuracy: " + str(float(accuracy_count) / len(images)))


def run_batch_network_with_sample_weights():
    (_, _), (images, labels) = mnist.load_mnist(normalize=False)
    with open("./lib/mnist/sample_weight.pkl", "rb") as f:
        net_data = pickle.load(f)
    net = NeuralNetwork()
    net.add(Layer(net_data["W1"], net_data["b1"], sigmoid))
    net.add(Layer(net_data["W2"], net_data["b2"], sigmoid))
    net.add(Layer(net_data["W3"], net_data["b3"], softmax))

    batch_size = 100
    accuracy_count = 0
    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        y = net.forward(batch_images)
        p = numpy.argmax(y, axis=1)
        accuracy_count += numpy.sum(p == batch_labels)
    print("Accuracy: " + str(float(accuracy_count) / len(images)))


def get_random_train_data(num):
    (images, labels), (_, _) = mnist.load_mnist(normalize=False, one_hot_label=True)
    random_choices = numpy.random.choice(images.shape[0], num)
    random_images = images[random_choices]
    random_labels = labels[random_choices]
    return random_images, random_labels


def plot_numerical_diff():
    x = numpy.arange(0.0, 20.0, 0.1)
    f = lambda x: 0.01 * x**2 + 0.1 * x
    y = f(x)
    print(y)
    pyplot.xlabel("x")
    pyplot.ylabel("f(x)")
    pyplot.plot(x, y)

    def get_tangent(f, x):
        d = numerical_diff(f, x)
        y = f(x) - d * x
        return lambda t: d * t + y

    t = get_tangent(f, 5)
    y2 = t(x)
    pyplot.plot(x, y2)

    pyplot.show()


def plot_numerical_gradient():
    x1 = numpy.arange(-2.0, 2.5, 0.25)
    x2 = x1.copy()
    xv1, xv2 = numpy.meshgrid(x1, x2)
    xv1 = xv1.flatten()
    xv2 = xv2.flatten()

    gradient = numerical_gradient(sum_of_square, numpy.array([xv1, xv2]).T).T
    vector = numpy.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
    u = -gradient[0] / vector
    v = -gradient[1] / vector

    pyplot.figure()
    pyplot.quiver(xv1, xv2, u, v, vector, cmap="jet")
    pyplot.xlabel("x0")
    pyplot.ylabel("x1")
    pyplot.grid()
    pyplot.draw()
    pyplot.show()


def plot_gradient_descent():
    x0 = []
    x1 = []
    for step in range(100):
        # gradient = gradient_descent(sum_of_square, [-3.0, 4.0], 10.0, step) # 学習率が大きすぎ
        # gradient = gradient_descent(sum_of_square, [-3.0, 4.0], 1e-10, step) # 学習率が小さすぎ
        gradient = gradient_descent(sum_of_square, [-3.0, 4.0], 0.1, step)
        x0.append(gradient[0])
        x1.append(gradient[1])
    pyplot.plot([-5.0, 5.0], [0.0, 0.0], "--b")
    pyplot.plot([0.0, 0.0], [-5.0, 5.0], "--b")
    pyplot.plot(x0, x1, "o")
    pyplot.xlabel("x0")
    pyplot.ylabel("x1")
    pyplot.show()


def gradient_without_side_effect():
    net = NeuralNetwork()
    # weight = numpy.random.randn(2, 3)
    weight = numpy.array(
        [
            [0.47355232, 0.9977393, 0.84668094],
            [0.85557411, 0.03563661, 0.69422093],
        ]
    )
    bias = 0.0
    net.add(Layer(weight, bias, identity_function))
    # x = numpy.random.randn(2)
    x = [0.6, 0.9]
    y = net.forward(x)
    t = [0, 0, 1]
    loss = net.loss(x, t)

    def f(w):
        net = NeuralNetwork()
        net.add(Layer(w, 0.0, identity_function))
        return net.loss(x, t)

    gradient = numerical_gradient(f, weight)
    print("------ weight -----", weight, sep="\n")
    print("------- bias ------", bias, sep="\n")
    print("-------- x --------", x, sep="\n")
    print("-------- t --------", t, sep="\n")
    print("-------- y --------", y, sep="\n")
    print("------- loss ------", loss, sep="\n")
    print("----- gradient ----", gradient, sep="\n")


def train_network():
    (x_train, t_train), (x_test, t_test) = mnist.load_mnist(one_hot_label=True)

    # ハイパーパラメータ
    iters_num = 10
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    net = NeuralNetwork()
    net.add(Layer(numpy.random.randn(784, 50) * 0.01, numpy.zeros(50), sigmoid))
    net.add(Layer(numpy.random.randn(50, 10) * 0.01, numpy.zeros(10), softmax))

    loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []
    iter_per_epoch = max(train_size / batch_size, 1)
    for i in range(iters_num):
        batch_mask = numpy.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        gradient = net.gradient(x_batch, t_batch)
        net.update(gradient, learning_rate)

        loss = net.loss(x_batch, t_batch)
        print(loss)
        loss_history.append(loss)

        if i % iter_per_epoch == 0:
            train_accuracy_history.append(net.accuracy(x_train, t_train))
            test_accuracy_history.append(net.accuracy(x_test, t_test))
            print(
                "train acc, test acc | "
                + str(train_accuracy_history)
                + ", "
                + str(test_accuracy_history)
            )

    pyplot.plot(numpy.arange(iters_num), loss_history)
    pyplot.xlabel("iteration")
    pyplot.ylabel("loss")
    pyplot.show()


def main():
    net = NeuralNetwork()


if __name__ == "__main__":
    main()
