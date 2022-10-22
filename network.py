import numpy as np
import random
from keras.datasets import mnist


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z) -> float:
    return sigmoid(z) * (1 - sigmoid(z))


class Network:
    def __init__(self, params) -> None:
        self.num_layers = len(params)
        self.sizes = params
        self.biases = [np.random.randn(y, 1) for y in params[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(params[:-1], params[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def gradient_descent(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def train(self, trainingdata, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
            print(f"Epoch {0}: {self.evaluate(test_data)} / {n_test}")
        n = len(trainingdata)

        for j in range(epochs):
            random.shuffle(trainingdata)
            mini_batches = [
                trainingdata[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.gradient_descent(mini_batch, eta)
            if test_data:
                print(f"Epoch {j+1}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j+1} complete")


def main():
    net = Network([784, 16, 16, 10])
    (x_train, train_y), (x_test, test_y) = mnist.load_data()
    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    training_inputs = [np.reshape(x, (784, 1)) for x in x_train]
    training_results = [vectorized_result(y) for y in train_y]
    training_data = [(x, y) for x, y in zip(training_inputs, training_results)]
    # validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    # validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in x_test]
    test_data = [(x, y) for x, y in zip(test_inputs, test_y)]

    net.train(training_data, 30, 10, 2)
    print(f"{net.evaluate(test_data)} / 10000")


if __name__ == "__main__":
    main()
