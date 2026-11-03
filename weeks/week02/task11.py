import numpy as np
from matplotlib import pyplot as plt

from dl_lib.activations.functional import sigmoid, relu, tanh


def main():
    activations_range = np.array([10, -10])
    activations_inputs = np.array([-5, 0, 5])
    print(f"Sigmoid range: {sigmoid(activations_range)}")
    print(f"ReLU range: {relu(activations_range)}")
    print(f"tanh range: {relu(activations_range)}")
    print()
    sigmoid_outputs = sigmoid(activations_inputs)
    relu_outputs = relu(activations_inputs)
    tanh_outputs = tanh(activations_inputs)
    print(f"At x=-5: sigmoid={sigmoid_outputs[0]}, relu={relu_outputs[0]}, tanh={tanh_outputs[0]}")
    print(f"At x=0: sigmoid={sigmoid_outputs[1]}, relu={relu_outputs[1]}, tanh={tanh_outputs[1]}")
    print(f"At x=5: sigmoid={sigmoid_outputs[2]}, relu={relu_outputs[2]}, tanh={tanh_outputs[2]}")
    x = np.linspace(-10, 10, 1000)
    plt.plot(x, sigmoid(x), label="sigmoid")
    plt.plot(x, relu(x), label="relu")
    plt.plot(x, tanh(x), label="tanh")
    plt.grid()
    plt.legend()
    plt.show()
