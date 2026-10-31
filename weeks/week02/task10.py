import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations.functional import sigmoid


def main():
    points = np.array([-10, 0, 10])
    sigmoid_array = sigmoid(points)
    print(f"sigmoid(-10) = {sigmoid_array[0]:.6f}")
    print(f"sigmoid(0) = {sigmoid_array[1]:.6f}")
    print(f"sigmoid(10) = {sigmoid_array[2]:.6f}")
    print(f"Output range: [{sigmoid_array[0]:.6f}, {sigmoid_array[2]:.6f}]")
    sigmoid_x = np.linspace(-10, 10, 1000)
    sigmoid_y = sigmoid(sigmoid_x)
    plt.plot(sigmoid_x, sigmoid_y, label="sigmoid")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.show()
    torch_sigmoid_points = torch.tensor([-10, 0, 10])
    torch.sigmoid(torch_sigmoid_points)
