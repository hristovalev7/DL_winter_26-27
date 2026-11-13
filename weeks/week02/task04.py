import numpy as np
from matplotlib import pyplot as plt

from dl_lib.losses.functional import mae, mse, rmse


def main():
    growing_values = np.linspace(0, 10, 101)
    y_true = np.zeros(4)
    fixed = np.array([1, -1, 0.5])
    maes = []
    mses = []
    rmses = []
    for fourth_error in growing_values:
        y_pred = np.append(fixed, fourth_error)
        current_mae = mae(y_true, y_pred)
        current_mse = mse(y_true, y_pred)
        current_rmse = rmse(y_true, y_pred)
        print(f"Fourth error {fourth_error}")
        print(f"MAE {current_mae}")
        print(f"MSE {current_mse}")
        print(f"RMSE {current_rmse}")
        print()
        maes.append(current_mae)
        mses.append(current_mse)
        rmses.append(current_rmse)
    print(maes)
    print(mses)
    print(rmses)

    plt.plot(growing_values, maes, label="MAE")
    plt.plot(growing_values, mses, label="MSE")
    plt.plot(growing_values, rmses, label="RMSE")
    plt.xlabel("Fourth error")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
