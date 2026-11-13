import numpy as np

from dl_lib.losses.functional import mae, mse


def main():
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2, 6, 8, 7])
    my_mse = mse(y_true, y_pred)
    print(f"MSE on Task 1 numbers: {my_mse}")
    y_pred_outlier = np.array([2, 6, 8, 100])
    mae_with_outlier = mae(y_true, y_pred_outlier)
    print(f"MAE on Task 1 numbers with an extreme outlier: {mae_with_outlier}")
    mse_with_outlier = mse(y_true, y_pred_outlier)
    print(f"MSE on Task 1 numbers with an extreme outlier: {mse_with_outlier}")
