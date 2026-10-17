import numpy as np


def main():
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2, 6, 8, 7])
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    squared_errors = errors * errors
    me = np.mean(errors)
    mae = np.mean(absolute_errors)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"errors: {errors}")
    print(f"absolute_errors: {absolute_errors}")
    print(f"squared_errors: {squared_errors}")
    print(f"ME: {me}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
