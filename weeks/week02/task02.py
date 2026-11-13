import numpy as np
import torch.nn.functional

from dl_lib.losses.functional import mae


def main():
    y_true = np.array([3.0, 5.0, 7.0, 9.0])
    y_pred = np.array([2.0, 6.0, 8.0, 7.0])
    mae_dl_lib = mae(y_true, y_pred)
    print(f"MAE(dl_lib):{mae_dl_lib}")
    torch_pred = torch.from_numpy(y_pred)
    torch_true = torch.from_numpy(y_true)
    mae_torch = torch.nn.functional.l1_loss(torch_pred, torch_true)
    print(f"MAE(torch):{mae_torch}")
