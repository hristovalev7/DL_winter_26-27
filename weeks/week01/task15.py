import numpy as np
import torch


def main():
    numpy_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("NumPy array:")
    print(numpy_array)
    print(f"dtype={numpy_array.dtype}, shape={numpy_array.shape}")
    pytorch_tensor = torch.from_numpy(numpy_array)
    print("PyTorch tensor:")
    print(pytorch_tensor)
    print(
        f"dtype={pytorch_tensor.dtype}, shape={pytorch_tensor.shape}, device={pytorch_tensor.device}"
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Selected device: {device}")
    moved_tensor = pytorch_tensor.to(device)
    print(f"Tensor moved to device: {moved_tensor.device}, dtype={moved_tensor.dtype}")
