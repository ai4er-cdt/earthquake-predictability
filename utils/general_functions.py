# Import relevant libraries
import numpy as np
import torch

### --------------------------------------------- ###
#               Functions for Pytorch               #
### --------------------------------------------- ###


def set_seed(seed):
    """
    Set the random seed for reproducibility in NumPy and PyTorch.

    Parameters:
        seed (int): The desired random seed.
    """
    np.random.seed(seed)  # Set NumPy random seed
    torch.manual_seed(seed)  # Set PyTorch random seed
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = (
        True  # Ensure deterministic behavior when using CUDA
    )


def set_torch_device():
    """
    Set the PyTorch device based on the availability of CUDA (NVIDIA GPU acceleration).

    Returns:
        torch.device: The PyTorch device (cuda or cpu).
    """
    # Check if CUDA (NVIDIA GPU acceleration) is available
    if torch.cuda.is_available():
        dev = "cuda"
        gpu_name = torch.cuda.get_device_name(torch.device("cuda"))
        _, max_memory = torch.cuda.mem_get_info()
        max_memory = max_memory / (1000**3)
        print(f"GPU name: {gpu_name}")
        print(f"Max GPU memory: {max_memory} GiB")
    else:
        dev = "cpu"
        print("No GPU available.")

    # Set PyTorch device based on the chosen device (cuda or cpu)
    device = torch.device(dev)

    return device
