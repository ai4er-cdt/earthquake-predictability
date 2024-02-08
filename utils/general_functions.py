# Import relevant libraries
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import f_oneway, ttest_ind
import matplotlib.pyplot as plt
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
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior when using CUDA


def set_torch_device():
    """
    Set the PyTorch device based on the availability of CUDA (NVIDIA GPU acceleration).

    Returns:
        torch.device: The PyTorch device (cuda or cpu).
    """
    # Check if CUDA (NVIDIA GPU acceleration) is available
    if torch.cuda.is_available():
        dev, map_location = "cuda", None  # Use GPU
        print(
            f"Total GPUs available: {torch.cuda.device_count()}"
        )  # Display GPU count
        # !nvidia-smi  # Display GPU details using nvidia-smi
    else:
        dev, map_location = "cpu", "cpu"  # Use CPU
        print("No GPU available.")

    # Set PyTorch device based on the chosen device (cuda or cpu)
    device = torch.device(dev)

    return device
