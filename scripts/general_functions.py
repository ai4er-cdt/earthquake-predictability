# Import relevant libraries
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import tqdm

def set_torch_device():
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

def moving_average_causal_filter(df, smoothing_window, downsampling_factor):
    downsampled_df = (
        df.rolling(window=int(smoothing_window*5), step=int(downsampling_factor), center=False).mean().dropna()
    )

    downsampled_df = downsampled_df.reset_index(drop=True)
    downsampled_df = downsampled_df.dropna()

    return downsampled_df

def create_dataset(dataset, lookback, forecast):
    """Transform a time series into a prediction dataset

    Args:
        dataset: Numpy array of time series (first dimension is the time steps).
        lookback: Size of the window for prediction.
        forecast: Number of time steps to predict into the future.
    Return:
        X_tensor: Pytorch tensor of the X windowed features
        y_tensor: Pytorch tensor of the y windowed targets
    """
    X, y = [], []

    # Create input features (X) and corresponding targets (y) for prediction
    for i in range(len(dataset) - lookback - forecast + 1):
        feature = dataset[i : i + lookback]
        target = dataset[i + lookback : i + lookback + forecast]
        X.append(feature)
        y.append(target)

    # Convert the lists to PyTorch tensors - note list->arrays->tensors is faster than list->tensors
    return torch.from_numpy(np.array(X, dtype=np.float32)), torch.from_numpy(
        np.array(y, dtype=np.float32)
    )

def split_train_test_forecast_windows(X, y, forecast, n_forecast_windows):
    n_forecast_windows = 30
    test_size = n_forecast_windows * forecast
    excess = X[:-test_size].shape[0] - forecast * (
        X[:-test_size].shape[0] // forecast
    )

    X_test, y_test = X[-test_size:], y[-test_size:]
    X_train, y_train = X[excess:-test_size], y[excess:-test_size]

    return X_train, y_train, X_test, y_test

def normalise_dataset(X_train, y_train, X_test, y_test):

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Note - we should only fit the scaler to the training set, not the test set!! Super important
    # We only transform the test set, and we will then do an inverse transform later when evaluating.
    X_train_sc, X_test_sc = scaler_X.fit_transform(X_train), scaler_X.transform(
        X_test
    )
    y_train_sc, y_test_sc = scaler_y.fit_transform(y_train), scaler_y.transform(
        y_test
    )

    # Turn all scaled arrays into float tensors
    X_train_sc, X_test_sc = (
        torch.from_numpy(X_train_sc).float(),
        torch.from_numpy(X_test_sc).float(),
    )
    y_train_sc, y_test_sc = (
        torch.from_numpy(y_train_sc).float(),
        torch.from_numpy(y_test_sc).float(),
    )

    data_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_sc": X_train_sc,
        "y_train_sc": y_train_sc,
        "X_test_sc": X_test_sc,
        "y_test_sc": y_test_sc
    }
    return data_dict, scaler_X, scaler_y