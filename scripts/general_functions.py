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
        - seed (int): The desired random seed.
    """
    np.random.seed(seed)  # Set NumPy random seed
    torch.manual_seed(seed)  # Set PyTorch random seed
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior when using CUDA


def set_torch_device():
    """
    Set the PyTorch device based on the availability of CUDA (NVIDIA GPU acceleration).

    Returns:
        - torch.device: The PyTorch device (cuda or cpu).
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




### --------------------------------------------- ###
#       Functions for dataset pre-processing        #
### --------------------------------------------- ###

def moving_average_causal_filter(df, smoothing_window, downsampling_factor):
    """
    Apply a causal moving average filter to a DataFrame.

    Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - smoothing_window (float): The size of the smoothing window.
        - downsampling_factor (float): The downsampling factor.

    Returns:
        - pd.DataFrame: The filtered and downsampled DataFrame.
    """
    # Apply rolling mean with causal filtering and downsampling
    downsampled_df = (
        df.rolling(window=int(smoothing_window * 5), step=int(downsampling_factor), center=False).mean().dropna()
    )

    # Reset index and drop any remaining NaN values
    downsampled_df = downsampled_df.reset_index(drop=True)
    downsampled_df = downsampled_df.dropna()

    return downsampled_df


def compare_feature_statistics(original_feature, downsampled_feature, significance_level=0.05):
    """
    Compare the mean and variance of two samples using t-test and F-test.

    Parameters:
        - original_feature (pd.Series): The original feature for comparison.
        - downsampled_feature (pd.Series): The downsampled feature for comparison.
        - significance_level (float, optional): The significance level for hypothesis testing.
                                              Default is 0.05.

    Returns:
        - dict: A dictionary containing the results of the mean and variance comparisons.
    """
    # Perform two-sample t-test for mean comparison
    t_statistic_mean, p_value_mean = ttest_ind(original_feature, downsampled_feature)

    # Perform F-test for variance comparison
    f_statistic, p_value_variance = f_oneway(original_feature, downsampled_feature)

    # Results dictionary
    results = {
        "mean_comparison": {
            "t_statistic": t_statistic_mean,
            "p_value": p_value_mean,
            "significant": p_value_mean < significance_level,
        },
        "variance_comparison": {
            "f_statistic": f_statistic,
            "p_value": p_value_variance,
            "significant": p_value_variance < significance_level,
        }
    }

    return results


def create_dataset(dataset, lookback, forecast):
    """Transform a time series into a prediction dataset.

    Parameters:
        - dataset (numpy.ndarray): Numpy array of time series (first dimension is the time steps).
        - lookback (int): Size of the window for prediction.
        - forecast (int): Number of time steps to predict into the future.

    Returns:
        - torch.Tensor: Pytorch tensor of the X windowed features
        - torch.Tensor: Pytorch tensor of the y windowed targets
    """
    X, y = [], []

    # Create input features (X) and corresponding targets (y) for prediction
    for i in range(len(dataset) - lookback - forecast + 1):
        # Extract feature and target windows from the time series
        feature = dataset[i : i + lookback]
        target = dataset[i + lookback : i + lookback + forecast]
        X.append(feature)
        y.append(target)

    # Convert the lists to PyTorch tensors
    # Note: list -> arrays -> tensors is faster than list -> tensors
    X_tensor = torch.from_numpy(np.array(X, dtype=np.float32))
    y_tensor = torch.from_numpy(np.array(y, dtype=np.float32))

    return X_tensor, y_tensor


def split_train_test_forecast_windows(X, y, forecast, n_forecast_windows):
    """
    Split input features (X) and target values (y) into training and testing sets
    for time series forecasting.

    Parameters:
        - X (torch.Tensor): Input features for time series forecasting.
        - y (torch.Tensor): Target values for time series forecasting.
        - forecast (int): Number of time steps to predict into the future.
        - n_forecast_windows (int): Number of forecast windows to reserve for testing.

    Returns:
        - tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing
        - X_train, y_train, X_test, y_test for training and testing the forecasting model.
    """
    # Calculate the total size of the test set in terms of forecast windows
    test_size = n_forecast_windows * forecast

    # Determine excess data that won't fit into complete forecast windows
    excess = X[:-test_size].shape[0] - forecast * (X[:-test_size].shape[0] // forecast)

    # Extract test set
    X_test, y_test = X[-test_size:], y[-test_size:]

    # Extract training set, excluding the test set and any excess data
    X_train, y_train = X[excess:-test_size], y[excess:-test_size]

    return X_train, y_train, X_test, y_test


def normalise_dataset(X_train, y_train, X_test, y_test):
    """
    Normalize the input and output datasets using Min-Max scaling.
    
    Parameters:
        - X_train (numpy.ndarray): Training input data.
        - y_train (numpy.ndarray): Training output data.
        - X_test (numpy.ndarray): Test input data.
        - y_test (numpy.ndarray): Test output data.

    Returns:
        - tuple: A tuple containing a dictionary with various dataset arrays,
               and MinMaxScaler instances for input and output data.
    """

    # Initialize MinMaxScaler for input and output
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit and transform the training set for both input and output
    X_train_sc, X_test_sc = scaler_X.fit_transform(X_train), scaler_X.transform(X_test)
    y_train_sc, y_test_sc = scaler_y.fit_transform(y_train), scaler_y.transform(y_test)

    # Convert scaled arrays to float tensors
    X_train_sc, X_test_sc = (
        torch.from_numpy(X_train_sc).float(),
        torch.from_numpy(X_test_sc).float(),
    )
    y_train_sc, y_test_sc = (
        torch.from_numpy(y_train_sc).float(),
        torch.from_numpy(y_test_sc).float(),
    )

    # Create a dictionary to store various dataset arrays
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




### --------------------------------------------- ###
#       Functions for ploting model results         #
### --------------------------------------------- ###

def plot_all_data(test_start_index, data_dict, results_dict, lookback, forecast, title, x_label, y_label, zoom_window):
    """
    Plot true values, training predictions, and testing predictions for time series data.

    Parameters:
        - test_start_index (int): Index where the test set starts.
        - data_dict (dict): Dictionary containing training and testing data arrays.
        - results_dict (dict): Dictionary containing the training and testing predictions.
        - lookback (int): Number of time steps to look back in the data.
        - forecast (int): Number of time steps ahead to forecast.
        - title (str): Plot title.
        - x_label (str): Label for the x-axis.
        - y_label (str): Label for the y-axis.
        - zoom_window (tuple): Tuple containing start and end indices for zooming into the plot (optional).
    """
    train_outputs = results_dict["y_train_pred"]
    test_outputs = results_dict["y_test_pred"]

    # Extract every 'forecast' time step for plotting
    train_plot = np.array(
        [train_outputs[idx] for idx in range(0, len(train_outputs), forecast)]
    ).reshape(-1, 1)

    test_plot = np.array(
        [test_outputs[idx] for idx in range(0, len(test_outputs), forecast)]
    ).reshape(-1, 1)

    combined_plot = np.concatenate((train_plot, test_plot))

    # Plot true values, training predictions, and testing predictions
    plt.figure(figsize=(25, 6))
    plt.plot(
        range(lookback, lookback + len(combined_plot)),
        np.concatenate((data_dict["y_train"][:, 0], data_dict["y_test"][:, 0])),
        label="True values",
    )

    plt.plot(
        range(lookback, lookback + len(train_plot)),
        train_plot,
        label="Training prediction",
    )
    plt.plot(
        range(lookback + len(train_plot), lookback + len(combined_plot)),
        test_plot,
        label="Testing prediction",
    )

    # Vertical line indicating the start of the test set
    plt.axvline(
        x=test_start_index, color="gray", linestyle="--", label="Test set start"
    )

    # Zoom into the specified window if provided
    if len(zoom_window) > 0:
        plt.xlim(zoom_window[0], zoom_window[1])

    # Set plot title, labels, and legend
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_metric(n_epochs, train_metric_list, test_metric_list, metric_label):
    """
    Plot a metric over epochs for training and testing sets.

    Parameters:
        n_epochs (int): Number of training epochs.
        train_metric_list (list): List of metric values for each training epoch.
        test_metric_list (list): List of metric values for each testing epoch.
        metric_label (str): Label for the metric being plotted.
    """
    # Plot metric over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, n_epochs), train_metric_list, label=f"Train {metric_label}")
    plt.plot(range(0, n_epochs), test_metric_list, label=f"Test {metric_label}")
    
    # Set plot labels, title, and legend
    plt.xlabel("Epochs")
    plt.ylabel(f"{metric_label}")
    plt.title(f"{metric_label} over Epochs")
    plt.legend()
    plt.show()
