# Import relevant libraries
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import f_oneway, ttest_ind
import torch

### --------------------------------------------- ###
#              Downsampling/smoothing               #
### --------------------------------------------- ###

def moving_average_causal_filter(df, smoothing_window, downsampling_factor):
    """
    Apply a causal moving average filter to a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        smoothing_window (float): The size of the smoothing window.
        downsampling_factor (float): The downsampling factor.

    Returns:
        pd.DataFrame: The filtered and downsampled DataFrame.
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
        original_feature (pd.Series): The original feature for comparison.
        downsampled_feature (pd.Series): The downsampled feature for comparison.
        significance_level (float, optional): The significance level for hypothesis testing.
                                              Default is 0.05.

    Returns:
        dict: A dictionary containing the results of the mean and variance comparisons.
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


### --------------------------------------------- ###
#           Dataset ready for modelling             #
### --------------------------------------------- ###

def create_dataset(dataset, lookback, forecast):
    """Transform a time series into a prediction dataset.

    Parameters:
        dataset (numpy.ndarray): Numpy array of time series (first dimension is the time steps).
        lookback (int): Size of the window for prediction.
        forecast (int): Number of time steps to predict into the future.

    Returns:
        torch.Tensor: Pytorch tensor of the X windowed features
        torch.Tensor: Pytorch tensor of the y windowed targets
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
        X (torch.Tensor): Input features for time series forecasting.
        y (torch.Tensor): Target values for time series forecasting.
        forecast (int): Number of time steps to predict into the future.
        n_forecast_windows (int): Number of forecast windows to reserve for testing.

    Returns:
        tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing
        X_train, y_train, X_test, y_test for training and testing the forecasting model.
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


### --------------------------------------------- ###
#        Normalise dataset on training set          #
### --------------------------------------------- ###

def normalise_dataset(X_train, y_train, X_test, y_test):
    """
    Normalize the input and output datasets using Min-Max scaling.
    
    Parameters:
        X_train (numpy.ndarray): Training input data.
        y_train (numpy.ndarray): Training output data.
        X_test (numpy.ndarray): Test input data.
        y_test (numpy.ndarray): Test output data.

    Returns:
        tuple: A tuple containing a dictionary with various dataset arrays,
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