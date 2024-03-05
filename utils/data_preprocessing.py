# Import relevant libraries
import numpy as np
import torch
from scipy.stats import f_oneway, ttest_ind
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
        df.rolling(
            window=int(smoothing_window),
            step=int(downsampling_factor),
            center=False,
        )
        .mean()
        .dropna()
    )

    # Reset index and drop any remaining NaN values
    downsampled_df = downsampled_df.reset_index(drop=True)
    downsampled_df = downsampled_df.dropna()

    return downsampled_df


def compare_feature_statistics(
    original_feature, downsampled_feature, significance_level=0.05
):
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
    t_statistic_mean, p_value_mean = ttest_ind(
        original_feature, downsampled_feature
    )

    # Perform F-test for variance comparison
    f_statistic, p_value_variance = f_oneway(
        original_feature, downsampled_feature
    )

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
        },
    }

    return results


### --------------------------------------------- ###
#               Feature Engineering                 #
### --------------------------------------------- ###

def find_peak_indices(data, threshold=100):

    """
    Finds the peaks of a univariate oscillating time series.

    Parameters:
        data (pd.Series): A univariate oscillating time series
        threshold (integer): The window size used before and after a specific index to determine if it is a peak

    Returns:
        peak_indices: A list of the indices of the peaks.
    """

    peak_indices = [0]
    for i in range(threshold, len(data) - threshold):
        if data[i] > max(data[i - threshold : i]) and data[i] >= max(
            data[i + 1 : i + threshold]
        ):
            peak_indices.append(i)
    return peak_indices


def create_features(df, column_name = "signal"):

    """
    Adds variance, first derivate, second derivative, time since last peak and time since last trough columns to a data frame containing a "signal" shear stress.

    Parameters:
        data (pd.Series): A data frame containing the time series on which features will be created
        column_name (string): The column name of the data frame on which features will be created

    Returns:
        dict: A data frame including the original signal and engineered features
    """

    # Calculate variance of shear stress
    df["variance"] = df[column_name].rolling(window=30).var()

    # Calculate first derivative of shear stress
    df["first_derivative"] = df[column_name].diff()

    # Calculate second derivative of shear stress
    df["second_derivative"] = df["first_derivative"].diff()

    peak_indices = find_peak_indices(df[column_name])
    trough_indices = find_peak_indices(-df[column_name])

    # Initialize the columns with NaNs
    df["steps_since_last_peak"] = np.nan
    df["steps_since_last_trough"] = np.nan

    # Calculate steps since last peak
    last_peak_index = None
    for i in range(len(df)):
        if i in peak_indices:
            last_peak_index = i
        if last_peak_index is not None:
            df.loc[i, "steps_since_last_peak"] = i - last_peak_index

    # Calculate steps since last trough
    last_trough_index = None
    for i in range(len(df)):
        if i in trough_indices:
            last_trough_index = i
        if last_trough_index is not None:
            df.loc[i, "steps_since_last_trough"] = i - last_trough_index

    df = df.dropna()
    df = df.reset_index(drop=True)

    return df

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


def split_train_test_forecast_windows(
    X, y, forecast, n_forecast_windows, n_validation_windows=0
):
    """
    Split input features (X) and target values (y) into training, optional validation, and testing sets
    for time series forecasting. The creation of a validation set is conditional based on the
    number of validation windows specified.

    Parameters:
        X (torch.Tensor): Input features for time series forecasting.
        y (torch.Tensor): Target values for time series forecasting.
        forecast (int): Number of time steps to predict into the future.
        n_forecast_windows (int): Number of forecast windows to reserve for testing.
        n_validation_windows (int, optional): Number of forecast windows to reserve for validation. Defaults to 0.

    Returns:
        tuple: A tuple containing X_train, y_train, and depending on the presence of a validation set, X_val, y_val, followed by X_test, y_test for training, validating (if applicable), and testing the forecasting model.
    """
    # Calculate the total size of the test set in terms of forecast windows
    test_size = n_forecast_windows * forecast
    val_size = n_validation_windows * forecast

    # Total size reserved for validation and test sets
    val_test_size = val_size + test_size

    # Determine excess data that won't fit into complete forecast windows
    excess = X[:-(val_test_size)].shape[0] - forecast * (
        X[:-(val_test_size)].shape[0] // forecast
    )

    # Extract test set
    X_test, y_test = X[-test_size:], y[-test_size:]

    # Initialize validation sets to None
    X_val, y_val = None, None

    # Conditionally extract validation set
    if n_validation_windows > 0:
        X_val, y_val = (
            X[-(val_test_size):-test_size],
            y[-(val_test_size):-test_size],
        )

    # Extract training set, excluding the validation (if any) and test sets and any excess data
    X_train, y_train = X[excess:-(val_test_size)], y[excess:-(val_test_size)]

    # Return tuple based on whether a validation set is present
    if n_validation_windows > 0:
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test


def select_features(data_dict_temp, feature_names):
    """
    Selects specific features from the dataset based on the given feature names.

    This function maps the desired feature names to their corresponding indices in the dataset arrays
    and creates a new data dictionary with datasets containing only the selected features.
    It also assumes that the target variable (y) is at index 0 for the outputs.

    Parameters:
    - data_dict_temp (dict): The original dataset dictionary containing 'X_train', 'X_test', 'X_val',
                             and their scaled versions ('_sc'), along with 'y_train', 'y_test', and 'y_val'.
    - feature_names (list of str): The names of the features to select for the model.

    Returns:
    - dict: A new dictionary containing the selected features for training, testing, and validation sets,
            along with their scaled versions and the target variables.
    """
    # Define a mapping from feature names to their indices in the dataset
    feature_map = {
        "signal": 0,
        "variance": 1,
        "first_derivative": 2,
        "second_derivative": 3,
        "steps_since_last_peak": 4,
        "steps_since_last_trough": 5,
    }

    # Convert the list of feature names to their corresponding indices
    feature_indices = [feature_map[name] for name in feature_names if name in feature_map]

    # Create a new data dictionary with only the selected features and the target variable at index 0
    data_dict = {
        "X_train": data_dict_temp["X_train"][:, :, feature_indices],
        "X_test": data_dict_temp["X_test"][:, :, feature_indices],
        "X_val": data_dict_temp["X_val"][:, :, feature_indices],
        "y_train": data_dict_temp["y_train"][:, :, 0],
        "y_test": data_dict_temp["y_test"][:, :, 0],
        "y_val": data_dict_temp["y_val"][:, :, 0],
        "X_train_sc": data_dict_temp["X_train_sc"][:, :, feature_indices],
        "X_test_sc": data_dict_temp["X_test_sc"][:, :, feature_indices],
        "X_val_sc": data_dict_temp["X_val_sc"][:, :, feature_indices],
        "y_train_sc": data_dict_temp["y_train_sc"][:, :, 0],
        "y_test_sc": data_dict_temp["y_test_sc"][:, :, 0],
        "y_val_sc": data_dict_temp["y_val_sc"][:, :, 0],
    }

    return data_dict


### --------------------------------------------- ###
#        Normalise dataset on training set          #
### --------------------------------------------- ###


def normalise_dataset(
    X_train, y_train, X_test, y_test, X_val=None, y_val=None
):
    """
    Normalize the input and output datasets using Min-Max scaling. Optionally handles validation data if provided.

    Parameters:
        X_train (numpy.ndarray): Training input data.
        y_train (numpy.ndarray): Training output data.
        X_test (numpy.ndarray): Test input data.
        y_test (numpy.ndarray): Test output data.
        X_val (numpy.ndarray, optional): Validation input data.
        y_val (numpy.ndarray, optional): Validation output data.

    Returns:
        tuple: A tuple containing a dictionary with various dataset arrays,
               and MinMaxScaler instances for input and output data.
               Optionally includes validation data if provided.
    """

    # Initialize MinMaxScaler for input and output
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit and transform the training set for both input and output
    X_train_sc = scaler_X.fit_transform(X_train.reshape(-1, 1))
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1))

    # Transform the test set using the fitted scalers
    X_test_sc = scaler_X.transform(X_test.reshape(-1, 1))
    y_test_sc = scaler_y.transform(y_test.reshape(-1, 1))

    # Extract all the shapes for train and test
    X_train_shape, X_test_shape, y_train_shape, y_test_shape = (
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
    )

    # Prepare the dictionary to include scaled datasets
    data_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_sc": torch.from_numpy(
            X_train_sc.reshape(*X_train_shape)
        ).float(),
        "y_train_sc": torch.from_numpy(
            y_train_sc.reshape(*y_train_shape)
        ).float(),
        "X_test_sc": torch.from_numpy(
            X_test_sc.reshape(*X_test_shape)
        ).float(),
        "y_test_sc": torch.from_numpy(
            y_test_sc.reshape(*y_test_shape)
        ).float(),
    }

    # Check if validation data is provided
    if X_val is not None and y_val is not None:
        # Transform the validation set using the fitted scalers
        X_val_sc = scaler_X.transform(X_val.reshape(-1, 1))
        y_val_sc = scaler_y.transform(y_val.reshape(-1, 1))

        # Extract all the shapes for val
        X_val_shape, y_val_shape = (
            X_val.shape,
            y_val.shape,
        )

        # Add the validation data to the dictionary
        data_dict.update(
            {
                "X_val": X_val,
                "y_val": y_val,
                "X_val_sc": torch.from_numpy(
                    X_val_sc.reshape(*X_val_shape)
                ).float(),
                "y_val_sc": torch.from_numpy(
                    y_val_sc.reshape(*y_val_shape)
                ).float(),
            }
        )

    return data_dict, scaler_X, scaler_y


def normalise_dataset_multi_feature(
    X_train, y_train, X_test, y_test, X_val=None, y_val=None, scaler_type="min-max"
):
    
    """
    Normalizes datasets using either standard or min-max scaling for each feature. Supports training, testing,
    and optional validation sets, allowing separate scalers for input (X) and output (y) data.
    
    Parameters:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    - X_val, y_val (optional): Validation data and labels.
    - scaler_type (str): Type of scaler to use ('standard' or 'min-max').

    Returns:
    - data_dict: Dictionary containing original and scaled datasets.
    - scalers_X: List of scalers used for X data.
    - scalers_y: List of scalers used for y data (same as scalers_X due to shared normalization).
    """

    num_features = X_train.shape[2]

    # Choose scaler type based on `scaler_type` parameter
    if scaler_type == "standard":
        scalers_X = [StandardScaler() for _ in range(num_features)]
    elif scaler_type == "min-max":
        scalers_X = [MinMaxScaler() for _ in range(num_features)]

    scalers_y = scalers_X  # Use same scalers for y for consistent scaling

    # Prepare scaled arrays
    X_train_sc, X_test_sc = np.zeros_like(X_train), np.zeros_like(X_test)
    X_val_sc = np.zeros_like(X_val) if X_val is not None else None
    y_train_sc, y_test_sc = np.zeros_like(y_train), np.zeros_like(y_test)
    y_val_sc = np.zeros_like(y_val) if y_val is not None else None

    # Normalize each feature in X datasets
    for i in range(num_features):
        X_train_reshaped = X_train[:, :, i].reshape(-1, 1)
        scalers_X[i].fit(X_train_reshaped)
        X_train_sc[:, :, i] = scalers_X[i].transform(X_train_reshaped).reshape(X_train.shape[0], X_train.shape[1])
        X_test_sc[:, :, i] = scalers_X[i].transform(X_test[:, :, i].reshape(-1, 1)).reshape(X_test.shape[0], X_test.shape[1])
        if X_val is not None:
            X_val_sc[:, :, i] = scalers_X[i].transform(X_val[:, :, i].reshape(-1, 1)).reshape(X_val.shape[0], X_val.shape[1])

    # Normalize each feature in y datasets using the same scalers as X
    for i in range(num_features):
        y_train_sc[:, :, i] = scalers_y[i].transform(y_train[:, :, i].reshape(-1, 1)).reshape(y_train.shape[0], y_train.shape[1])
        y_test_sc[:, :, i] = scalers_y[i].transform(y_test[:, :, i].reshape(-1, 1)).reshape(y_test.shape[0], y_test.shape[1])
        if y_val is not None:
            y_val_sc[:, :, i] = scalers_y[i].transform(y_val[:, :, i].reshape(-1, 1)).reshape(y_val.shape[0], y_val.shape[1])

    # Package data into a dictionary, converting to PyTorch tensors
    data_dict = {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_train_sc": torch.tensor(X_train_sc, dtype=torch.float),
        "y_train_sc": torch.tensor(y_train_sc, dtype=torch.float),
        "X_test_sc": torch.tensor(X_test_sc, dtype=torch.float),
        "y_test_sc": torch.tensor(y_test_sc, dtype=torch.float),
    }

    # Include validation data if available
    if X_val is not None and y_val is not None:
        data_dict.update({
            "X_val": X_val, "y_val": y_val,
            "X_val_sc": torch.tensor(X_val_sc, dtype=torch.float) if X_val_sc is not None else None,
            "y_val_sc": torch.tensor(y_val_sc, dtype=torch.float) if y_val_sc is not None else None,
        })

    return data_dict, scalers_X, scalers_y