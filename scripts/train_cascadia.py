# Import relevant libraries and local modules

from dataclasses import dataclass

import tyro
from models.lstm_oneshot_multistep import MultiStepLSTMMultiLayer
from models.tcn_oneshot_multistep import MultiStepTCN

from utils.data_preprocessing import (
    create_dataset,
    moving_average_causal_filter,
    normalise_dataset,
    split_train_test_forecast_windows,
)
from utils.dataset import SlowEarthquakeDataset
from utils.eval import record_metrics
from utils.general_functions import set_seed, set_torch_device
from utils.nn_io import save_model
from utils.nn_train import train_model
from utils.plotting import (
    PLOTS_DIRECTORY,
    plot_all_data_results,
    plot_metric_results,
)

# from typing import List


### ------ Parameters Definition ------ ###


@dataclass
class ExperimentConfig:
    """
    Configuration settings for the experiment,
    including general settings, data processing parameters, model specifications,
    and plotting options.
    """

    # General config options

    seed: int = 17
    """random seed for the dataset and model to ensure reproducibility."""
    exp: str = "cascadia_1_seg"
    """experiment name or identifier."""
    record: bool = True
    """flag to indicate whether results should be recorded."""
    plot: bool = True
    """flag to indicate whether to plot the results."""

    # Preprocessing config options

    smoothing_window: int = 5
    """moving average window size for data smoothing."""
    downsampling_factor: int = 1
    """factor by which to downsample the data."""
    lookback: int = 90
    """number of past observations to consider for forecasting."""
    forecast: int = 30
    """number of future observations to forecast."""
    n_forecast_windows: int = 40
    """number of forecasted windows in the test set."""

    # Model config options

    model: str = "TCN"
    """model type to use"""
    n_variates: int = 1
    """number of variates in the dataset (e.g., univariate or multivariate)."""
    hidden_size: int = 50
    """size of the hidden layers in the LSTM model."""
    n_layers: int = 1
    """number of layers in the LSTM model."""
    kernel_size: int = 3
    """size of the kernel in the convolutional layers of the TCN model."""
    epochs: int = 50
    """number of epochs for training the model."""
    dropout: int = 0
    """fraction of neurons to drop in model"""

    # Plotting config options

    plot_title: str = (
        "Original Time Series and Model Predictions of Segment 1 sum"
    )
    """title for the plot."""
    plot_xlabel: str = "Time (days)"
    """label for the x-axis of the plot."""
    plot_ylabel: str = "Displacement potency (?)"
    """label for the y-axis of the plot."""
    zoom_min: int = 3200
    """minimum x-axis value for zooming in on the plot."""
    zoom_max: int = 3400
    """maximum x-axis value for zooming in on the plot."""

    def __post_init__(self):
        self.output_size = self.forecast
        self.zoom_window = [self.zoom_min, self.zoom_max]


args = tyro.cli(ExperimentConfig)

### ------ Set up ------ ###

# Set random seed
set_seed(args.seed)

# Set torch device
device = set_torch_device()


### ------ Load and pre-process data ------ ###

# Load dataset and convert to dataframe
dataset = SlowEarthquakeDataset([args.exp])
df = SlowEarthquakeDataset.convert_to_df(dataset, args.exp)
df_seg_1 = df["seg_avg"] / 1e8

# Smooth and pre-process the data into windows
df_smoothed = moving_average_causal_filter(
    df_seg_1, args.smoothing_window, args.downsampling_factor
)
X, y = create_dataset(df_smoothed, args.lookback, args.forecast)

# Split into train and test sets and normalise it
X_train, y_train, X_test, y_test = split_train_test_forecast_windows(
    X, y, args.forecast, args.n_forecast_windows
)
data_dict, scaler_X, scaler_y = normalise_dataset(
    X_train, y_train, X_test, y_test
)


### ------ Train LSTM ------ ###

# Choose model
if args.model == "LSTM":
    model = MultiStepLSTMMultiLayer(
        args.n_variates,
        args.hidden_size,
        args.n_layers,
        args.output_size,
        device,
    )
elif args.model == "TCN":
    model = MultiStepTCN(
        args.n_variates,
        args.lookback,
        args.output_size,
        [args.hidden_size],
        args.kernel_size,
        args.dropout,
    )

# Train the model
results_dict = train_model(model, args.epochs, data_dict, scaler_y, device)


if args.record:
    model_dir = save_model(
        model,
        df_smoothed.values[-len(y_test) :],
        results_dict,
        range(0, len(y_test)),
        model_name=f"{args.model}_cascadia",
        model_params=args,
    )

    record_metrics(
        model,
        {"y_test": y_test, "y_pred": results_dict["y_test_pred"]},
        "cascadia",
        model_dir,
    )


### ------ Plot Results ------ ###

if args.plot:
    # Plot predictions against true values
    test_start_idx = len(df_smoothed) - len(y_test)

    plot_all_data_results(
        test_start_idx,
        data_dict,
        results_dict,
        args.lookback,
        args.forecast,
        args.plot_title,
        args.plot_xlabel,
        args.plot_ylabel,
        [],
    )

    plot_all_data_results(
        test_start_idx,
        data_dict,
        results_dict,
        args.lookback,
        args.forecast,
        args.plot_title,
        args.plot_xlabel,
        args.plot_ylabel,
        args.zoom_window,
    )

    # Plot RMSE and R^2
    plot_metric_results(
        args.epochs,
        results_dict["train_rmse_list"],
        results_dict["test_rmse_list"],
        "RMSE",
    )
    plot_metric_results(
        args.epochs,
        results_dict["train_r2_list"],
        results_dict["test_r2_list"],
        "R$^2$",
    )

    print(f"plots saved in {PLOTS_DIRECTORY}")
