# Import relevant libraries and local modules

import pickle
from dataclasses import dataclass

import tyro

from scripts.models.lstm_oneshot_multistep import MultiStepLSTMMultiLayer
from scripts.models.tcn_oneshot_multistep import MultiStepTCN
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
from utils.nn_train import eval_model_on_test_set, train_model
from utils.paths import MAIN_DIRECTORY, PLOTS_DIR
from utils.plotting import plot_all_data_results, plot_metric_results

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
    exp: str = "b698"
    """experiment name or identifier."""
    record: bool = True
    """flag to indicate whether results should be recorded."""
    plot: bool = True
    """flag to indicate whether to plot the results."""

    # Optuna config options

    optuna: bool = False
    """flag to indicate whether to use optuna for hyperparameter optimization."""
    optuna_id: int = 0
    """optuna study id for saving a study."""

    # Preprocessing config options

    smoothing_window: int = 2
    """moving average window size for data smoothing."""
    downsampling_factor: int = 2
    """factor by which to downsample the data."""
    lookback: int = 180
    """number of past observations to consider for forecasting."""
    forecast: int = 30
    """number of future observations to forecast."""
    n_forecast_windows: int = 15
    """number of forecasted windows in the test set."""
    n_validation_windows: int = 15
    """number of validation windows in the train set."""

    # Model config options

    model: str = "TCN"  # Or "LSTM"
    """model type to use"""
    n_variates: int = 1
    """number of variates in the dataset (e.g., univariate or multivariate)."""
    hidden_size: int = 50
    """size of the hidden layers in the LSTM model."""
    n_layers: int = 1
    """number of layers in the LSTM model."""
    kernel_size: int = 3
    """size of the kernel in the convolutional layers of the TCN model."""
    epochs: int = 75
    """number of epochs for training the model."""
    dropout: float = 0
    """fraction of neurons to drop in model"""

    # Plotting config options

    plot_title: str = "Original Time Series and Model Predictions"
    """title for the plot."""
    plot_xlabel: str = "Time (days)"
    """label for the x-axis of the plot."""
    plot_ylabel: str = "Shear Stress (mPa)"
    """label for the y-axis of the plot."""
    zoom_min: int = 9010
    """minimum x-axis value for zooming in on the plot."""
    zoom_max: int = 9500
    """maximum x-axis value for zooming in on the plot."""
    save_plots: bool = True
    """flag to indicate whether to save the plots."""

    def __post_init__(self):
        self.output_size = self.forecast
        self.zoom_window = [self.zoom_min, self.zoom_max]


args = tyro.cli(ExperimentConfig)

### ------ Set up ------ ###

# Set random seed
set_seed(args.seed)

# Set torch device
device = set_torch_device()


### ------ Load and pre-process simulated data ------ ###

EXP = "b698"

dataset = SlowEarthquakeDataset([EXP])
df = SlowEarthquakeDataset.convert_to_df(dataset, EXP)
df_shear_stress = df["obs_shear_stress"]

# Print sample rate from df['time']
# sample_rate = 1 / np.mean(np.diff(df["time"]))
# print(f"Raw sample rate: {sample_rate}")

# Smooth and pre-process the data into windows
df_smoothed = moving_average_causal_filter(
    df_shear_stress, args.smoothing_window, args.downsampling_factor
)
X, y = create_dataset(df_smoothed, args.lookback, args.forecast)

# Print sample rate from based on the downsampling factor
# downsampled_sample_rate = sample_rate * (1 / args.downsampling_factor)
# print(f"Downsampled sample rate: {downsampled_sample_rate}")

# Split into train and test sets and normalise it
(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
) = split_train_test_forecast_windows(
    X, y, args.forecast, args.n_forecast_windows, args.n_validation_windows
)
data_dict, scaler_X, scaler_y = normalise_dataset(
    X_train, y_train, X_test, y_test, X_val, y_val
)

### ------ Train Models ------ ###

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
results_dict = eval_model_on_test_set(
    model, results_dict, data_dict, scaler_y, device
)

if args.optuna:
    with open(
        f"{MAIN_DIRECTORY}/scripts/tmp/results_dict_{args.optuna_id}.tmp", "wb"
    ) as handle:
        pickle.dump(results_dict, handle)

    args.record = False
    args.plot = False

if args.record:
    model_dir = save_model(
        model,
        df_smoothed.values[-len(y_test) :],
        results_dict,
        range(0, len(y_test)),
        model_name=f"{args.model}_b698",
        model_params=args,
    )

    record_metrics(
        model,
        {"y_test": y_test, "y_pred": results_dict["y_test_pred"]},
        "b698",
        model_dir,
    )


### ------ Plot Results ------ ###

if args.plot:
    # Plot predictions against true values
    plot_all_data_results(
        data_dict,
        results_dict,
        args.lookback,
        args.forecast,
        args.plot_title,
        args.plot_xlabel,
        args.plot_ylabel,
        [],
        save_plot=args.save_plots,
    )

    plot_all_data_results(
        data_dict,
        results_dict,
        args.lookback,
        args.forecast,
        args.plot_title,
        args.plot_xlabel,
        args.plot_ylabel,
        args.zoom_window,
        plot_type="scatter",
        save_plot=args.save_plots,
    )

    # Plot RMSE and R^2
    plot_metric_results(
        args.epochs,
        results_dict["train_rmse_list"],
        results_dict["val_rmse_list"],
        "RMSE",
        args.save_plots,
    )
    plot_metric_results(
        args.epochs,
        results_dict["train_r2_list"],
        results_dict["val_r2_list"],
        "R$^2$",
        args.save_plots,
    )

    if args.save_plots:
        print(f"Plots saved in {PLOTS_DIR}")
