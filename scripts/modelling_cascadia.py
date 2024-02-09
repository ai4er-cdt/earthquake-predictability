# Import relevant libraries

# Import local modules
from scripts.lstm_oneshot_multistep import MultiStepLstmMultiLayer
from scripts.tcn_oneshot_multistep import MultiStepTCN
from utils.data_preprocessing import (
    create_dataset,
    moving_average_causal_filter,
    normalise_dataset,
    split_train_test_forecast_windows,
)
from utils.dataset import SlowEarthquakeDataset
from utils.eval import record_metrics
from utils.general_functions import set_seed, set_torch_device
from utils.nn_io import load_model, save_model
from utils.nn_train import train_model
from utils.plotting import plot_all_data_results, plot_metric_results

###------ Parameters Definition ------###
# General
SEED = 17  # random seed for the dataset and model
EXP = "cascadia_1_seg"

# Data smoothing
SMOOTHING_WINDOW = 5  # moving average window size
DOWNSAMPLING_FACTOR = 1

# Dataset parameters
LOOKBACK, FORECAST = 90, 30  # lookback and forecast values
N_FORECAST_WINDOWS = 40  # n forecasted windows in test set

# Choose model type
RECORDING = True  # Record results? Bool
MODEL = "TCN"  # or "LSTM"

# For general model config
N_VARIATES = 1
OUTPUT_SIZE = FORECAST

# For LSTM config
HIDDEN_SIZE = 50
N_LAYERS = 1

# For TCN config
N_CHANNELS = [64, 64, 64, 64]
KERNEL_SIZE = 3

# For model training
N_EPOCHS = 75

# For plotting results
PLOTTING = True  # Plot graphs? Bool
TITLE = "Original Time Series and Model Predictions of Segment 1 sum"
X_LABEL = "Time (days)"
Y_LABEL = "Displacement potency (?)"
ZOOM_MIN = 3200
ZOOM_MAX = 3400
ZOOM_WINDOW = [ZOOM_MIN, ZOOM_MAX]


###------ Set up ------###
# Set random seed
set_seed(SEED)

# Set torch device
device = set_torch_device()


###------ Load and pre-process data ------###
# Load dataset and convert to dataframe
dataset = SlowEarthquakeDataset([EXP])
df = SlowEarthquakeDataset.convert_to_df(dataset, EXP)
df_seg_1 = df["seg_avg"] / 1e8

# Smooth and pre-process the data into windows
df_smoothed = moving_average_causal_filter(
    df_seg_1, SMOOTHING_WINDOW, DOWNSAMPLING_FACTOR
)
X, y = create_dataset(df_smoothed, LOOKBACK, FORECAST)

# Split into train and test sets and normalise it
X_train, y_train, X_test, y_test = split_train_test_forecast_windows(
    X, y, FORECAST, N_FORECAST_WINDOWS
)
data_dict, scaler_X, scaler_y = normalise_dataset(
    X_train, y_train, X_test, y_test
)


###------ Train LSTM ------###
# Choose model
if MODEL == "LSTM":
    model = MultiStepLstmMultiLayer(
        N_VARIATES, HIDDEN_SIZE, N_LAYERS, OUTPUT_SIZE, device
    )
elif MODEL == "TCN":
    model = MultiStepTCN(
        N_VARIATES, N_CHANNELS, KERNEL_SIZE, OUTPUT_SIZE, device
    )

# Train the modle
results_dict = train_model(model, N_EPOCHS, data_dict, scaler_y, device)

if RECORDING:
    model_dir = save_model(
        model,
        df_smoothed.values[-len(y_test) :],
        results_dict,
        range(0, len(y_test)),
        model_name=f"{MODEL}_cascadia",
        gluon_ts=True,
    )

    model, data = load_model(model, model_dir, gluon_ts=True)

    record_metrics(model, results_dict, model_dir)


###------ Plot Results ------###
if PLOTTING:
    # Plot predictions against true values
    TEST_START_IDX = len(df_smoothed) - len(y_test)
    plot_all_data_results(
        TEST_START_IDX,
        data_dict,
        results_dict,
        LOOKBACK,
        FORECAST,
        TITLE,
        X_LABEL,
        Y_LABEL,
        [],
    )
    plot_all_data_results(
        TEST_START_IDX,
        data_dict,
        results_dict,
        LOOKBACK,
        FORECAST,
        TITLE,
        X_LABEL,
        Y_LABEL,
        ZOOM_WINDOW,
    )

    # Plot RMSE and R2
    plot_metric_results(
        N_EPOCHS,
        results_dict["train_rmse_list"],
        results_dict["test_rmse_list"],
        "RMSE",
    )
    plot_metric_results(
        N_EPOCHS,
        results_dict["train_r2_list"],
        results_dict["test_r2_list"],
        "R$^2$",
    )
