# Import relevant libraries
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
import torch.nn as nn
import os

sys.path.append(os.getcwd())

# Import local modules
import notebooks.local_paths as local_paths
from utils.dataset import SlowEarthquakeDataset
import general_functions as gfn
import lstm_oneshot_multistep as lstm

MAIN_DICT = local_paths.MAIN_DIRECTORY
sys.path.append(local_paths.MAIN_DIRECTORY)

###------ Parameters Definition ------###
# General
SEED = 17 # random seed for the dataset and model
EXP = "cascadia_1_seg"

# Data smoothing
SMOOTHING_WINDOW = 5 # moving average window size
DOWNSAMPLING_FACTOR = 1

# Dataset parameters
LOOKBACK, FORECAST = 150, 14 # lookback and forecast values
N_FORECAST_WINDOWS = 40 # n forecasted windows in test set

# For LSTM config
N_VARIATES = 1
HIDDEN_SIZE = 50
N_LAYERS = 1
OUTPUT_SIZE = FORECAST

# For LSTM training
N_EPOCHS = 75

# For plotting results
PLOTTING = True
TITLE = "Original Time Series and Model Predictions of Segment 1 sum"
X_LABEL = "Time (days)"
Y_LABEL = "Displacement potency (?)"
ZOOM_MIN = 3200
ZOOM_MAX = 3400
ZOOM_WINDOW = [ZOOM_MIN, ZOOM_MAX]


###------ Set up ------###
# Set random seed
gfn.set_seed(SEED)

# Set torch device
device = gfn.set_torch_device()


###------ Load and pre-process data ------###
# Load dataset and convert to dataframe
dataset = SlowEarthquakeDataset([EXP])
df = SlowEarthquakeDataset.convert_to_df(dataset, EXP)
df_seg_1 = df["seg_avg"]/1e8

# Smooth and pre-process the data into windows
df_smoothed = gfn.moving_average_causal_filter(df_seg_1, SMOOTHING_WINDOW, DOWNSAMPLING_FACTOR)
X, y = gfn.create_dataset(df_smoothed, LOOKBACK, FORECAST)

# Split into train and test sets and normalise it
X_train, y_train, X_test, y_test = gfn.split_train_test_forecast_windows(X, y, FORECAST, N_FORECAST_WINDOWS)
data_dict, scaler_X, scaler_y = gfn.normalise_dataset(X_train, y_train, X_test, y_test)


###------ Train LSTM ------###
model = lstm.MultiStepLstmSingleLayer(N_VARIATES, HIDDEN_SIZE, N_LAYERS, OUTPUT_SIZE, device).to(device)
results_dict = lstm.train_lstm(model, N_EPOCHS, data_dict, scaler_y, device)


###------ Plot Results ------###
if PLOTTING:
    # Plot predictions against true values
    TEST_START_IDX = len(df_smoothed) - len(y_test)
    gfn.plot_all_data(TEST_START_IDX, data_dict, results_dict, LOOKBACK, FORECAST, TITLE, X_LABEL, Y_LABEL, [])
    gfn.plot_all_data(TEST_START_IDX, data_dict, results_dict, LOOKBACK, FORECAST, TITLE, X_LABEL, Y_LABEL, ZOOM_WINDOW)

    # Plot RMSE and R2
    gfn.plot_metric(N_EPOCHS, results_dict["train_rmse_list"], results_dict["test_rmse_list"], "RMSE")
    gfn.plot_metric(N_EPOCHS, results_dict["train_r2_list"], results_dict["test_r2_list"], "R$^2$")