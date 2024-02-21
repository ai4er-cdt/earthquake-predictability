# Import relevant libraries
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from utils.paths import PLOTS_DIR, username

### --------------------------------------------- ###
#         Functions for plotting dataset            #
### --------------------------------------------- ###

def plot_original_vs_processed_data(
    original_df, processed_df, plot_type="scatter", processing_label="Smoothed", save_plot=False
):
    """
    Plots relevant features in the original and processed datasets.

    Parameters:
        original_df (DataFrame): Original dataset to be plotted.
        processed_df (DataFrame): Processed dataset to be plotted.
        plot_type (str): Type of plot. Options: "scatter" or "line".
        processing_label (str): Label describing the processing applied to the data.
        save_plot (bool, optional): If True, saves the plot to the specified directory. Defaults to False.
    """

    # Create a figure to hold the plots
    plt.figure(figsize=(12, 6))

    # Determine the plotting function based on the plot type
    plot_fn = plt.scatter if plot_type == "scatter" else plt.plot


    # Plot original data in the first subplot
    plt.subplot(1, 2, 1)
    plot_fn(range(len(original_df)), original_df, label="Original Data")
    plt.title("Original Data")

    # Plot processed data in the second subplot
    plt.subplot(1, 2, 2)
    plot_fn(
        range(len(processed_df)),
        processed_df,
        label="Processed Data"
    )
    plt.title(f"{processing_label} Data")
    plt.tight_layout()

    # Display the plot
    plt.show()

    # Save the plot if specified
    if save_plot:   
        current_time = datetime.now().isoformat(timespec="seconds")
        plt.savefig(
            f"{PLOTS_DIR}/{username}_{current_time}_og_vs_proc.png",
            bbox_inches="tight",
        )


def plot_example_sample(X, y, select_window, lookback, forecast, plot_type="scatter", save_plot=False):
    """
    Plots an example of input and target data to visualize the lookback and forecast periods.

    Parameters:
        X (array): Input dataset corresponding to the lookack windows.
        y (array): Target dataset corresponding to the forecast windows.
        select_window (int): Index within X from which to start the plot, representing the selected sample.
        lookback (int): Number of time steps to look back in the input data.
        forecast (int): Number of time steps in the forecast period for the target data.
        plot_type (str, optional): Determines the type of plot to create ('scatter' or 'line'). Defaults to "scatter".
        save_plot (bool, optional): If True, saves the plot to a file. Defaults to False.
    """
    plt.figure(figsize=(15, 5))

    # Determine the plotting function based on the plot type
    plot_fn = plt.scatter if plot_type == "scatter" else plt.plot

    # Plot the lookback data
    plot_fn(range(lookback), X[select_window], label="Lookback")
    # Plot the forecast data shifted by the length of the lookback
    plot_fn(
        range(lookback, lookback + forecast),
        y[select_window],
        label="Forecast",
    )
    plt.title("Lookback and forecast of the sample")
    plt.xlabel("Time (days)")
    plt.ylabel("Displacement potency (?)")
    plt.legend()

    # Display the plot
    plt.show()

    # Save the plot if specified
    if save_plot:
        current_time = datetime.now().isoformat(timespec="seconds")
        plt.savefig(
            f"{PLOTS_DIR}/{username}_{current_time}_eg_sample.png",
            bbox_inches="tight",
        )


### --------------------------------------------- ###
#        Functions for plotting model results       #
### --------------------------------------------- ###

def plot_single_seg_result(
        data_dict,
        results_dict,
        lookback,
        forecast,
        chosen_seg,
        title,
        x_label,
        y_label,
        plot_type = "scatter",
        save_plot=False
):
    """
    Plot the true values and testing prediction for a single segment of time series data.

    Parameters:
        data_dict (dict): A dictionary containing 'y_test' key with test data.
        results_dict (dict): A dictionary containing 'y_train_pred' and 'y_test_pred' keys with predictions.
        lookback (int): The number of past observations to consider for a prediction.
        forecast (int): The number of future observations to predict.
        title (str): The title of the plot.
        chosen_seg (int): The index of the segment to plot, e.g. 3200.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        plot_type (str): Type of plot. Options: "scatter" or "line". Defaults to "scatter".
        save_plot (bool, optional): If True, saves the plot to a file. Defaults to False.
    """

    # Convert predictions from PyTorch tensors to NumPy arrays
    train_outputs = results_dict["y_train_pred"].cpu().detach().numpy()
    test_outputs = results_dict["y_test_pred"].cpu().detach().numpy()

    # Initialize segment index and sizes
    i_seg = chosen_seg  # Example segment index
    seg_size = lookback + forecast  # Total size of the segment
    i_seg_test = i_seg - len(train_outputs)  # Index for test segment
    i_seg_lb = i_seg * (forecast + lookback)  # Adjusted index for plotting

    # Create a figure for plotting
    plt.figure(figsize=(15, 6))

    # Determine the plotting function based on the plot type
    plot_fn = plt.scatter if plot_type == "scatter" else plt.plot

    # Plot true values for the segment
    plot_fn(
        np.arange(seg_size) + i_seg_lb,
        data_dict["y_test"][i_seg_test:i_seg_test + seg_size, 0],
        label="True values",
    )
    
    # Plot true values for the forecast period
    plot_fn(
        np.arange(forecast) + (i_seg_lb + seg_size),
        data_dict["y_test"][i_seg_test + seg_size:i_seg_test + seg_size + forecast, 0],
        label="True forecast values",
    )
    
    # Plot testing prediction for the forecast period
    plot_fn(
        np.arange(forecast) + (i_seg_lb + seg_size),
        test_outputs[i_seg_test + seg_size],
        label="Testing prediction",
    )


    # Vertical line indicating the start of the test set
    plt.axvline(
        x=(i_seg_lb+seg_size),
        color="gray",
        linestyle="--",
        label="New forecast window",
    )

    # Set plot title, labels, and legend
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="upper left")

    # Display the plot
    plt.show()

    # Save the plot if specified
    if save_plot:
        current_time = datetime.now().isoformat(timespec="seconds")
        plt.savefig(
            f"{PLOTS_DIR}/{username}_{current_time}_single_window_{i_seg}.png",
            bbox_inches="tight",
        )


def plot_all_data_results(
    data_dict,
    results_dict,
    lookback,
    forecast,
    title,
    x_label,
    y_label,
    zoom_window,
    plot_type = "line",
    save_plot=False
):
    """
    Plot true values, training predictions, and testing predictions for time series data.

    Parameters:
        test_start_index (int): Index where the test set starts.
        data_dict (dict): Dictionary containing training and testing data arrays.
        results_dict (dict): Dictionary containing the training and testing predictions.
        lookback (int): Number of time steps to look back in the data.
        forecast (int): Number of time steps ahead to forecast.
        title (str): Plot title.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        plot_type (str): Type of plot. Options: "scatter" or "line". Defaults to "line".
        zoom_window (tuple): Tuple containing start and end indices for zooming into the plot (optional).
        save_plot (bool, optional): If True, saves the plot to a file. Defaults to False.
    """

    train_outputs = results_dict["y_train_pred"].cpu().detach().numpy()
    test_outputs = results_dict["y_test_pred"].cpu().detach().numpy()

    # Extract every 'forecast' time step for plotting
    train_plot = np.array(
        [train_outputs[idx] for idx in range(0, len(train_outputs), forecast)]
    ).reshape(-1, 1)

    test_plot = np.array(
        [test_outputs[idx] for idx in range(0, len(test_outputs), forecast)]
    ).reshape(-1, 1)

    # Calculate starting indices for test data and forecast
    test_start_index = len(train_plot)
    test_forecast_start_index = len(train_plot) + lookback

    # Combine training and testing data for plotting
    combined_plot = np.concatenate((train_plot, test_plot))

    # Determine the plotting function based on the plot type
    plot_fn = plt.scatter if plot_type == "scatter" else plt.plot

    # Initialize plot
    plt.figure(figsize=(25, 6))

    # Plot true values
    plot_fn(
        range(lookback, lookback + len(combined_plot)),
        np.concatenate(
            (data_dict["y_train"][:, 0], data_dict["y_test"][:, 0])
        ),
        label="True values",
    )
    # Plot training predictions
    plot_fn(
        range(lookback, lookback + len(train_plot)),
        train_plot,
        label="Training prediction",
    )
    # Plot testing predictions
    plot_fn(
        range(lookback + len(train_plot), lookback + len(combined_plot)),
        test_plot,
        label="Testing prediction",
    )

    # Vertical axis to indicate start of test data
    plt.axvline(
        x=test_start_index,
        color="gray",
        label="Test set start",
    )
    # Vertical axis to indicate start of first test forecast window
    plt.axvline(
        x=test_forecast_start_index,
        color="gray",
        linestyle="--",
        label="New forecast window",
    )

    # If a zoom range is provided
    if len(zoom_window) > 0:

        # Add vertical lines to indicate forecast window starts
        n_forecast_windows = int(len(test_plot)/forecast)
        for i in range(n_forecast_windows):
            x = test_forecast_start_index + i * forecast
            plt.axvline(x=x, color="grey", linestyle="--")
        
        # Zoom into the specified range
        plt.xlim(zoom_window[0], zoom_window[1])

    # Set plot title, labels, and legend
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    # Display the plot
    plt.show()

    # Save the plot if specified
    if save_plot:
        current_time = datetime.now().isoformat(timespec="seconds")
        plt.savefig(
            f"{PLOTS_DIR}/{username}_{current_time}_all_data.png",
            bbox_inches="tight",
        )



def plot_metric_results(
    n_epochs, train_metric_list, test_metric_list, metric_label, save_plot=False
):
    """
    Plot a metric over epochs for training and testing sets.

    Parameters:
        n_epochs (int): Number of training epochs.
        train_metric_list (list): List of metric values for each training epoch.
        test_metric_list (list): List of metric values for each testing epoch.
        metric_label (str): Label for the metric being plotted.
        save_plot (bool, optional): If True, saves the plot to a file. Defaults to False.
    """
    # Plot metric over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(0, n_epochs), train_metric_list, label=f"Train {metric_label}"
    )
    plt.plot(
        range(0, n_epochs), test_metric_list, label=f"Test {metric_label}"
    )

    # Set plot labels, title, and legend
    plt.xlabel("Epochs")
    plt.ylabel(f"{metric_label}")
    plt.title(f"{metric_label} over Epochs")
    plt.legend()

    # Display the plot
    plt.show()

    # Save the plot if specified
    if save_plot:
        current_time = datetime.now().isoformat(timespec="seconds")
        plt.savefig(
            f"{PLOTS_DIR}/{username}_{current_time}_metrics.png",
            bbox_inches="tight",
        )
