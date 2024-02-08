# Import relevant libraries
import numpy as np
import matplotlib.pyplot as plt


### --------------------------------------------- ###
#         Functions for plotting dataset            #
### --------------------------------------------- ###

def plot_original_vs_processed_data(original_df, processed_df, plot_type, processing_label="Smoothed"):
    """
    Plots relevant features in the original and processed datasets.

    Parameters:
        original_df (DataFrame): Original dataset to be plotted.
        processed_df (DataFrame): Processed dataset to be plotted.
        processing_label (str): Label describing the processing applied to the data.
        plot_type (str): Type of plot. Options: "scatter" or "line".
    """
    
    # Create a figure to hold the plots
    plt.figure(figsize=(12, 6))

    # Plot scatter plots
    if plot_type == "scatter":
        # Plot original data
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(original_df)), original_df, label="Original Data", s=10)
        plt.title("Original Data")

        # Plot processed data
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(processed_df)), processed_df, label="Processed Data", s=10)
        plt.title(f"{processing_label} Data")

    # Plot line plots
    elif plot_type == "line":
        # Plot original data
        plt.subplot(1, 2, 1)
        plt.plot(range(len(original_df)), original_df, label="Original Data")
        plt.title("Original Data")

        # Plot processed data
        plt.subplot(1, 2, 2)
        plt.plot(range(len(processed_df)), processed_df, label="Processed Data")
        plt.title(f"{processing_label} Data")

    else:
        # Handle unsupported plot types
        print("Unsupported plot type. Choose either 'scatter' or 'line'.")

    # Display the plot
    plt.show()


def plot_example_sample(X, y, select_window, lookback, forecast):
    """
    Plots an example sample of X and y with the specified lookback and forecast.

    Parameters:
        X (array_like): Input data.
        y (array_like): Target data.
        select_window (int): Index of the sample to plot.
        lookback (int): Length of the lookback period.
        forecast (int): Length of the forecast period.
    """
    plt.figure(figsize=(15, 5))
    # Plot the lookback data
    plt.plot(X[select_window], ".", label="Lookback")
    # Plot the forecast data shifted by the length of the lookback
    plt.plot(np.arange(lookback, lookback + forecast), y[select_window], ".", label="Forecast")
    plt.title(f"Lookback and forecast of the sample")
    plt.xlabel("Time (days)")
    plt.ylabel("Displacement potency (?)")
    plt.legend()
    plt.show()



### --------------------------------------------- ###
#        Functions for plotting model results       #
### --------------------------------------------- ###

def plot_all_data_results(test_start_index, data_dict, results_dict, lookback, forecast, title, x_label, y_label, zoom_window):
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
        zoom_window (tuple): Tuple containing start and end indices for zooming into the plot (optional).
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


def plot_metric_results(n_epochs, train_metric_list, test_metric_list, metric_label):
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
