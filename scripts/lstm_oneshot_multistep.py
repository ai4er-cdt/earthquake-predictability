# Import relevant libraries
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import r2_score


# Import local modules
from notebooks import local_paths
from utils.dataset import SlowEarthquakeDataset
import general_functions as gfn

MAIN_DICT = local_paths.MAIN_DIRECTORY
sys.path.append(local_paths.MAIN_DIRECTORY)


class MultiStepLSTM(nn.Module):
    """Subclass of nn.Module"""

    def __init__(
        self, n_variates, hidden_size, n_layers, output_size, device
    ):
        super().__init__()
        self.n_variates = n_variates
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.device = device

        # LSTM layer with specified input size, hidden size, and batch_first
        self.lstm = nn.LSTM(
            input_size=self.n_variates,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
        )

        # Linear layer mapping the LSTM output to the forecasted values
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """Forward pass through the LSTM layer."""
        # Initialise hidden state and cell state
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)

        # LSTM layer
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Extract the last time step output from the LSTM output
        lstm_out = lstm_out[:, -1, :]

        # Linear layer for the final output (forecasted values)
        output = self.linear(lstm_out)

        return output
    

def train_lstm(model, N_EPOCHS, data_dict, scaler_y):

    # Define Adam optimizer and Mean Squared Error (MSE) loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    # Create a DataLoader for training batches
    loader = data.DataLoader(
        data.TensorDataset(data_dict["X_train_sc"], data_dict["y_train_sc"]), shuffle=True, batch_size=32
    )

    # Lists to store RMSE values for plotting
    train_rmse_list = []; test_rmse_list = []
    train_r2_list = []; test_r2_list = []

    # Progress bar length
    pbar = tqdm.tqdm(range(N_EPOCHS))

    # Training loop
    for epoch in tqdm.tqdm(range(N_EPOCHS)):
        model.train()

        # Iterate through batches in the DataLoader
        for X_batch, y_batch in loader:
            # Reshape input for univariate (add a dimension) and model
            y_pred = model(X_batch.unsqueeze(-1))
            loss = loss_fn(y_pred, y_batch)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():  # do not consider gradient in evaluating - no backprop
            # Evaluate model on training data
            y_train_pred = model(data_dict["X_train_sc"].unsqueeze(-1))
            y_train_pred = torch.Tensor(
                scaler_y.inverse_transform(y_train_pred.cpu())
            )
            train_rmse = np.sqrt(loss_fn(y_train_pred, data_dict["y_train"]))
            train_rmse_list.append(train_rmse.item())
            train_r2 = r2_score(data_dict["y_train"], y_train_pred)
            train_r2_list.append(train_r2.item())

            # Evaluate model on testing data
            y_test_pred = model(data_dict["X_test_sc"].unsqueeze(-1))
            y_test_pred = torch.Tensor(
                scaler_y.inverse_transform(y_test_pred.cpu())
            )
            test_rmse = np.sqrt(loss_fn(y_test_pred, data_dict["y_test"]))
            test_rmse_list.append(test_rmse.item())
            test_r2 = r2_score(data_dict["y_test"], y_test_pred)
            test_r2_list.append(test_r2.item())

        # Update progress bar with training and testing RMSE
        pbar.set_description(
            f"Epoch [{epoch+1}/{N_EPOCHS}], Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}, Last Batch Loss: {loss.item():.4f}"
        )

    results_dict = {
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_rmse_list": train_rmse_list,
        "test_rmse_list": test_rmse_list,
        "train_r2_list": train_r2_list,
        "test_r2_list": test_r2_list

    }

    return results_dict

def plot_all_data(test_start_index, data_dict, results_dict, lookback, forecast, title, x_label, y_label, zoom_window):
    train_outputs = results_dict["y_train_pred"]
    test_outputs = results_dict["y_test_pred"]

    train_plot = np.array(
        [train_outputs[idx] for idx in range(0, len(train_outputs), forecast)]
    ).reshape(-1, 1)

    test_plot = np.array(
        [test_outputs[idx] for idx in range(0, len(test_outputs), forecast)]
    ).reshape(-1, 1)

    combined_plot = np.concatenate((train_plot, test_plot))

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

    plt.axvline(
        x=test_start_index, color="gray", linestyle="--", label="Test set start"
    )

    if len(zoom_window)>0:
        plt.xlim(zoom_window[0], zoom_window[1])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def plot_metric(n_epochs, train_metric_list, test_metric_list, metric_label):
    # Plot metric over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, n_epochs), train_metric_list, label=f"Train {metric_label}")
    plt.plot(range(0, n_epochs), test_metric_list, label=f"Test {metric_label}")
    plt.xlabel("Epochs")
    plt.ylabel(f"{metric_label}")
    plt.title(f"{metric_label} over Epochs")
    plt.legend()
    plt.show()