# Import relevant libraries
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import r2_score


class MultiStepLstmSingleLayer(nn.Module):
    """
    A PyTorch neural network model using an LSTM for multi-step time series forecasting.


    Attributes:
        n_variates (int): Number of input variables (features).
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        output_size (int): Number of features in the output/forecasted values.
        device (str): Device on which the model is being run (e.g., 'cuda' or 'cpu').

    Methods:
        forward(x):
            Performs a forward pass through the LSTM layer.

    Example:
        model = MultiStepLSTM(n_variates=5, hidden_size=64, n_layers=2, output_size=1, device='cuda')
    """

    def __init__(self, n_variates, hidden_size, n_layers, output_size, device):
        """
        Initializes the MultiStepLSTM model.

        Parameters:
            - n_variates (int): Number of input variables (features).
            - hidden_size (int): Number of features in the hidden state of the LSTM.
            - n_layers (int): Number of recurrent layers in the LSTM.
            - output_size (int): Number of features in the output/forecasted values.
            - device (str): Device on which the model is being run (e.g., 'cuda' or 'cpu').
        """
        super().__init__()

        # Set model attributes
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
        """
        Performs a forward pass through the LSTM layer.

        Parameters:
            - x (torch.Tensor): Input data tensor with shape (batch_size, seq_length, n_variates).

        Returns:
            - torch.Tensor: Output tensor with shape (batch_size, output_size).
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)

        # LSTM layer
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Extract the last time step output from the LSTM output
        lstm_out = lstm_out[:, -1, :]

        # Linear layer for the final output (forecasted values)
        output = self.linear(lstm_out)

        return output
    


class MultiStepLstmMultiLayer(nn.Module):
    def __init__(self, n_variates, hidden_size, n_layers, output_size, device):
        """
        Initializes the MultiStepLSTM model.

        Parameters:
            - n_variates (int): Number of input variables (features).
            - hidden_size (int): Number of features in the hidden state of the LSTM.
            - n_layers (int): Number of recurrent layers in the LSTM.
            - output_size (int): Number of features in the output/forecasted values.
            - device (str): Device on which the model is being run (e.g., 'cuda' or 'cpu').
        """
        super().__init__()
        # Set model attributes
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

        self.fc1 = nn.Linear(n_layers * hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU6()

    def forward(self, x):
        """
        Performs a forward pass through the LSTM layer.

        Parameters:
            - x (torch.Tensor): Input data tensor with shape (batch_size, seq_length, n_variates).

        Returns:
            - torch.Tensor: Output tensor with shape (batch_size, output_size).
        """
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(
            x.device
        )

        _, (hn, cn) = self.lstm(x, (h0, c0))
        hn = hn.view(x.size(0), self.n_layers * self.hidden_size)
        out = self.relu(hn)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_lstm(model, N_EPOCHS, data_dict, scaler_y, device):
    """
    Train an LSTM model for time series forecasting.

    Parameters:
        - model (nn.Module): LSTM model to be trained.
        - N_EPOCHS (int): Number of training epochs.
        - data_dict (dict): Dictionary containing training and testing data arrays.
        - scaler_y: Scaler for the target variable.

    Returns:
        - dict: Dictionary containing the training and testing predictions,
              as well as lists of training and testing RMSE and R2 values.
    """

    # Move training and testing data to the specified device (cuda or cpu)
    X_train_sc = data_dict["X_train_sc"].to(device)
    y_train_sc = data_dict["y_train_sc"].to(device)
    X_test_sc = data_dict["X_test_sc"].to(device)
    y_test_sc = data_dict["y_test_sc"].to(device)

    # Define Adam optimizer and Mean Squared Error (MSE) loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    # Create a DataLoader for training batches
    loader = data.DataLoader(
        data.TensorDataset(X_train_sc, y_train_sc), shuffle=True, batch_size=32
    )

    # Lists to store RMSE values for plotting
    train_rmse_list = []; test_rmse_list = []
    train_r2_list = []; test_r2_list = []

    # Progress bar length
    pbar = tqdm.tqdm(range(N_EPOCHS))

    # Training loop
    for epoch in pbar:
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
            y_train_pred = model(X_train_sc.unsqueeze(-1))
            y_train_pred = torch.Tensor(
                scaler_y.inverse_transform(y_train_pred.cpu())
            )
            train_rmse = np.sqrt(loss_fn(y_train_pred, data_dict["y_train"]))
            train_rmse_list.append(train_rmse.item())
            train_r2 = r2_score(data_dict["y_train"], y_train_pred)
            train_r2_list.append(train_r2.item())

            # Evaluate model on testing data
            y_test_pred = model(X_test_sc.unsqueeze(-1))
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

