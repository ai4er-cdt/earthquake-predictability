# Import relevant libraries
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
import torch.utils.data as data
from sklearn.metrics import r2_score



class Chomp1d(nn.Module):
    """Removes the trailing chomp_size from the input tensor."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Defines a single temporal block consisting of a convolutional layer, chomping, and activation."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Weight normalized convolutional layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # Chomp1d removes excess padding to ensure output size matches input size
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        return out


class TemporalConvNet(nn.Module):
    """Defines the Temporal Convolutional Network architecture."""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        # Stack multiple temporal blocks
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Append temporal block to the network
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        # Sequentially connect the layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiStepTCN(nn.Module):
    """Multi-step Temporal Convolutional Network model for forecasting."""
    def __init__(self, n_variates, num_channels, kernel_size, output_size, device):
        """
        Initializes the MultiStepTCN model.

        Parameters:
            - n_variates (int): Number of input variables (features).
            - num_channels (list): List of integers specifying the number of channels in each TCN block.
            - kernel_size (int): Size of the convolutional kernel.
            - output_size (int): Number of features in the output/forecasted values.
            - device (str): Device on which the model is being run (e.g., 'cuda' or 'cpu').
        """
        super(MultiStepTCN, self).__init__()
        self.tcn = TemporalConvNet(n_variates, num_channels, kernel_size)
        self.fc1 = nn.Linear(num_channels[-1], 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Performs a forward pass through the MultiStepTCN model.

        Parameters:
            - x (torch.Tensor): Input data tensor with shape (batch_size, seq_length, n_variates).

        Returns:
            - torch.Tensor: Output tensor with shape (batch_size, output_size).
        """
        out = self.tcn(x.transpose(1, 2))  # TCN expects shape (batch_size, input_dim, seq_length)
        out = out[:, :, -1]  # taking the output of the last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_model(model, N_EPOCHS, data_dict, scaler_y, device):
    """
    Train a model for time series forecasting.

    Parameters:
        - model (nn.Module): model to be trained.
        - N_EPOCHS (int): Number of training epochs.
        - data_dict (dict): Dictionary containing training and testing data arrays.
        - scaler_y: Scaler for the target variable.

    Returns:
        - dict: Dictionary containing the training and testing predictions,
              as well as lists of training and testing RMSE and R2 values.
    """

    # Move model and training and testing data to the specified device (cuda or cpu)
    model = model.to(device)
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

