
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import r2_score

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
            f"Epoch [{epoch+1}/{N_EPOCHS}], RMSE Train: {train_rmse:.4f}, RMSE Test: {test_rmse:.4f}, MAE Train: {train_r2:.4f}, MAE Test: {test_r2:.4f},"
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