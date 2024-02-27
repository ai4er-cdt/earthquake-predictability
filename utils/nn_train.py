import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import tqdm
from sklearn.metrics import r2_score


# def train_model(model, n_epochs, data_dict, scaler_y, device):
#     """
#     Train a model for time series forecasting.

#     Parameters:
#         - model (nn.Module): model to be trained.
#         - n_epochs (int): Number of training epochs.
#         - data_dict (dict): Dictionary containing training and testing data arrays.
#         - scaler_y: Scaler for the target variable.

#     Returns:
#         - dict: Dictionary containing the training and testing predictions,
#               as well as lists of training and testing RMSE and R2 values.
#     """

#     # Declare training device
#     print(f"Training model on {device}")

#     # Move model and training and testing data to the specified device (cuda or cpu)
#     model = model.to(device)
#     X_train_sc = data_dict["X_train_sc"].to(device)
#     y_train_sc = data_dict["y_train_sc"].to(device)
#     X_test_sc = data_dict["X_test_sc"].to(device)
#     # y_test_sc = data_dict["y_test_sc"].to(device)

#     # Define Adam optimizer and Mean Squared Error (MSE) loss function
#     optimizer = optim.Adam(model.parameters())
#     loss_fn = nn.MSELoss()

#     # Create a DataLoader for training batches
#     loader = data.DataLoader(
#         data.TensorDataset(X_train_sc, y_train_sc), shuffle=True, batch_size=32
#     )

#     # Lists to store RMSE values for plotting
#     train_rmse_list = []; test_rmse_list = []
#     train_r2_list = []; test_r2_list = []

#     # Progress bar length
#     pbar = tqdm.tqdm(range(n_epochs))

#     # Training loop
#     for epoch in pbar:
#         model.train()

#         # Iterate through batches in the DataLoader
#         for X_batch, y_batch in loader:
#             # Reshape input for univariate (add a dimension) and model
#             y_pred = model(X_batch.unsqueeze(-1))
#             loss = loss_fn(y_pred, y_batch)

#             # Backward pass and optimization step
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         model.eval()

#         with torch.no_grad():  # do not consider gradient in evaluating - no backprop
#             # Evaluate model on training data
#             y_train_pred = model(X_train_sc.unsqueeze(-1))
#             y_train_pred = torch.Tensor(
#                 scaler_y.inverse_transform(y_train_pred.cpu())
#             )
#             train_rmse = np.sqrt(loss_fn(y_train_pred, data_dict["y_train"]))
#             train_rmse_list.append(train_rmse.item())
#             train_r2 = r2_score(data_dict["y_train"], y_train_pred)
#             train_r2_list.append(train_r2.item())

#             # Evaluate model on testing data
#             y_test_pred = model(X_test_sc.unsqueeze(-1))
#             y_test_pred = torch.Tensor(
#                 scaler_y.inverse_transform(y_test_pred.cpu())
#             )
#             test_rmse = np.sqrt(loss_fn(y_test_pred, data_dict["y_test"]))
#             test_rmse_list.append(test_rmse.item())
#             test_r2 = r2_score(data_dict["y_test"], y_test_pred)
#             test_r2_list.append(test_r2.item())

#         # Update progress bar with training and testing RMSE
#         pbar.set_description(
#             f"Epoch [{epoch+1}/{n_epochs}], RMSE Train: {train_rmse:.4f}, RMSE Test: {test_rmse:.4f}"
#         )

#     results_dict = {
#         "y_train_pred": y_train_pred,
#         "y_test_pred": y_test_pred,
#         "train_rmse_list": train_rmse_list,
#         "test_rmse_list": test_rmse_list,
#         "train_r2_list": train_r2_list,
#         "test_r2_list": test_r2_list,
#     }

#     return results_dict



def train_model(model, n_epochs, data_dict, scaler_y, device):
    """
    Train a model for time series forecasting, optionally using a validation set.

    Parameters:
        model (nn.Module): model to be trained.
        n_epochs (int): Number of training epochs.
        data_dict (dict): Dictionary containing training, optional validation, and testing data arrays.
        scaler_y: Scaler for the target variable.
        device: Device (CPU or CUDA) to train the model on.

    Returns:
        dict: Dictionary containing the training, optional validation, and testing predictions,
                as well as lists of training, validation (if applicable), and testing RMSE and R2 values.
    """

    print(f"Training model on {device}")

    # Move model to the specified device
    model = model.to(device)

    # Prepare data
    X_train_sc = data_dict["X_train_sc"].to(device)
    y_train_sc = data_dict["y_train_sc"].to(device)

    # Lists for storing metrics
    train_rmse_list, train_r2_list = [], []

    # Check for validation set
    has_val = "X_val_sc" in data_dict and "y_val_sc" in data_dict
    if has_val:
        X_val_sc = data_dict["X_val_sc"].to(device)
        val_rmse_list, val_r2_list = [], []
    else:
        X_test_sc = data_dict["X_test_sc"].to(device)
        test_rmse_list, test_r2_list = [], []

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    # DataLoader for training data
    loader = data.DataLoader(data.TensorDataset(X_train_sc, y_train_sc), shuffle=True, batch_size=32)

    # Training loop
    pbar = tqdm.tqdm(range(n_epochs))
    for epoch in pbar:
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch.unsqueeze(-1))
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate performance
        model.eval()
        with torch.no_grad():
            # Training data evaluation
            y_train_pred = model(X_train_sc.unsqueeze(-1))
            y_train_pred_inv = torch.Tensor(scaler_y.inverse_transform(y_train_pred.cpu().detach().numpy()))
            # train_rmse = np.sqrt(loss_fn(y_train_pred_inv, data_dict["y_train"].to(device))).item()
            train_rmse = np.sqrt(loss_fn(y_train_pred_inv.cpu(), data_dict["y_train"].cpu())).item()
            train_r2 = r2_score(data_dict["y_train"], y_train_pred_inv.cpu().detach().numpy())
            
            # Update lists and progress bar
            train_rmse_list.append(train_rmse)
            train_r2_list.append(train_r2)

            # Validation data evaluation (if available)
            if has_val:
                y_val_pred = model(X_val_sc.unsqueeze(-1))
                y_val_pred_inv = torch.Tensor(scaler_y.inverse_transform(y_val_pred.cpu().detach().numpy()))
                # val_rmse = np.sqrt(loss_fn(y_val_pred_inv, data_dict["y_val"].to(device))).item()
                val_rmse = np.sqrt(loss_fn(y_val_pred_inv.cpu(), data_dict["y_val"].cpu())).item()
                val_r2 = r2_score(data_dict["y_val"], y_val_pred_inv.cpu().detach().numpy())

                # Update lists and progress bar
                val_rmse_list.append(val_rmse)
                val_r2_list.append(val_r2)
            
            else: # Testing data evaluation
                y_test_pred = model(X_test_sc.unsqueeze(-1))
                y_test_pred_inv = torch.Tensor(scaler_y.inverse_transform(y_test_pred.cpu().detach().numpy()))
                # test_rmse = np.sqrt(loss_fn(y_test_pred_inv, data_dict["y_test"].to(device))).item()
                test_rmse = np.sqrt(loss_fn(y_test_pred_inv.cpu(), data_dict["y_test"].cpu())).item()
                test_r2 = r2_score(data_dict["y_test"], y_test_pred_inv.cpu().detach().numpy())

                # Update lists and progress bar
                test_rmse_list.append(test_rmse)
                test_r2_list.append(test_r2)


        if has_val:
            pbar_desc = f"Epoch [{epoch+1}/{n_epochs}], Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}"
        else:
            pbar_desc = f"Epoch [{epoch+1}/{n_epochs}], Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}"

        pbar.set_description(pbar_desc)

    # Compile results
    results_dict = {
        "y_train_pred": y_train_pred_inv.cpu(),
        "train_rmse_list": train_rmse_list,
        "train_r2_list": train_r2_list,
    }

    # Add val results
    if has_val:
        results_dict.update({
            "y_val_pred": y_val_pred_inv.cpu(),
            "val_rmse_list": val_rmse_list,
            "val_r2_list": val_r2_list,
        })
    else: # Add test results
        results_dict.update({
            "y_test_pred": y_test_pred_inv.cpu(),
            "test_rmse_list": test_rmse_list,
            "test_r2_list": test_r2_list,
        })

    return results_dict

# TODO: Test the below!!!

def eval_model_on_test_set(model, results_dict, data_dict, scaler_y, device):
    """
    Evaluate the model on the test set and return RMSE and R^2 metrics.

    Parameters:
        model (torch.nn.Module): Trained model.
        results_dict (dict): Dictionary containing training and validation predictions and metrics.
        data_dict (dict): Contains scaled test features ('X_test_sc') and original labels ('y_test').
        scaler_y (MinMaxScaler): Scaler for inverse transforming predictions.
        device (str): Device for computation ('cpu' or 'cuda').

    Returns:
        dict: Contains inverse transformed predictions ('y_test_pred'), RMSE ('test_rmse_list'), and R^2 ('test_r2_list').
    """

    # Load scaled test features
    X_test_sc = data_dict["X_test_sc"].to(device)

    # Predict and inverse transform to original scale
    y_test_pred = model(X_test_sc.unsqueeze(-1))

    # Define loss function as mean squared error
    loss_fn = nn.MSELoss()

    # Calculate RMSE and R^2
    y_test_pred_inv = torch.Tensor(scaler_y.inverse_transform(y_test_pred.cpu().detach().numpy()))
    # test_rmse = np.sqrt(loss_fn(y_test_pred_inv, data_dict["y_test"].to(device))).item()
    test_rmse = np.sqrt(loss_fn(y_test_pred_inv.cpu(), data_dict["y_test"].cpu())).item()
    test_r2 = r2_score(data_dict["y_test"], y_test_pred_inv.cpu().detach().numpy())

    # Compile results
    results_dict.update({
            "y_test_pred": y_test_pred_inv.cpu(),
            "test_rmse_list": test_rmse,
            "test_r2_list": test_r2,
        })
    return results_dict