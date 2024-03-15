import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import tqdm
from sklearn.metrics import r2_score

from utils.general_functions import set_seed


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
    batch_size = 32

    # Check if conv2dlstm is the model else keep unsqueeze dim the same
    unsqueeze_dim = -1
    model_class_name = model.__class__.__name__.lower()
    if "conv2d" in model_class_name:
        unsqueeze_dim = 1

    # Training loop
    pbar = tqdm.tqdm(range(n_epochs))
    for epoch in pbar:
        model.train()

        loader = data.DataLoader(
            data.TensorDataset(X_train_sc, y_train_sc),
            shuffle=True,
            batch_size=batch_size,
        )

        for X_batch, y_batch in loader:
            y_pred = model(X_batch.unsqueeze(unsqueeze_dim))
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # success = False
        # while not success:
        #     try:
        #         loader = data.DataLoader(
        #             data.TensorDataset(X_train_sc, y_train_sc), shuffle=True, batch_size=batch_size
        #         )

        #         for X_batch, y_batch in loader:
        #             y_pred = model(X_batch.unsqueeze(unsqueeze_dim))
        #             loss = loss_fn(y_pred, y_batch)

        #             optimizer.zero_grad()
        #             loss.backward()
        #             optimizer.step()

        #         success = True  # Training succeeded
        #     except torch.cuda.OutOfMemoryError:
        #         torch.cuda.empty_cache()  # Attempt to free some memory
        #         if batch_size > 1:
        #             batch_size = max(batch_size // 2, 1)  # Reduce batch size, ensure it's at least 1
        #             print(f"Reducing batch size to {batch_size} due to memory constraints.")
        #         else:
        #             print("Warning: Minimum batch size reached, and out of memory. Skipping epoch.")
        #             success = True  # Allow the pipeline to continue without raising an error
        #             return np.nan

        # Evaluate performance
        model.eval()
        with torch.no_grad():
            # Training data evaluation
            y_train_pred = model(X_train_sc.unsqueeze(unsqueeze_dim))
            y_train_pred_inv = torch.Tensor(
                scaler_y.inverse_transform(
                    y_train_pred.cpu().reshape(-1, 1).detach().numpy()
                )
            )
            y_train_pred_inv = y_train_pred_inv.reshape(
                *data_dict["y_train"].shape
            )
            # train_rmse = np.sqrt(loss_fn(y_train_pred_inv, data_dict["y_train"].to(device))).item()
            train_rmse = np.sqrt(
                loss_fn(y_train_pred_inv.cpu(), data_dict["y_train"].cpu())
            ).item()
            train_r2 = r2_score(
                data_dict["y_train"].reshape(-1, 1),
                y_train_pred_inv.cpu().reshape(-1, 1).detach().numpy(),
            )

            # Update lists and progress bar
            train_rmse_list.append(train_rmse)
            train_r2_list.append(train_r2)

            # Validation data evaluation (if available)
            if has_val:
                y_val_pred = model(X_val_sc.unsqueeze(unsqueeze_dim))
                y_val_pred_inv = torch.Tensor(
                    scaler_y.inverse_transform(
                        y_val_pred.cpu().reshape(-1, 1).detach().numpy()
                    )
                )
                y_val_pred_inv = y_val_pred_inv.reshape(
                    *data_dict["y_val"].shape
                )
                val_rmse = np.sqrt(
                    loss_fn(y_val_pred_inv.cpu(), data_dict["y_val"].cpu())
                ).item()
                val_r2 = r2_score(
                    data_dict["y_val"].reshape(-1, 1),
                    y_val_pred_inv.cpu().reshape(-1, 1).detach().numpy(),
                )

                # Update lists and progress bar
                val_rmse_list.append(val_rmse)
                val_r2_list.append(val_r2)

            else:  # Testing data evaluation
                y_test_pred = model(X_test_sc.unsqueeze(unsqueeze_dim))
                y_test_pred_inv = torch.Tensor(
                    scaler_y.inverse_transform(
                        y_test_pred.cpu().reshape(-1, 1).detach().numpy()
                    )
                )
                y_test_pred_inv = y_test_pred_inv.reshape(
                    *data_dict["y_test"].shape
                )
                test_rmse = np.sqrt(
                    loss_fn(y_test_pred_inv.cpu(), data_dict["y_test"].cpu())
                ).item()
                test_r2 = r2_score(
                    data_dict["y_test"].reshape(-1, 1),
                    y_test_pred_inv.cpu().reshape(-1, 1).detach().numpy(),
                )

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
        results_dict.update(
            {
                "y_val_pred": y_val_pred_inv.cpu(),
                "val_rmse_list": val_rmse_list,
                "val_r2_list": val_r2_list,
            }
        )
    else:  # Add test results
        results_dict.update(
            {
                "y_test_pred": y_test_pred_inv.cpu(),
                "test_rmse_list": test_rmse_list,
                "test_r2_list": test_r2_list,
            }
        )

    return results_dict


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

    # Check if conv2dlstm is the model else keep unsqueeze dim the same
    unsqueeze_dim = -1
    model_class_name = model.__class__.__name__.lower()
    if "conv2d" in model_class_name:
        unsqueeze_dim = 1

    # Predict and inverse transform to original scale
    y_test_pred = model(X_test_sc.unsqueeze(unsqueeze_dim))

    # Define loss function as mean squared error
    loss_fn = nn.MSELoss()

    # Calculate RMSE and R^2
    y_test_pred_inv = torch.Tensor(
        scaler_y.inverse_transform(
            y_test_pred.cpu().reshape(-1, 1).detach().numpy()
        )
    )
    y_test_pred_inv = y_test_pred_inv.reshape(*data_dict["y_test"].shape)

    test_rmse = np.sqrt(
        loss_fn(y_test_pred_inv.cpu(), data_dict["y_test"].cpu())
    ).item()
    test_r2 = r2_score(
        data_dict["y_test"].reshape(-1, 1),
        y_test_pred_inv.cpu().reshape(-1, 1).detach().numpy(),
    )

    # Compile results
    results_dict.update(
        {
            "y_test_pred": y_test_pred_inv.cpu(),
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        }
    )
    return results_dict


def train_model_multi_feature(
    model, n_epochs, data_dict, scaler_y, device, SEED=42
):
    """
    Train a model for time series forecasting, optionally using a validation set.
    *** Adjusted to work with multiple input features ***

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
    set_seed(SEED)

    # Move model to the specified device
    model = model.to(device)

    # Initialize optimizer, loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    min_loss = np.Inf
    best_epoch = 0

    # Prepare data
    X_train_sc = data_dict["X_train_sc"].to(device)
    y_train_sc = data_dict["y_train_sc"].to(device)
    X_val_sc = data_dict["X_val_sc"].to(device)
    X_test_sc = data_dict["X_test_sc"].to(device)
    results_dict = {}

    # Lists for storing metrics
    train_rmse_list, train_r2_list = [], []
    val_rmse_list, val_r2_list = [], []
    test_rmse_list, test_r2_list = [], []

    # DataLoader for training data
    loader = data.DataLoader(
        data.TensorDataset(X_train_sc, y_train_sc), shuffle=True, batch_size=32
    )

    # Training loop
    pbar = tqdm.tqdm(range(n_epochs))
    pbar.set_description("Initialising...")

    for epoch in pbar:
        # Train for one epoch
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate performance
        model.eval()
        with torch.no_grad():
            # Training data evaluation
            y_train_pred = model(X_train_sc)
            y_train_pred_inv = torch.Tensor(
                scaler_y.inverse_transform(y_train_pred.cpu().detach().numpy())
            )
            train_rmse = np.sqrt(
                loss_fn(y_train_pred_inv.cpu(), data_dict["y_train"].cpu())
            ).item()
            train_r2 = r2_score(
                data_dict["y_train"], y_train_pred_inv.cpu().detach().numpy()
            )
            # Update list
            train_rmse_list.append(train_rmse)
            train_r2_list.append(train_r2)

            # Validation data evaluation
            y_val_pred = model(X_val_sc)
            y_val_pred_inv = torch.Tensor(
                scaler_y.inverse_transform(y_val_pred.cpu().detach().numpy())
            )
            val_rmse = np.sqrt(
                loss_fn(y_val_pred_inv.cpu(), data_dict["y_val"].cpu())
            ).item()
            val_r2 = r2_score(
                data_dict["y_val"], y_val_pred_inv.cpu().detach().numpy()
            )
            # Update lists
            val_rmse_list.append(val_rmse)
            val_r2_list.append(val_r2)

            # Test data evaluation
            y_test_pred = model(X_test_sc)
            y_test_pred_inv = torch.Tensor(
                scaler_y.inverse_transform(y_test_pred.cpu().detach().numpy())
            )
            test_rmse = np.sqrt(
                loss_fn(y_test_pred_inv.cpu(), data_dict["y_test"].cpu())
            ).item()
            test_r2 = r2_score(
                data_dict["y_test"], y_test_pred_inv.cpu().detach().numpy()
            )
            # Update lists
            test_rmse_list.append(test_rmse)
            test_r2_list.append(test_r2)

        curr_val_loss = val_rmse

        # Check if this is the best model so far (based on validation loss)
        if curr_val_loss < min_loss:
            min_loss = curr_val_loss
            best_epoch = epoch + 1
            best_state_dict = model.state_dict()
            torch.save(model.state_dict(), "best_model.pt")

            best_train_rmse = train_rmse
            best_train_r2 = train_r2
            best_val_rmse = val_rmse
            best_val_r2 = val_r2
            best_test_rmse = test_rmse
            best_test_r2 = test_r2

            results_dict.update(
                {
                    "best_train_rmse": best_train_rmse,
                    "best_train_r2": best_train_r2,
                    "best_val_rmse": best_val_rmse,
                    "best_val_r2": best_val_r2,
                    "best_test_rmse": best_test_rmse,
                    "best_test_r2": best_test_r2,
                }
            )

        pbar_desc = f"Best Epoch: {best_epoch}, Val RMSE: {min_loss:.4f} | Last Epoch: [{epoch+1}/{n_epochs}], Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}"

        pbar.set_description(pbar_desc)

    model.load_state_dict(best_state_dict)

    # Compile results
    results_dict.update(
        {
            "y_train_pred": y_train_pred_inv.cpu(),
            "train_rmse_list": train_rmse_list,
            "train_r2_list": train_r2_list,
            "y_val_pred": y_val_pred_inv.cpu(),
            "val_rmse_list": val_rmse_list,
            "val_r2_list": val_r2_list,
            "y_test_pred": y_test_pred_inv.cpu(),
            "test_rmse_list": test_rmse_list,
            "test_r2_list": test_r2_list,
        }
    )

    return results_dict


def eval_model_on_test_set_multi_feature(
    model, results_dict, data_dict, scaler_y, device
):
    """
    Evaluate the model on the test and validation set, return RMSE and R^2 metrics for both scaled and original data,
    along with inverse transformed predictions.

    Parameters:
        model (torch.nn.Module): Best trained model.
        best_results_dict (dict): Dictionary containing training and validation predictions and metrics from the best epoch.
        data_dict (dict): Contains scaled features and labels for train, test, and validation sets.
        scaler_y (Scaler): Scaler for inverse transforming predictions.
        device (str): Device for computation ('cpu' or 'cuda').

    Returns:
        dict: Updated best_results_dict with added metrics and predictions for test and validation sets.
    """

    # Load scaled test features
    # X_test_sc = data_dict["X_test_sc"].to(device)

    # Predict and inverse transform to original scale
    # y_test_pred = model(X_test_sc)

    # Define loss function as mean squared error
    loss_fn = nn.MSELoss()

    # Helper function to perform predictions and metric calculations
    def predict_and_calculate_metrics(X_sc, y, y_sc):
        y_pred = model(X_sc.to(device))
        y_pred_inv = torch.Tensor(
            scaler_y.inverse_transform(y_pred.cpu().detach().numpy())
        )

        # RMSE and R^2 for unscaled data
        rmse = np.sqrt(loss_fn(y_pred_inv, y.cpu()).item())
        r2 = r2_score(
            y.cpu().detach().numpy(), y_pred_inv.cpu().detach().numpy()
        )

        # RMSE and R^2 for scaled data
        rmse_sc = np.sqrt(loss_fn(y_pred.detach(), y_sc.to(device)).item())
        r2_sc = r2_score(
            y_sc.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        )

        return y_pred_inv.cpu(), rmse, r2, rmse_sc, r2_sc

    # Predict and calculate metrics for test set
    (
        y_test_pred_inv,
        test_rmse,
        test_r2,
        test_rmse_sc,
        test_r2_sc,
    ) = predict_and_calculate_metrics(
        data_dict["X_test_sc"], data_dict["y_test"], data_dict["y_test_sc"]
    )

    # Predict and calculate metrics for validation set
    (
        y_val_pred_inv,
        val_rmse,
        val_r2,
        val_rmse_sc,
        val_r2_sc,
    ) = predict_and_calculate_metrics(
        data_dict["X_val_sc"], data_dict["y_val"], data_dict["y_val_sc"]
    )

    # Update results dictionary
    results_dict.update(
        {
            "eval_y_test_pred": y_test_pred_inv,
            "eval_test_rmse": test_rmse,
            "eval_test_r2": test_r2,
            "eval_test_rmse_sc": test_rmse_sc,
            "eval_test_r2_sc": test_r2_sc,
            "eval_y_val_pred": y_val_pred_inv,
            "eval_val_rmse": val_rmse,
            "eval_val_r2": val_r2,
            "eval_val_rmse_sc": val_rmse_sc,
            "eval_val_r2_sc": val_r2_sc,
        }
    )

    return results_dict
