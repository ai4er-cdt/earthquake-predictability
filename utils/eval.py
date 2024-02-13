import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate

from utils.paths import MAIN_DIRECTORY, username


def record_metrics(model, data, exp_type, model_dir):
    """
    Record metrics of a model and update the leaderboard.

    Args:
    - model: The trained model.
    - data: A dictionary containing test data and predictions.
    - exp_type: Type of experiment (lab, synthetic and cascadia).
    - model_dir: The directory where the model is saved.

    Returns:
    None
    """
    score_dict = {}
    score_dict["user"] = username  # Reusing 'username' from your system
    score_dict["r2"] = r2_score(
        data["y_test"], data["y_pred"]
    )  # Calculate R^2 score
    score_dict["mae"] = mean_absolute_error(
        data["y_test"], data["y_pred"]
    )  # Calculate MAE
    score_dict["rmse"] = np.sqrt(
        mean_squared_error(data["y_test"], data["y_pred"])  # Calculate RMSE
    )
    score_dict["exp_type"] = exp_type  # Save the exp_type being evaluated
    score_dict["model_dir"] = model_dir  # Save the model directory

    csv_file_path = f"{MAIN_DIRECTORY}/leaderboard_{exp_type}.csv"
    try:
        df = pd.read_csv(csv_file_path)  # Try to read the leaderboard CSV file
    except Exception:
        df = (
            pd.DataFrame()
        )  # If file doesn't exist or error reading, create an empty DataFrame

    new_df = pd.DataFrame(
        [score_dict]
    )  # Create a DataFrame from the score dictionary
    combined_df = pd.concat(
        [df, new_df], ignore_index=True
    )  # Concatenate existing DataFrame with new one
    sorted_df = combined_df.sort_values(
        by="r2", ascending=False
    )  # Sort DataFrame by R^2 score
    sorted_df = sorted_df.drop_duplicates(
        subset="model_dir", keep="last"
    ).reset_index(
        drop=True
    )  # Remove duplicate models based on model directory
    sorted_df.to_csv(
        csv_file_path, index=False
    )  # Write updated DataFrame to CSV
    rank = sorted_df.loc[sorted_df["model_dir"] == model_dir].index[
        0
    ]  # Find the rank of the current model

    # Print messages to the user
    print("[1] Thanks for submitting your model to the leaderboard 🍰")
    print(
        f"[2] ../leaderboard_{exp_type}.csv has been updated and sorted by r2_score 🍾"
    )
    print(
        f"[3] Congratulations! you are rank {rank+1} out of {len(sorted_df)} 🥂"
    )
    print(tabulate(new_df.T, tablefmt="psql"))  # Print the new metrics table
