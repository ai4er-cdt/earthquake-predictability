import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate

from utils.paths import MAIN_DIRECTORY, username


def record_metrics(model, data, model_dir):
    score_dict = {}
    score_dict["user"] = username
    score_dict["r2"] = r2_score(data["y_test"], data["y_pred"])
    score_dict["mae"] = mean_absolute_error(data["y_test"], data["y_pred"])
    score_dict["rmse"] = np.sqrt(
        mean_squared_error(data["y_test"], data["y_pred"])
    )
    score_dict["model_dir"] = model_dir

    csv_file_path = f"{MAIN_DIRECTORY}/leaderboard.csv"
    try:
        df = pd.read_csv(csv_file_path)
    except Exception:
        df = pd.DataFrame()

    new_df = pd.DataFrame([score_dict])
    combined_df = pd.concat([df, new_df], ignore_index=True)
    sorted_df = combined_df.sort_values(by="r2", ascending=False)
    sorted_df = sorted_df.drop_duplicates(
        subset="model_dir", keep="last"
    ).reset_index(drop=True)
    sorted_df.to_csv(csv_file_path, index=False)
    rank = sorted_df.loc[sorted_df["model_dir"] == model_dir].index[0]

    print("[1] Thanks for submitting your model to the leaderboard 🍰")
    print("[2] ../leaderboard.csv has been updated and sorted by r2_score 🍾")
    print(
        f"[3] Congratulations! you are rank {rank+1} out of {len(sorted_df)} 🥂"
    )
    print(tabulate(new_df.T, tablefmt="psql"))
