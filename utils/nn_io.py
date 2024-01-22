import getpass
import os
import pickle
import sys
from datetime import datetime

import torch

MAIN_DICT = "/gws/nopw/j04/ai4er/users/pn341/earthquake-predictability"
RESULTS_DIRECTORY = f"{MAIN_DICT}/results"

sys.path.append(MAIN_DICT)


def save_model(
    model,
    input,
    pred,
    pred_index,
    model_name=None,
    directory=RESULTS_DIRECTORY,
):
    current_time = datetime.now().isoformat(timespec="seconds")

    base_filename = model_name if model_name else getpass.getuser()
    base_filename += "_" + current_time

    model_path = os.path.join(directory, base_filename + "_model.pth")
    torch.save(model.state_dict(), model_path)

    data = {
        "input": input,
        "pred": pred,
        "pred_index": pred_index,
    }

    data_path = os.path.join(directory, base_filename + "_data.pt")
    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    print(f"model saved to {model_path}")
    print(f"input, predictions and forecasts saved to {data_path}")


def load_model(model, model_path, data_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    print(f"model loaded from {model_path}")
    print(f"input & predictions loaded from {data_path}")

    return model, data
