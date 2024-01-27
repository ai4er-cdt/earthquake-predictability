import getpass
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import torch
from gluonts.model.predictor import Predictor

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
    gluon_ts=False,
):
    current_time = datetime.now().isoformat(timespec="seconds")

    base_filename = model_name if model_name else getpass.getuser()
    base_filename += "_" + current_time

    if gluon_ts:
        model_dir = os.path.join(directory, base_filename + "_gluonts")
        os.makedirs(model_dir)
        data_path = model_dir + "/data.pkl"
        model.serialize(Path(model_dir))
    else:
        model_dir = os.path.join(directory, base_filename + "_torch")
        os.makedirs(model_dir)
        model_path = model_dir + "/model.pt"
        data_path = model_dir + "/data.pkl"
        torch.save(model.state_dict(), model_path)

    data = {
        "input": input,
        "pred": pred,
        "pred_index": pred_index,
    }

    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    print(f"model and data saved to {model_dir}")


def load_model(model, model_dir, gluon_ts=False):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    model_path = model_dir + "/model.pt"
    data_path = model_dir + "/data.pkl"

    if gluon_ts:
        model = Predictor.deserialize(Path(model_dir))
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    print(f"model and data loaded from {model_dir}")

    return model, data
