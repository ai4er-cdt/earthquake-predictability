import os
import pickle
from datetime import datetime
from pathlib import Path

import torch
from gluonts.model.predictor import Predictor

from utils.paths import RESULTS_DIR, username

def save_model(
    model,
    y_test,
    y_pred,
    y_pred_index,
    model_name=None,
    directory=RESULTS_DIR,
    gluon_ts=False,
    uncertainty=None,
    model_params=None,
):
    current_time = datetime.now().isoformat(timespec="seconds")

    base_filename = username
    base_filename += "_" + model_name if model_name else None
    base_filename += "_" + current_time

    if gluon_ts:
        model_dir = os.path.join(directory, base_filename + "_gluonts")
        os.makedirs(model_dir)
        model.serialize(Path(model_dir))
    else:
        model_dir = os.path.join(directory, base_filename + "_torch")
        os.makedirs(model_dir)
        model_path = model_dir + "/model.pt"
        torch.save(model.state_dict(), model_path)

    data_path = model_dir + "/data.pkl"
    params_path = model_dir + "/params.pkl"

    data = {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_index": y_pred_index,
        "uncertainty": uncertainty,
    }

    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    if model_params:
        with open(params_path, "wb") as f:
            pickle.dump(model_params, f)

    print(f"model and data [+ params] saved to {model_dir}")
    return model_dir


def load_model(model, model_dir, gluon_ts=False):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    model_path = model_dir + "/model.pt"
    data_path = model_dir + "/data.pkl"
    params_path = model_dir + "/params.pkl"

    if gluon_ts:
        model = Predictor.deserialize(Path(model_dir))
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    model_params = None
    try:
        with open(params_path, "rb") as f:
            model_params = pickle.load(f)
    except Exception:
        pass

    print(f"model and data [+ params] loaded from {model_dir}")

    return model, data, model_params
