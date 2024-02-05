import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from notebooks import local_paths
from utils.load import load_data
from utils.params import set_param

MAIN_DICT = local_paths.MAIN_DIRECTORY # Activate when local_paths works and has been changed for your device
# MAIN_DICT = "/gws/nopw/j04/ai4er/users/pn341/earthquake-predictability"
sys.path.append(MAIN_DICT)

EXPERIMENTS = [
    "b726",
    "b698",
    "i417",
    "p4679",
    "p4581",
    "cascadia",
    "sim_b726",
    "sim_b698",
    "sim_i417",
]

EXPERIMENTS += [f"cascadia_{x}_seg" for x in range(0, 14)]

CPU_COUNT = 8


def create_sequences(data, lookback, forecast):
    xs, ys = [], []
    for i in range(len(data) - lookback - forecast + 1):
        x = data[i : (i + lookback)]
        y = data[(i + lookback) : (i + lookback + forecast)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


class Detrender:
    def __init__(self):
        self.p = None

    def fit(self, x):
        t = np.array(range(0, len(x)))
        self.p = np.polyfit(t, x[:, 0], deg=1)

    def transform(self, x):
        t = np.array(range(0, len(x)))
        x_det = x - (self.p[0] * t + self.p[1]).reshape(-1, 1)
        return x_det

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class SlowEarthquakeDataset:
    def __init__(self, exp_list):
        self.exp_list = exp_list
        self.dataset = {}

    def _load_experiment(self, exp):
        if exp in self.dataset:
            return self.dataset[exp]

        if exp in EXPERIMENTS:
            params = set_param("cascadia" if "_seg" in exp else exp)
            if "_seg" in exp:
                params["segment"] = int(exp.split("_")[1])
            dirs = {"main": MAIN_DICT}
            dirs["data"] = dirs["main"] + local_paths.REL_DATA_DIR + params["dir_data"]
            X, Y, t, dt, vl = load_data(exp, dirs, params)
            dataset = {
                "X": X,
                "Y": Y,
                "t": t,
                "dt": dt,
                "vl": vl,
            }

            if params["data_type"] == "lab":
                dataset["hdrs"] = {
                    "X": "obs_shear_stress",
                    "Y": [
                        "obs_normal_stress",
                        "obs_ecdisp",
                        "obs_shear_strain",
                    ],
                    "t": "time",
                }
            elif params["data_type"] == "synthetic":
                dataset["hdrs"] = {
                    "X": "obs_shear_stress",
                    "Y": [
                        "obs_normal_stress",
                    ],
                    "t": "time",
                }
            elif params["data_type"] == "nature":
                dataset["hdrs"] = {
                    "X": "seg_avg",
                    "Y": [f"seg_{x}" for x in range(Y.shape[1])],
                    "t": "time",
                }
            else:
                dataset["hdrs"] = None
            self.dataset[exp] = dataset
            return dataset

        else:
            raise Exception(f"{exp} not found")

    def load(self):
        max_workers = min(len(self.exp_list), CPU_COUNT)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._load_experiment, exp)
                for exp in self.exp_list
            ]
            [future.result() for idx, future in enumerate(futures)]

    def __getitem__(self, key):
        if key not in self.dataset:
            self.dataset[key] = self._load_experiment(key)
        return self.dataset[key]

    def convert_to_df(self, experiment):
        ds_exp = self.__getitem__(experiment)
        
        X, Y, t = ds_exp["X"], ds_exp["Y"], ds_exp["t"]

        df = pd.DataFrame(
            np.hstack((X, Y, t.reshape(-1, 1))),
            columns=[ds_exp["hdrs"]["X"], *ds_exp["hdrs"]["Y"], ds_exp["hdrs"]["t"]],
            )

        df = df.dropna(axis=1)

        return df



