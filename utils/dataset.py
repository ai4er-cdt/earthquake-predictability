import sys
from concurrent.futures import ProcessPoolExecutor

from utils.load import load_data
from utils.params import set_param

MAIN_DICT = "/gws/nopw/j04/ai4er/users/pn341/earthquake-predictability"
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

CPU_COUNT = 4


class SlowEarthquakeDataset:
    def __init__(self, exp_list):
        self.exp_list = exp_list
        self.dataset = {}

    def _load_experiment(self, exp):
        if exp in self.dataset:
            return self.dataset[exp]

        if exp in EXPERIMENTS:
            params = set_param(exp)
            dirs = {"main": MAIN_DICT}
            dirs["data"] = dirs["main"] + "/data/" + params["dir_data"]
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
                    "X": "det_shear_stress",
                    "Y": [
                        "obs_shear_stress",
                        "obs_normal_stress",
                        "obs_ecdisp",
                        "obs_shear_strain",
                    ],
                    "t": "time",
                }
            elif params["data_type"] == "synthetic":
                dataset["hdrs"] = {
                    "X": "det_shear_stress",
                    "Y": [
                        "det_shear_stress",
                        "det_normal_stress",
                    ],
                    "t": "time",
                }
            elif params["data_type"] == "nature":
                dataset["hdrs"] = {
                    "X": "seg_avg",
                    "Y": [f"seg_{x}" for x in range(196)],
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
