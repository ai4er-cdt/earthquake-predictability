import subprocess

from utils.paths import MAIN_DIRECTORY

n_trials = 20

# Train the tcn with suggested hyperparameters
cmd_tcn = f"python {MAIN_DIRECTORY}/scripts/train_cascadia_with_optuna.py --n_trials_optuna {n_trials} --model {'TCN'}"
subprocess.run(cmd_tcn.split())

# Train the lstm with suggested hyperparameters
cmd_lstm = f"python {MAIN_DIRECTORY}/scripts/train_cascadia_with_optuna.py --n_trials_optuna {n_trials} --model {'LSTM'}"
subprocess.run(cmd_lstm.split())
