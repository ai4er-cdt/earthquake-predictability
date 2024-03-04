import csv
import os
import pickle
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime

import optuna

from utils.paths import MAIN_DIRECTORY, username


@dataclass
class OptunaExperimentConfig:
    """
    Defines the configuration settings for an Optuna-based hyperparameter optimization experiment.

    Attributes:
        n_trials_optuna (int): Number of optimization trials to conduct.
        n_jobs_optuna (int): Number of parallel jobs for executing optimization trials.
        model (str): Type of model to optimize. Supports "TCN" or "LSTM".
        results_format (str): Format to save the results in. Supports "csv" or "pkl".
    """

    n_trials_optuna: int = 4
    n_jobs_optuna: int = 2  # -1 to automatically take number of cores, but keep fixed for JASMIN
    model: str = "Conv2DLSTM"
    results_format: str = (
        "csv"  # Format to save the results in ("csv" or "pkl")
    )


opt_args = OptunaExperimentConfig()


### ------ Optuna Hyperparameter Tuning ------ ###


def objective(trial):
    """
    Objective function that Optuna uses to optimize hyperparameters.

    This function suggests hyperparameters based on the model type specified in the OptunaExperimentConfig,
    trains the model using these parameters, and then returns the model's performance metric to be minimized.

    Args:
        trial (optuna.trial._trial.Trial): A trial object to suggest hyperparameters.

    Returns:
        float: The performance metric (e.g., RMSE) of the model trained with the suggested hyperparameters.
    """
    # Generate a unique ID for each trial to avoid overwriting results
    optuna_id = random.randint(1000000000, 9999999999)

    # Different hyperparameter suggestion logic based on model type
    if opt_args.model == "Conv2DLSTM":
        # Define and suggest hyperparameters for Conv2DLSTM model
        lookback = trial.suggest_int("lookback", 180, 600)
        hidden_size = trial.suggest_categorical(
            "hidden_size", [16, 32, 64, 128]
        )
        kernel_size = trial.suggest_int("kernel_size", 3, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # Train the model with suggested hyperparameters
        cmd = f"python {MAIN_DIRECTORY}/scripts/train_sptmp_cascadia.py --optuna --optuna_id {optuna_id} --lookback {lookback} --model {opt_args.model} --hidden_size {hidden_size} --kernel_size {kernel_size} --dropout {dropout}"
        subprocess.run(cmd.split())

    # Wait for the training script to generate results
    results_path = f"{MAIN_DIRECTORY}/scripts/tmp/results_dict_{optuna_id}.tmp"
    while not os.path.exists(results_path):
        time.sleep(1)  # Check for the file every second

    # Load and return the model's performance metric
    with open(results_path, "rb") as f:
        results_dict = pickle.load(f)

    # Check for validation set
    has_val = "val_rmse_list" in results_dict

    # Use the final test RMSE as the objective to minimize
    final_test_rmse = results_dict[
        f"{'val' if has_val else 'test'}_rmse_list"
    ][-1]

    return final_test_rmse


def run_optuna_optimization():
    """
    Initiates the Optuna optimization process to find the best hyperparameters for the model specified in the OptunaExperimentConfig.

    Creates an Optuna study, executes the specified number of trials with parallel job execution, and returns the best hyperparameters found.

    Returns:
        A dictionary of the best hyperparameters found during the optimization.
    """
    # Create an Optuna study object to minimize the objective function
    study = optuna.create_study(direction="minimize")

    # Print out the number of trials before starting optimization
    print(
        f"Starting Optuna optimization with {opt_args.n_trials_optuna} trials..."
    )

    # Start the optimization process
    study.optimize(
        lambda trial: objective(trial),
        n_trials=opt_args.n_trials_optuna,
        n_jobs=opt_args.n_jobs_optuna,
    )

    # Print the results of the best trial
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params


# Running the optimization process and saving the best result
best_results_dict = run_optuna_optimization()

# Construct filename with user, model type, and current time for uniqueness
optuna_results_dir = f"{MAIN_DIRECTORY}/scripts/optuna_results"
current_time = datetime.now().isoformat(timespec="seconds")
base_filename = f"{username}_best_{opt_args.model}_cascadia_1to6_{current_time}.{opt_args.results_format}"
results_filepath = os.path.join(optuna_results_dir, base_filename)

# Save the best hyperparameters to a file
if opt_args.results_format == "csv":
    # Save the best hyperparameters to a CSV file
    with open(results_filepath, "w") as f:
        writer = csv.writer(f)
        for key, value in best_results_dict.items():
            writer.writerow([key, value])
elif opt_args.results_format == "pkl":
    with open(results_filepath, "wb") as f:
        pickle.dump(best_results_dict, f)
