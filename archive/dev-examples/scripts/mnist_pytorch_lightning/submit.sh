#!/bin/bash
#SBATCH --job-name=mnist-training
#SBATCH --output=/gws/nopw/j04/ai4er/users/pn341/earthquake-predictability/dev-examples/scripts/mnist_pytorch_lightning/output_%j.log
#SBATCH --error=/gws/nopw/j04/ai4er/users/pn341/earthquake-predictability/dev-examples/scripts/mnist_pytorch_lightning/error_%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --partition=orchid
#SBATCH --account=orchid

# Load modules or set environment variables here
conda activate venv

# Run the Python script
srun python /gws/nopw/j04/ai4er/users/pn341/earthquake-predictability/dev-examples/scripts/mnist_pytorch_lightning/main.py

# TO RUN THIS SCRIPT USE:
# sbatch submit.sh
