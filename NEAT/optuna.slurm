#!/bin/bash
#SBATCH --job-name=rstdpmasspole
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --output=output.out
#SBATCH --error=output.err
#SBATCH --cpus-per-task=25


python3 optuna_script.py
rm -rf __pycache__
rm -rf annarchy