#!/bin/bash
#SBATCH --job-name=ll-b-iz-i
#SBATCH --output=/home/XI/bloyola/ll/optuna/b-iz-i/output.txt
#SBATCH --error=/home/XI/bloyola/ll/optuna/b-iz-i/error.txt
#SBATCH --partition=batch
#SBATCH --cpus-per-task=50  # Cada tarea usará 4 CPU

# Ir al directorio que contiene el ejecutable NEAT
cd /home/XI/bloyola/ll/optuna/b-iz-i
mkdir results

# Ejecutar el ejecutable NEAT
python3 optuna_script.py

# Mover output.txt y error.txt a la carpeta results
mv output.txt results/
mv error.txt results/
mv example-study.db results/



