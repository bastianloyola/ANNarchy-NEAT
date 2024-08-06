#!/bin/bash
#SBATCH --job-name=dicyt-t5
#SBATCH --output=/home/XI/bloyola/neat-tests3/tests/NEAT-5/output.txt
#SBATCH --error=/home/XI/bloyola/neat-tests3/tests/NEAT-5/error.txt                            
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Ir al directorio que contiene el ejecutable NEAT
cd /home/XI/bloyola/neat-tests3/tests/NEAT-5
mkdir results

# Ejecutar el ejecutable NEAT
python3 optuna_script.py

# Mover output.txt y error.txt a la carpeta results
mv output.txt results/
mv error.txt results/


# Comandos para limpiar el directorio
rm -rf __pycache__
rm -rf annarchy

