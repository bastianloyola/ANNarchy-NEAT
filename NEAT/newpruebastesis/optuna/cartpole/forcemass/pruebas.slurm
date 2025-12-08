#!/bin/bash
#SBATCH --job-name=forcemass
#SBATCH --partition=batch
#SBATCH --output=output.out
#SBATCH --error=output.err
#SBATCH --cpus-per-task=1

# Ejecutar el ejecutable ./NEAT
python3 neatsolo.py
rm -rf __pycache__
rm -rf annarchy
