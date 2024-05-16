#!/bin/bash
#SBATCH --job-name=dicyt-neat
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Ir al directorio que contiene el ejecutable NEAT
cd /home/XI/bloyola/prueba1

# Ejecutar el ejecutable NEAT
./NEAT