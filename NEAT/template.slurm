#!/bin/bash
#SBATCH --job-name=dicyt-t1
#SBATCH --output=/home/XI/gfernandez/neat-tests/tests/NEAT-1/output.txt
#SBATCH --error=/home/XI/gfernandez/neat-tests/tests/NEAT-1/error.txt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Ir al directorio que contiene el ejecutable NEAT
cd /home/XI/gfernandez/neat-tests/tests/NEAT-1
mkdir results

# Ejecutar el ejecutable NEAT
./NEAT

# Mover output.txt y error.txt a la carpeta results
mv output.txt results/
mv error.txt results/

# Generar un nombre único para la carpeta results
RESULTS_DIR=~/neat-tests/runs-results
UNIQUE_RESULTS_DIR="${RESULTS_DIR}/results"
COUNTER=1

while [ -d "${UNIQUE_RESULTS_DIR}" ]; do
    UNIQUE_RESULTS_DIR="${RESULTS_DIR}/results_${COUNTER}"
    COUNTER=$((COUNTER + 1))
done

# Mover la carpeta results a la ubicación deseada con un nombre único
mv results "${UNIQUE_RESULTS_DIR}"

# Comandos para limpiar el directorio
rm -rf __pycache__
rm -rf annarchy
