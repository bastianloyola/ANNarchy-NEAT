#!/bin/bash
#SBATCH --job-name=acrobotns
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=1-31:1
#SBATCH --output=outputs/job-%A_%a.out
#SBATCH --error=outputs/job-%A_%a.err
#SBATCH --cpus-per-task=50

# Ejecutar el ejecutable ./NEAT
./NEAT $SLURM_ARRAY_TASK_ID
rm -rf __pycache__
rm -rf annarchy



