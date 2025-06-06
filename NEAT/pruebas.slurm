#!/bin/bash
#SBATCH --job-name=cartpole
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --cpus-per-task=1

# Ejecutar el ejecutable ./NEAT
cd force
python3 force.py
cd ..
cd base-force-rstdp
python3 force-rstdp.py
cd ..
cd force-rstdp-hidden1
python3 force-rstdp.py
cd ..
cd neatcartpole
python3 neatsolo.py
cd ..
cd rstdphidden2
python3 rstdp.py
cd ..
cd force-rstdp-hidden2
python3 force-rstdp.py
cd ..
cd rstdprand1
python3 rstdp.py
cd ..
cd force-rstdp-rand1
python3 force-rstdp.py
cd ..
cd rstdp
python3 rstdp.py
cd ..
cd force-rstdp
python3 force-rstdp.py
cd ..
cd neatbase-rstdp
python3 neatrstdp.py
cd ..
cd rstdphidden1
python3 rstdp.py
cd ..


