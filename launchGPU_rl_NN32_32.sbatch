#!/bin/bash -l
#SBATCH --job-name="90"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --mem=50GB
#SBATCH --output=out-90.o
#SBATCH --error=err-90.e
#SBATCH --account="ich048"

module load daint-gpu

echo $(hostname)
source activate opensim-rl

python coupled_exo_optimization_rl.py
echo "Program finished"
