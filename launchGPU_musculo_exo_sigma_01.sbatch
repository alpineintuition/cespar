#!/bin/bash -l
#SBATCH --job-name="ME_sigma_0.1"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --mem=50GB
#SBATCH --output=out-%j.o
#SBATCH --error=err-%j.e
#SBATCH --account="ich048"

module load daint-gpu

echo $(hostname)
source activate opensim-rl

srun mpirun -np 20 python coupled_gait_optimization_cmaes.py -g 20000 --duration=60 --init_speed=1.6 --tgt_speed=1.3 --sigma-fb=0.1 --file control/params_2D.txt

echo "Program finished"
