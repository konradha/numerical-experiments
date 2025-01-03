#!/bin/bash -l
#SBATCH --job-name="gpu-benchmark"
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=32G

python benchmark_solvers.py

